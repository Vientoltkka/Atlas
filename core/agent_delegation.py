"""Typed deterministic delegation between specialized Atlas agents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import inspect
import json
import math
import types
from types import MappingProxyType
from typing import Any

from core.agent_context import AgentContextBuilder
from core.agent_executor import AgentExecutionRequest, AgentExecutionResult, AgentExecutionStatus, AgentExecutor
from core.agent_registry import AgentDefinition, AgentNotFoundError, AgentRegistry, AgentType, validate_agent_id
from core.agent_resolver import AgentResolutionRequest, AgentResolutionResult, AgentResolutionStatus, AgentResolver


MAX_AGENT_DELEGATION_IDS = 32
MAX_AGENT_DELEGATION_DEPTH = 8
MAX_AGENT_DELEGATIONS = 32
MAX_AGENT_DELEGATION_METADATA_ITEMS = 16
MAX_AGENT_DELEGATION_MAPPING_ITEMS = 32
MAX_AGENT_DELEGATION_SEQUENCE_ITEMS = 32
MAX_AGENT_DELEGATION_TOTAL_ITEMS = 256
MAX_AGENT_DELEGATION_STRING_LENGTH = 1_000
MAX_AGENT_DELEGATION_EVENTS = 32
_PERMISSION_IDS = frozenset(
    {
        "can_read_project",
        "can_write_files",
        "can_execute_tools",
        "can_modify_memory",
        "can_use_network",
        "requires_confirmation",
    }
)
_SENSITIVE_KEY_PARTS = (
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "private_key",
    "credential",
    "prompt",
)


class AgentDelegationError(RuntimeError):
    """Base error for safe agent delegation."""


class InvalidAgentDelegationRequestError(AgentDelegationError):
    """Raised when a delegation request or policy is malformed."""


class AgentDelegationStatus(str, Enum):
    """Structured statuses for one controlled delegation."""

    SUCCESS = "SUCCESS"
    DISABLED = "DISABLED"
    INVALID_REQUEST = "INVALID_REQUEST"
    ORIGIN_AGENT_NOT_FOUND = "ORIGIN_AGENT_NOT_FOUND"
    ORIGIN_AGENT_DISABLED = "ORIGIN_AGENT_DISABLED"
    TARGET_AGENT_NOT_FOUND = "TARGET_AGENT_NOT_FOUND"
    TARGET_AGENT_DISABLED = "TARGET_AGENT_DISABLED"
    SELF_DELEGATION_DENIED = "SELF_DELEGATION_DENIED"
    TARGET_NOT_ALLOWED = "TARGET_NOT_ALLOWED"
    TARGET_DENIED = "TARGET_DENIED"
    MISSING_CAPABILITIES = "MISSING_CAPABILITIES"
    MISSING_PERMISSIONS = "MISSING_PERMISSIONS"
    TYPE_INCOMPATIBLE = "TYPE_INCOMPATIBLE"
    NO_MATCHING_AGENT = "NO_MATCHING_AGENT"
    AMBIGUOUS_AGENT_SELECTION = "AMBIGUOUS_AGENT_SELECTION"
    MAX_DEPTH_REACHED = "MAX_DEPTH_REACHED"
    MAX_DELEGATIONS_REACHED = "MAX_DELEGATIONS_REACHED"
    CONTEXT_REJECTED = "CONTEXT_REJECTED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentDelegationFailureMode(str, Enum):
    """Closed failure handling modes for future extension."""

    FAIL_CLOSED = "FAIL_CLOSED"


@dataclass(frozen=True, slots=True)
class AgentDelegationPolicy:
    """Immutable policy that gates one delegation request."""

    enabled: bool = False
    allow_self_delegation: bool = False
    require_origin_agent_enabled: bool = True
    require_target_agent_enabled: bool = True
    max_delegation_depth: int = 3
    max_total_delegations: int = 8
    allowed_target_agent_ids: tuple[str, ...] = ()
    denied_target_agent_ids: tuple[str, ...] = ()
    allowed_target_agent_types: tuple[AgentType | str, ...] = ()
    required_target_capability_ids: tuple[str, ...] = ()
    required_target_permission_ids: tuple[str, ...] = ()
    propagate_shared_context: bool = False
    propagate_structured_input: bool = True
    propagate_metadata: bool = False
    failure_mode: AgentDelegationFailureMode | str = AgentDelegationFailureMode.FAIL_CLOSED

    def __post_init__(self) -> None:
        for field_name in (
            "enabled",
            "allow_self_delegation",
            "require_origin_agent_enabled",
            "require_target_agent_enabled",
            "propagate_shared_context",
            "propagate_structured_input",
            "propagate_metadata",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise InvalidAgentDelegationRequestError(f"{field_name} must be a bool.")
        object.__setattr__(
            self,
            "max_delegation_depth",
            _bounded_int(self.max_delegation_depth, "max_delegation_depth", MAX_AGENT_DELEGATION_DEPTH),
        )
        object.__setattr__(
            self,
            "max_total_delegations",
            _bounded_int(self.max_total_delegations, "max_total_delegations", MAX_AGENT_DELEGATIONS),
        )
        object.__setattr__(
            self,
            "allowed_target_agent_ids",
            _agent_id_tuple(self.allowed_target_agent_ids, "allowed_target_agent_ids"),
        )
        object.__setattr__(
            self,
            "denied_target_agent_ids",
            _agent_id_tuple(self.denied_target_agent_ids, "denied_target_agent_ids"),
        )
        object.__setattr__(
            self,
            "allowed_target_agent_types",
            _agent_type_tuple(self.allowed_target_agent_types, "allowed_target_agent_types"),
        )
        object.__setattr__(
            self,
            "required_target_capability_ids",
            _identifier_tuple(self.required_target_capability_ids, "required_target_capability_ids"),
        )
        object.__setattr__(
            self,
            "required_target_permission_ids",
            _permission_tuple(self.required_target_permission_ids, "required_target_permission_ids"),
        )
        object.__setattr__(self, "failure_mode", _failure_mode(self.failure_mode))
        overlap = set(self.allowed_target_agent_ids).intersection(self.denied_target_agent_ids)
        if overlap:
            raise InvalidAgentDelegationRequestError("allowed and denied target ids cannot overlap.")


@dataclass(frozen=True, slots=True)
class AgentDelegationRequest:
    """Declarative request for one agent to delegate work to one other agent."""

    origin_agent_id: str
    target_agent_id: str | None = None
    required_agent_types: tuple[AgentType | str, ...] = ()
    required_capability_ids: tuple[str, ...] = ()
    required_permission_ids: tuple[str, ...] = ()
    structured_input: Mapping[str, object] | None = None
    shared_context: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    execution_id: str | None = None
    parent_execution_id: str | None = None
    reason_code: str | None = None
    policy: AgentDelegationPolicy = field(default_factory=AgentDelegationPolicy)
    delegation_depth: int = 0
    delegation_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_agent_id", validate_agent_id(self.origin_agent_id))
        if self.target_agent_id is not None:
            object.__setattr__(self, "target_agent_id", validate_agent_id(self.target_agent_id))
        object.__setattr__(
            self,
            "required_agent_types",
            _agent_type_tuple(self.required_agent_types, "required_agent_types"),
        )
        object.__setattr__(
            self,
            "required_capability_ids",
            _identifier_tuple(self.required_capability_ids, "required_capability_ids"),
        )
        object.__setattr__(
            self,
            "required_permission_ids",
            _permission_tuple(self.required_permission_ids, "required_permission_ids"),
        )
        object.__setattr__(self, "structured_input", _optional_safe_mapping(self.structured_input, "structured_input"))
        object.__setattr__(self, "shared_context", _optional_safe_mapping(self.shared_context, "shared_context"))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))
        if self.execution_id is not None:
            object.__setattr__(self, "execution_id", _identifier(self.execution_id, "execution_id"))
        if self.parent_execution_id is not None:
            object.__setattr__(self, "parent_execution_id", _identifier(self.parent_execution_id, "parent_execution_id"))
        if self.reason_code is not None:
            object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        if not isinstance(self.policy, AgentDelegationPolicy):
            raise InvalidAgentDelegationRequestError("policy must be AgentDelegationPolicy.")
        if isinstance(self.delegation_depth, bool) or not isinstance(self.delegation_depth, int):
            raise InvalidAgentDelegationRequestError("delegation_depth must be an integer.")
        if self.delegation_depth < 0 or self.delegation_depth > MAX_AGENT_DELEGATION_DEPTH:
            raise InvalidAgentDelegationRequestError("delegation_depth is outside the allowed range.")
        path = _agent_id_tuple(self.delegation_path, "delegation_path") if self.delegation_path else (self.origin_agent_id,)
        if path[-1] != self.origin_agent_id:
            raise InvalidAgentDelegationRequestError("delegation_path must end with origin_agent_id.")
        if self.delegation_depth != len(path) - 1:
            raise InvalidAgentDelegationRequestError("delegation_depth is inconsistent with delegation_path.")
        object.__setattr__(self, "delegation_path", path)


@dataclass(frozen=True, slots=True)
class AgentDelegationDecision:
    """Explain the deterministic target chosen for a delegation."""

    target_agent_id: str | None
    status: AgentDelegationStatus
    resolution_result: AgentResolutionResult | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.target_agent_id is not None:
            object.__setattr__(self, "target_agent_id", validate_agent_id(self.target_agent_id))
        object.__setattr__(self, "status", _status(self.status))
        if self.resolution_result is not None and not isinstance(self.resolution_result, AgentResolutionResult):
            raise InvalidAgentDelegationRequestError("resolution_result must be AgentResolutionResult or None.")
        if self.reason is not None:
            object.__setattr__(self, "reason", _safe_message(self.reason))


@dataclass(frozen=True, slots=True)
class AgentDelegationEvent:
    """Safe event emitted by the delegation service."""

    name: str
    status: AgentDelegationStatus
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "event name"))
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(self, "details", MappingProxyType(_safe_mapping(self.details, "event details")))


@dataclass(frozen=True, slots=True)
class AgentDelegationResult:
    """Immutable result for one controlled delegation."""

    status: AgentDelegationStatus
    request_signature: str
    origin_agent_id: str | None = None
    target_agent_id: str | None = None
    execution_id: str | None = None
    parent_execution_id: str | None = None
    delegation_depth: int = 0
    delegation_path: tuple[str, ...] = ()
    resolution_result: AgentDelegationDecision | None = None
    agent_execution_result: AgentExecutionResult | None = None
    safe_output: Mapping[str, object] | None = None
    error_code: str | None = None
    error_message: str | None = None
    events: tuple[AgentDelegationEvent, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        if self.origin_agent_id is not None:
            object.__setattr__(self, "origin_agent_id", validate_agent_id(self.origin_agent_id))
        if self.target_agent_id is not None:
            object.__setattr__(self, "target_agent_id", validate_agent_id(self.target_agent_id))
        if self.execution_id is not None:
            object.__setattr__(self, "execution_id", _identifier(self.execution_id, "execution_id"))
        if self.parent_execution_id is not None:
            object.__setattr__(self, "parent_execution_id", _identifier(self.parent_execution_id, "parent_execution_id"))
        if isinstance(self.delegation_depth, bool) or not isinstance(self.delegation_depth, int):
            raise InvalidAgentDelegationRequestError("delegation_depth must be an integer.")
        object.__setattr__(self, "delegation_path", _agent_id_tuple(self.delegation_path, "delegation_path"))
        if self.safe_output is not None:
            object.__setattr__(self, "safe_output", MappingProxyType(_safe_mapping(self.safe_output, "safe_output")))
        if self.error_message is not None:
            object.__setattr__(self, "error_message", _safe_message(self.error_message))
        object.__setattr__(self, "events", tuple(self.events))
        if not all(isinstance(event, AgentDelegationEvent) for event in self.events):
            raise InvalidAgentDelegationRequestError("events must be AgentDelegationEvent values.")
        object.__setattr__(self, "metrics", MappingProxyType(_metric_mapping(self.metrics)))

    @property
    def success(self) -> bool:
        """Return whether the delegated execution completed successfully."""

        return self.status is AgentDelegationStatus.SUCCESS


class AgentDelegationService:
    """Validate and execute one controlled delegation through AgentExecutor."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        agent_resolver: AgentResolver,
        agent_context_builder: AgentContextBuilder,
        agent_executor: AgentExecutor,
    ) -> None:
        if not isinstance(agent_registry, AgentRegistry):
            raise AgentDelegationError("agent_registry must be AgentRegistry.")
        if not isinstance(agent_resolver, AgentResolver):
            raise AgentDelegationError("agent_resolver must be AgentResolver.")
        if not isinstance(agent_context_builder, AgentContextBuilder):
            raise AgentDelegationError("agent_context_builder must be AgentContextBuilder.")
        if not isinstance(agent_executor, AgentExecutor):
            raise AgentDelegationError("agent_executor must be AgentExecutor.")
        self._agent_registry = agent_registry
        self._agent_resolver = agent_resolver
        self._agent_context_builder = agent_context_builder
        self._agent_executor = agent_executor

    def delegate(
        self,
        request: AgentDelegationRequest,
    ) -> AgentDelegationResult:
        """Validate and execute exactly one target agent through AgentExecutor."""

        events: list[AgentDelegationEvent] = []
        metrics = _base_metrics()
        try:
            if not isinstance(request, AgentDelegationRequest):
                raise InvalidAgentDelegationRequestError("request must be AgentDelegationRequest.")
            signature = agent_delegation_request_signature(request)
        except (AgentDelegationError, TypeError, ValueError) as error:
            metrics["agent_delegations_failed"] = 1
            metrics["agent_delegation_validation_failures"] = 1
            return _result(
                AgentDelegationStatus.INVALID_REQUEST,
                request_signature="",
                events=events,
                metrics=metrics,
                error_code="INVALID_REQUEST",
                error_message=str(error),
            )

        metrics["agent_delegations_requested"] = 1
        events.append(_event("agent_delegation_requested", AgentDelegationStatus.SUCCESS, request=request))
        events.append(_event("agent_delegation_validation_started", AgentDelegationStatus.SUCCESS, request=request))

        try:
            blocked = self._preflight(request, events, metrics)
            if blocked is not None:
                return blocked
            origin = self._agent_registry.get(request.origin_agent_id)
            events.append(_event("agent_delegation_origin_validated", AgentDelegationStatus.SUCCESS, request=request))
            decision = self._resolve_target(request, origin, events, metrics)
            if decision.status is not AgentDelegationStatus.SUCCESS or decision.target_agent_id is None:
                return _blocked_result(
                    decision.status,
                    request,
                    signature,
                    events,
                    metrics,
                    decision=decision,
                    error_code=decision.status.value,
                    error_message=decision.reason or decision.status.value,
                )
            target = self._agent_registry.get(decision.target_agent_id)
            target_check = self._validate_target(request, target)
            if target_check is not None:
                return _blocked_result(
                    target_check,
                    request,
                    signature,
                    events,
                    metrics,
                    decision=decision,
                    target_agent_id=target.agent_id,
                    error_code=target_check.value,
                    error_message=target_check.value,
                )

            execution_request = self._execution_request(request, target)
            events.append(_event("agent_delegation_execution_started", AgentDelegationStatus.SUCCESS, request=request, target=target))
            execution = self._agent_executor.execute(execution_request)
            if execution.status is AgentExecutionStatus.COMPLETED:
                events.append(_event("agent_delegation_execution_succeeded", AgentDelegationStatus.SUCCESS, request=request, target=target))
                events.append(_event("agent_delegation_completed", AgentDelegationStatus.SUCCESS, request=request, target=target))
                metrics["agent_delegations_succeeded"] = 1
                return _result(
                    AgentDelegationStatus.SUCCESS,
                    request_signature=signature,
                    request=request,
                    target_agent_id=target.agent_id,
                    decision=decision,
                    agent_execution_result=execution,
                    safe_output=execution.output,
                    events=events,
                    metrics=metrics,
                )

            status = _status_from_execution(execution.status)
            events.append(_event("agent_delegation_execution_failed", status, request=request, target=target))
            metrics["agent_delegations_failed"] = 1
            metrics["agent_delegation_execution_failures"] = 1
            return _result(
                status,
                request_signature=signature,
                request=request,
                target_agent_id=target.agent_id,
                decision=decision,
                agent_execution_result=execution,
                events=events,
                metrics=metrics,
                error_code=execution.error_code or status.value,
                error_message=execution.safe_message or status.value,
            )
        except AgentNotFoundError as error:
            return _blocked_result(
                AgentDelegationStatus.ORIGIN_AGENT_NOT_FOUND,
                request,
                signature,
                events,
                metrics,
                error_code="ORIGIN_AGENT_NOT_FOUND",
                error_message=str(error),
            )
        except (AgentDelegationError, TypeError, ValueError, RuntimeError) as error:
            metrics["agent_delegations_failed"] = 1
            return _result(
                AgentDelegationStatus.INTERNAL_ERROR,
                request_signature=signature,
                request=request,
                events=events,
                metrics=metrics,
                error_code=type(error).__name__,
                error_message=str(error),
            )

    def _preflight(
        self,
        request: AgentDelegationRequest,
        events: list[AgentDelegationEvent],
        metrics: dict[str, int],
    ) -> AgentDelegationResult | None:
        signature = agent_delegation_request_signature(request)
        policy = request.policy
        if not policy.enabled:
            metrics["agent_delegations_blocked"] = 1
            return _blocked_result(
                AgentDelegationStatus.DISABLED,
                request,
                signature,
                events,
                metrics,
                error_code="DISABLED",
                error_message="delegation policy is disabled.",
            )
        if request.delegation_depth >= policy.max_delegation_depth:
            metrics["agent_delegation_max_depth_reached"] = 1
            return _blocked_result(
                AgentDelegationStatus.MAX_DEPTH_REACHED,
                request,
                signature,
                events,
                metrics,
                error_code="MAX_DEPTH_REACHED",
                error_message="maximum delegation depth reached.",
            )
        if len(request.delegation_path) - 1 >= policy.max_total_delegations:
            metrics["agent_delegation_max_total_reached"] = 1
            return _blocked_result(
                AgentDelegationStatus.MAX_DELEGATIONS_REACHED,
                request,
                signature,
                events,
                metrics,
                error_code="MAX_DELEGATIONS_REACHED",
                error_message="maximum total delegations reached.",
            )
        if not policy.allow_self_delegation and len(set(request.delegation_path)) != len(request.delegation_path):
            return _blocked_result(
                AgentDelegationStatus.INVALID_REQUEST,
                request,
                signature,
                events,
                metrics,
                error_code="INVALID_DELEGATION_PATH",
                error_message="delegation path contains a cycle.",
            )
        try:
            origin = self._agent_registry.get(request.origin_agent_id)
        except AgentNotFoundError as error:
            return _blocked_result(
                AgentDelegationStatus.ORIGIN_AGENT_NOT_FOUND,
                request,
                signature,
                events,
                metrics,
                error_code="ORIGIN_AGENT_NOT_FOUND",
                error_message=str(error),
            )
        if policy.require_origin_agent_enabled and not origin.enabled:
            return _blocked_result(
                AgentDelegationStatus.ORIGIN_AGENT_DISABLED,
                request,
                signature,
                events,
                metrics,
                error_code="ORIGIN_AGENT_DISABLED",
                error_message="origin agent is disabled.",
            )
        return None

    def _resolve_target(
        self,
        request: AgentDelegationRequest,
        origin: AgentDefinition,
        events: list[AgentDelegationEvent],
        metrics: dict[str, int],
    ) -> AgentDelegationDecision:
        events.append(_event("agent_delegation_target_resolution_started", AgentDelegationStatus.SUCCESS, request=request))
        if request.target_agent_id is not None:
            if request.target_agent_id == origin.agent_id and not request.policy.allow_self_delegation:
                metrics["agent_delegation_self_denied"] = 1
                decision = AgentDelegationDecision(
                    request.target_agent_id,
                    AgentDelegationStatus.SELF_DELEGATION_DENIED,
                    reason="self delegation is denied.",
                )
                events.append(_event("agent_delegation_target_resolution_failed", decision.status, request=request))
                return decision
            if request.target_agent_id in request.delegation_path and not request.policy.allow_self_delegation:
                decision = AgentDelegationDecision(
                    request.target_agent_id,
                    AgentDelegationStatus.INVALID_REQUEST,
                    reason="target creates a delegation cycle.",
                )
                events.append(_event("agent_delegation_target_resolution_failed", decision.status, request=request))
                return decision
            try:
                self._agent_registry.get(request.target_agent_id)
            except AgentNotFoundError:
                decision = AgentDelegationDecision(
                    request.target_agent_id,
                    AgentDelegationStatus.TARGET_AGENT_NOT_FOUND,
                    reason="target agent was not found.",
                )
                events.append(_event("agent_delegation_target_resolution_failed", decision.status, request=request))
                return decision
            decision = AgentDelegationDecision(request.target_agent_id, AgentDelegationStatus.SUCCESS, reason="exact target selected.")
            events.append(_event("agent_delegation_target_resolved", AgentDelegationStatus.SUCCESS, request=request, target_id=request.target_agent_id))
            return decision

        resolution_request = self._resolution_request(request)
        resolution = self._agent_resolver.resolve(resolution_request)
        if resolution.status is AgentResolutionStatus.RESOLVED and resolution.selected_agent_id is not None:
            if resolution.selected_agent_id in request.delegation_path and not request.policy.allow_self_delegation:
                decision = AgentDelegationDecision(
                    resolution.selected_agent_id,
                    AgentDelegationStatus.INVALID_REQUEST,
                    resolution,
                    reason="resolved target creates a delegation cycle.",
                )
                events.append(_event("agent_delegation_target_resolution_failed", decision.status, request=request))
                return decision
            decision = AgentDelegationDecision(
                resolution.selected_agent_id,
                AgentDelegationStatus.SUCCESS,
                resolution,
                reason="target resolved by AgentResolver.",
            )
            events.append(_event("agent_delegation_target_resolved", AgentDelegationStatus.SUCCESS, request=request, target_id=resolution.selected_agent_id))
            return decision
        status = _status_from_resolution(resolution.status)
        if status in (AgentDelegationStatus.NO_MATCHING_AGENT, AgentDelegationStatus.AMBIGUOUS_AGENT_SELECTION):
            metrics["agent_delegation_resolution_failures"] = 1
        decision = AgentDelegationDecision(None, status, resolution, reason=resolution.error_message or status.value)
        events.append(_event("agent_delegation_target_resolution_failed", status, request=request))
        return decision

    def _validate_target(
        self,
        request: AgentDelegationRequest,
        target: AgentDefinition,
    ) -> AgentDelegationStatus | None:
        policy = request.policy
        if policy.require_target_agent_enabled and not target.enabled:
            return AgentDelegationStatus.TARGET_AGENT_DISABLED
        if policy.allowed_target_agent_ids and target.agent_id not in policy.allowed_target_agent_ids:
            return AgentDelegationStatus.TARGET_NOT_ALLOWED
        if target.agent_id in policy.denied_target_agent_ids:
            return AgentDelegationStatus.TARGET_DENIED
        required_types = policy.allowed_target_agent_types or request.required_agent_types
        if required_types and target.agent_type not in required_types:
            return AgentDelegationStatus.TYPE_INCOMPATIBLE
        required_capabilities = set(request.required_capability_ids).union(policy.required_target_capability_ids)
        if any(capability not in target.capabilities.capabilities for capability in required_capabilities):
            return AgentDelegationStatus.MISSING_CAPABILITIES
        required_permissions = set(request.required_permission_ids).union(policy.required_target_permission_ids)
        if any(not bool(getattr(target.permissions, permission)) for permission in required_permissions):
            return AgentDelegationStatus.MISSING_PERMISSIONS
        return None

    def _execution_request(
        self,
        request: AgentDelegationRequest,
        target: AgentDefinition,
    ) -> AgentExecutionRequest:
        resolution_request = AgentResolutionRequest(
            required_agent_ids=(target.agent_id,),
            required_capability_ids=tuple(
                sorted(set(request.required_capability_ids).union(request.policy.required_target_capability_ids))
            ),
            required_agent_types=(target.agent_type,),
            required_permission_ids=tuple(
                sorted(set(request.required_permission_ids).union(request.policy.required_target_permission_ids))
            ),
            enabled_only=request.policy.require_target_agent_enabled,
            require_unique_top_score=True,
            metadata={"source": "agent_delegation"},
        )
        metadata: Mapping[str, object] = MappingProxyType({})
        if request.policy.propagate_metadata:
            metadata = request.metadata
        metadata = MappingProxyType(
            {
                **dict(metadata),
                "origin_agent_id": request.origin_agent_id,
                "parent_execution_id": request.parent_execution_id or "",
                "delegation_depth": request.delegation_depth + 1,
                "reason_code": request.reason_code or "",
            }
        )
        return AgentExecutionRequest(
            resolution_request=resolution_request,
            execution_id=request.execution_id,
            correlation_id=request.parent_execution_id,
            structured_input=request.structured_input if request.policy.propagate_structured_input else None,
            shared_context=request.shared_context if request.policy.propagate_shared_context else None,
            metadata=metadata,
            required_capability_ids=resolution_request.required_capability_ids,
            required_permission_ids=resolution_request.required_permission_ids,
        )

    def _resolution_request(self, request: AgentDelegationRequest) -> AgentResolutionRequest:
        required_capabilities = tuple(
            sorted(set(request.required_capability_ids).union(request.policy.required_target_capability_ids))
        )
        required_permissions = tuple(
            sorted(set(request.required_permission_ids).union(request.policy.required_target_permission_ids))
        )
        required_agent_types = request.policy.allowed_target_agent_types or request.required_agent_types
        excluded = tuple(sorted(set(request.policy.denied_target_agent_ids).union(request.delegation_path)))
        return AgentResolutionRequest(
            required_capability_ids=required_capabilities,
            required_permission_ids=required_permissions,
            required_agent_types=required_agent_types,
            required_agent_ids=request.policy.allowed_target_agent_ids,
            excluded_agent_ids=excluded,
            enabled_only=request.policy.require_target_agent_enabled,
            require_unique_top_score=True,
            metadata={"source": "agent_delegation", "origin_agent_id": request.origin_agent_id},
        )


def agent_delegation_request_signature(
    request: AgentDelegationRequest,
) -> str:
    """Return a deterministic SHA-256 signature for a delegation request."""

    if not isinstance(request, AgentDelegationRequest):
        raise InvalidAgentDelegationRequestError("request must be AgentDelegationRequest.")
    payload = {
        "origin_agent_id": request.origin_agent_id,
        "target_agent_id": request.target_agent_id,
        "required_agent_types": sorted(agent_type.value for agent_type in request.required_agent_types),
        "required_capability_ids": sorted(request.required_capability_ids),
        "required_permission_ids": sorted(request.required_permission_ids),
        "structured_input": _jsonable(request.structured_input),
        "shared_context": _jsonable(request.shared_context),
        "metadata": _jsonable(request.metadata),
        "execution_id": request.execution_id,
        "parent_execution_id": request.parent_execution_id,
        "reason_code": request.reason_code,
        "policy": _policy_payload(request.policy),
        "delegation_depth": request.delegation_depth,
        "delegation_path": request.delegation_path,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _blocked_result(
    status: AgentDelegationStatus,
    request: AgentDelegationRequest,
    request_signature: str,
    events: list[AgentDelegationEvent],
    metrics: dict[str, int],
    *,
    decision: AgentDelegationDecision | None = None,
    target_agent_id: str | None = None,
    error_code: str,
    error_message: str,
) -> AgentDelegationResult:
    metrics["agent_delegations_failed"] = 1
    if metrics["agent_delegation_max_depth_reached"] == 0 and status is AgentDelegationStatus.MAX_DEPTH_REACHED:
        metrics["agent_delegation_max_depth_reached"] = 1
    if metrics["agent_delegation_max_total_reached"] == 0 and status is AgentDelegationStatus.MAX_DELEGATIONS_REACHED:
        metrics["agent_delegation_max_total_reached"] = 1
    if status in (
        AgentDelegationStatus.INVALID_REQUEST,
        AgentDelegationStatus.ORIGIN_AGENT_NOT_FOUND,
        AgentDelegationStatus.ORIGIN_AGENT_DISABLED,
        AgentDelegationStatus.TARGET_AGENT_NOT_FOUND,
        AgentDelegationStatus.TARGET_AGENT_DISABLED,
        AgentDelegationStatus.SELF_DELEGATION_DENIED,
        AgentDelegationStatus.TARGET_NOT_ALLOWED,
        AgentDelegationStatus.TARGET_DENIED,
        AgentDelegationStatus.MISSING_CAPABILITIES,
        AgentDelegationStatus.MISSING_PERMISSIONS,
        AgentDelegationStatus.TYPE_INCOMPATIBLE,
        AgentDelegationStatus.MAX_DEPTH_REACHED,
        AgentDelegationStatus.MAX_DELEGATIONS_REACHED,
    ):
        metrics["agent_delegations_blocked"] = 1
    if status is AgentDelegationStatus.INVALID_REQUEST:
        metrics["agent_delegation_validation_failures"] = 1
    events.append(_event("agent_delegation_validation_failed", status, request=request, target_id=target_agent_id))
    events.append(_event("agent_delegation_blocked", status, request=request, target_id=target_agent_id))
    return _result(
        status,
        request_signature=request_signature,
        request=request,
        target_agent_id=target_agent_id,
        decision=decision,
        events=events,
        metrics=metrics,
        error_code=error_code,
        error_message=error_message,
    )


def _result(
    status: AgentDelegationStatus,
    *,
    request_signature: str,
    request: AgentDelegationRequest | None = None,
    target_agent_id: str | None = None,
    decision: AgentDelegationDecision | None = None,
    agent_execution_result: AgentExecutionResult | None = None,
    safe_output: Mapping[str, object] | None = None,
    events: list[AgentDelegationEvent] | tuple[AgentDelegationEvent, ...] = (),
    metrics: Mapping[str, int] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> AgentDelegationResult:
    if len(events) > MAX_AGENT_DELEGATION_EVENTS:
        events = tuple(events)[-MAX_AGENT_DELEGATION_EVENTS:]
    return AgentDelegationResult(
        status=status,
        request_signature=request_signature,
        origin_agent_id=request.origin_agent_id if request is not None else None,
        target_agent_id=target_agent_id,
        execution_id=request.execution_id if request is not None else None,
        parent_execution_id=request.parent_execution_id if request is not None else None,
        delegation_depth=request.delegation_depth if request is not None else 0,
        delegation_path=request.delegation_path if request is not None else (),
        resolution_result=decision,
        agent_execution_result=agent_execution_result,
        safe_output=safe_output,
        error_code=error_code,
        error_message=error_message,
        events=tuple(events),
        metrics=metrics or _base_metrics(),
    )


def _event(
    name: str,
    status: AgentDelegationStatus,
    *,
    request: AgentDelegationRequest | None = None,
    target: AgentDefinition | None = None,
    target_id: str | None = None,
) -> AgentDelegationEvent:
    details: dict[str, object] = {}
    if request is not None:
        details["origin_agent_id"] = request.origin_agent_id
        details["execution_id"] = request.execution_id or ""
        details["delegation_depth"] = request.delegation_depth
    if target is not None:
        details["target_agent_id"] = target.agent_id
    elif target_id is not None:
        details["target_agent_id"] = target_id
    return AgentDelegationEvent(name=name, status=status, details=details)


def _base_metrics() -> dict[str, int]:
    return {
        "agent_delegations_requested": 0,
        "agent_delegations_succeeded": 0,
        "agent_delegations_failed": 0,
        "agent_delegations_blocked": 0,
        "agent_delegation_validation_failures": 0,
        "agent_delegation_resolution_failures": 0,
        "agent_delegation_execution_failures": 0,
        "agent_delegation_self_denied": 0,
        "agent_delegation_max_depth_reached": 0,
        "agent_delegation_max_total_reached": 0,
    }


def _status_from_resolution(status: AgentResolutionStatus) -> AgentDelegationStatus:
    if status is AgentResolutionStatus.AMBIGUOUS:
        return AgentDelegationStatus.AMBIGUOUS_AGENT_SELECTION
    if status in (
        AgentResolutionStatus.NO_AGENTS,
        AgentResolutionStatus.NO_MATCHING_AGENTS,
        AgentResolutionStatus.BELOW_MINIMUM_SCORE,
    ):
        return AgentDelegationStatus.NO_MATCHING_AGENT
    if status is AgentResolutionStatus.INVALID_REQUEST:
        return AgentDelegationStatus.INVALID_REQUEST
    return AgentDelegationStatus.INTERNAL_ERROR


def _status_from_execution(status: AgentExecutionStatus) -> AgentDelegationStatus:
    if status is AgentExecutionStatus.AGENT_DISABLED:
        return AgentDelegationStatus.TARGET_AGENT_DISABLED
    if status in (AgentExecutionStatus.PERMISSION_DENIED,):
        return AgentDelegationStatus.MISSING_PERMISSIONS
    if status in (AgentExecutionStatus.CAPABILITY_NOT_ALLOWED,):
        return AgentDelegationStatus.MISSING_CAPABILITIES
    if status is AgentExecutionStatus.CONTEXT_BUILD_FAILED:
        return AgentDelegationStatus.CONTEXT_REJECTED
    if status in (AgentExecutionStatus.NO_AGENT_CANDIDATES,):
        return AgentDelegationStatus.NO_MATCHING_AGENT
    if status is AgentExecutionStatus.AGENT_AMBIGUOUS:
        return AgentDelegationStatus.AMBIGUOUS_AGENT_SELECTION
    if status in (AgentExecutionStatus.INVALID_REQUEST, AgentExecutionStatus.CANCELLED):
        return AgentDelegationStatus.INVALID_REQUEST
    if status is AgentExecutionStatus.EXECUTION_FAILED:
        return AgentDelegationStatus.EXECUTION_FAILED
    return AgentDelegationStatus.EXECUTION_FAILED


def _bounded_int(value: int, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidAgentDelegationRequestError(f"{field_name} must be an integer.")
    if value <= 0 or value > maximum:
        raise InvalidAgentDelegationRequestError(f"{field_name} is outside the allowed range.")
    return value


def _agent_id_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentDelegationRequestError(f"{field_name} must be an iterable of agent ids.")
    normalized = tuple(dict.fromkeys(validate_agent_id(value) for value in values))
    if len(normalized) > MAX_AGENT_DELEGATION_IDS:
        raise InvalidAgentDelegationRequestError(f"{field_name} has too many items.")
    return normalized


def _identifier_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentDelegationRequestError(f"{field_name} must be an iterable of strings.")
    normalized = tuple(dict.fromkeys(_identifier(value, field_name) for value in values))
    if len(normalized) > MAX_AGENT_DELEGATION_IDS:
        raise InvalidAgentDelegationRequestError(f"{field_name} has too many items.")
    return normalized


def _permission_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    normalized = _identifier_tuple(values, field_name)
    unknown = tuple(value for value in normalized if value not in _PERMISSION_IDS)
    if unknown:
        raise InvalidAgentDelegationRequestError(f"{field_name} contains an unknown permission id.")
    return normalized


def _agent_type_tuple(values: Iterable[AgentType | str], field_name: str) -> tuple[AgentType, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentDelegationRequestError(f"{field_name} must be an iterable.")
    normalized: list[AgentType] = []
    for value in values:
        if isinstance(value, AgentType):
            agent_type = value
        elif isinstance(value, str):
            try:
                agent_type = AgentType(value.strip().lower())
            except ValueError as error:
                raise InvalidAgentDelegationRequestError(f"{field_name} contains an invalid AgentType.") from error
        else:
            raise InvalidAgentDelegationRequestError(f"{field_name} values must be AgentType or str.")
        if agent_type not in normalized:
            normalized.append(agent_type)
    if len(normalized) > MAX_AGENT_DELEGATION_IDS:
        raise InvalidAgentDelegationRequestError(f"{field_name} has too many items.")
    return tuple(normalized)


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidAgentDelegationRequestError(f"{field_name} must be a string.")
    if not value or value.strip() != value:
        raise InvalidAgentDelegationRequestError(f"{field_name} cannot be empty or padded.")
    if "/" in value or "\\" in value or ".." in value:
        raise InvalidAgentDelegationRequestError(f"{field_name} cannot be path-like.")
    if any(ord(character) < 32 for character in value):
        raise InvalidAgentDelegationRequestError(f"{field_name} cannot contain control characters.")
    if not all(character.isalnum() or character in "_.-" for character in value):
        raise InvalidAgentDelegationRequestError(f"{field_name} contains unsupported characters.")
    if len(value) > 128:
        raise InvalidAgentDelegationRequestError(f"{field_name} exceeds the length limit.")
    if _is_sensitive_key(value):
        raise InvalidAgentDelegationRequestError(f"{field_name} cannot contain sensitive content.")
    return value


def _optional_safe_mapping(value: Mapping[str, object] | None, field_name: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return MappingProxyType(_safe_mapping(value, field_name))


def _safe_mapping(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentDelegationRequestError(f"{field_name} must be a mapping.")
    if len(value) > MAX_AGENT_DELEGATION_METADATA_ITEMS and field_name == "metadata":
        raise InvalidAgentDelegationRequestError("metadata has too many items.")
    counter = {"nodes": 0, "chars": 0}
    return _safe_mapping_inner(value, field_name, depth=0, counter=counter)


def _safe_mapping_inner(
    value: Mapping[str, object],
    field_name: str,
    *,
    depth: int,
    counter: dict[str, int],
) -> dict[str, object]:
    if depth > MAX_AGENT_DELEGATION_DEPTH:
        raise InvalidAgentDelegationRequestError(f"{field_name} is too deep.")
    if len(value) > MAX_AGENT_DELEGATION_MAPPING_ITEMS:
        raise InvalidAgentDelegationRequestError(f"{field_name} has too many items.")
    result: dict[str, object] = {}
    for raw_key in sorted(value):
        key = _key(raw_key, field_name)
        if _is_sensitive_key(key):
            raise InvalidAgentDelegationRequestError(f"{field_name} cannot contain sensitive keys.")
        result[key] = _safe_value(value[raw_key], field_name, depth=depth + 1, counter=counter)
    return result


def _safe_sequence(
    value: Sequence[object],
    field_name: str,
    *,
    depth: int,
    counter: dict[str, int],
) -> tuple[object, ...]:
    if len(value) > MAX_AGENT_DELEGATION_SEQUENCE_ITEMS:
        raise InvalidAgentDelegationRequestError(f"{field_name} has too many items.")
    return tuple(_safe_value(item, field_name, depth=depth + 1, counter=counter) for item in value)


def _safe_value(value: object, field_name: str, *, depth: int, counter: dict[str, int]) -> object:
    counter["nodes"] += 1
    if counter["nodes"] > MAX_AGENT_DELEGATION_TOTAL_ITEMS:
        raise InvalidAgentDelegationRequestError(f"{field_name} is too large.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentDelegationRequestError(f"{field_name} floats must be finite.")
        return value
    if isinstance(value, str):
        if len(value) > MAX_AGENT_DELEGATION_STRING_LENGTH:
            raise InvalidAgentDelegationRequestError(f"{field_name} strings are too long.")
        counter["chars"] += len(value)
        if counter["chars"] > MAX_AGENT_DELEGATION_STRING_LENGTH * MAX_AGENT_DELEGATION_MAPPING_ITEMS:
            raise InvalidAgentDelegationRequestError(f"{field_name} string budget exceeded.")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_mapping_inner(value, field_name, depth=depth, counter=counter))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _safe_sequence(value, field_name, depth=depth, counter=counter)
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value) or isinstance(value, types.ModuleType):
        raise InvalidAgentDelegationRequestError(f"{field_name} cannot contain executable objects.")
    raise InvalidAgentDelegationRequestError(f"{field_name} contains an unsupported value.")


def _key(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidAgentDelegationRequestError(f"{field_name} keys must be strings.")
    return _identifier(value, f"{field_name} key")


def _metric_mapping(metrics: Mapping[str, int]) -> dict[str, int]:
    safe: dict[str, int] = {}
    for key, value in metrics.items():
        name = _identifier(str(key), "metric name")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidAgentDelegationRequestError("metric values must be non-negative integers.")
        safe[name] = value
    return safe


def _policy_payload(policy: AgentDelegationPolicy) -> Mapping[str, object]:
    return {
        "enabled": policy.enabled,
        "allow_self_delegation": policy.allow_self_delegation,
        "require_origin_agent_enabled": policy.require_origin_agent_enabled,
        "require_target_agent_enabled": policy.require_target_agent_enabled,
        "max_delegation_depth": policy.max_delegation_depth,
        "max_total_delegations": policy.max_total_delegations,
        "allowed_target_agent_ids": sorted(policy.allowed_target_agent_ids),
        "denied_target_agent_ids": sorted(policy.denied_target_agent_ids),
        "allowed_target_agent_types": sorted(agent_type.value for agent_type in policy.allowed_target_agent_types),
        "required_target_capability_ids": sorted(policy.required_target_capability_ids),
        "required_target_permission_ids": sorted(policy.required_target_permission_ids),
        "propagate_shared_context": policy.propagate_shared_context,
        "propagate_structured_input": policy.propagate_structured_input,
        "propagate_metadata": policy.propagate_metadata,
        "failure_mode": policy.failure_mode.value,
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidAgentDelegationRequestError("non-finite floats are not allowed.")
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value) or isinstance(value, types.ModuleType):
        raise InvalidAgentDelegationRequestError("executable objects are not allowed.")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise InvalidAgentDelegationRequestError("unsupported signature value.")
    return value


def _safe_message(value: str) -> str:
    text = " ".join(str(value).split())
    lowered = text.lower()
    if "[redacted]" in lowered or any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        text = "[redacted]"
    return text[:240]


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _failure_mode(value: AgentDelegationFailureMode | str) -> AgentDelegationFailureMode:
    if isinstance(value, AgentDelegationFailureMode):
        return value
    if isinstance(value, str):
        try:
            return AgentDelegationFailureMode(value.upper())
        except ValueError as error:
            raise InvalidAgentDelegationRequestError("failure_mode is invalid.") from error
    raise InvalidAgentDelegationRequestError("failure_mode must be AgentDelegationFailureMode.")


def _status(value: AgentDelegationStatus | str) -> AgentDelegationStatus:
    if isinstance(value, AgentDelegationStatus):
        return value
    if isinstance(value, str):
        return AgentDelegationStatus(value)
    raise InvalidAgentDelegationRequestError("status must be AgentDelegationStatus.")

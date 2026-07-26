"""Controlled deterministic chained delegation for specialized Atlas agents."""

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

from core.agent_context import AgentContextBuilder
from core.agent_delegation import (
    AgentDelegationPolicy,
    AgentDelegationRequest,
    AgentDelegationResult,
    AgentDelegationService,
    AgentDelegationStatus,
)
from core.agent_executor import AgentExecutor
from core.agent_registry import AgentDefinition, AgentNotFoundError, AgentRegistry, AgentType, validate_agent_id
from core.agent_resolver import AgentResolutionRequest, AgentResolutionStatus, AgentResolver


MAX_CHAIN_STEPS = 20
MAX_CHAIN_DEPTH = 10
MAX_CHAIN_TOTAL_DELEGATIONS = 20
MAX_CHAIN_IDS = 32
MAX_CHAIN_METADATA_ITEMS = 16
MAX_CHAIN_MAPPING_ITEMS = 32
MAX_CHAIN_SEQUENCE_ITEMS = 32
MAX_CHAIN_TOTAL_ITEMS = 256
MAX_CHAIN_STRING_LENGTH = 1_000
MAX_CHAIN_EVENTS = 64
PREVIOUS_OUTPUT_KEY = "previous_delegation_output"
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


class AgentDelegationChainError(RuntimeError):
    """Base error for controlled chained delegation."""


class InvalidAgentDelegationChainRequestError(AgentDelegationChainError):
    """Raised when a chain request or policy is malformed."""


class AgentDelegationChainStatus(str, Enum):
    """Structured terminal statuses for chained delegation."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    DISABLED = "DISABLED"
    INVALID_REQUEST = "INVALID_REQUEST"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    MAX_DEPTH_REACHED = "MAX_DEPTH_REACHED"
    MAX_TOTAL_DELEGATIONS_REACHED = "MAX_TOTAL_DELEGATIONS_REACHED"
    CYCLE_DETECTED = "CYCLE_DETECTED"
    REPEATED_AGENT_DENIED = "REPEATED_AGENT_DENIED"
    SOURCE_AGENT_NOT_FOUND = "SOURCE_AGENT_NOT_FOUND"
    TARGET_AGENT_NOT_FOUND = "TARGET_AGENT_NOT_FOUND"
    TARGET_RESOLUTION_FAILED = "TARGET_RESOLUTION_FAILED"
    TARGET_RESOLUTION_AMBIGUOUS = "TARGET_RESOLUTION_AMBIGUOUS"
    DELEGATION_BLOCKED = "DELEGATION_BLOCKED"
    DELEGATION_FAILED = "DELEGATION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentDelegationChainFailureMode(str, Enum):
    """Conservative failure mode for chain execution."""

    FAIL_CLOSED = "FAIL_CLOSED"


@dataclass(frozen=True, slots=True)
class AgentDelegationChainPolicy:
    """Immutable policy for a declarative delegation chain."""

    enabled: bool = False
    max_steps: int = 5
    max_depth: int = 5
    max_total_delegations: int = 5
    allow_repeated_agents: bool = False
    allow_cycles: bool = False
    stop_on_failure: bool = True
    propagate_previous_output: bool = False
    propagate_shared_context: bool = False
    sanitize_intermediate_outputs: bool = True
    allowed_agent_ids: tuple[str, ...] = ()
    denied_agent_ids: tuple[str, ...] = ()
    allowed_agent_types: tuple[AgentType | str, ...] = ()
    denied_agent_types: tuple[AgentType | str, ...] = ()
    failure_mode: AgentDelegationChainFailureMode | str = AgentDelegationChainFailureMode.FAIL_CLOSED

    def __post_init__(self) -> None:
        for field_name in (
            "enabled",
            "allow_repeated_agents",
            "allow_cycles",
            "stop_on_failure",
            "propagate_previous_output",
            "propagate_shared_context",
            "sanitize_intermediate_outputs",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise InvalidAgentDelegationChainRequestError(f"{field_name} must be a bool.")
        object.__setattr__(self, "max_steps", _bounded_int(self.max_steps, "max_steps", MAX_CHAIN_STEPS))
        object.__setattr__(self, "max_depth", _bounded_int(self.max_depth, "max_depth", MAX_CHAIN_DEPTH))
        object.__setattr__(
            self,
            "max_total_delegations",
            _bounded_int(self.max_total_delegations, "max_total_delegations", MAX_CHAIN_TOTAL_DELEGATIONS),
        )
        object.__setattr__(self, "allowed_agent_ids", _agent_id_tuple(self.allowed_agent_ids, "allowed_agent_ids"))
        object.__setattr__(self, "denied_agent_ids", _agent_id_tuple(self.denied_agent_ids, "denied_agent_ids"))
        object.__setattr__(self, "allowed_agent_types", _agent_type_tuple(self.allowed_agent_types, "allowed_agent_types"))
        object.__setattr__(self, "denied_agent_types", _agent_type_tuple(self.denied_agent_types, "denied_agent_types"))
        object.__setattr__(self, "failure_mode", _failure_mode(self.failure_mode))
        if set(self.allowed_agent_ids).intersection(self.denied_agent_ids):
            raise InvalidAgentDelegationChainRequestError("allowed and denied agent ids cannot overlap.")
        if set(self.allowed_agent_types).intersection(self.denied_agent_types):
            raise InvalidAgentDelegationChainRequestError("allowed and denied agent types cannot overlap.")


@dataclass(frozen=True, slots=True)
class AgentDelegationChainStep:
    """One declarative step in a finite delegation chain."""

    source_agent_id: str
    target_agent_id: str | None = None
    required_agent_types: tuple[AgentType | str, ...] = ()
    required_capability_ids: tuple[str, ...] = ()
    required_permission_ids: tuple[str, ...] = ()
    preferred_agent_ids: tuple[str, ...] = ()
    preferred_agent_types: tuple[AgentType | str, ...] = ()
    excluded_agent_ids: tuple[str, ...] = ()
    enabled_only: bool = True
    structured_input: Mapping[str, object] | None = None
    shared_context: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    execution_required_capability_ids: tuple[str, ...] = ()
    execution_required_permission_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_agent_id", validate_agent_id(self.source_agent_id))
        if self.target_agent_id is not None:
            object.__setattr__(self, "target_agent_id", validate_agent_id(self.target_agent_id))
        object.__setattr__(self, "required_agent_types", _agent_type_tuple(self.required_agent_types, "required_agent_types"))
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
        object.__setattr__(self, "preferred_agent_ids", _agent_id_tuple(self.preferred_agent_ids, "preferred_agent_ids"))
        object.__setattr__(
            self,
            "preferred_agent_types",
            _agent_type_tuple(self.preferred_agent_types, "preferred_agent_types"),
        )
        object.__setattr__(self, "excluded_agent_ids", _agent_id_tuple(self.excluded_agent_ids, "excluded_agent_ids"))
        if type(self.enabled_only) is not bool:
            raise InvalidAgentDelegationChainRequestError("enabled_only must be a bool.")
        object.__setattr__(self, "structured_input", _optional_safe_mapping(self.structured_input, "structured_input"))
        object.__setattr__(self, "shared_context", _optional_safe_mapping(self.shared_context, "shared_context"))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))
        object.__setattr__(
            self,
            "execution_required_capability_ids",
            _identifier_tuple(self.execution_required_capability_ids, "execution_required_capability_ids"),
        )
        object.__setattr__(
            self,
            "execution_required_permission_ids",
            _permission_tuple(self.execution_required_permission_ids, "execution_required_permission_ids"),
        )
        automatic = bool(
            self.required_agent_types
            or self.required_capability_ids
            or self.required_permission_ids
            or self.preferred_agent_ids
            or self.preferred_agent_types
        )
        if self.target_agent_id is not None and automatic:
            raise InvalidAgentDelegationChainRequestError("step cannot mix explicit target and automatic criteria.")
        if self.target_agent_id is None and not automatic:
            raise InvalidAgentDelegationChainRequestError("step requires a target_agent_id or resolution criteria.")


@dataclass(frozen=True, slots=True)
class AgentDelegationChainRequest:
    """Declarative finite chain of delegation steps."""

    steps: Sequence[AgentDelegationChainStep]
    policy: AgentDelegationChainPolicy = field(default_factory=AgentDelegationChainPolicy)
    initial_input: Mapping[str, object] | None = None
    shared_context: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    execution_id: str | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.steps, (str, bytes)) or not isinstance(self.steps, Sequence):
            raise InvalidAgentDelegationChainRequestError("steps must be a sequence.")
        steps = tuple(self.steps)
        if not steps:
            raise InvalidAgentDelegationChainRequestError("steps cannot be empty.")
        if len(steps) > MAX_CHAIN_STEPS:
            raise InvalidAgentDelegationChainRequestError("steps exceeds the absolute limit.")
        if not all(isinstance(step, AgentDelegationChainStep) for step in steps):
            raise InvalidAgentDelegationChainRequestError("steps must contain AgentDelegationChainStep values.")
        object.__setattr__(self, "steps", steps)
        if not isinstance(self.policy, AgentDelegationChainPolicy):
            raise InvalidAgentDelegationChainRequestError("policy must be AgentDelegationChainPolicy.")
        object.__setattr__(self, "initial_input", _optional_safe_mapping(self.initial_input, "initial_input"))
        object.__setattr__(self, "shared_context", _optional_safe_mapping(self.shared_context, "shared_context"))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))
        if self.execution_id is not None:
            object.__setattr__(self, "execution_id", _identifier(self.execution_id, "execution_id"))
        if self.correlation_id is not None:
            object.__setattr__(self, "correlation_id", _identifier(self.correlation_id, "correlation_id"))


@dataclass(frozen=True, slots=True)
class AgentDelegationChainEvent:
    """Safe event emitted by the chain service."""

    name: str
    status: AgentDelegationChainStatus
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "event name"))
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(self, "details", MappingProxyType(_safe_mapping(self.details, "event details")))


@dataclass(frozen=True, slots=True)
class AgentDelegationChainStepResult:
    """Structured result for one chain step."""

    step_index: int
    source_agent_id: str
    resolved_target_agent_id: str | None
    status: AgentDelegationChainStatus
    delegation_result: AgentDelegationResult | None = None
    safe_output: Mapping[str, object] | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0:
            raise InvalidAgentDelegationChainRequestError("step_index must be a non-negative integer.")
        object.__setattr__(self, "source_agent_id", validate_agent_id(self.source_agent_id))
        if self.resolved_target_agent_id is not None:
            object.__setattr__(self, "resolved_target_agent_id", validate_agent_id(self.resolved_target_agent_id))
        object.__setattr__(self, "status", _status(self.status))
        if self.safe_output is not None:
            object.__setattr__(self, "safe_output", MappingProxyType(_safe_mapping(self.safe_output, "safe_output")))
        if self.error_message is not None:
            object.__setattr__(self, "error_message", _safe_message(self.error_message))


@dataclass(frozen=True, slots=True)
class AgentDelegationChainResult:
    """Immutable result for a finite delegation chain."""

    status: AgentDelegationChainStatus
    request_signature: str
    completed_steps: int = 0
    failed_step_index: int | None = None
    step_results: tuple[AgentDelegationChainStepResult, ...] = ()
    final_output: Mapping[str, object] | None = None
    execution_id: str | None = None
    correlation_id: str | None = None
    events: tuple[AgentDelegationChainEvent, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        if isinstance(self.completed_steps, bool) or not isinstance(self.completed_steps, int) or self.completed_steps < 0:
            raise InvalidAgentDelegationChainRequestError("completed_steps must be a non-negative integer.")
        if self.failed_step_index is not None and (
            isinstance(self.failed_step_index, bool)
            or not isinstance(self.failed_step_index, int)
            or self.failed_step_index < 0
        ):
            raise InvalidAgentDelegationChainRequestError("failed_step_index must be a non-negative integer or None.")
        object.__setattr__(self, "step_results", tuple(self.step_results))
        if not all(isinstance(result, AgentDelegationChainStepResult) for result in self.step_results):
            raise InvalidAgentDelegationChainRequestError("step_results must contain AgentDelegationChainStepResult values.")
        if self.final_output is not None:
            object.__setattr__(self, "final_output", MappingProxyType(_safe_mapping(self.final_output, "final_output")))
        if self.execution_id is not None:
            object.__setattr__(self, "execution_id", _identifier(self.execution_id, "execution_id"))
        if self.correlation_id is not None:
            object.__setattr__(self, "correlation_id", _identifier(self.correlation_id, "correlation_id"))
        object.__setattr__(self, "events", tuple(self.events))
        if not all(isinstance(event, AgentDelegationChainEvent) for event in self.events):
            raise InvalidAgentDelegationChainRequestError("events must be AgentDelegationChainEvent values.")
        object.__setattr__(self, "metrics", MappingProxyType(_metric_mapping(self.metrics)))
        if self.error_message is not None:
            object.__setattr__(self, "error_message", _safe_message(self.error_message))


class AgentDelegationChainService:
    """Execute a predeclared finite chain through AgentDelegationService."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        agent_resolver: AgentResolver,
        agent_context_builder: AgentContextBuilder,
        agent_executor: AgentExecutor,
        agent_delegation_service: AgentDelegationService,
    ) -> None:
        if not isinstance(agent_registry, AgentRegistry):
            raise AgentDelegationChainError("agent_registry must be AgentRegistry.")
        if not isinstance(agent_resolver, AgentResolver):
            raise AgentDelegationChainError("agent_resolver must be AgentResolver.")
        if not isinstance(agent_context_builder, AgentContextBuilder):
            raise AgentDelegationChainError("agent_context_builder must be AgentContextBuilder.")
        if not isinstance(agent_executor, AgentExecutor):
            raise AgentDelegationChainError("agent_executor must be AgentExecutor.")
        if not isinstance(agent_delegation_service, AgentDelegationService):
            raise AgentDelegationChainError("agent_delegation_service must be AgentDelegationService.")
        self._agent_registry = agent_registry
        self._agent_resolver = agent_resolver
        self._agent_context_builder = agent_context_builder
        self._agent_executor = agent_executor
        self._agent_delegation_service = agent_delegation_service

    def execute(self, request: AgentDelegationChainRequest) -> AgentDelegationChainResult:
        """Execute the declared chain sequentially."""

        events: list[AgentDelegationChainEvent] = []
        metrics = _base_metrics()
        try:
            if not isinstance(request, AgentDelegationChainRequest):
                raise InvalidAgentDelegationChainRequestError("request must be AgentDelegationChainRequest.")
            signature = agent_delegation_chain_request_signature(request)
        except (AgentDelegationChainError, TypeError, ValueError) as error:
            metrics["delegation_chains_failed"] = 1
            return _result(
                AgentDelegationChainStatus.INVALID_REQUEST,
                request_signature="",
                events=events,
                metrics=metrics,
                error_code="INVALID_REQUEST",
                error_message=str(error),
            )

        metrics["delegation_chains_requested"] = 1
        events.append(_event("agent_delegation_chain_requested", AgentDelegationChainStatus.SUCCESS, request=request))
        events.append(_event("agent_delegation_chain_validation_started", AgentDelegationChainStatus.SUCCESS, request=request))

        validation = self._validate_request(request, signature, events, metrics)
        if validation is not None:
            return validation
        events.append(_event("agent_delegation_chain_validation_succeeded", AgentDelegationChainStatus.SUCCESS, request=request))
        events.append(_event("agent_delegation_chain_started", AgentDelegationChainStatus.SUCCESS, request=request))
        metrics["delegation_chains_started"] = 1

        seen_targets: set[str] = set()
        edges: list[tuple[str, str]] = []
        source_history: list[str] = []
        step_results: list[AgentDelegationChainStepResult] = []
        previous_output: Mapping[str, object] | None = None
        final_output: Mapping[str, object] | None = None

        for index, step in enumerate(request.steps):
            events.append(_event("agent_delegation_chain_step_started", AgentDelegationChainStatus.SUCCESS, request=request, step_index=index))
            metrics["delegation_chain_steps_started"] += 1
            resolved = self._resolve_step_target(step)
            if resolved.status is not AgentDelegationChainStatus.SUCCESS or resolved.agent is None:
                step_result = AgentDelegationChainStepResult(
                    step_index=index,
                    source_agent_id=step.source_agent_id,
                    resolved_target_agent_id=resolved.agent.agent_id if resolved.agent is not None else step.target_agent_id,
                    status=resolved.status,
                    error_code=resolved.status.value,
                    error_message=resolved.message,
                )
                step_results.append(step_result)
                metrics["delegation_chain_steps_failed"] += 1
                events.append(_event("agent_delegation_chain_step_failed", resolved.status, request=request, step_index=index))
                return self._failure_result(request, signature, step_results, events, metrics, index, resolved.status, resolved.message)

            target = resolved.agent
            blocked = self._edge_block(request, step, target, seen_targets, edges)
            if blocked is not None:
                status, message = blocked
                step_result = AgentDelegationChainStepResult(
                    step_index=index,
                    source_agent_id=step.source_agent_id,
                    resolved_target_agent_id=target.agent_id,
                    status=status,
                    error_code=status.value,
                    error_message=message,
                )
                step_results.append(step_result)
                metrics["delegation_chain_steps_failed"] += 1
                if status is AgentDelegationChainStatus.CYCLE_DETECTED:
                    metrics["delegation_chain_cycles_detected"] += 1
                    events.append(_event("agent_delegation_chain_cycle_detected", status, request=request, step_index=index))
                events.append(_event("agent_delegation_chain_step_failed", status, request=request, step_index=index))
                return self._failure_result(request, signature, step_results, events, metrics, index, status, message)

            events.append(_event("agent_delegation_chain_target_resolved", AgentDelegationChainStatus.SUCCESS, request=request, step_index=index, target_id=target.agent_id))
            metrics["delegation_chain_targets_resolved"] += 1
            try:
                delegation_request = self._delegation_request(
                    request,
                    step,
                    target,
                    index,
                    previous_output,
                    tuple(source_history),
                )
            except InvalidAgentDelegationChainRequestError as error:
                step_result = AgentDelegationChainStepResult(
                    step_index=index,
                    source_agent_id=step.source_agent_id,
                    resolved_target_agent_id=target.agent_id,
                    status=AgentDelegationChainStatus.INVALID_REQUEST,
                    error_code="INVALID_REQUEST",
                    error_message=str(error),
                )
                step_results.append(step_result)
                metrics["delegation_chain_steps_failed"] += 1
                events.append(_event("agent_delegation_chain_step_failed", AgentDelegationChainStatus.INVALID_REQUEST, request=request, step_index=index, target_id=target.agent_id))
                return self._failure_result(
                    request,
                    signature,
                    step_results,
                    events,
                    metrics,
                    index,
                    AgentDelegationChainStatus.INVALID_REQUEST,
                    str(error),
                )
            delegation_result = self._agent_delegation_service.delegate(delegation_request)
            step_status = _status_from_delegation(delegation_result.status)
            safe_output = delegation_result.safe_output if delegation_result.success else None
            if delegation_result.success:
                metrics["delegation_chain_steps_succeeded"] += 1
                step_results.append(
                    AgentDelegationChainStepResult(
                        step_index=index,
                        source_agent_id=step.source_agent_id,
                        resolved_target_agent_id=target.agent_id,
                        status=AgentDelegationChainStatus.SUCCESS,
                        delegation_result=delegation_result,
                        safe_output=safe_output,
                    )
                )
                events.append(_event("agent_delegation_chain_step_succeeded", AgentDelegationChainStatus.SUCCESS, request=request, step_index=index, target_id=target.agent_id))
                previous_output = _safe_mapping(safe_output or {}, "previous_output") if request.policy.sanitize_intermediate_outputs else safe_output
                final_output = previous_output
                seen_targets.add(target.agent_id)
                source_history.append(step.source_agent_id)
                edges.append((step.source_agent_id, target.agent_id))
                continue

            metrics["delegation_chain_steps_failed"] += 1
            step_results.append(
                AgentDelegationChainStepResult(
                    step_index=index,
                    source_agent_id=step.source_agent_id,
                    resolved_target_agent_id=target.agent_id,
                    status=step_status,
                    delegation_result=delegation_result,
                    error_code=delegation_result.error_code or step_status.value,
                    error_message=delegation_result.error_message or step_status.value,
                )
            )
            events.append(_event("agent_delegation_chain_step_failed", step_status, request=request, step_index=index, target_id=target.agent_id))
            if request.policy.stop_on_failure:
                return self._failure_result(
                    request,
                    signature,
                    step_results,
                    events,
                    metrics,
                    index,
                    step_status,
                    delegation_result.error_message or step_status.value,
                )
            seen_targets.add(target.agent_id)
            source_history.append(step.source_agent_id)
            edges.append((step.source_agent_id, target.agent_id))

        failures = [result for result in step_results if result.status is not AgentDelegationChainStatus.SUCCESS]
        if failures:
            events.append(_event("agent_delegation_chain_partial", AgentDelegationChainStatus.PARTIAL_SUCCESS, request=request))
            metrics["delegation_chains_partial"] = 1
            return _result(
                AgentDelegationChainStatus.PARTIAL_SUCCESS,
                request_signature=signature,
                request=request,
                step_results=tuple(step_results),
                completed_steps=sum(1 for result in step_results if result.status is AgentDelegationChainStatus.SUCCESS),
                failed_step_index=failures[0].step_index,
                final_output=final_output,
                events=events,
                metrics=metrics,
                error_code="PARTIAL_SUCCESS",
                error_message="one or more delegation chain steps failed.",
            )

        events.append(_event("agent_delegation_chain_completed", AgentDelegationChainStatus.SUCCESS, request=request))
        metrics["delegation_chains_succeeded"] = 1
        return _result(
            AgentDelegationChainStatus.SUCCESS,
            request_signature=signature,
            request=request,
            step_results=tuple(step_results),
            completed_steps=len(step_results),
            final_output=final_output,
            events=events,
            metrics=metrics,
        )

    def _validate_request(
        self,
        request: AgentDelegationChainRequest,
        signature: str,
        events: list[AgentDelegationChainEvent],
        metrics: dict[str, int],
    ) -> AgentDelegationChainResult | None:
        policy = request.policy
        if not policy.enabled:
            return _blocked_result(AgentDelegationChainStatus.DISABLED, request, signature, events, metrics, "chain policy is disabled.")
        if len(request.steps) > policy.max_steps:
            metrics["delegation_chain_limits_reached"] = 1
            events.append(_event("agent_delegation_chain_limit_reached", AgentDelegationChainStatus.MAX_STEPS_REACHED, request=request))
            return _blocked_result(AgentDelegationChainStatus.MAX_STEPS_REACHED, request, signature, events, metrics, "maximum chain steps reached.")
        if len(request.steps) > policy.max_depth:
            metrics["delegation_chain_limits_reached"] = 1
            events.append(_event("agent_delegation_chain_limit_reached", AgentDelegationChainStatus.MAX_DEPTH_REACHED, request=request))
            return _blocked_result(AgentDelegationChainStatus.MAX_DEPTH_REACHED, request, signature, events, metrics, "maximum chain depth reached.")
        if len(request.steps) > policy.max_total_delegations:
            metrics["delegation_chain_limits_reached"] = 1
            events.append(
                _event("agent_delegation_chain_limit_reached", AgentDelegationChainStatus.MAX_TOTAL_DELEGATIONS_REACHED, request=request)
            )
            return _blocked_result(
                AgentDelegationChainStatus.MAX_TOTAL_DELEGATIONS_REACHED,
                request,
                signature,
                events,
                metrics,
                "maximum total delegations reached.",
            )
        for step in request.steps:
            try:
                self._agent_registry.get(step.source_agent_id)
            except AgentNotFoundError:
                return _blocked_result(
                    AgentDelegationChainStatus.SOURCE_AGENT_NOT_FOUND,
                    request,
                    signature,
                    events,
                    metrics,
                    "source agent was not found.",
                )
        return None

    def _resolve_step_target(self, step: AgentDelegationChainStep) -> _ResolvedTarget:
        if step.target_agent_id is not None:
            try:
                return _ResolvedTarget(AgentDelegationChainStatus.SUCCESS, self._agent_registry.get(step.target_agent_id))
            except AgentNotFoundError:
                return _ResolvedTarget(AgentDelegationChainStatus.TARGET_AGENT_NOT_FOUND, None, "target agent was not found.")

        resolution = self._agent_resolver.resolve(
            AgentResolutionRequest(
                required_capability_ids=step.required_capability_ids,
                preferred_capability_ids=(),
                required_agent_types=step.required_agent_types,
                preferred_agent_types=step.preferred_agent_types,
                required_permission_ids=step.required_permission_ids,
                preferred_agent_ids=step.preferred_agent_ids,
                excluded_agent_ids=step.excluded_agent_ids,
                enabled_only=step.enabled_only,
                require_unique_top_score=True,
                metadata={"source": "agent_delegation_chain"},
            )
        )
        if resolution.status is AgentResolutionStatus.RESOLVED and resolution.selected_agent is not None:
            return _ResolvedTarget(AgentDelegationChainStatus.SUCCESS, resolution.selected_agent)
        if resolution.status is AgentResolutionStatus.AMBIGUOUS:
            return _ResolvedTarget(AgentDelegationChainStatus.TARGET_RESOLUTION_AMBIGUOUS, None, resolution.error_message)
        return _ResolvedTarget(AgentDelegationChainStatus.TARGET_RESOLUTION_FAILED, None, resolution.error_message)

    def _edge_block(
        self,
        request: AgentDelegationChainRequest,
        step: AgentDelegationChainStep,
        target: AgentDefinition,
        seen_targets: set[str],
        edges: list[tuple[str, str]],
    ) -> tuple[AgentDelegationChainStatus, str] | None:
        policy = request.policy
        for agent in (step.source_agent_id, target.agent_id):
            definition = self._agent_registry.get(agent)
            if policy.allowed_agent_ids and agent not in policy.allowed_agent_ids:
                return AgentDelegationChainStatus.DELEGATION_BLOCKED, "agent is not allowed by chain policy."
            if agent in policy.denied_agent_ids:
                return AgentDelegationChainStatus.DELEGATION_BLOCKED, "agent is denied by chain policy."
            if policy.allowed_agent_types and definition.agent_type not in policy.allowed_agent_types:
                return AgentDelegationChainStatus.DELEGATION_BLOCKED, "agent type is not allowed by chain policy."
            if definition.agent_type in policy.denied_agent_types:
                return AgentDelegationChainStatus.DELEGATION_BLOCKED, "agent type is denied by chain policy."
        if not policy.allow_cycles and _creates_cycle(edges, step.source_agent_id, target.agent_id):
            return AgentDelegationChainStatus.CYCLE_DETECTED, "delegation edge creates a cycle."
        if not policy.allow_repeated_agents and (target.agent_id in seen_targets or step.source_agent_id == target.agent_id):
            return AgentDelegationChainStatus.REPEATED_AGENT_DENIED, "agent repetition is denied."
        return None

    def _delegation_request(
        self,
        request: AgentDelegationChainRequest,
        step: AgentDelegationChainStep,
        target: AgentDefinition,
        index: int,
        previous_output: Mapping[str, object] | None,
        source_history: tuple[str, ...],
    ) -> AgentDelegationRequest:
        structured_input = _merge_structured_input(
            request.initial_input,
            step.structured_input,
            previous_output,
            propagate_previous_output=request.policy.propagate_previous_output,
        )
        shared_context = step.shared_context
        if request.policy.propagate_shared_context:
            shared_context = _merge_mapping(request.shared_context, step.shared_context, "shared_context")
        metadata = _merge_mapping(request.metadata, step.metadata, "metadata")
        return AgentDelegationRequest(
            origin_agent_id=step.source_agent_id,
            target_agent_id=target.agent_id,
            required_agent_types=(target.agent_type,),
            required_capability_ids=step.execution_required_capability_ids or step.required_capability_ids,
            required_permission_ids=step.execution_required_permission_ids or step.required_permission_ids,
            structured_input=structured_input,
            shared_context=shared_context if request.policy.propagate_shared_context else None,
            metadata=metadata,
            execution_id=_child_execution_id(request.execution_id, index),
            parent_execution_id=request.correlation_id,
            reason_code="chain_delegation",
            delegation_depth=index,
            delegation_path=tuple((*source_history, step.source_agent_id)),
            policy=AgentDelegationPolicy(
                enabled=True,
                allow_self_delegation=request.policy.allow_cycles or request.policy.allow_repeated_agents,
                max_delegation_depth=request.policy.max_depth,
                max_total_delegations=request.policy.max_total_delegations,
                propagate_shared_context=request.policy.propagate_shared_context,
                propagate_structured_input=True,
                propagate_metadata=True,
            ),
        )

    def _failure_result(
        self,
        request: AgentDelegationChainRequest,
        signature: str,
        step_results: list[AgentDelegationChainStepResult],
        events: list[AgentDelegationChainEvent],
        metrics: dict[str, int],
        failed_step_index: int,
        status: AgentDelegationChainStatus,
        message: str | None,
    ) -> AgentDelegationChainResult:
        if request.policy.stop_on_failure:
            events.append(_event("agent_delegation_chain_failed", status, request=request))
            metrics["delegation_chains_failed"] = 1
            return _result(
                AgentDelegationChainStatus.FAILED if status not in _TERMINAL_BLOCK_STATUSES else status,
                request_signature=signature,
                request=request,
                step_results=tuple(step_results),
                completed_steps=sum(1 for result in step_results if result.status is AgentDelegationChainStatus.SUCCESS),
                failed_step_index=failed_step_index,
                events=events,
                metrics=metrics,
                error_code=status.value,
                error_message=message or status.value,
            )
        return _result(
            AgentDelegationChainStatus.PARTIAL_SUCCESS,
            request_signature=signature,
            request=request,
            step_results=tuple(step_results),
            completed_steps=sum(1 for result in step_results if result.status is AgentDelegationChainStatus.SUCCESS),
            failed_step_index=failed_step_index,
            events=events,
            metrics={**metrics, "delegation_chains_partial": 1},
            error_code=status.value,
            error_message=message or status.value,
        )


@dataclass(frozen=True, slots=True)
class _ResolvedTarget:
    status: AgentDelegationChainStatus
    agent: AgentDefinition | None
    message: str | None = None


_TERMINAL_BLOCK_STATUSES = frozenset(
    {
        AgentDelegationChainStatus.DISABLED,
        AgentDelegationChainStatus.MAX_STEPS_REACHED,
        AgentDelegationChainStatus.MAX_DEPTH_REACHED,
        AgentDelegationChainStatus.MAX_TOTAL_DELEGATIONS_REACHED,
        AgentDelegationChainStatus.CYCLE_DETECTED,
        AgentDelegationChainStatus.REPEATED_AGENT_DENIED,
        AgentDelegationChainStatus.SOURCE_AGENT_NOT_FOUND,
        AgentDelegationChainStatus.TARGET_AGENT_NOT_FOUND,
        AgentDelegationChainStatus.TARGET_RESOLUTION_FAILED,
        AgentDelegationChainStatus.TARGET_RESOLUTION_AMBIGUOUS,
    }
)


def agent_delegation_chain_request_signature(request: AgentDelegationChainRequest) -> str:
    """Return a deterministic SHA-256 signature for a chain request."""

    if not isinstance(request, AgentDelegationChainRequest):
        raise InvalidAgentDelegationChainRequestError("request must be AgentDelegationChainRequest.")
    payload = {
        "steps": [_step_payload(step) for step in request.steps],
        "policy": _policy_payload(request.policy),
        "initial_input": _jsonable(request.initial_input),
        "shared_context": _jsonable(request.shared_context),
        "metadata": _jsonable(request.metadata),
        "execution_id": request.execution_id,
        "correlation_id": request.correlation_id,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _step_payload(step: AgentDelegationChainStep) -> Mapping[str, object]:
    return {
        "source_agent_id": step.source_agent_id,
        "target_agent_id": step.target_agent_id,
        "required_agent_types": sorted(agent_type.value for agent_type in step.required_agent_types),
        "required_capability_ids": sorted(step.required_capability_ids),
        "required_permission_ids": sorted(step.required_permission_ids),
        "preferred_agent_ids": sorted(step.preferred_agent_ids),
        "preferred_agent_types": sorted(agent_type.value for agent_type in step.preferred_agent_types),
        "excluded_agent_ids": sorted(step.excluded_agent_ids),
        "enabled_only": step.enabled_only,
        "structured_input": _jsonable(step.structured_input),
        "shared_context": _jsonable(step.shared_context),
        "metadata": _jsonable(step.metadata),
        "execution_required_capability_ids": sorted(step.execution_required_capability_ids),
        "execution_required_permission_ids": sorted(step.execution_required_permission_ids),
    }


def _policy_payload(policy: AgentDelegationChainPolicy) -> Mapping[str, object]:
    return {
        "enabled": policy.enabled,
        "max_steps": policy.max_steps,
        "max_depth": policy.max_depth,
        "max_total_delegations": policy.max_total_delegations,
        "allow_repeated_agents": policy.allow_repeated_agents,
        "allow_cycles": policy.allow_cycles,
        "stop_on_failure": policy.stop_on_failure,
        "propagate_previous_output": policy.propagate_previous_output,
        "propagate_shared_context": policy.propagate_shared_context,
        "sanitize_intermediate_outputs": policy.sanitize_intermediate_outputs,
        "allowed_agent_ids": sorted(policy.allowed_agent_ids),
        "denied_agent_ids": sorted(policy.denied_agent_ids),
        "allowed_agent_types": sorted(agent_type.value for agent_type in policy.allowed_agent_types),
        "denied_agent_types": sorted(agent_type.value for agent_type in policy.denied_agent_types),
        "failure_mode": policy.failure_mode.value,
    }


def _blocked_result(
    status: AgentDelegationChainStatus,
    request: AgentDelegationChainRequest,
    signature: str,
    events: list[AgentDelegationChainEvent],
    metrics: dict[str, int],
    message: str,
) -> AgentDelegationChainResult:
    events.append(_event("agent_delegation_chain_validation_failed", status, request=request))
    metrics["delegation_chains_failed"] = 1
    return _result(
        status,
        request_signature=signature,
        request=request,
        events=events,
        metrics=metrics,
        error_code=status.value,
        error_message=message,
    )


def _result(
    status: AgentDelegationChainStatus,
    *,
    request_signature: str,
    request: AgentDelegationChainRequest | None = None,
    completed_steps: int = 0,
    failed_step_index: int | None = None,
    step_results: tuple[AgentDelegationChainStepResult, ...] = (),
    final_output: Mapping[str, object] | None = None,
    events: list[AgentDelegationChainEvent] | tuple[AgentDelegationChainEvent, ...] = (),
    metrics: Mapping[str, int] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> AgentDelegationChainResult:
    if len(events) > MAX_CHAIN_EVENTS:
        events = tuple(events)[-MAX_CHAIN_EVENTS:]
    return AgentDelegationChainResult(
        status=status,
        request_signature=request_signature,
        completed_steps=completed_steps,
        failed_step_index=failed_step_index,
        step_results=step_results,
        final_output=final_output,
        execution_id=request.execution_id if request is not None else None,
        correlation_id=request.correlation_id if request is not None else None,
        events=tuple(events),
        metrics=metrics or _base_metrics(),
        error_code=error_code,
        error_message=error_message,
    )


def _event(
    name: str,
    status: AgentDelegationChainStatus,
    *,
    request: AgentDelegationChainRequest | None = None,
    step_index: int | None = None,
    target_id: str | None = None,
) -> AgentDelegationChainEvent:
    details: dict[str, object] = {}
    if request is not None:
        details["execution_id"] = request.execution_id or ""
        details["steps"] = len(request.steps)
    if step_index is not None:
        details["step_index"] = step_index
    if target_id is not None:
        details["target_agent_id"] = target_id
    return AgentDelegationChainEvent(name=name, status=status, details=details)


def _base_metrics() -> dict[str, int]:
    return {
        "delegation_chains_requested": 0,
        "delegation_chains_started": 0,
        "delegation_chains_succeeded": 0,
        "delegation_chains_failed": 0,
        "delegation_chains_partial": 0,
        "delegation_chain_steps_started": 0,
        "delegation_chain_steps_succeeded": 0,
        "delegation_chain_steps_failed": 0,
        "delegation_chain_cycles_detected": 0,
        "delegation_chain_limits_reached": 0,
        "delegation_chain_targets_resolved": 0,
    }


def _status_from_delegation(status: AgentDelegationStatus) -> AgentDelegationChainStatus:
    if status in (AgentDelegationStatus.SUCCESS,):
        return AgentDelegationChainStatus.SUCCESS
    if status in (
        AgentDelegationStatus.TARGET_AGENT_NOT_FOUND,
        AgentDelegationStatus.ORIGIN_AGENT_NOT_FOUND,
    ):
        return AgentDelegationChainStatus.TARGET_AGENT_NOT_FOUND
    if status in (AgentDelegationStatus.NO_MATCHING_AGENT,):
        return AgentDelegationChainStatus.TARGET_RESOLUTION_FAILED
    if status in (AgentDelegationStatus.AMBIGUOUS_AGENT_SELECTION,):
        return AgentDelegationChainStatus.TARGET_RESOLUTION_AMBIGUOUS
    if status in (
        AgentDelegationStatus.DISABLED,
        AgentDelegationStatus.SELF_DELEGATION_DENIED,
        AgentDelegationStatus.TARGET_NOT_ALLOWED,
        AgentDelegationStatus.TARGET_DENIED,
        AgentDelegationStatus.MISSING_CAPABILITIES,
        AgentDelegationStatus.MISSING_PERMISSIONS,
        AgentDelegationStatus.TYPE_INCOMPATIBLE,
        AgentDelegationStatus.MAX_DEPTH_REACHED,
        AgentDelegationStatus.MAX_DELEGATIONS_REACHED,
        AgentDelegationStatus.CONTEXT_REJECTED,
    ):
        return AgentDelegationChainStatus.DELEGATION_BLOCKED
    return AgentDelegationChainStatus.DELEGATION_FAILED


def _creates_cycle(edges: list[tuple[str, str]], source: str, target: str) -> bool:
    if source == target:
        return True
    adjacency: dict[str, set[str]] = {}
    for edge_source, edge_target in (*edges, (source, target)):
        adjacency.setdefault(edge_source, set()).add(edge_target)
    stack = [target]
    visited: set[str] = set()
    while stack:
        current = stack.pop()
        if current == source:
            return True
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency.get(current, ()))
    return False


def _merge_structured_input(
    initial: Mapping[str, object] | None,
    step: Mapping[str, object] | None,
    previous_output: Mapping[str, object] | None,
    *,
    propagate_previous_output: bool,
) -> Mapping[str, object] | None:
    merged = _merge_mapping(initial, step, "structured_input")
    if not propagate_previous_output or previous_output is None:
        return merged
    base = dict(merged or {})
    if PREVIOUS_OUTPUT_KEY in base:
        raise InvalidAgentDelegationChainRequestError("previous_delegation_output is reserved.")
    base[PREVIOUS_OUTPUT_KEY] = _safe_mapping(previous_output, PREVIOUS_OUTPUT_KEY)
    return MappingProxyType(_safe_mapping(base, "structured_input"))


def _merge_mapping(
    first: Mapping[str, object] | None,
    second: Mapping[str, object] | None,
    field_name: str,
) -> Mapping[str, object] | None:
    if first is None and second is None:
        return None
    merged: dict[str, object] = {}
    if first is not None:
        merged.update(_safe_mapping(first, field_name))
    if second is not None:
        overlap = set(merged).intersection(second)
        if overlap:
            raise InvalidAgentDelegationChainRequestError(f"{field_name} cannot overwrite existing keys.")
        merged.update(_safe_mapping(second, field_name))
    return MappingProxyType(_safe_mapping(merged, field_name))


def _child_execution_id(execution_id: str | None, index: int) -> str | None:
    if execution_id is None:
        return None
    return f"{execution_id}.step-{index}"


def _bounded_int(value: int, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidAgentDelegationChainRequestError(f"{field_name} must be an integer.")
    if value <= 0 or value > maximum:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} is outside the allowed range.")
    return value


def _agent_id_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentDelegationChainRequestError(f"{field_name} must be an iterable of agent ids.")
    normalized = tuple(dict.fromkeys(validate_agent_id(value) for value in values))
    if len(normalized) > MAX_CHAIN_IDS:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} has too many items.")
    return normalized


def _identifier_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentDelegationChainRequestError(f"{field_name} must be an iterable of strings.")
    normalized = tuple(dict.fromkeys(_identifier(value, field_name) for value in values))
    if len(normalized) > MAX_CHAIN_IDS:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} has too many items.")
    return normalized


def _permission_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    normalized = _identifier_tuple(values, field_name)
    unknown = tuple(value for value in normalized if value not in _PERMISSION_IDS)
    if unknown:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} contains an unknown permission id.")
    return normalized


def _agent_type_tuple(values: Iterable[AgentType | str], field_name: str) -> tuple[AgentType, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentDelegationChainRequestError(f"{field_name} must be an iterable.")
    normalized: list[AgentType] = []
    for value in values:
        if isinstance(value, AgentType):
            agent_type = value
        elif isinstance(value, str):
            try:
                agent_type = AgentType(value.strip().lower())
            except ValueError as error:
                raise InvalidAgentDelegationChainRequestError(f"{field_name} contains an invalid AgentType.") from error
        else:
            raise InvalidAgentDelegationChainRequestError(f"{field_name} values must be AgentType or str.")
        if agent_type not in normalized:
            normalized.append(agent_type)
    if len(normalized) > MAX_CHAIN_IDS:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} has too many items.")
    return tuple(normalized)


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidAgentDelegationChainRequestError(f"{field_name} must be a string.")
    if not value or value.strip() != value:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} cannot be empty or padded.")
    if "/" in value or "\\" in value or ".." in value:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} cannot be path-like.")
    if any(ord(character) < 32 for character in value):
        raise InvalidAgentDelegationChainRequestError(f"{field_name} cannot contain control characters.")
    if not all(character.isalnum() or character in "_.-" for character in value):
        raise InvalidAgentDelegationChainRequestError(f"{field_name} contains unsupported characters.")
    if len(value) > 128:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} exceeds the length limit.")
    if _is_sensitive_key(value):
        raise InvalidAgentDelegationChainRequestError(f"{field_name} cannot contain sensitive content.")
    return value


def _optional_safe_mapping(value: Mapping[str, object] | None, field_name: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return MappingProxyType(_safe_mapping(value, field_name))


def _safe_mapping(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentDelegationChainRequestError(f"{field_name} must be a mapping.")
    if len(value) > MAX_CHAIN_METADATA_ITEMS and field_name == "metadata":
        raise InvalidAgentDelegationChainRequestError("metadata has too many items.")
    return _safe_mapping_inner(value, field_name, depth=0, counter={"nodes": 0, "chars": 0})


def _safe_mapping_inner(
    value: Mapping[str, object],
    field_name: str,
    *,
    depth: int,
    counter: dict[str, int],
) -> dict[str, object]:
    if depth > MAX_CHAIN_DEPTH:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} is too deep.")
    if len(value) > MAX_CHAIN_MAPPING_ITEMS:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} has too many items.")
    result: dict[str, object] = {}
    for raw_key in sorted(value):
        if isinstance(raw_key, str) and _is_sensitive_key(raw_key):
            continue
        key = _key(raw_key, field_name)
        if _is_sensitive_key(key):
            continue
        result[key] = _safe_value(value[raw_key], field_name, depth=depth + 1, counter=counter)
    return result


def _safe_sequence(value: Sequence[object], field_name: str, *, depth: int, counter: dict[str, int]) -> tuple[object, ...]:
    if len(value) > MAX_CHAIN_SEQUENCE_ITEMS:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} has too many items.")
    return tuple(_safe_value(item, field_name, depth=depth + 1, counter=counter) for item in value)


def _safe_value(value: object, field_name: str, *, depth: int, counter: dict[str, int]) -> object:
    counter["nodes"] += 1
    if counter["nodes"] > MAX_CHAIN_TOTAL_ITEMS:
        raise InvalidAgentDelegationChainRequestError(f"{field_name} is too large.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentDelegationChainRequestError(f"{field_name} floats must be finite.")
        return value
    if isinstance(value, str):
        if len(value) > MAX_CHAIN_STRING_LENGTH:
            raise InvalidAgentDelegationChainRequestError(f"{field_name} strings are too long.")
        counter["chars"] += len(value)
        if counter["chars"] > MAX_CHAIN_STRING_LENGTH * MAX_CHAIN_MAPPING_ITEMS:
            raise InvalidAgentDelegationChainRequestError(f"{field_name} string budget exceeded.")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_mapping_inner(value, field_name, depth=depth, counter=counter))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _safe_sequence(value, field_name, depth=depth, counter=counter)
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value) or isinstance(value, types.ModuleType):
        raise InvalidAgentDelegationChainRequestError(f"{field_name} cannot contain executable objects.")
    raise InvalidAgentDelegationChainRequestError(f"{field_name} contains an unsupported value.")


def _key(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidAgentDelegationChainRequestError(f"{field_name} keys must be strings.")
    return _identifier(value, f"{field_name} key")


def _metric_mapping(metrics: Mapping[str, int]) -> dict[str, int]:
    safe: dict[str, int] = {}
    for key, value in metrics.items():
        name = _identifier(str(key), "metric name")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidAgentDelegationChainRequestError("metric values must be non-negative integers.")
        safe[name] = value
    return safe


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
        raise InvalidAgentDelegationChainRequestError("non-finite floats are not allowed.")
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value) or isinstance(value, types.ModuleType):
        raise InvalidAgentDelegationChainRequestError("executable objects are not allowed.")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise InvalidAgentDelegationChainRequestError("unsupported signature value.")
    return value


def _safe_message(value: str | None) -> str:
    text = " ".join(str(value or "").split())
    lowered = text.lower()
    if "[redacted]" in lowered or any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        text = "[redacted]"
    return text[:240]


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _failure_mode(value: AgentDelegationChainFailureMode | str) -> AgentDelegationChainFailureMode:
    if isinstance(value, AgentDelegationChainFailureMode):
        return value
    if isinstance(value, str):
        try:
            return AgentDelegationChainFailureMode(value.upper())
        except ValueError as error:
            raise InvalidAgentDelegationChainRequestError("failure_mode is invalid.") from error
    raise InvalidAgentDelegationChainRequestError("failure_mode must be AgentDelegationChainFailureMode.")


def _status(value: AgentDelegationChainStatus | str) -> AgentDelegationChainStatus:
    if isinstance(value, AgentDelegationChainStatus):
        return value
    if isinstance(value, str):
        return AgentDelegationChainStatus(value)
    raise InvalidAgentDelegationChainRequestError("status must be AgentDelegationChainStatus.")

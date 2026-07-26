"""Deterministic structured router for Atlas service requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable

from core.agent_executor import AgentExecutionRequest, AgentExecutionResult, AgentExecutionStatus
from core.agent_registry import AgentType, validate_agent_id
from core.agent_resolver import AgentResolutionRequest, AgentResolutionResult, AgentResolutionStatus
from core.agent_system import AgentSystem
from core.capability_execution_service import (
    CapabilityExecutionRequest,
    CapabilityExecutionResult,
    CapabilityExecutionService,
    CapabilityExecutionStatus,
)


MAX_ATLAS_ROUTING_METADATA_ITEMS = 32
MAX_ATLAS_ROUTING_PAYLOAD_DEPTH = 6
MAX_ATLAS_ROUTING_PAYLOAD_NODES = 128
SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)


class AtlasRoutingError(RuntimeError):
    """Base error for deterministic Atlas routing."""


class InvalidAtlasRoutingRequestError(AtlasRoutingError):
    """Raised when an Atlas routing request is malformed."""


class AtlasRouteType(str, Enum):
    """Explicit high-level Atlas route types."""

    CAPABILITY = "capability"
    CONVERSATION = "conversation"
    TOOL = "tool"
    WORKFLOW = "workflow"
    AGENT = "agent"
    UNKNOWN = "unknown"


class AtlasRoutingStatus(str, Enum):
    """Stable states returned by AtlasRouter."""

    COMPLETED = "completed"
    ROUTE_UNAVAILABLE = "route_unavailable"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN_ROUTE = "unknown_route"
    SERVICE_UNAVAILABLE = "service_unavailable"
    EXECUTION_FAILED = "execution_failed"
    INTERNAL_ERROR = "internal_error"


class AgentSelectionStatus(str, Enum):
    """Structured statuses for AGENT route selection."""

    SELECTED = "SELECTED"
    EXPLICIT_AGENT_SELECTED = "EXPLICIT_AGENT_SELECTED"
    AUTOMATIC_AGENT_SELECTED = "AUTOMATIC_AGENT_SELECTED"
    INVALID_SELECTION_REQUEST = "INVALID_SELECTION_REQUEST"
    INSUFFICIENT_SELECTION_CRITERIA = "INSUFFICIENT_SELECTION_CRITERIA"
    NO_MATCHING_AGENT = "NO_MATCHING_AGENT"
    AMBIGUOUS_SELECTION = "AMBIGUOUS_SELECTION"
    AGENT_SYSTEM_UNAVAILABLE = "AGENT_SYSTEM_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class AtlasRoutingEvent:
    """Safe routing event with no payload or output values."""

    name: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidAtlasRoutingRequestError("event name must be a non-empty string.")
        if not isinstance(self.status, str) or not self.status.strip():
            raise InvalidAtlasRoutingRequestError("event status must be a non-empty string.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "status", self.status.strip())
        object.__setattr__(self, "details", MappingProxyType(_safe_metadata(self.details)))


@dataclass(frozen=True, slots=True)
class AtlasRoutingRequest:
    """Structured request for the deterministic Atlas router."""

    route_type: AtlasRouteType | str
    payload: object | None = None
    request_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_type", _validate_route_type(self.route_type))
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _safe_identifier(self.request_id, "request_id"))
        object.__setattr__(self, "payload", _safe_payload(self.payload))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class AtlasAgentRoutingRequest:
    """Explicit structured request for routing one known agent id."""

    agent_id: str | None = None
    payload: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    shared_context: Mapping[str, object] | None = None
    required_capabilities: tuple[str, ...] = ()
    required_permissions: tuple[str, ...] = ()
    preferred_agent_types: tuple[AgentType | str, ...] = ()
    preferred_agent_ids: tuple[str, ...] = ()
    excluded_agent_ids: tuple[str, ...] = ()
    enabled_only: bool = True
    task_id: str | None = None
    execution_id: str | None = None
    correlation_id: str | None = None
    session_id: str | None = None
    user_input: str | None = None

    def __post_init__(self) -> None:
        if self.agent_id is not None:
            try:
                object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
            except Exception as error:
                raise InvalidAtlasRoutingRequestError("agent_id is invalid.") from error
        object.__setattr__(self, "payload", _optional_safe_mapping(self.payload, "payload"))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))
        object.__setattr__(self, "shared_context", _optional_safe_mapping(self.shared_context, "shared_context"))
        object.__setattr__(
            self,
            "required_capabilities",
            _safe_sorted_identifier_tuple(self.required_capabilities, "required_capabilities"),
        )
        object.__setattr__(
            self,
            "required_permissions",
            _safe_sorted_identifier_tuple(self.required_permissions, "required_permissions"),
        )
        object.__setattr__(
            self,
            "preferred_agent_types",
            _safe_agent_type_tuple(self.preferred_agent_types, "preferred_agent_types"),
        )
        object.__setattr__(
            self,
            "preferred_agent_ids",
            _safe_agent_id_tuple(self.preferred_agent_ids, "preferred_agent_ids"),
        )
        object.__setattr__(
            self,
            "excluded_agent_ids",
            _safe_sorted_agent_id_tuple(self.excluded_agent_ids, "excluded_agent_ids"),
        )
        if not isinstance(self.enabled_only, bool):
            raise InvalidAtlasRoutingRequestError("enabled_only must be a bool.")
        for field_name in ("task_id", "execution_id", "correlation_id", "session_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _safe_identifier(value, field_name))
        if self.user_input is not None and not isinstance(self.user_input, str):
            raise InvalidAtlasRoutingRequestError("user_input must be a string or None.")


@dataclass(frozen=True, slots=True)
class AtlasRoutingResult:
    """Immutable safe result returned by AtlasRouter."""

    status: AtlasRoutingStatus
    route_type: AtlasRouteType
    output: object | None = None
    error_code: str | None = None
    message: str | None = None
    request_id: str | None = None
    capability_result: CapabilityExecutionResult | None = None
    agent_result: AgentExecutionResult | None = None
    agent_resolution_result: AgentResolutionResult | None = None
    events: tuple[AtlasRoutingEvent, ...] = ()
    request_signature: str | None = None
    metrics: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _validate_status(self.status))
        object.__setattr__(self, "route_type", _validate_route_type(self.route_type))
        object.__setattr__(self, "output", _safe_output(self.output))
        object.__setattr__(self, "message", _safe_message(self.message))
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _safe_identifier(self.request_id, "request_id"))
        if self.capability_result is not None and not isinstance(
            self.capability_result,
            CapabilityExecutionResult,
        ):
            raise InvalidAtlasRoutingRequestError("capability_result must be CapabilityExecutionResult or None.")
        if self.agent_result is not None and not isinstance(self.agent_result, AgentExecutionResult):
            raise InvalidAtlasRoutingRequestError("agent_result must be AgentExecutionResult or None.")
        if self.agent_resolution_result is not None and not isinstance(
            self.agent_resolution_result,
            AgentResolutionResult,
        ):
            raise InvalidAtlasRoutingRequestError("agent_resolution_result must be AgentResolutionResult or None.")
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "metrics", MappingProxyType(_safe_metrics(self.metrics)))

    @property
    def completed(self) -> bool:
        """Return whether routing completed successfully."""

        return self.status is AtlasRoutingStatus.COMPLETED


RoutingObserver = Callable[[AtlasRoutingEvent], None]


class AtlasRouter:
    """Route pre-classified structured requests to explicit Atlas services."""

    _UNAVAILABLE_ROUTES = {
        AtlasRouteType.CONVERSATION,
        AtlasRouteType.TOOL,
        AtlasRouteType.WORKFLOW,
    }

    def __init__(
        self,
        capability_execution_service: CapabilityExecutionService | None = None,
        *,
        agent_system: AgentSystem | None = None,
        observer: RoutingObserver | None = None,
    ) -> None:
        if capability_execution_service is not None and not isinstance(
            capability_execution_service,
            CapabilityExecutionService,
        ):
            raise AtlasRoutingError("capability_execution_service must be CapabilityExecutionService or None.")
        if agent_system is not None and not isinstance(agent_system, AgentSystem):
            raise AtlasRoutingError("agent_system must be AgentSystem or None.")
        if observer is not None and not callable(observer):
            raise AtlasRoutingError("observer must be callable or None.")
        self._capability_execution_service = capability_execution_service
        self._agent_system = agent_system
        self._observer = observer

    def route_agent_request(
        self,
        request: AtlasAgentRoutingRequest,
        *,
        request_id: str | None = None,
    ) -> AtlasRoutingResult:
        """Route one explicit agent request through the structured AGENT route."""

        return self.route(AtlasRoutingRequest(AtlasRouteType.AGENT, payload=request, request_id=request_id))

    def route(self, request: AtlasRoutingRequest) -> AtlasRoutingResult:
        """Route one already-classified request without interpreting text."""

        events: list[AtlasRoutingEvent] = []
        if not isinstance(request, AtlasRoutingRequest):
            _record(events, self._observer, "atlas_routing_started", "failed")
            return _result(
                AtlasRoutingStatus.INVALID_REQUEST,
                AtlasRouteType.UNKNOWN,
                events,
                error_code="INVALID_REQUEST",
                message="request must be AtlasRoutingRequest.",
            )

        signature = atlas_routing_request_signature(request)
        _record(
            events,
            self._observer,
            "atlas_routing_started",
            "started",
            {"route_type": request.route_type.value},
        )
        _record(
            events,
            self._observer,
            "atlas_route_selected",
            "finished",
            {"route_type": request.route_type.value},
        )

        if request.route_type is AtlasRouteType.UNKNOWN:
            _record(events, self._observer, "atlas_routing_failed", "failed", {"reason": "unknown_route"})
            return _result(
                AtlasRoutingStatus.UNKNOWN_ROUTE,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code="UNKNOWN_ROUTE",
                message="Atlas route is unknown.",
            )

        if request.route_type in self._UNAVAILABLE_ROUTES:
            _record(events, self._observer, "atlas_route_unavailable", "finished")
            return _result(
                AtlasRoutingStatus.ROUTE_UNAVAILABLE,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code="ROUTE_UNAVAILABLE",
                message=f"Atlas route '{request.route_type.value}' is not available.",
            )

        if request.route_type is AtlasRouteType.CAPABILITY:
            return self._route_capability(request, events, signature)

        if request.route_type is AtlasRouteType.AGENT:
            return self._route_agent(request, events, signature)

        _record(events, self._observer, "atlas_routing_failed", "failed", {"reason": "unknown_route"})
        return _result(
            AtlasRoutingStatus.UNKNOWN_ROUTE,
            request.route_type,
            events,
            request_id=request.request_id,
            request_signature=signature,
            error_code="UNKNOWN_ROUTE",
            message="Atlas route is unknown.",
        )

    def _route_capability(
        self,
        request: AtlasRoutingRequest,
        events: list[AtlasRoutingEvent],
        signature: str,
    ) -> AtlasRoutingResult:
        if not isinstance(request.payload, CapabilityExecutionRequest):
            _record(events, self._observer, "atlas_routing_failed", "failed", {"reason": "invalid_payload"})
            return _result(
                AtlasRoutingStatus.INVALID_REQUEST,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code="INVALID_CAPABILITY_PAYLOAD",
                message="CAPABILITY payload must be CapabilityExecutionRequest.",
            )

        if self._capability_execution_service is None:
            _record(events, self._observer, "atlas_route_unavailable", "finished", {"reason": "service_unavailable"})
            return _result(
                AtlasRoutingStatus.SERVICE_UNAVAILABLE,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code="CAPABILITY_EXECUTION_SERVICE_UNAVAILABLE",
                message="Capability execution service is not configured.",
            )

        try:
            capability_result = self._capability_execution_service.execute(request.payload)
        except (ValueError, TypeError, RuntimeError):
            _record(events, self._observer, "atlas_routing_failed", "failed", {"reason": "capability_exception"})
            return _result(
                AtlasRoutingStatus.INTERNAL_ERROR,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code="INTERNAL_ERROR",
                message="Capability route failed before returning a structured result.",
            )

        if not isinstance(capability_result, CapabilityExecutionResult):
            _record(events, self._observer, "atlas_routing_failed", "failed", {"reason": "invalid_result"})
            return _result(
                AtlasRoutingStatus.INTERNAL_ERROR,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code="INVALID_CAPABILITY_RESULT",
                message="Capability route returned an invalid result.",
            )

        routing_status = _routing_status_for_capability(capability_result.status)
        event_name = "atlas_routing_succeeded" if routing_status is AtlasRoutingStatus.COMPLETED else "atlas_routing_failed"
        _record(
            events,
            self._observer,
            event_name,
            "finished" if routing_status is AtlasRoutingStatus.COMPLETED else "failed",
            {"capability_status": capability_result.status.value},
        )
        return _result(
            routing_status,
            request.route_type,
            events,
            output=capability_result.output if routing_status is AtlasRoutingStatus.COMPLETED else None,
            error_code=capability_result.error_code,
            message=capability_result.message,
            request_id=request.request_id,
            capability_result=capability_result,
            request_signature=signature,
        )

    def _route_agent(
        self,
        request: AtlasRoutingRequest,
        events: list[AtlasRoutingEvent],
        signature: str,
    ) -> AtlasRoutingResult:
        try:
            agent_request = _agent_routing_request(request.payload)
        except InvalidAtlasRoutingRequestError as error:
            _record(events, self._observer, "atlas_routing_failed", "failed", {"reason": "invalid_agent_payload"})
            return _result(
                AtlasRoutingStatus.INVALID_REQUEST,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code="INVALID_AGENT_PAYLOAD",
                message=str(error),
                metrics=_agent_metrics(invalid=1),
            )

        if self._agent_system is None:
            _record(events, self._observer, "atlas_agent_route_unavailable", "finished", {"reason": "service_unavailable"})
            return _result(
                AtlasRoutingStatus.SERVICE_UNAVAILABLE,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code="AGENT_SYSTEM_UNAVAILABLE",
                message="AgentSystem is not configured.",
                metrics=_agent_metrics(unavailable=1),
            )

        if agent_request.agent_id is None:
            return self._route_agent_automatic(request, agent_request, events, signature)

        _record(
            events,
            self._observer,
            "agent_explicit_selection_succeeded",
            "finished",
            {"selection_status": AgentSelectionStatus.EXPLICIT_AGENT_SELECTED.value, "agent_id": agent_request.agent_id},
        )
        try:
            execution_request = AgentExecutionRequest(
                resolution_request=AgentResolutionRequest(
                    required_agent_ids=(agent_request.agent_id,),
                    enabled_only=False,
                    require_unique_top_score=False,
                    metadata={"route": "agent"},
                ),
                task_id=agent_request.task_id,
                execution_id=agent_request.execution_id,
                correlation_id=agent_request.correlation_id,
                session_id=agent_request.session_id,
                user_input=agent_request.user_input,
                structured_input=agent_request.payload,
                shared_context=agent_request.shared_context,
                metadata=agent_request.metadata,
                required_capability_ids=agent_request.required_capabilities,
                required_permission_ids=agent_request.required_permissions,
            )
        except (ValueError, TypeError, RuntimeError) as error:
            _record(events, self._observer, "atlas_routing_failed", "failed", {"reason": "invalid_agent_request"})
            return _result(
                AtlasRoutingStatus.INVALID_REQUEST,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code=AgentSelectionStatus.INVALID_SELECTION_REQUEST.value,
                message=str(error),
                metrics=_agent_metrics(invalid=1),
            )

        return self._execute_agent_request(
            request,
            execution_request,
            events,
            signature,
            selected_agent_id=agent_request.agent_id,
            selection_metrics=_agent_metrics(),
        )

    def _route_agent_automatic(
        self,
        request: AtlasRoutingRequest,
        agent_request: AtlasAgentRoutingRequest,
        events: list[AtlasRoutingEvent],
        signature: str,
    ) -> AtlasRoutingResult:
        if not _has_automatic_selection_criteria(agent_request):
            _record(
                events,
                self._observer,
                "agent_auto_selection_failed",
                "failed",
                {"selection_status": AgentSelectionStatus.INSUFFICIENT_SELECTION_CRITERIA.value},
            )
            return _result(
                AtlasRoutingStatus.INVALID_REQUEST,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code=AgentSelectionStatus.INSUFFICIENT_SELECTION_CRITERIA.value,
                message="AGENT automatic selection requires declarative criteria.",
                metrics=_agent_metrics(invalid=1, auto_requested=1, auto_failed=1),
            )

        try:
            resolution_request = AgentResolutionRequest(
                required_capability_ids=agent_request.required_capabilities,
                required_permission_ids=agent_request.required_permissions,
                preferred_agent_types=agent_request.preferred_agent_types,
                preferred_agent_ids=agent_request.preferred_agent_ids,
                excluded_agent_ids=agent_request.excluded_agent_ids,
                enabled_only=agent_request.enabled_only,
                require_unique_top_score=True,
                metadata={"route": "agent", "selection": "automatic"},
            )
        except (ValueError, TypeError, RuntimeError) as error:
            _record(
                events,
                self._observer,
                "agent_auto_selection_failed",
                "failed",
                {"selection_status": AgentSelectionStatus.INVALID_SELECTION_REQUEST.value},
            )
            return _result(
                AtlasRoutingStatus.INVALID_REQUEST,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code=AgentSelectionStatus.INVALID_SELECTION_REQUEST.value,
                message=str(error),
                metrics=_agent_metrics(invalid=1, auto_requested=1, auto_failed=1),
            )
        _record(
            events,
            self._observer,
            "agent_auto_selection_requested",
            "started",
            {"selection_status": AgentSelectionStatus.SELECTED.value},
        )
        try:
            resolution_result = self._agent_system.agent_resolver.resolve(resolution_request)
        except (ValueError, TypeError, RuntimeError):
            _record(events, self._observer, "atlas_routing_failed", "failed", {"reason": "invalid_agent_result"})
            return _result(
                AtlasRoutingStatus.INTERNAL_ERROR,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code=AgentSelectionStatus.INTERNAL_ERROR.value,
                message="Agent automatic selection failed before returning a structured result.",
                metrics=_agent_metrics(failed=1, auto_requested=1, auto_failed=1),
            )

        _record_agent_selection_events(events, self._observer, resolution_result)
        if resolution_result.status is AgentResolutionStatus.NO_AGENTS or resolution_result.status in (
            AgentResolutionStatus.NO_MATCHING_AGENTS,
            AgentResolutionStatus.BELOW_MINIMUM_SCORE,
        ):
            _record(
                events,
                self._observer,
                "agent_auto_selection_no_match",
                "failed",
                {"selection_status": AgentSelectionStatus.NO_MATCHING_AGENT.value},
            )
            return _result(
                AtlasRoutingStatus.EXECUTION_FAILED,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code=AgentSelectionStatus.NO_MATCHING_AGENT.value,
                message="No registered agent matched the declarative criteria.",
                agent_resolution_result=resolution_result,
                metrics=_agent_metrics(
                    failed=1,
                    auto_requested=1,
                    auto_failed=1,
                    auto_no_match=1,
                    candidates_evaluated=resolution_result.scanned_agents,
                    candidates_rejected=len(resolution_result.rejections),
                ),
            )
        if resolution_result.status is AgentResolutionStatus.AMBIGUOUS:
            _record(
                events,
                self._observer,
                "agent_auto_selection_ambiguous",
                "failed",
                {"selection_status": AgentSelectionStatus.AMBIGUOUS_SELECTION.value},
            )
            return _result(
                AtlasRoutingStatus.EXECUTION_FAILED,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code=AgentSelectionStatus.AMBIGUOUS_SELECTION.value,
                message="Multiple agents matched the declarative criteria with no decisive preference.",
                agent_resolution_result=resolution_result,
                metrics=_agent_metrics(
                    failed=1,
                    auto_requested=1,
                    auto_failed=1,
                    auto_ambiguous=1,
                    candidates_evaluated=resolution_result.scanned_agents,
                    candidates_rejected=len(resolution_result.rejections),
                ),
            )
        if resolution_result.status is not AgentResolutionStatus.RESOLVED or resolution_result.selected_agent_id is None:
            _record(
                events,
                self._observer,
                "agent_auto_selection_failed",
                "failed",
                {"selection_status": AgentSelectionStatus.INVALID_SELECTION_REQUEST.value},
            )
            return _result(
                AtlasRoutingStatus.INTERNAL_ERROR,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code=resolution_result.error_code or AgentSelectionStatus.INTERNAL_ERROR.value,
                message=resolution_result.error_message or "Agent automatic selection failed.",
                agent_resolution_result=resolution_result,
                metrics=_agent_metrics(
                    failed=1,
                    auto_requested=1,
                    auto_failed=1,
                    candidates_evaluated=resolution_result.scanned_agents,
                    candidates_rejected=len(resolution_result.rejections),
                ),
            )

        _record(
            events,
            self._observer,
            "agent_auto_selection_succeeded",
            "finished",
            {
                "selection_status": AgentSelectionStatus.AUTOMATIC_AGENT_SELECTED.value,
                "agent_id": resolution_result.selected_agent_id,
            },
        )
        try:
            execution_request = AgentExecutionRequest(
                resolution_request=AgentResolutionRequest(
                    required_agent_ids=(resolution_result.selected_agent_id,),
                    enabled_only=False,
                    require_unique_top_score=False,
                    metadata={"route": "agent", "selection": "automatic_execution"},
                ),
                task_id=agent_request.task_id,
                execution_id=agent_request.execution_id,
                correlation_id=agent_request.correlation_id,
                session_id=agent_request.session_id,
                user_input=agent_request.user_input,
                structured_input=agent_request.payload,
                shared_context=agent_request.shared_context,
                metadata=agent_request.metadata,
                required_capability_ids=agent_request.required_capabilities,
                required_permission_ids=agent_request.required_permissions,
            )
        except (ValueError, TypeError, RuntimeError) as error:
            _record(events, self._observer, "agent_auto_selection_failed", "failed", {"reason": "invalid_execution_request"})
            return _result(
                AtlasRoutingStatus.INVALID_REQUEST,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code=AgentSelectionStatus.INVALID_SELECTION_REQUEST.value,
                message=str(error),
                agent_resolution_result=resolution_result,
                metrics=_agent_metrics(invalid=1, auto_requested=1, auto_failed=1),
            )
        return self._execute_agent_request(
            request,
            execution_request,
            events,
            signature,
            selected_agent_id=resolution_result.selected_agent_id,
            agent_resolution_result=resolution_result,
            selection_metrics=_agent_metrics(
                auto_requested=1,
                auto_succeeded=1,
                candidates_evaluated=resolution_result.scanned_agents,
                candidates_rejected=len(resolution_result.rejections),
            ),
        )

    def _execute_agent_request(
        self,
        request: AtlasRoutingRequest,
        execution_request: AgentExecutionRequest,
        events: list[AtlasRoutingEvent],
        signature: str,
        *,
        selected_agent_id: str,
        agent_resolution_result: AgentResolutionResult | None = None,
        selection_metrics: Mapping[str, int],
    ) -> AtlasRoutingResult:
        _record(events, self._observer, "atlas_agent_execution_started", "started", {"agent_id": selected_agent_id})
        try:
            agent_result = self._agent_system.agent_executor.execute(execution_request)
        except (ValueError, TypeError, RuntimeError):
            _record(events, self._observer, "atlas_routing_failed", "failed", {"reason": "agent_exception"})
            return _result(
                AtlasRoutingStatus.INTERNAL_ERROR,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code="INTERNAL_ERROR",
                message="Agent route failed before returning a structured result.",
                metrics=_merge_metrics(selection_metrics, _agent_metrics(failed=1)),
            )

        if not isinstance(agent_result, AgentExecutionResult):
            _record(events, self._observer, "atlas_routing_failed", "failed", {"reason": "invalid_agent_result"})
            return _result(
                AtlasRoutingStatus.INTERNAL_ERROR,
                request.route_type,
                events,
                request_id=request.request_id,
                request_signature=signature,
                error_code="INVALID_AGENT_RESULT",
                message="Agent route returned an invalid result.",
                metrics=_merge_metrics(selection_metrics, _agent_metrics(failed=1)),
            )

        routing_status = _routing_status_for_agent(agent_result.status)
        event_name = "atlas_routing_succeeded" if routing_status is AtlasRoutingStatus.COMPLETED else "atlas_routing_failed"
        _record(
            events,
            self._observer,
            event_name,
            "finished" if routing_status is AtlasRoutingStatus.COMPLETED else "failed",
            {"agent_status": agent_result.status.value, "agent_id": agent_result.agent_id or selected_agent_id},
        )
        return _result(
            routing_status,
            request.route_type,
            events,
            output=agent_result.output if routing_status is AtlasRoutingStatus.COMPLETED else None,
            error_code=agent_result.error_code,
            message=agent_result.safe_message,
            request_id=request.request_id,
            agent_result=agent_result,
            agent_resolution_result=agent_resolution_result or agent_result.resolution_result,
            request_signature=signature,
            metrics=_merge_metrics(
                selection_metrics,
                _agent_metrics(completed=1) if routing_status is AtlasRoutingStatus.COMPLETED else _agent_metrics(failed=1),
            ),
        )


def atlas_routing_request_signature(request: AtlasRoutingRequest) -> str:
    """Return a stable signature for a safe routing request structure."""

    if not isinstance(request, AtlasRoutingRequest):
        raise InvalidAtlasRoutingRequestError("request must be AtlasRoutingRequest.")
    payload = {
        "route_type": request.route_type.value,
        "payload": _signature_payload(request.payload),
        "request_id": request.request_id,
        "metadata": _jsonable_mapping(request.metadata),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _routing_status_for_capability(status: CapabilityExecutionStatus) -> AtlasRoutingStatus:
    if status is CapabilityExecutionStatus.COMPLETED:
        return AtlasRoutingStatus.COMPLETED
    if status is CapabilityExecutionStatus.SERVICE_UNAVAILABLE:
        return AtlasRoutingStatus.SERVICE_UNAVAILABLE
    if status is CapabilityExecutionStatus.INVALID_REQUEST:
        return AtlasRoutingStatus.INVALID_REQUEST
    if status is CapabilityExecutionStatus.INTERNAL_ERROR:
        return AtlasRoutingStatus.INTERNAL_ERROR
    return AtlasRoutingStatus.EXECUTION_FAILED


def _routing_status_for_agent(status: AgentExecutionStatus) -> AtlasRoutingStatus:
    if status is AgentExecutionStatus.COMPLETED:
        return AtlasRoutingStatus.COMPLETED
    if status is AgentExecutionStatus.INVALID_REQUEST:
        return AtlasRoutingStatus.INVALID_REQUEST
    if status is AgentExecutionStatus.INTERNAL_ERROR:
        return AtlasRoutingStatus.INTERNAL_ERROR
    return AtlasRoutingStatus.EXECUTION_FAILED


def _record(
    events: list[AtlasRoutingEvent],
    observer: RoutingObserver | None,
    name: str,
    status: str,
    details: Mapping[str, object] | None = None,
) -> None:
    event = AtlasRoutingEvent(name, status, {} if details is None else details)
    events.append(event)
    if observer is not None:
        observer(event)


def _result(
    status: AtlasRoutingStatus,
    route_type: AtlasRouteType,
    events: list[AtlasRoutingEvent],
    *,
    output: object | None = None,
    error_code: str | None = None,
    message: str | None = None,
    request_id: str | None = None,
    capability_result: CapabilityExecutionResult | None = None,
    agent_result: AgentExecutionResult | None = None,
    agent_resolution_result: AgentResolutionResult | None = None,
    request_signature: str | None = None,
    metrics: Mapping[str, int] | None = None,
) -> AtlasRoutingResult:
    return AtlasRoutingResult(
        status=status,
        route_type=route_type,
        output=output,
        error_code=error_code,
        message=message,
        request_id=request_id,
        capability_result=capability_result,
        agent_result=agent_result,
        agent_resolution_result=agent_resolution_result,
        events=tuple(events),
        request_signature=request_signature,
        metrics={} if metrics is None else metrics,
    )


def _validate_route_type(value: AtlasRouteType | str) -> AtlasRouteType:
    if isinstance(value, AtlasRouteType):
        return value
    if isinstance(value, str):
        try:
            return AtlasRouteType(value.strip().lower())
        except ValueError as error:
            raise InvalidAtlasRoutingRequestError("invalid route_type.") from error
    raise InvalidAtlasRoutingRequestError("route_type must be AtlasRouteType or str.")


def _validate_status(value: AtlasRoutingStatus | str) -> AtlasRoutingStatus:
    if isinstance(value, AtlasRoutingStatus):
        return value
    if isinstance(value, str):
        try:
            return AtlasRoutingStatus(value)
        except ValueError as error:
            raise InvalidAtlasRoutingRequestError("invalid routing status.") from error
    raise InvalidAtlasRoutingRequestError("status must be AtlasRoutingStatus.")


def _safe_payload(value: object) -> object:
    if value is None or isinstance(value, (CapabilityExecutionRequest, AtlasAgentRoutingRequest)):
        return value
    return _copy_safe_payload(value, depth=0, counter={"nodes": 0})


def _safe_output(value: object) -> object:
    return _copy_safe_output(value, depth=0, counter={"nodes": 0})


def _copy_safe_output(value: object, *, depth: int, counter: dict[str, int]) -> object:
    if depth > MAX_ATLAS_ROUTING_PAYLOAD_DEPTH:
        return None
    counter["nodes"] += 1
    if counter["nodes"] > MAX_ATLAS_ROUTING_PAYLOAD_NODES:
        return None
    if _is_safe_primitive(value):
        return value
    if isinstance(value, Mapping):
        safe: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if not key:
                continue
            if _is_sensitive_key(key):
                safe[key] = "[redacted]"
                continue
            safe[key] = _copy_safe_output(raw_value, depth=depth + 1, counter=counter)
        return MappingProxyType(safe)
    if isinstance(value, (tuple, list)):
        return tuple(_copy_safe_output(item, depth=depth + 1, counter=counter) for item in value)
    return None


def _copy_safe_payload(value: object, *, depth: int, counter: dict[str, int]) -> object:
    if depth > MAX_ATLAS_ROUTING_PAYLOAD_DEPTH:
        raise InvalidAtlasRoutingRequestError("payload is too deep.")
    counter["nodes"] += 1
    if counter["nodes"] > MAX_ATLAS_ROUTING_PAYLOAD_NODES:
        raise InvalidAtlasRoutingRequestError("payload is too large.")
    if _is_safe_primitive(value):
        return value
    if isinstance(value, Mapping):
        safe: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise InvalidAtlasRoutingRequestError("payload keys must be non-empty strings.")
            key = raw_key.strip()
            if _is_sensitive_key(key):
                raise InvalidAtlasRoutingRequestError("payload cannot contain sensitive keys.")
            safe[key] = _copy_safe_payload(raw_value, depth=depth + 1, counter=counter)
        return MappingProxyType(safe)
    if isinstance(value, (tuple, list)):
        return tuple(_copy_safe_payload(item, depth=depth + 1, counter=counter) for item in value)
    raise InvalidAtlasRoutingRequestError("payload contains an unsupported value.")


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidAtlasRoutingRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_ATLAS_ROUTING_METADATA_ITEMS:
        raise InvalidAtlasRoutingRequestError("metadata has too many items.")
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise InvalidAtlasRoutingRequestError("metadata keys must be non-empty strings.")
        safe_key = key.strip()
        if _is_sensitive_key(safe_key):
            raise InvalidAtlasRoutingRequestError("metadata cannot contain sensitive keys.")
        if not _is_safe_primitive(value):
            raise InvalidAtlasRoutingRequestError("metadata values must be primitive safe values.")
        safe[safe_key] = value
    return safe


def _safe_metrics(metrics: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(metrics, Mapping):
        raise InvalidAtlasRoutingRequestError("metrics must be a mapping.")
    safe: dict[str, int] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise InvalidAtlasRoutingRequestError("metric keys must be non-empty strings.")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidAtlasRoutingRequestError("metric values must be non-negative integers.")
        safe[key.strip()] = value
    return safe


def _optional_safe_mapping(value: Mapping[str, object] | None, field_name: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise InvalidAtlasRoutingRequestError(f"{field_name} must be a mapping or None.")
    return _copy_safe_payload(value, depth=0, counter={"nodes": 0})


def _safe_identifier_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise InvalidAtlasRoutingRequestError(f"{field_name} must be a sequence of strings.")
    normalized = tuple(dict.fromkeys(_safe_identifier(value, field_name) for value in values))
    if len(normalized) > MAX_ATLAS_ROUTING_METADATA_ITEMS:
        raise InvalidAtlasRoutingRequestError(f"{field_name} has too many items.")
    return normalized


def _safe_sorted_identifier_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    return tuple(sorted(_safe_identifier_tuple(values, field_name)))


def _safe_agent_id_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise InvalidAtlasRoutingRequestError(f"{field_name} must be a sequence of strings.")
    try:
        normalized = tuple(dict.fromkeys(validate_agent_id(value) for value in values))
    except Exception as error:
        raise InvalidAtlasRoutingRequestError(f"{field_name} contains an invalid agent id.") from error
    if len(normalized) > MAX_ATLAS_ROUTING_METADATA_ITEMS:
        raise InvalidAtlasRoutingRequestError(f"{field_name} has too many items.")
    return normalized


def _safe_sorted_agent_id_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    return tuple(sorted(_safe_agent_id_tuple(values, field_name)))


def _safe_agent_type_tuple(values: Sequence[AgentType | str], field_name: str) -> tuple[AgentType, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise InvalidAtlasRoutingRequestError(f"{field_name} must be a sequence of agent types.")
    normalized: list[AgentType] = []
    for value in values:
        if isinstance(value, AgentType):
            agent_type = value
        elif isinstance(value, str):
            try:
                agent_type = AgentType(value.strip().lower())
            except ValueError as error:
                raise InvalidAtlasRoutingRequestError(f"{field_name} contains an invalid agent type.") from error
        else:
            raise InvalidAtlasRoutingRequestError(f"{field_name} contains an invalid agent type.")
        if agent_type not in normalized:
            normalized.append(agent_type)
    if len(normalized) > MAX_ATLAS_ROUTING_METADATA_ITEMS:
        raise InvalidAtlasRoutingRequestError(f"{field_name} has too many items.")
    return tuple(normalized)


def _agent_routing_request(value: object) -> AtlasAgentRoutingRequest:
    if isinstance(value, AtlasAgentRoutingRequest):
        return value
    if not isinstance(value, Mapping):
        raise InvalidAtlasRoutingRequestError("AGENT payload must be AtlasAgentRoutingRequest or mapping.")
    allowed = {
        "agent_id",
        "payload",
        "metadata",
        "shared_context",
        "required_capabilities",
        "required_permissions",
        "preferred_agent_types",
        "preferred_agent_ids",
        "excluded_agent_ids",
        "enabled_only",
        "task_id",
        "execution_id",
        "correlation_id",
        "session_id",
        "user_input",
    }
    unknown = tuple(sorted(str(key) for key in value.keys() if key not in allowed))
    if unknown:
        raise InvalidAtlasRoutingRequestError("AGENT payload contains unsupported fields.")
    agent_id = value.get("agent_id")
    if agent_id is not None and not isinstance(agent_id, str):
        raise InvalidAtlasRoutingRequestError("agent_id must be a string.")
    return AtlasAgentRoutingRequest(
        agent_id=agent_id,
        payload=value.get("payload"),  # type: ignore[arg-type]
        metadata=value.get("metadata", {}),  # type: ignore[arg-type]
        shared_context=value.get("shared_context"),  # type: ignore[arg-type]
        required_capabilities=tuple(value.get("required_capabilities", ())),  # type: ignore[arg-type]
        required_permissions=tuple(value.get("required_permissions", ())),  # type: ignore[arg-type]
        preferred_agent_types=tuple(value.get("preferred_agent_types", ())),  # type: ignore[arg-type]
        preferred_agent_ids=tuple(value.get("preferred_agent_ids", ())),  # type: ignore[arg-type]
        excluded_agent_ids=tuple(value.get("excluded_agent_ids", ())),  # type: ignore[arg-type]
        enabled_only=value.get("enabled_only", True),  # type: ignore[arg-type]
        task_id=value.get("task_id"),  # type: ignore[arg-type]
        execution_id=value.get("execution_id"),  # type: ignore[arg-type]
        correlation_id=value.get("correlation_id"),  # type: ignore[arg-type]
        session_id=value.get("session_id"),  # type: ignore[arg-type]
        user_input=value.get("user_input"),  # type: ignore[arg-type]
    )


def _agent_metrics(
    *,
    completed: int = 0,
    failed: int = 0,
    invalid: int = 0,
    unavailable: int = 0,
    auto_requested: int = 0,
    auto_succeeded: int = 0,
    auto_failed: int = 0,
    auto_no_match: int = 0,
    auto_ambiguous: int = 0,
    candidates_evaluated: int = 0,
    candidates_rejected: int = 0,
) -> Mapping[str, int]:
    return {
        "agent_route_completed": completed,
        "agent_route_failed": failed,
        "agent_route_invalid": invalid,
        "agent_route_unavailable": unavailable,
        "agent_auto_selections_requested": auto_requested,
        "agent_auto_selections_succeeded": auto_succeeded,
        "agent_auto_selections_failed": auto_failed,
        "agent_auto_selections_no_match": auto_no_match,
        "agent_auto_selections_ambiguous": auto_ambiguous,
        "agent_candidates_evaluated": candidates_evaluated,
        "agent_candidates_rejected": candidates_rejected,
    }


def _merge_metrics(*metrics: Mapping[str, int]) -> Mapping[str, int]:
    merged: dict[str, int] = {}
    for metric in metrics:
        for key, value in metric.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def _has_automatic_selection_criteria(request: AtlasAgentRoutingRequest) -> bool:
    return bool(
        request.required_capabilities
        or request.required_permissions
        or request.preferred_agent_types
        or request.preferred_agent_ids
    )


def _record_agent_selection_events(
    events: list[AtlasRoutingEvent],
    observer: RoutingObserver | None,
    result: AgentResolutionResult,
) -> None:
    for candidate in result.candidates:
        _record(
            events,
            observer,
            "agent_candidate_evaluated",
            "finished",
            {"agent_id": candidate.agent.agent_id, "score": candidate.score},
        )
        _record(
            events,
            observer,
            "agent_candidate_accepted",
            "finished",
            {"agent_id": candidate.agent.agent_id, "score": candidate.score},
        )
    for rejection in result.rejections:
        _record(
            events,
            observer,
            "agent_candidate_rejected",
            "finished",
            {"agent_id": rejection.agent_id, "reason": rejection.reason_code.value},
        )


def _signature_payload(value: object) -> object:
    if isinstance(value, AtlasAgentRoutingRequest):
        return {
            "type": "AtlasAgentRoutingRequest",
            "agent_id": value.agent_id,
            "payload": _signature_payload(value.payload),
            "metadata": _jsonable_mapping(value.metadata),
            "shared_context": None if value.shared_context is None else _jsonable_mapping(value.shared_context),
            "required_capabilities": tuple(sorted(value.required_capabilities)),
            "required_permissions": tuple(sorted(value.required_permissions)),
            "preferred_agent_types": tuple(item.value for item in value.preferred_agent_types),
            "preferred_agent_ids": tuple(value.preferred_agent_ids),
            "excluded_agent_ids": tuple(sorted(value.excluded_agent_ids)),
            "enabled_only": value.enabled_only,
            "task_id": value.task_id,
            "execution_id": value.execution_id,
            "correlation_id": value.correlation_id,
            "session_id": value.session_id,
            "user_input": value.user_input,
        }
    if isinstance(value, CapabilityExecutionRequest):
        return {
            "type": "CapabilityExecutionRequest",
            "objective": value.objective,
            "capability_id": value.capability_id,
            "capability_type": value.capability_type.value,
            "categories": tuple(value.categories),
            "excluded_categories": tuple(value.excluded_categories),
            "required_tags": tuple(value.required_tags),
            "preferred_tags": tuple(value.preferred_tags),
            "required_inputs": tuple(value.required_inputs),
            "required_outputs": tuple(value.required_outputs),
            "preferred_workflow_reference": (
                None
                if value.preferred_workflow_reference is None
                else {
                    "plan_id": value.preferred_workflow_reference.plan_id,
                    "version": value.preferred_workflow_reference.version,
                }
            ),
            "minimum_score": value.minimum_score,
            "minimum_workflow_score": value.minimum_workflow_score,
            "require_unique_top_score": value.require_unique_top_score,
            "enabled_only": value.enabled_only,
            "confirmation_granted": value.confirmation_granted,
            "metadata": _jsonable_mapping(value.metadata),
        }
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, tuple):
        return tuple(_signature_payload(item) for item in value)
    if _is_safe_primitive(value):
        return value
    return None


def _jsonable_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    return {key: _jsonable_value(value) for key, value in sorted(mapping.items(), key=lambda item: item[0])}


def _jsonable_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    return value


def _safe_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidAtlasRoutingRequestError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if _is_sensitive_key(normalized):
        raise InvalidAtlasRoutingRequestError(f"{field_name} cannot contain sensitive content.")
    return normalized[:200]


def _safe_message(message: str | None) -> str | None:
    if message is None:
        return None
    text = " ".join(str(message).split())[:300]
    for part in SENSITIVE_KEY_PARTS:
        text = text.replace(part, "[redacted]")
    return text


def _is_safe_primitive(value: object) -> bool:
    if value is None or isinstance(value, (str, int, bool)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)

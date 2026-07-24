"""Deterministic structured router for Atlas service requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable

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
class AtlasRoutingResult:
    """Immutable safe result returned by AtlasRouter."""

    status: AtlasRoutingStatus
    route_type: AtlasRouteType
    output: object | None = None
    error_code: str | None = None
    message: str | None = None
    request_id: str | None = None
    capability_result: CapabilityExecutionResult | None = None
    events: tuple[AtlasRoutingEvent, ...] = ()
    request_signature: str | None = None

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
        object.__setattr__(self, "events", tuple(self.events))

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
        AtlasRouteType.AGENT,
    }

    def __init__(
        self,
        capability_execution_service: CapabilityExecutionService | None = None,
        *,
        observer: RoutingObserver | None = None,
    ) -> None:
        if capability_execution_service is not None and not isinstance(
            capability_execution_service,
            CapabilityExecutionService,
        ):
            raise AtlasRoutingError("capability_execution_service must be CapabilityExecutionService or None.")
        if observer is not None and not callable(observer):
            raise AtlasRoutingError("observer must be callable or None.")
        self._capability_execution_service = capability_execution_service
        self._observer = observer

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
    request_signature: str | None = None,
) -> AtlasRoutingResult:
    return AtlasRoutingResult(
        status=status,
        route_type=route_type,
        output=output,
        error_code=error_code,
        message=message,
        request_id=request_id,
        capability_result=capability_result,
        events=tuple(events),
        request_signature=request_signature,
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
    if value is None or isinstance(value, CapabilityExecutionRequest):
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


def _signature_payload(value: object) -> object:
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

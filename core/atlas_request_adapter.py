"""Pure adaptation from structured Atlas requests to routing requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType

from core.atlas_router import (
    AtlasRouteType,
    AtlasRoutingRequest,
    AtlasRoutingResult,
    AtlasRoutingStatus,
)
from core.capability_execution_service import CapabilityExecutionRequest
from core.execution_plan_registry import ExecutionPlanReference


MAX_STRUCTURED_ATLAS_DEPTH = 8
MAX_STRUCTURED_ATLAS_NODES = 256
MAX_STRUCTURED_ATLAS_COLLECTION_ITEMS = 64
MAX_STRUCTURED_ATLAS_STRING_LENGTH = 500
MAX_STRUCTURED_ATLAS_METADATA_ITEMS = 32
SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)


class AtlasRequestAdaptationError(RuntimeError):
    """Base error for Atlas request adaptation."""


class InvalidStructuredAtlasRequestError(AtlasRequestAdaptationError):
    """Raised when a structured Atlas request is malformed."""


class AtlasRequestAdaptationStatus(str, Enum):
    """Stable outcomes for structured request adaptation."""

    ADAPTED = "adapted"
    INVALID_REQUEST = "invalid_request"
    INVALID_ROUTE_TYPE = "invalid_route_type"
    INVALID_PAYLOAD = "invalid_payload"
    INVALID_METADATA = "invalid_metadata"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class StructuredAtlasRequest:
    """Already-classified structured request before Atlas routing."""

    route_type: AtlasRouteType | str
    payload: Mapping[str, object] | None = None
    request_id: str | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _safe_request_id(self.request_id))
        object.__setattr__(
            self,
            "payload",
            None if self.payload is None else MappingProxyType(_safe_mapping(self.payload, field_name="payload")),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_safe_metadata({} if self.metadata is None else self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class AtlasRequestAdaptationResult:
    """Result of adapting a structured Atlas request to AtlasRoutingRequest."""

    status: AtlasRequestAdaptationStatus
    routing_request: AtlasRoutingRequest | None = None
    error_code: str | None = None
    message: str | None = None
    request_id: str | None = None
    route_type: AtlasRouteType | None = None
    request_signature: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _validate_status(self.status))
        if self.routing_request is not None and not isinstance(self.routing_request, AtlasRoutingRequest):
            raise InvalidStructuredAtlasRequestError("routing_request must be AtlasRoutingRequest or None.")
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _safe_request_id(self.request_id))
        if self.route_type is not None and not isinstance(self.route_type, AtlasRouteType):
            raise InvalidStructuredAtlasRequestError("route_type must be AtlasRouteType or None.")
        object.__setattr__(self, "message", _safe_message(self.message))

    @property
    def adapted(self) -> bool:
        """Return whether adaptation produced an AtlasRoutingRequest."""

        return self.status is AtlasRequestAdaptationStatus.ADAPTED


class AtlasRequestAdapter:
    """Validate and adapt already-classified structured Atlas requests."""

    def adapt(self, request: StructuredAtlasRequest) -> AtlasRequestAdaptationResult:
        """Adapt one structured request without calling Router or services."""

        if not isinstance(request, StructuredAtlasRequest):
            return AtlasRequestAdaptationResult(
                AtlasRequestAdaptationStatus.INVALID_REQUEST,
                error_code="INVALID_REQUEST",
                message="request must be StructuredAtlasRequest.",
            )

        route_type = _normalize_route_type(request.route_type)
        if route_type is None:
            return AtlasRequestAdaptationResult(
                AtlasRequestAdaptationStatus.INVALID_ROUTE_TYPE,
                error_code="INVALID_ROUTE_TYPE",
                message="route_type is invalid.",
                request_id=request.request_id,
            )

        signature = structured_atlas_request_signature(request)
        try:
            payload = self._payload_for_route(route_type, request.payload)
            routing_request = AtlasRoutingRequest(
                route_type=route_type,
                payload=payload,
                request_id=request.request_id,
                metadata=request.metadata or {},
            )
        except (InvalidStructuredAtlasRequestError, ValueError, TypeError) as error:
            return AtlasRequestAdaptationResult(
                AtlasRequestAdaptationStatus.INVALID_PAYLOAD,
                error_code="INVALID_PAYLOAD",
                message=str(error),
                request_id=request.request_id,
                route_type=route_type,
                request_signature=signature,
            )

        return AtlasRequestAdaptationResult(
            AtlasRequestAdaptationStatus.ADAPTED,
            routing_request=routing_request,
            request_id=request.request_id,
            route_type=route_type,
            request_signature=signature,
        )

    def _payload_for_route(
        self,
        route_type: AtlasRouteType,
        payload: Mapping[str, object] | None,
    ) -> object | None:
        if route_type is AtlasRouteType.CAPABILITY:
            return _capability_execution_request_from_payload(payload)
        return payload


def unavailable_atlas_request_adapter_result(
    request: StructuredAtlasRequest | object,
) -> AtlasRoutingResult:
    """Return a routing-compatible result when no adapter is configured."""

    route_type = request.route_type if isinstance(request, StructuredAtlasRequest) else AtlasRouteType.UNKNOWN
    normalized = _normalize_route_type(route_type) or AtlasRouteType.UNKNOWN
    request_id = request.request_id if isinstance(request, StructuredAtlasRequest) else None
    return AtlasRoutingResult(
        status=AtlasRoutingStatus.SERVICE_UNAVAILABLE,
        route_type=normalized,
        request_id=request_id,
        error_code="ATLAS_REQUEST_ADAPTER_UNAVAILABLE",
        message="Atlas request adapter is not configured.",
    )


def adaptation_failure_to_routing_result(
    result: AtlasRequestAdaptationResult,
) -> AtlasRoutingResult:
    """Translate a failed adaptation into a safe routing result."""

    if result.status is AtlasRequestAdaptationStatus.ADAPTED and result.routing_request is not None:
        raise InvalidStructuredAtlasRequestError("adapted result cannot be converted to failure routing result.")
    route_type = result.route_type or AtlasRouteType.UNKNOWN
    status = (
        AtlasRoutingStatus.SERVICE_UNAVAILABLE
        if result.status is AtlasRequestAdaptationStatus.ADAPTER_UNAVAILABLE
        else AtlasRoutingStatus.INVALID_REQUEST
    )
    return AtlasRoutingResult(
        status=status,
        route_type=route_type,
        request_id=result.request_id,
        error_code=result.error_code or result.status.value.upper(),
        message=result.message or "Structured Atlas request could not be adapted.",
        request_signature=result.request_signature,
    )


def structured_atlas_request_signature(request: StructuredAtlasRequest) -> str:
    """Return a deterministic signature for a safe structured Atlas request."""

    if not isinstance(request, StructuredAtlasRequest):
        raise InvalidStructuredAtlasRequestError("request must be StructuredAtlasRequest.")
    route_type = _normalize_route_type(request.route_type)
    if route_type is None:
        route_value = str(request.route_type).strip().lower() if isinstance(request.route_type, str) else None
    else:
        route_value = route_type.value
    payload = {
        "route_type": route_value,
        "payload": _jsonable_mapping({} if request.payload is None else request.payload),
        "request_id": request.request_id,
        "metadata": _jsonable_mapping(request.metadata or {}),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capability_execution_request_from_payload(
    payload: Mapping[str, object] | None,
) -> CapabilityExecutionRequest:
    data = {} if payload is None else dict(payload)
    preferred = data.get("preferred_workflow_reference")
    preferred_reference = None
    if preferred is not None:
        if not isinstance(preferred, Mapping):
            raise InvalidStructuredAtlasRequestError("preferred_workflow_reference must be a mapping.")
        plan_id = preferred.get("plan_id")
        version = preferred.get("version")
        if not isinstance(plan_id, str) or not plan_id.strip():
            raise InvalidStructuredAtlasRequestError("preferred_workflow_reference.plan_id must be a string.")
        if version is not None and not isinstance(version, str):
            raise InvalidStructuredAtlasRequestError("preferred_workflow_reference.version must be a string or None.")
        preferred_reference = ExecutionPlanReference(plan_id, version)

    return CapabilityExecutionRequest(
        objective=_string_value(data.get("objective"), "execute capability"),
        capability_id=_optional_string(data.get("capability_id"), "capability_id"),
        capability_type=_optional_string(data.get("capability_type"), "capability_type"),
        categories=_string_tuple(data.get("categories")),
        excluded_categories=_string_tuple(data.get("excluded_categories")),
        required_tags=_string_tuple(data.get("required_tags")),
        preferred_tags=_string_tuple(data.get("preferred_tags")),
        required_inputs=_string_tuple(data.get("required_inputs")),
        required_outputs=_string_tuple(data.get("required_outputs")),
        preferred_workflow_reference=preferred_reference,
        minimum_score=_int_value(data.get("minimum_score"), "minimum_score", 0),
        minimum_workflow_score=_int_value(data.get("minimum_workflow_score"), "minimum_workflow_score", 0),
        require_unique_top_score=_bool_value(data.get("require_unique_top_score"), "require_unique_top_score", True),
        enabled_only=_bool_value(data.get("enabled_only"), "enabled_only", True),
        confirmation_granted=_bool_value(data.get("confirmation_granted"), "confirmation_granted", False),
        inputs=_execution_inputs_from_payload(data),
        metadata=_metadata_from_payload(data.get("metadata")),
    )


_CAPABILITY_REQUEST_FIELDS = frozenset(
    {
        "objective",
        "capability_id",
        "capability_type",
        "categories",
        "excluded_categories",
        "required_tags",
        "preferred_tags",
        "required_inputs",
        "required_outputs",
        "preferred_workflow_reference",
        "minimum_score",
        "minimum_workflow_score",
        "require_unique_top_score",
        "enabled_only",
        "confirmation_granted",
        "metadata",
        "inputs",
    }
)


def _execution_inputs_from_payload(data: Mapping[str, object]) -> Mapping[str, object]:
    explicit = data.get("inputs")
    inputs: dict[str, object] = {}
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            raise InvalidStructuredAtlasRequestError("inputs must be a mapping.")
        inputs.update(dict(explicit))
    for key, value in data.items():
        if key in _CAPABILITY_REQUEST_FIELDS:
            continue
        if key in inputs:
            raise InvalidStructuredAtlasRequestError("inputs cannot duplicate direct payload fields.")
        inputs[key] = value
    return inputs


def _normalize_route_type(value: AtlasRouteType | str) -> AtlasRouteType | None:
    if isinstance(value, AtlasRouteType):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return AtlasRouteType(value.strip().lower())
    except ValueError:
        return None


def _validate_status(value: AtlasRequestAdaptationStatus | str) -> AtlasRequestAdaptationStatus:
    if isinstance(value, AtlasRequestAdaptationStatus):
        return value
    if isinstance(value, str):
        try:
            return AtlasRequestAdaptationStatus(value)
        except ValueError as error:
            raise InvalidStructuredAtlasRequestError("invalid adaptation status.") from error
    raise InvalidStructuredAtlasRequestError("status must be AtlasRequestAdaptationStatus.")


def _safe_mapping(mapping: Mapping[str, object], *, field_name: str) -> dict[str, object]:
    if not isinstance(mapping, Mapping):
        raise InvalidStructuredAtlasRequestError(f"{field_name} must be a mapping.")
    safe = _copy_safe_value(mapping, field_name=field_name, depth=0, counter={"nodes": 0})
    if not isinstance(safe, Mapping):
        raise InvalidStructuredAtlasRequestError(f"{field_name} must be a mapping.")
    return dict(safe)


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if len(metadata) > MAX_STRUCTURED_ATLAS_METADATA_ITEMS:
        raise InvalidStructuredAtlasRequestError("metadata has too many items.")
    return _safe_mapping(metadata, field_name="metadata")


def _copy_safe_value(
    value: object,
    *,
    field_name: str,
    depth: int,
    counter: dict[str, int],
) -> object:
    if depth > MAX_STRUCTURED_ATLAS_DEPTH:
        raise InvalidStructuredAtlasRequestError(f"{field_name} is too deep.")
    counter["nodes"] += 1
    if counter["nodes"] > MAX_STRUCTURED_ATLAS_NODES:
        raise InvalidStructuredAtlasRequestError(f"{field_name} has too many nodes.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidStructuredAtlasRequestError(f"{field_name} floats must be finite.")
        return value
    if isinstance(value, str):
        if len(value) > MAX_STRUCTURED_ATLAS_STRING_LENGTH:
            raise InvalidStructuredAtlasRequestError(f"{field_name} strings are too long.")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_STRUCTURED_ATLAS_COLLECTION_ITEMS:
            raise InvalidStructuredAtlasRequestError(f"{field_name} has too many items.")
        copied: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise InvalidStructuredAtlasRequestError(f"{field_name} keys must be non-empty strings.")
            key = raw_key.strip()
            if _is_sensitive_key(key):
                raise InvalidStructuredAtlasRequestError(f"{field_name} cannot contain sensitive keys.")
            copied[key] = _copy_safe_value(raw_value, field_name=field_name, depth=depth + 1, counter=counter)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_STRUCTURED_ATLAS_COLLECTION_ITEMS:
            raise InvalidStructuredAtlasRequestError(f"{field_name} has too many items.")
        return tuple(
            _copy_safe_value(item, field_name=field_name, depth=depth + 1, counter=counter)
            for item in value
        )
    raise InvalidStructuredAtlasRequestError(f"{field_name} contains unsupported value.")


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (tuple, list)):
        raise InvalidStructuredAtlasRequestError("expected a string list.")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise InvalidStructuredAtlasRequestError("expected non-empty string values.")
        result.append(item)
    return tuple(result)


def _metadata_from_payload(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InvalidStructuredAtlasRequestError("capability metadata must be a mapping.")
    return dict(value)


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidStructuredAtlasRequestError(f"{field_name} must be a string or None.")
    return value


def _string_value(value: object, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise InvalidStructuredAtlasRequestError("objective must be a string.")
    return value


def _int_value(value: object, field_name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidStructuredAtlasRequestError(f"{field_name} must be an int.")
    return value


def _bool_value(value: object, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise InvalidStructuredAtlasRequestError(f"{field_name} must be a bool.")
    return value


def _safe_request_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidStructuredAtlasRequestError("request_id must be a non-empty string.")
    if len(value) > 200:
        raise InvalidStructuredAtlasRequestError("request_id is too long.")
    if _is_sensitive_key(value):
        raise InvalidStructuredAtlasRequestError("request_id cannot contain sensitive content.")
    return value.strip()


def _safe_message(message: str | None) -> str | None:
    if message is None:
        return None
    text = " ".join(str(message).split())[:300]
    for part in SENSITIVE_KEY_PARTS:
        text = text.replace(part, "[redacted]")
    return text


def _jsonable_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    return {key: _jsonable_value(value) for key, value in sorted(mapping.items(), key=lambda item: item[0])}


def _jsonable_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)

"""Deterministic classification for structured Atlas inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType

from core.atlas_request_adapter import (
    StructuredAtlasRequest,
    adaptation_failure_to_routing_result,
    unavailable_atlas_request_adapter_result,
)
from core.atlas_router import AtlasRouteType, AtlasRoutingResult, AtlasRoutingStatus


MAX_STRUCTURED_INPUT_DEPTH = 8
MAX_STRUCTURED_INPUT_NODES = 256
MAX_STRUCTURED_INPUT_COLLECTION_ITEMS = 64
MAX_STRUCTURED_INPUT_STRING_LENGTH = 500
MAX_STRUCTURED_INPUT_METADATA_ITEMS = 32
SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)


class AtlasRequestClassificationError(RuntimeError):
    """Base error for Atlas request classification."""


class InvalidStructuredInputError(AtlasRequestClassificationError):
    """Raised when a structured input contains unsafe values."""


class AtlasRequestClassificationStatus(str, Enum):
    """Stable outcomes for deterministic structured input classification."""

    CLASSIFIED = "classified"
    UNKNOWN = "unknown"
    INVALID_INPUT = "invalid_input"
    INVALID_ROUTE = "invalid_route"
    INVALID_METADATA = "invalid_metadata"
    INVALID_PAYLOAD = "invalid_payload"
    CLASSIFIER_UNAVAILABLE = "classifier_unavailable"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class StructuredInput:
    """Already-structured input before Atlas request adaptation."""

    kind: str | None = None
    capability_id: str | None = None
    workflow_id: str | None = None
    tool_name: str | None = None
    route: AtlasRouteType | str | None = None
    metadata: Mapping[str, object] | None = None
    payload: Mapping[str, object] | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _optional_identifier(self.kind, "kind", allow_route_name=True))
        object.__setattr__(self, "capability_id", _optional_identifier(self.capability_id, "capability_id"))
        object.__setattr__(self, "workflow_id", _optional_identifier(self.workflow_id, "workflow_id"))
        object.__setattr__(self, "tool_name", _optional_identifier(self.tool_name, "tool_name"))
        if self.route is not None and not isinstance(self.route, (AtlasRouteType, str)):
            raise InvalidStructuredInputError("route must be AtlasRouteType, str, or None.")
        if isinstance(self.route, str) and len(self.route) > MAX_STRUCTURED_INPUT_STRING_LENGTH:
            raise InvalidStructuredInputError("route is too long.")
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _safe_request_id(self.request_id))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_safe_metadata({} if self.metadata is None else self.metadata)),
        )
        object.__setattr__(
            self,
            "payload",
            None if self.payload is None else MappingProxyType(_safe_mapping(self.payload, field_name="payload")),
        )


@dataclass(frozen=True, slots=True)
class AtlasRequestClassificationResult:
    """Result of classifying one StructuredInput."""

    status: AtlasRequestClassificationStatus
    structured_request: StructuredAtlasRequest | None = None
    route_type: AtlasRouteType | None = None
    error_code: str | None = None
    message: str | None = None
    request_id: str | None = None
    input_signature: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _validate_status(self.status))
        if self.structured_request is not None and not isinstance(self.structured_request, StructuredAtlasRequest):
            raise InvalidStructuredInputError("structured_request must be StructuredAtlasRequest or None.")
        if self.route_type is not None and not isinstance(self.route_type, AtlasRouteType):
            raise InvalidStructuredInputError("route_type must be AtlasRouteType or None.")
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _safe_request_id(self.request_id))
        object.__setattr__(self, "message", _safe_message(self.message))

    @property
    def classified(self) -> bool:
        """Return whether classification produced a StructuredAtlasRequest."""

        return self.status is AtlasRequestClassificationStatus.CLASSIFIED


class AtlasRequestClassifier:
    """Classify structured inputs through explicit deterministic rules only."""

    def classify(self, structured_input: StructuredInput) -> AtlasRequestClassificationResult:
        """Classify one structured input without calling Adapter, Router, or services."""

        if not isinstance(structured_input, StructuredInput):
            return AtlasRequestClassificationResult(
                AtlasRequestClassificationStatus.INVALID_INPUT,
                error_code="INVALID_INPUT",
                message="structured_input must be StructuredInput.",
            )

        signature = atlas_request_classification_signature(structured_input)
        route = _route_from_explicit_fields(structured_input)
        if route is _InvalidRoute:
            return AtlasRequestClassificationResult(
                AtlasRequestClassificationStatus.INVALID_ROUTE,
                error_code="INVALID_ROUTE",
                message="route is invalid.",
                request_id=structured_input.request_id,
                input_signature=signature,
            )
        if route is None:
            return AtlasRequestClassificationResult(
                AtlasRequestClassificationStatus.UNKNOWN,
                route_type=AtlasRouteType.UNKNOWN,
                error_code="UNKNOWN_ROUTE",
                message="No deterministic route could be selected.",
                request_id=structured_input.request_id,
                input_signature=signature,
            )

        try:
            request = StructuredAtlasRequest(
                route,
                payload=_payload_for_route(route, structured_input),
                request_id=structured_input.request_id,
                metadata=structured_input.metadata,
            )
        except (InvalidStructuredInputError, ValueError, TypeError) as error:
            return AtlasRequestClassificationResult(
                AtlasRequestClassificationStatus.INTERNAL_ERROR,
                route_type=route,
                error_code="INTERNAL_ERROR",
                message=str(error),
                request_id=structured_input.request_id,
                input_signature=signature,
            )

        return AtlasRequestClassificationResult(
            AtlasRequestClassificationStatus.CLASSIFIED,
            structured_request=request,
            route_type=route,
            request_id=structured_input.request_id,
            input_signature=signature,
        )


def unavailable_atlas_request_classifier_result(
    structured_input: StructuredInput | object,
) -> AtlasRoutingResult:
    """Return a routing-compatible result when no classifier is configured."""

    request_id = structured_input.request_id if isinstance(structured_input, StructuredInput) else None
    return AtlasRoutingResult(
        status=AtlasRoutingStatus.SERVICE_UNAVAILABLE,
        route_type=AtlasRouteType.UNKNOWN,
        request_id=request_id,
        error_code="ATLAS_REQUEST_CLASSIFIER_UNAVAILABLE",
        message="Atlas request classifier is not configured.",
    )


def classification_failure_to_routing_result(
    result: AtlasRequestClassificationResult,
) -> AtlasRoutingResult:
    """Translate failed classification into a safe routing result."""

    if result.status is AtlasRequestClassificationStatus.CLASSIFIED and result.structured_request is not None:
        raise InvalidStructuredInputError("classified result cannot be converted to failure routing result.")
    status = (
        AtlasRoutingStatus.SERVICE_UNAVAILABLE
        if result.status is AtlasRequestClassificationStatus.CLASSIFIER_UNAVAILABLE
        else AtlasRoutingStatus.INVALID_REQUEST
    )
    return AtlasRoutingResult(
        status=status,
        route_type=result.route_type or AtlasRouteType.UNKNOWN,
        request_id=result.request_id,
        error_code=result.error_code or result.status.value.upper(),
        message=result.message or "Structured input could not be classified.",
        request_signature=result.input_signature,
    )


def atlas_request_classification_signature(structured_input: StructuredInput) -> str:
    """Return a deterministic signature for safe structured input fields."""

    if not isinstance(structured_input, StructuredInput):
        raise InvalidStructuredInputError("structured_input must be StructuredInput.")
    route_value = (
        structured_input.route.value
        if isinstance(structured_input.route, AtlasRouteType)
        else structured_input.route
    )
    payload = {
        "kind": structured_input.kind,
        "capability_id": structured_input.capability_id,
        "workflow_id": structured_input.workflow_id,
        "tool_name": structured_input.tool_name,
        "route": route_value.strip().lower() if isinstance(route_value, str) else route_value,
        "metadata": _jsonable_mapping(structured_input.metadata or {}),
        "payload": _jsonable_mapping(structured_input.payload or {}),
        "request_id": structured_input.request_id,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_InvalidRoute = object()


def _route_from_explicit_fields(structured_input: StructuredInput) -> AtlasRouteType | object | None:
    if structured_input.capability_id is not None:
        return AtlasRouteType.CAPABILITY
    if structured_input.route is not None:
        route = _normalize_route(structured_input.route)
        return route if route is not None else _InvalidRoute
    if structured_input.kind is not None:
        route = _normalize_route(structured_input.kind)
        if route is not None:
            return route
    if structured_input.workflow_id is not None:
        return AtlasRouteType.WORKFLOW
    if structured_input.tool_name is not None:
        return AtlasRouteType.TOOL
    return None


def _payload_for_route(route: AtlasRouteType, structured_input: StructuredInput) -> Mapping[str, object] | None:
    payload = {} if structured_input.payload is None else dict(structured_input.payload)
    if route is AtlasRouteType.CAPABILITY and structured_input.capability_id is not None:
        payload.setdefault("capability_id", structured_input.capability_id)
    if route is AtlasRouteType.WORKFLOW and structured_input.workflow_id is not None:
        payload.setdefault("workflow_id", structured_input.workflow_id)
    if route is AtlasRouteType.TOOL and structured_input.tool_name is not None:
        payload.setdefault("tool_name", structured_input.tool_name)
    return payload or None


def _normalize_route(value: AtlasRouteType | str) -> AtlasRouteType | None:
    if isinstance(value, AtlasRouteType):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return AtlasRouteType(value.strip().lower())
    except ValueError:
        return None


def _validate_status(value: AtlasRequestClassificationStatus | str) -> AtlasRequestClassificationStatus:
    if isinstance(value, AtlasRequestClassificationStatus):
        return value
    if isinstance(value, str):
        try:
            return AtlasRequestClassificationStatus(value)
        except ValueError as error:
            raise InvalidStructuredInputError("invalid classification status.") from error
    raise InvalidStructuredInputError("status must be AtlasRequestClassificationStatus.")


def _safe_mapping(mapping: Mapping[str, object], *, field_name: str) -> dict[str, object]:
    if not isinstance(mapping, Mapping):
        raise InvalidStructuredInputError(f"{field_name} must be a mapping.")
    safe = _copy_safe_value(mapping, field_name=field_name, depth=0, counter={"nodes": 0})
    if not isinstance(safe, Mapping):
        raise InvalidStructuredInputError(f"{field_name} must be a mapping.")
    return dict(safe)


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if len(metadata) > MAX_STRUCTURED_INPUT_METADATA_ITEMS:
        raise InvalidStructuredInputError("metadata has too many items.")
    return _safe_mapping(metadata, field_name="metadata")


def _copy_safe_value(
    value: object,
    *,
    field_name: str,
    depth: int,
    counter: dict[str, int],
) -> object:
    if depth > MAX_STRUCTURED_INPUT_DEPTH:
        raise InvalidStructuredInputError(f"{field_name} is too deep.")
    counter["nodes"] += 1
    if counter["nodes"] > MAX_STRUCTURED_INPUT_NODES:
        raise InvalidStructuredInputError(f"{field_name} has too many nodes.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidStructuredInputError(f"{field_name} floats must be finite.")
        return value
    if isinstance(value, str):
        if len(value) > MAX_STRUCTURED_INPUT_STRING_LENGTH:
            raise InvalidStructuredInputError(f"{field_name} strings are too long.")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_STRUCTURED_INPUT_COLLECTION_ITEMS:
            raise InvalidStructuredInputError(f"{field_name} has too many items.")
        copied: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise InvalidStructuredInputError(f"{field_name} keys must be non-empty strings.")
            key = raw_key.strip()
            if _is_sensitive_key(key):
                raise InvalidStructuredInputError(f"{field_name} cannot contain sensitive keys.")
            copied[key] = _copy_safe_value(raw_value, field_name=field_name, depth=depth + 1, counter=counter)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_STRUCTURED_INPUT_COLLECTION_ITEMS:
            raise InvalidStructuredInputError(f"{field_name} has too many items.")
        return tuple(
            _copy_safe_value(item, field_name=field_name, depth=depth + 1, counter=counter)
            for item in value
        )
    raise InvalidStructuredInputError(f"{field_name} contains unsupported value.")


def _optional_identifier(value: str | None, field_name: str, *, allow_route_name: bool = False) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InvalidStructuredInputError(f"{field_name} must be a non-empty string or None.")
    if len(value) > MAX_STRUCTURED_INPUT_STRING_LENGTH:
        raise InvalidStructuredInputError(f"{field_name} is too long.")
    if _is_sensitive_key(value):
        raise InvalidStructuredInputError(f"{field_name} cannot contain sensitive content.")
    normalized = value.strip()
    return normalized.lower() if allow_route_name else normalized


def _safe_request_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidStructuredInputError("request_id must be a non-empty string.")
    if len(value) > 200:
        raise InvalidStructuredInputError("request_id is too long.")
    if _is_sensitive_key(value):
        raise InvalidStructuredInputError("request_id cannot contain sensitive content.")
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

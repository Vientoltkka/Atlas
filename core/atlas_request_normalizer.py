"""Canonical normalization for structured Atlas inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType

from core.atlas_request_classifier import StructuredInput
from core.atlas_router import AtlasRouteType, AtlasRoutingResult, AtlasRoutingStatus


MAX_ATLAS_REQUEST_NORMALIZATION_DEPTH = 8
MAX_ATLAS_REQUEST_NORMALIZATION_NODES = 256
MAX_ATLAS_REQUEST_NORMALIZATION_COLLECTION_ITEMS = 64
MAX_ATLAS_REQUEST_NORMALIZATION_STRING_LENGTH = 500
MAX_ATLAS_REQUEST_NORMALIZATION_METADATA_ITEMS = 32
MAX_ATLAS_REQUEST_NORMALIZATION_REQUEST_ID_LENGTH = 200
SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)


class AtlasRequestNormalizationError(RuntimeError):
    """Base error for Atlas request normalization."""


class InvalidAtlasRequestNormalizationInputError(AtlasRequestNormalizationError):
    """Raised when structured input cannot be normalized safely."""


class AtlasRequestNormalizationStatus(str, Enum):
    """Stable outcomes for deterministic structured input normalization."""

    NORMALIZED = "normalized"
    INVALID_INPUT = "invalid_input"
    NORMALIZER_UNAVAILABLE = "normalizer_unavailable"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class AtlasRequestNormalizationResult:
    """Result of canonicalizing one StructuredInput."""

    status: AtlasRequestNormalizationStatus
    structured_input: StructuredInput | None = None
    error_code: str | None = None
    message: str | None = None
    request_id: str | None = None
    input_signature: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _validate_status(self.status))
        if self.structured_input is not None and not isinstance(self.structured_input, StructuredInput):
            raise InvalidAtlasRequestNormalizationInputError(
                "structured_input must be StructuredInput or None."
            )
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _normalize_request_id(self.request_id))
        object.__setattr__(self, "message", _safe_message(self.message))

    @property
    def normalized(self) -> bool:
        """Return whether normalization produced canonical StructuredInput."""

        return self.status is AtlasRequestNormalizationStatus.NORMALIZED


class AtlasRequestNormalizer:
    """Normalize already-structured inputs without interpreting free text."""

    def normalize(self, structured_input: StructuredInput) -> AtlasRequestNormalizationResult:
        """Return a canonical, immutable StructuredInput or a structured failure."""

        if not isinstance(structured_input, StructuredInput):
            return AtlasRequestNormalizationResult(
                AtlasRequestNormalizationStatus.INVALID_INPUT,
                error_code="INVALID_INPUT",
                message="structured_input must be StructuredInput.",
            )

        try:
            normalized = StructuredInput(
                kind=_normalize_identifier(
                    structured_input.kind,
                    "kind",
                    allow_route_name=True,
                ),
                capability_id=_normalize_identifier(structured_input.capability_id, "capability_id"),
                workflow_id=_normalize_identifier(structured_input.workflow_id, "workflow_id"),
                tool_name=_normalize_identifier(structured_input.tool_name, "tool_name"),
                route=_normalize_route_name(structured_input.route),
                metadata=_normalize_metadata(structured_input.metadata or {}),
                payload=(
                    None
                    if structured_input.payload is None
                    else _normalize_mapping(structured_input.payload, field_name="payload")
                ),
                request_id=(
                    None
                    if structured_input.request_id is None
                    else _normalize_request_id(structured_input.request_id)
                ),
            )
            signature = atlas_request_normalization_signature(normalized)
            return AtlasRequestNormalizationResult(
                AtlasRequestNormalizationStatus.NORMALIZED,
                structured_input=normalized,
                request_id=normalized.request_id,
                input_signature=signature,
            )
        except InvalidAtlasRequestNormalizationInputError as error:
            return AtlasRequestNormalizationResult(
                AtlasRequestNormalizationStatus.INVALID_INPUT,
                error_code="INVALID_INPUT",
                message=str(error),
                request_id=getattr(structured_input, "request_id", None),
            )
        except (TypeError, ValueError) as error:
            return AtlasRequestNormalizationResult(
                AtlasRequestNormalizationStatus.INVALID_INPUT,
                error_code="INVALID_INPUT",
                message=str(error),
                request_id=getattr(structured_input, "request_id", None),
            )
        except Exception:
            return AtlasRequestNormalizationResult(
                AtlasRequestNormalizationStatus.INTERNAL_ERROR,
                error_code="INTERNAL_ERROR",
                message="Atlas request normalization failed.",
                request_id=getattr(structured_input, "request_id", None),
            )


def unavailable_atlas_request_normalizer_result(
    structured_input: StructuredInput | object,
) -> AtlasRoutingResult:
    """Return a routing-compatible result when no normalizer is configured."""

    request_id = structured_input.request_id if isinstance(structured_input, StructuredInput) else None
    return AtlasRoutingResult(
        status=AtlasRoutingStatus.SERVICE_UNAVAILABLE,
        route_type=AtlasRouteType.UNKNOWN,
        request_id=request_id,
        error_code="ATLAS_REQUEST_NORMALIZER_UNAVAILABLE",
        message="Atlas request normalizer is not configured.",
    )


def normalization_failure_to_routing_result(
    result: AtlasRequestNormalizationResult,
) -> AtlasRoutingResult:
    """Translate failed normalization into a safe routing result."""

    if result.status is AtlasRequestNormalizationStatus.NORMALIZED and result.structured_input is not None:
        raise InvalidAtlasRequestNormalizationInputError(
            "normalized result cannot be converted to failure routing result."
        )
    if result.status is AtlasRequestNormalizationStatus.NORMALIZER_UNAVAILABLE:
        routing_status = AtlasRoutingStatus.SERVICE_UNAVAILABLE
    elif result.status is AtlasRequestNormalizationStatus.INTERNAL_ERROR:
        routing_status = AtlasRoutingStatus.INTERNAL_ERROR
    else:
        routing_status = AtlasRoutingStatus.INVALID_REQUEST
    return AtlasRoutingResult(
        status=routing_status,
        route_type=AtlasRouteType.UNKNOWN,
        request_id=result.request_id,
        error_code=result.error_code or result.status.value.upper(),
        message=result.message or "Structured input could not be normalized.",
        request_signature=result.input_signature,
    )


def atlas_request_normalization_signature(structured_input: StructuredInput) -> str:
    """Return a deterministic signature for canonical structured input."""

    if not isinstance(structured_input, StructuredInput):
        raise InvalidAtlasRequestNormalizationInputError("structured_input must be StructuredInput.")
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
        "payload": (
            None
            if structured_input.payload is None
            else _jsonable_mapping(structured_input.payload)
        ),
        "request_id": structured_input.request_id,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_status(
    value: AtlasRequestNormalizationStatus | str,
) -> AtlasRequestNormalizationStatus:
    if isinstance(value, AtlasRequestNormalizationStatus):
        return value
    if isinstance(value, str):
        try:
            return AtlasRequestNormalizationStatus(value)
        except ValueError as error:
            raise InvalidAtlasRequestNormalizationInputError("invalid normalization status.") from error
    raise InvalidAtlasRequestNormalizationInputError("status must be AtlasRequestNormalizationStatus.")


def _normalize_identifier(
    value: str | None,
    field_name: str,
    *,
    allow_route_name: bool = False,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidAtlasRequestNormalizationInputError(f"{field_name} must be a string or None.")
    normalized = value.strip()
    if not normalized:
        raise InvalidAtlasRequestNormalizationInputError(f"{field_name} must be a non-empty string or None.")
    if len(normalized) > MAX_ATLAS_REQUEST_NORMALIZATION_STRING_LENGTH:
        raise InvalidAtlasRequestNormalizationInputError(f"{field_name} is too long.")
    if _is_sensitive_key(normalized):
        raise InvalidAtlasRequestNormalizationInputError(f"{field_name} cannot contain sensitive content.")
    return normalized.lower() if allow_route_name else normalized


def _normalize_request_id(value: str) -> str:
    if not isinstance(value, str):
        raise InvalidAtlasRequestNormalizationInputError("request_id must be a string.")
    normalized = value.strip()
    if not normalized:
        raise InvalidAtlasRequestNormalizationInputError("request_id must be a non-empty string.")
    if len(normalized) > MAX_ATLAS_REQUEST_NORMALIZATION_REQUEST_ID_LENGTH:
        raise InvalidAtlasRequestNormalizationInputError("request_id is too long.")
    if _is_sensitive_key(normalized):
        raise InvalidAtlasRequestNormalizationInputError("request_id cannot contain sensitive content.")
    return normalized


def _normalize_route_name(value: AtlasRouteType | str | None) -> AtlasRouteType | str | None:
    if value is None or isinstance(value, AtlasRouteType):
        return value
    if not isinstance(value, str):
        raise InvalidAtlasRequestNormalizationInputError("route must be AtlasRouteType, str, or None.")
    normalized = value.strip().lower()
    if not normalized:
        raise InvalidAtlasRequestNormalizationInputError("route must be a non-empty string or None.")
    if len(normalized) > MAX_ATLAS_REQUEST_NORMALIZATION_STRING_LENGTH:
        raise InvalidAtlasRequestNormalizationInputError("route is too long.")
    if _is_sensitive_key(normalized):
        raise InvalidAtlasRequestNormalizationInputError("route cannot contain sensitive content.")
    return normalized


def _normalize_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    if len(metadata) > MAX_ATLAS_REQUEST_NORMALIZATION_METADATA_ITEMS:
        raise InvalidAtlasRequestNormalizationInputError("metadata has too many items.")
    return _normalize_mapping(metadata, field_name="metadata")


def _normalize_mapping(mapping: Mapping[str, object], *, field_name: str) -> Mapping[str, object]:
    if not isinstance(mapping, Mapping):
        raise InvalidAtlasRequestNormalizationInputError(f"{field_name} must be a mapping.")
    normalized = _normalize_value(mapping, field_name=field_name, depth=0, counter={"nodes": 0})
    if not isinstance(normalized, Mapping):
        raise InvalidAtlasRequestNormalizationInputError(f"{field_name} must be a mapping.")
    return normalized


def _normalize_value(
    value: object,
    *,
    field_name: str,
    depth: int,
    counter: dict[str, int],
) -> object:
    if depth > MAX_ATLAS_REQUEST_NORMALIZATION_DEPTH:
        raise InvalidAtlasRequestNormalizationInputError(f"{field_name} is too deep.")
    counter["nodes"] += 1
    if counter["nodes"] > MAX_ATLAS_REQUEST_NORMALIZATION_NODES:
        raise InvalidAtlasRequestNormalizationInputError(f"{field_name} has too many nodes.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAtlasRequestNormalizationInputError(f"{field_name} floats must be finite.")
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if len(normalized) > MAX_ATLAS_REQUEST_NORMALIZATION_STRING_LENGTH:
            raise InvalidAtlasRequestNormalizationInputError(f"{field_name} strings are too long.")
        return normalized
    if isinstance(value, Mapping):
        if len(value) > MAX_ATLAS_REQUEST_NORMALIZATION_COLLECTION_ITEMS:
            raise InvalidAtlasRequestNormalizationInputError(f"{field_name} has too many items.")
        copied: dict[str, object] = {}
        for raw_key, raw_value in sorted(value.items(), key=lambda item: _canonical_key(item[0], field_name)):
            key = _canonical_key(raw_key, field_name)
            if key in copied:
                raise InvalidAtlasRequestNormalizationInputError(f"{field_name} has duplicate canonical keys.")
            if _is_sensitive_key(key):
                raise InvalidAtlasRequestNormalizationInputError(f"{field_name} cannot contain sensitive keys.")
            copied[key] = _normalize_value(
                raw_value,
                field_name=field_name,
                depth=depth + 1,
                counter=counter,
            )
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_ATLAS_REQUEST_NORMALIZATION_COLLECTION_ITEMS:
            raise InvalidAtlasRequestNormalizationInputError(f"{field_name} has too many items.")
        return tuple(
            _normalize_value(item, field_name=field_name, depth=depth + 1, counter=counter)
            for item in value
        )
    raise InvalidAtlasRequestNormalizationInputError(f"{field_name} contains unsupported value.")


def _canonical_key(key: object, field_name: str) -> str:
    if not isinstance(key, str):
        raise InvalidAtlasRequestNormalizationInputError(f"{field_name} keys must be strings.")
    normalized = key.strip()
    if not normalized:
        raise InvalidAtlasRequestNormalizationInputError(f"{field_name} keys must be non-empty strings.")
    if len(normalized) > MAX_ATLAS_REQUEST_NORMALIZATION_STRING_LENGTH:
        raise InvalidAtlasRequestNormalizationInputError(f"{field_name} keys are too long.")
    return normalized


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

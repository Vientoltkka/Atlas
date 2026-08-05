"""Safe declarative skill manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType

from core.skill_registry import (
    InvalidSkillDefinitionError,
    SkillDefinition,
    SkillExecutionTargetType,
    SkillFieldDefinition,
    SkillLimits,
)


SKILL_SCHEMA_VERSION = "1.0"
MAX_SKILL_MANIFEST_BYTES = 64_000
_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "skill_id",
        "name",
        "version",
        "description",
        "enabled",
        "required_capability_ids",
        "required_permission_ids",
        "allowed_agent_types",
        "input_names",
        "output_names",
        "input_fields",
        "output_fields",
        "execution_target",
        "execution_target_type",
        "limits",
        "metadata",
        "tags",
        "handler_id",
        "workflow_reference",
    }
)
_ALLOWED_SKILL_FIELD_FIELDS = frozenset({"name", "type", "required"})
_SENSITIVE_KEY_PARTS = (
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "authorization",
    "cookie",
    "private_key",
    "credential",
)


class SkillManifestStatus(str, Enum):
    """Manifest validation status."""

    VALID = "VALID"
    INVALID = "INVALID"


class SkillManifestError(RuntimeError):
    """Base manifest error."""


class InvalidSkillManifestError(SkillManifestError):
    """Raised when a skill manifest is malformed."""


@dataclass(frozen=True, slots=True)
class SkillManifestValidationResult:
    """Structured validation result for a manifest."""

    status: SkillManifestStatus
    errors: tuple[str, ...] = ()
    definition: SkillDefinition | None = None
    signature: str = ""

    @property
    def valid(self) -> bool:
        return self.status is SkillManifestStatus.VALID


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """Immutable manifest payload before conversion to SkillDefinition."""

    schema_version: str
    data: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SKILL_SCHEMA_VERSION:
            raise InvalidSkillManifestError("schema_version is unsupported.")
        object.__setattr__(self, "data", MappingProxyType(_safe_mapping(self.data)))


class SkillManifestLoader:
    """Load, validate, and convert skill manifests without registration."""

    def load(self, data: Mapping[str, object] | str | bytes) -> SkillManifestValidationResult:
        try:
            raw = _load_data(data)
            definition = self.to_definition(raw)
            return SkillManifestValidationResult(
                SkillManifestStatus.VALID,
                definition=definition,
                signature=skill_manifest_signature(raw),
            )
        except (SkillManifestError, InvalidSkillDefinitionError, ValueError, TypeError, json.JSONDecodeError) as error:
            return SkillManifestValidationResult(SkillManifestStatus.INVALID, errors=(str(error),))

    def to_definition(self, data: Mapping[str, object]) -> SkillDefinition:
        raw = _safe_mapping(data)
        unknown = tuple(sorted(key for key in raw if key not in _ALLOWED_FIELDS))
        if unknown:
            raise InvalidSkillManifestError("manifest contains unknown fields.")
        if raw.get("schema_version") != SKILL_SCHEMA_VERSION:
            raise InvalidSkillManifestError("schema_version is unsupported.")
        if "input_names" in raw and "input_fields" in raw:
            raise InvalidSkillManifestError("input_names cannot be combined with input_fields.")
        if "output_names" in raw and "output_fields" in raw:
            raise InvalidSkillManifestError("output_names cannot be combined with output_fields.")
        limits = raw.get("limits") or {}
        if not isinstance(limits, Mapping):
            raise InvalidSkillManifestError("limits must be a mapping.")
        return SkillDefinition(
            skill_id=_string(raw, "skill_id"),
            name=_string(raw, "name"),
            version=_string(raw, "version"),
            description=_string(raw, "description"),
            enabled=_bool(raw.get("enabled", True), "enabled"),
            required_capability_ids=_tuple(raw.get("required_capability_ids")),
            required_permission_ids=_tuple(raw.get("required_permission_ids")),
            allowed_agent_types=_tuple(raw.get("allowed_agent_types")),
            input_names=_tuple(raw.get("input_names")),
            output_names=_tuple(raw.get("output_names")),
            input_fields=_skill_fields(raw.get("input_fields"), "input_fields"),
            output_fields=_skill_fields(raw.get("output_fields"), "output_fields"),
            execution_target=_string(raw, "execution_target"),
            execution_target_type=_string_value(raw.get("execution_target_type"), SkillExecutionTargetType.TOOL.value),
            limits=SkillLimits(
                timeout_seconds=_int(limits.get("timeout_seconds"), 30),
                max_inputs=_int(limits.get("max_inputs"), 16),
                max_outputs=_int(limits.get("max_outputs"), 16),
                max_result_items=_int(limits.get("max_result_items"), 64),
            ),
            metadata=raw.get("metadata") or {},
            tags=_tuple(raw.get("tags")),
            handler_id=_optional_string(raw.get("handler_id")),
            workflow_reference=_optional_string(raw.get("workflow_reference")),
        )


def skill_manifest_signature(data: Mapping[str, object]) -> str:
    safe = _safe_mapping(data)
    encoded = json.dumps(_jsonable(safe), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_data(data: Mapping[str, object] | str | bytes) -> Mapping[str, object]:
    if isinstance(data, bytes):
        if len(data) > MAX_SKILL_MANIFEST_BYTES:
            raise InvalidSkillManifestError("manifest is too large.")
        decoded = data.decode("utf-8")
        loaded = json.loads(decoded)
    elif isinstance(data, str):
        if len(data.encode("utf-8")) > MAX_SKILL_MANIFEST_BYTES:
            raise InvalidSkillManifestError("manifest is too large.")
        loaded = json.loads(data)
    else:
        loaded = data
    if not isinstance(loaded, Mapping):
        raise InvalidSkillManifestError("manifest must be a JSON object.")
    return loaded


def _safe_mapping(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidSkillManifestError("value must be a mapping.")
    return {str(key): _safe_value(key, child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}


def _safe_value(key: object, value: object) -> object:
    if not isinstance(key, str) or not key.strip():
        raise InvalidSkillManifestError("manifest keys must be non-empty strings.")
    if _is_sensitive_key(key):
        raise InvalidSkillManifestError("manifest contains a sensitive key.")
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidSkillManifestError("manifest floats must be finite.")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_mapping(value))
    if isinstance(value, (tuple, list)):
        return tuple(_safe_value(key, item) for item in value)
    raise InvalidSkillManifestError("manifest contains unsupported values.")


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _string(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise InvalidSkillManifestError(f"{key} must be a string.")
    return value


def _string_value(value: object, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise InvalidSkillManifestError("expected a string.")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidSkillManifestError("expected a string or None.")
    return value


def _tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (tuple, list)):
        raise InvalidSkillManifestError("expected a string list.")
    if not all(isinstance(item, str) for item in value):
        raise InvalidSkillManifestError("expected string list values.")
    return tuple(value)


def _skill_fields(
    value: object,
    field_name: str,
) -> tuple[SkillFieldDefinition, ...]:
    if value is None:
        return ()
    if not isinstance(value, (tuple, list)):
        raise InvalidSkillManifestError(f"{field_name} must be a list.")
    fields: list[SkillFieldDefinition] = []
    for raw_field in value:
        if not isinstance(raw_field, Mapping):
            raise InvalidSkillManifestError(f"{field_name} must contain objects.")
        unknown = tuple(
            sorted(key for key in raw_field if key not in _ALLOWED_SKILL_FIELD_FIELDS)
        )
        if unknown:
            raise InvalidSkillManifestError(f"{field_name} contains unknown fields.")
        fields.append(
            SkillFieldDefinition(
                name=_string(raw_field, "name"),
                type_name=_string(raw_field, "type"),
                required=_bool(raw_field.get("required", True), "required"),
            )
        )
    return tuple(fields)


def _bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidSkillManifestError(f"{field_name} must be a bool.")
    return value


def _int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidSkillManifestError("limit values must be integers.")
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)

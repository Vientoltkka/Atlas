"""Safe declarative manifests for specialized Atlas agents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import inspect
import json
import math
import re
import types
from types import MappingProxyType
from typing import Any

from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentLimits,
    AgentMemoryPolicy,
    AgentPermissions,
    AgentSecurityPolicy,
    AgentType,
    InvalidAgentDefinitionError,
    validate_agent_id,
)


MAX_MANIFEST_ITEMS = 64
MAX_MANIFEST_METADATA_ITEMS = 32
MAX_MANIFEST_TEXT_LENGTH = 1_000
MANIFEST_SCHEMA_VERSION = "1.0"
_VERSION_PATTERN = re.compile(r"^[0-9]+[.][0-9]+(?:[.][0-9]+)?$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SENSITIVE_KEY_PARTS = (
    "secret",
    "api_key",
    "apikey",
    "password",
    "token",
    "authorization",
    "cookie",
    "private_key",
    "credential",
)
_REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "agent_id",
        "name",
        "description",
        "version",
        "enabled",
        "agent_type",
        "capabilities",
        "permissions",
        "limits",
        "context_policy",
        "memory_policy",
        "security_policy",
        "handler_id",
        "tags",
        "metadata",
    }
)


class InvalidAgentManifestError(RuntimeError):
    """Raised when an agent manifest is malformed or unsafe."""


class AgentManifestConflictError(RuntimeError):
    """Raised when a manifest conflicts with another known manifest."""


class AgentManifestValidationStatus(str, Enum):
    """Structured manifest validation status."""

    VALID = "VALID"
    INVALID = "INVALID"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class AgentManifest:
    """Immutable declarative manifest for one specialized agent."""

    schema_version: str
    agent_id: str
    name: str
    description: str
    version: str
    enabled: bool
    agent_type: AgentType | str
    capabilities: Iterable[str] = field(default_factory=tuple)
    permissions: Mapping[str, object] = field(default_factory=dict)
    limits: Mapping[str, object] = field(default_factory=dict)
    context_policy: Mapping[str, object] = field(default_factory=dict)
    memory_policy: Mapping[str, object] = field(default_factory=dict)
    security_policy: Mapping[str, object] = field(default_factory=dict)
    handler_id: str = ""
    tags: Iterable[str] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version, "schema_version"))
        object.__setattr__(self, "agent_id", _agent_id(self.agent_id))
        object.__setattr__(self, "name", _safe_text(self.name, "name", MAX_MANIFEST_TEXT_LENGTH))
        object.__setattr__(
            self,
            "description",
            _safe_text(self.description, "description", MAX_MANIFEST_TEXT_LENGTH),
        )
        object.__setattr__(self, "version", _version(self.version, "version"))
        if type(self.enabled) is not bool:
            raise InvalidAgentManifestError("enabled must be a bool.")
        object.__setattr__(self, "agent_type", _agent_type(self.agent_type))
        object.__setattr__(self, "capabilities", _identifier_tuple(self.capabilities, "capabilities"))
        object.__setattr__(self, "handler_id", _identifier(self.handler_id, "handler_id"))
        object.__setattr__(self, "tags", _identifier_tuple(self.tags, "tags"))
        object.__setattr__(self, "permissions", MappingProxyType(_safe_mapping(self.permissions, "permissions")))
        object.__setattr__(self, "limits", MappingProxyType(_safe_mapping(self.limits, "limits")))
        object.__setattr__(
            self,
            "context_policy",
            MappingProxyType(_safe_mapping(self.context_policy, "context_policy")),
        )
        object.__setattr__(
            self,
            "memory_policy",
            MappingProxyType(_safe_mapping(self.memory_policy, "memory_policy")),
        )
        object.__setattr__(
            self,
            "security_policy",
            MappingProxyType(_safe_mapping(self.security_policy, "security_policy")),
        )
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))
        try:
            _build_definition(self)
        except InvalidAgentDefinitionError as error:
            raise InvalidAgentManifestError(str(error)) from error

    def to_agent_definition(self) -> AgentDefinition:
        """Convert the manifest to an immutable AgentDefinition without registering it."""

        return _build_definition(self)


@dataclass(frozen=True, slots=True)
class AgentManifestValidationResult:
    """Structured validation result for manifest loading."""

    status: AgentManifestValidationStatus
    manifest: AgentManifest | None = None
    agent_definition: AgentDefinition | None = None
    signature: str = ""
    error_code: str | None = None
    safe_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))


class AgentManifestLoader:
    """Validate and convert manifests without registering or executing agents."""

    def __init__(
        self,
        *,
        known_agent_ids: Iterable[str] = (),
        known_handler_ids: Iterable[str] = (),
    ) -> None:
        self._known_agent_ids = frozenset(validate_agent_id(agent_id) for agent_id in known_agent_ids)
        self._known_handler_ids = frozenset(_identifier(handler_id, "handler_id") for handler_id in known_handler_ids)

    def validate(
        self,
        value: AgentManifest | Mapping[str, object],
    ) -> AgentManifestValidationResult:
        """Validate one manifest value and return a structured result."""

        try:
            manifest = self.load(value)
            definition = manifest.to_agent_definition()
            return AgentManifestValidationResult(
                status=AgentManifestValidationStatus.VALID,
                manifest=manifest,
                agent_definition=definition,
                signature=agent_manifest_signature(manifest),
            )
        except AgentManifestConflictError as error:
            return _result_error(AgentManifestValidationStatus.CONFLICT, "CONFLICT", error)
        except (InvalidAgentManifestError, InvalidAgentDefinitionError, TypeError, ValueError) as error:
            return _result_error(AgentManifestValidationStatus.INVALID, type(error).__name__, error)

    def validate_json(
        self,
        payload: str,
    ) -> AgentManifestValidationResult:
        """Validate a JSON manifest string and return a structured result."""

        try:
            return self.validate(self._decode_json(payload))
        except InvalidAgentManifestError as error:
            return _result_error(AgentManifestValidationStatus.INVALID, "InvalidAgentManifestError", error)

    def load(
        self,
        value: AgentManifest | Mapping[str, object],
    ) -> AgentManifest:
        """Return a normalized immutable manifest or raise a structured error."""

        manifest = value if isinstance(value, AgentManifest) else _manifest_from_mapping(value)
        self._check_conflict(manifest, seen_agent_ids=frozenset(), seen_handler_ids=frozenset())
        return manifest

    def load_json(
        self,
        payload: str,
    ) -> AgentManifest:
        """Decode JSON and return a normalized immutable manifest."""

        return self.load(self._decode_json(payload))

    def to_agent_definition(
        self,
        value: AgentManifest | Mapping[str, object],
    ) -> AgentDefinition:
        """Validate and convert one manifest without registering it."""

        return self.load(value).to_agent_definition()

    def load_many(
        self,
        values: Iterable[AgentManifest | Mapping[str, object]],
    ) -> tuple[AgentManifest, ...]:
        """Validate multiple manifests and reject duplicate ids or handlers."""

        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise InvalidAgentManifestError("values must be an iterable of manifests.")
        manifests: list[AgentManifest] = []
        seen_agent_ids: set[str] = set()
        seen_handler_ids: set[str] = set()
        for value in values:
            manifest = value if isinstance(value, AgentManifest) else _manifest_from_mapping(value)
            self._check_conflict(
                manifest,
                seen_agent_ids=frozenset(seen_agent_ids),
                seen_handler_ids=frozenset(seen_handler_ids),
            )
            seen_agent_ids.add(manifest.agent_id)
            seen_handler_ids.add(manifest.handler_id)
            manifests.append(manifest)
        return tuple(manifests)

    def to_agent_definitions(
        self,
        values: Iterable[AgentManifest | Mapping[str, object]],
    ) -> tuple[AgentDefinition, ...]:
        """Validate and convert manifests without registering them."""

        return tuple(manifest.to_agent_definition() for manifest in self.load_many(values))

    def _check_conflict(
        self,
        manifest: AgentManifest,
        *,
        seen_agent_ids: frozenset[str],
        seen_handler_ids: frozenset[str],
    ) -> None:
        if manifest.agent_id in self._known_agent_ids or manifest.agent_id in seen_agent_ids:
            raise AgentManifestConflictError(f"agent_id already declared: {manifest.agent_id}")
        if manifest.handler_id in self._known_handler_ids or manifest.handler_id in seen_handler_ids:
            raise AgentManifestConflictError(f"handler_id already declared: {manifest.handler_id}")

    @staticmethod
    def _decode_json(payload: str) -> Mapping[str, object]:
        if not isinstance(payload, str):
            raise InvalidAgentManifestError("json payload must be a string.")
        try:
            decoded = json.loads(payload, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as error:
            raise InvalidAgentManifestError("invalid manifest JSON.") from error
        if not isinstance(decoded, Mapping):
            raise InvalidAgentManifestError("manifest JSON must decode to an object.")
        return decoded


def agent_manifest_signature(
    manifest: AgentManifest | Mapping[str, object],
) -> str:
    """Return a deterministic SHA-256 signature for a safe normalized manifest."""

    normalized = manifest if isinstance(manifest, AgentManifest) else _manifest_from_mapping(manifest)
    payload = {
        "schema_version": normalized.schema_version,
        "agent_id": normalized.agent_id,
        "name": normalized.name,
        "description": normalized.description,
        "version": normalized.version,
        "enabled": normalized.enabled,
        "agent_type": normalized.agent_type.value,
        "capabilities": normalized.capabilities,
        "permissions": normalized.permissions,
        "limits": normalized.limits,
        "context_policy": normalized.context_policy,
        "memory_policy": normalized.memory_policy,
        "security_policy": normalized.security_policy,
        "handler_id": normalized.handler_id,
        "tags": normalized.tags,
        "metadata": normalized.metadata,
    }
    encoded = json.dumps(_jsonable(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_from_mapping(
    value: Mapping[str, object],
) -> AgentManifest:
    if not isinstance(value, Mapping):
        raise InvalidAgentManifestError("manifest must be a mapping.")
    unknown = set(value).difference(_REQUIRED_MANIFEST_KEYS)
    missing = _REQUIRED_MANIFEST_KEYS.difference(value)
    if missing:
        raise InvalidAgentManifestError(f"manifest is missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise InvalidAgentManifestError(f"manifest contains unsupported fields: {', '.join(sorted(unknown))}")
    _reject_unsafe_objects(value)
    return AgentManifest(**{key: value[key] for key in sorted(_REQUIRED_MANIFEST_KEYS)})  # type: ignore[arg-type]


def _build_definition(
    manifest: AgentManifest,
) -> AgentDefinition:
    permissions = AgentPermissions(**_allowed_dataclass_payload(manifest.permissions, AgentPermissions, "permissions"))
    limits = AgentLimits(**_allowed_dataclass_payload(manifest.limits, AgentLimits, "limits"))
    context_policy = AgentContextPolicy(
        **_allowed_dataclass_payload(manifest.context_policy, AgentContextPolicy, "context_policy")
    )
    memory_policy = AgentMemoryPolicy(**_allowed_dataclass_payload(manifest.memory_policy, AgentMemoryPolicy, "memory_policy"))
    security_policy = AgentSecurityPolicy(
        **_allowed_dataclass_payload(manifest.security_policy, AgentSecurityPolicy, "security_policy")
    )
    metadata = dict(manifest.metadata)
    metadata["manifest_schema_version"] = manifest.schema_version
    metadata["manifest_version"] = manifest.version
    metadata["handler_id"] = manifest.handler_id
    metadata["manifest_signature"] = agent_manifest_signature(manifest)
    return AgentDefinition(
        agent_id=manifest.agent_id,
        agent_type=manifest.agent_type,
        name=manifest.name,
        description=manifest.description,
        permissions=permissions,
        limits=limits,
        capabilities=AgentCapabilities(capabilities=manifest.capabilities, tags=manifest.tags),
        context_policy=context_policy,
        memory_policy=memory_policy,
        security_policy=security_policy,
        enabled=manifest.enabled,
        metadata=metadata,
    )


def _allowed_dataclass_payload(
    value: Mapping[str, object],
    dataclass_type: type,
    field_name: str,
) -> dict[str, object]:
    fields = frozenset(getattr(dataclass_type, "__dataclass_fields__", ()))
    unknown = set(value).difference(fields)
    if unknown:
        raise InvalidAgentManifestError(f"{field_name} contains unsupported fields: {', '.join(sorted(unknown))}")
    return dict(value)


def _safe_mapping(
    value: Mapping[str, object],
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentManifestError(f"{field_name} must be a mapping.")
    if len(value) > MAX_MANIFEST_METADATA_ITEMS:
        raise InvalidAgentManifestError(f"{field_name} has too many items.")
    result: dict[str, object] = {}
    for raw_key in sorted(value):
        key = _identifier(raw_key, f"{field_name} key")
        if _is_sensitive_key(key):
            raise InvalidAgentManifestError(f"{field_name} contains a forbidden sensitive key.")
        result[key] = _safe_value(value[raw_key], depth=0)
    return result


def _safe_value(
    value: object,
    *,
    depth: int,
) -> object:
    if depth > 4:
        raise InvalidAgentManifestError("manifest nested value exceeds depth limit.")
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentManifestError("non-finite floats are not allowed.")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_mapping(value, "nested metadata"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_MANIFEST_ITEMS:
            raise InvalidAgentManifestError("manifest sequence exceeds item limit.")
        return tuple(_safe_value(item, depth=depth + 1) for item in value)
    _reject_executable_object(value)
    raise InvalidAgentManifestError("manifest values must be JSON-safe primitives, sequences, or mappings.")


def _reject_unsafe_objects(
    value: object,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise InvalidAgentManifestError("manifest keys must be strings.")
            if _is_sensitive_key(key):
                raise InvalidAgentManifestError("manifest contains a forbidden sensitive key.")
            _reject_unsafe_objects(child)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_unsafe_objects(child)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidAgentManifestError("non-finite floats are not allowed.")
    _reject_executable_object(value)


def _reject_executable_object(
    value: object,
) -> None:
    if (
        inspect.isfunction(value)
        or inspect.ismethod(value)
        or inspect.isclass(value)
        or isinstance(value, types.ModuleType)
        or callable(value)
    ):
        raise InvalidAgentManifestError("executable objects are not allowed in manifests.")


def _jsonable(
    value: object,
) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidAgentManifestError("non-finite floats are not allowed.")
    if value is not None and type(value) not in (bool, int, float, str):
        raise InvalidAgentManifestError("manifest contains unsupported values.")
    return value


def _identifier_tuple(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentManifestError(f"{field_name} must be an iterable of strings.")
    normalized = tuple(dict.fromkeys(_identifier(value, field_name) for value in values))
    if len(normalized) > MAX_MANIFEST_ITEMS:
        raise InvalidAgentManifestError(f"{field_name} exceeds the item limit.")
    return normalized


def _agent_id(
    value: str,
) -> str:
    try:
        return validate_agent_id(value)
    except InvalidAgentDefinitionError as error:
        raise InvalidAgentManifestError(str(error)) from error


def _identifier(
    value: object,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise InvalidAgentManifestError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise InvalidAgentManifestError(f"{field_name} cannot be empty.")
    if normalized != value:
        raise InvalidAgentManifestError(f"{field_name} cannot contain surrounding whitespace.")
    if not _IDENTIFIER_PATTERN.fullmatch(normalized):
        raise InvalidAgentManifestError(f"{field_name} contains unsupported characters.")
    return normalized


def _safe_text(
    value: object,
    field_name: str,
    max_length: int,
) -> str:
    if not isinstance(value, str):
        raise InvalidAgentManifestError(f"{field_name} must be a string.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise InvalidAgentManifestError(f"{field_name} cannot be empty.")
    if len(normalized) > max_length:
        raise InvalidAgentManifestError(f"{field_name} exceeds the length limit.")
    return normalized


def _version(
    value: object,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise InvalidAgentManifestError(f"{field_name} must be a string.")
    normalized = value.strip()
    if normalized != value or not _VERSION_PATTERN.fullmatch(normalized):
        raise InvalidAgentManifestError(f"{field_name} must use numeric dotted version format.")
    return normalized


def _agent_type(
    value: AgentType | str,
) -> AgentType:
    if isinstance(value, AgentType):
        return value
    if isinstance(value, str):
        try:
            return AgentType(value.strip().lower())
        except ValueError as error:
            raise InvalidAgentManifestError("invalid agent_type.") from error
    raise InvalidAgentManifestError("agent_type must be AgentType or str.")


def _is_sensitive_key(
    key: str,
) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _reject_json_constant(
    value: str,
) -> None:
    raise InvalidAgentManifestError(f"invalid JSON constant: {value}")


def _result_error(
    status: AgentManifestValidationStatus,
    code: str,
    error: Exception,
) -> AgentManifestValidationResult:
    return AgentManifestValidationResult(
        status=status,
        error_code=code,
        safe_message=_safe_message(str(error)),
    )


def _safe_message(
    value: str,
) -> str:
    message = " ".join(value.split())
    for key in _SENSITIVE_KEY_PARTS:
        message = message.replace(key, "[redacted]")
    return message[:240]


def _status(
    value: AgentManifestValidationStatus | str,
) -> AgentManifestValidationStatus:
    if isinstance(value, AgentManifestValidationStatus):
        return value
    if isinstance(value, str):
        return AgentManifestValidationStatus(value)
    raise InvalidAgentManifestError("status must be AgentManifestValidationStatus.")

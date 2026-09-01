"""Declarative registry for reusable Atlas skills."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from types import MappingProxyType

from core.agent_registry import AgentType


MAX_SKILL_ITEMS = 64
MAX_SKILL_TEXT_LENGTH = 240
MAX_SKILL_DESCRIPTION_LENGTH = 1_000
MAX_SKILL_LIMIT = 10_000
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+){0,2}$")
_SKILL_FIELD_TYPE_NAMES = frozenset(
    {"string", "integer", "number", "boolean", "object", "array"}
)
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


class SkillRegistryError(RuntimeError):
    """Base error for skill definitions and registry operations."""


class InvalidSkillDefinitionError(SkillRegistryError):
    """Raised when a skill definition is malformed."""


class SkillAlreadyRegisteredError(SkillRegistryError):
    """Raised when a duplicate skill id is registered."""


class SkillNotFoundError(SkillRegistryError):
    """Raised when a skill id is absent."""


class SkillExecutionTargetType(str, Enum):
    """Allowed concrete execution targets for a skill."""

    TOOL = "tool"
    CAPABILITY = "capability"
    AGENT = "agent"
    HANDLER = "handler"


@dataclass(frozen=True, slots=True)
class SkillLimits:
    """Declarative non-dynamic execution limits for a skill."""

    timeout_seconds: int = 30
    max_inputs: int = 16
    max_outputs: int = 16
    max_result_items: int = 64

    def __post_init__(self) -> None:
        for name in ("timeout_seconds", "max_inputs", "max_outputs", "max_result_items"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > MAX_SKILL_LIMIT:
                raise InvalidSkillDefinitionError(f"{name} is outside the allowed range.")


@dataclass(frozen=True, slots=True)
class SkillFieldDefinition:
    """One deterministic field in a typed skill contract."""

    name: str
    type_name: str
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "skill field name"))
        object.__setattr__(self, "type_name", _skill_field_type(self.type_name))
        if not isinstance(self.required, bool):
            raise InvalidSkillDefinitionError("skill field required must be a bool.")


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    """Immutable normalized definition of one reusable Atlas skill."""

    skill_id: str
    name: str
    version: str
    description: str
    enabled: bool = True
    required_capability_ids: tuple[str, ...] = ()
    required_permission_ids: tuple[str, ...] = ()
    allowed_agent_types: tuple[AgentType | str, ...] = ()
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    execution_target: str = ""
    execution_target_type: SkillExecutionTargetType | str = SkillExecutionTargetType.TOOL
    limits: SkillLimits = field(default_factory=SkillLimits)
    metadata: Mapping[str, object] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    handler_id: str | None = None
    workflow_reference: str | None = None
    input_fields: tuple[SkillFieldDefinition, ...] = ()
    output_fields: tuple[SkillFieldDefinition, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", validate_skill_id(self.skill_id))
        object.__setattr__(self, "name", _text(self.name, "name", MAX_SKILL_TEXT_LENGTH))
        object.__setattr__(self, "version", _version(self.version))
        object.__setattr__(self, "description", _text(self.description, "description", MAX_SKILL_DESCRIPTION_LENGTH))
        if not isinstance(self.enabled, bool):
            raise InvalidSkillDefinitionError("enabled must be a bool.")
        object.__setattr__(self, "required_capability_ids", _identifier_tuple(self.required_capability_ids, "required_capability_ids"))
        object.__setattr__(self, "required_permission_ids", _identifier_tuple(self.required_permission_ids, "required_permission_ids"))
        object.__setattr__(self, "allowed_agent_types", _agent_type_tuple(self.allowed_agent_types, "allowed_agent_types"))
        object.__setattr__(self, "input_names", _identifier_tuple(self.input_names, "input_names"))
        object.__setattr__(self, "output_names", _identifier_tuple(self.output_names, "output_names"))
        object.__setattr__(self, "input_fields", _skill_field_tuple(self.input_fields, "input_fields"))
        object.__setattr__(self, "output_fields", _skill_field_tuple(self.output_fields, "output_fields"))
        if self.input_names and self.input_fields:
            raise InvalidSkillDefinitionError("input_names cannot be combined with input_fields.")
        if self.output_names and self.output_fields:
            raise InvalidSkillDefinitionError("output_names cannot be combined with output_fields.")
        object.__setattr__(self, "execution_target", _identifier(self.execution_target, "execution_target"))
        object.__setattr__(self, "execution_target_type", _target_type(self.execution_target_type))
        if not isinstance(self.limits, SkillLimits):
            raise InvalidSkillDefinitionError("limits must be SkillLimits.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))
        object.__setattr__(self, "tags", _identifier_tuple(self.tags, "tags"))
        if self.handler_id is not None:
            object.__setattr__(self, "handler_id", _identifier(self.handler_id, "handler_id"))
        if self.workflow_reference is not None:
            object.__setattr__(self, "workflow_reference", _identifier(self.workflow_reference, "workflow_reference"))

    @property
    def id(self) -> str:
        return self.skill_id


class SkillRegistry:
    """Deterministic in-memory registry for reusable skills."""

    def __init__(self, definitions: Iterable[SkillDefinition] = ()) -> None:
        self._definitions: OrderedDict[str, SkillDefinition] = OrderedDict()
        for definition in definitions:
            self.register(definition)

    def register(self, definition: SkillDefinition, *, replace: bool = False) -> SkillDefinition:
        if not isinstance(definition, SkillDefinition):
            raise InvalidSkillDefinitionError("definition must be SkillDefinition.")
        if definition.skill_id in self._definitions and not replace:
            raise SkillAlreadyRegisteredError(f"skill id already registered: {definition.skill_id}")
        self._definitions[definition.skill_id] = definition
        return definition

    def get(self, skill_id: str) -> SkillDefinition:
        normalized = validate_skill_id(skill_id)
        try:
            return self._definitions[normalized]
        except KeyError as error:
            raise SkillNotFoundError(f"skill id not found: {normalized}") from error

    def contains(self, skill_id: str) -> bool:
        return validate_skill_id(skill_id) in self._definitions

    def list_skills(self, *, enabled_only: bool = False) -> tuple[SkillDefinition, ...]:
        return tuple(
            definition for definition in self._definitions.values() if definition.enabled or not enabled_only
        )

    def find_by_capability(self, capability_id: str, *, enabled_only: bool = True) -> tuple[SkillDefinition, ...]:
        capability = _identifier(capability_id, "capability_id")
        return tuple(
            definition
            for definition in self._definitions.values()
            if capability in definition.required_capability_ids and (definition.enabled or not enabled_only)
        )

    def find_by_tag(self, tag: str, *, enabled_only: bool = True) -> tuple[SkillDefinition, ...]:
        normalized = _identifier(tag, "tag")
        return tuple(
            definition
            for definition in self._definitions.values()
            if normalized in definition.tags and (definition.enabled or not enabled_only)
        )

    def find_by_agent_type(self, agent_type: AgentType | str, *, enabled_only: bool = True) -> tuple[SkillDefinition, ...]:
        normalized = _agent_type(agent_type)
        return tuple(
            definition
            for definition in self._definitions.values()
            if normalized in definition.allowed_agent_types and (definition.enabled or not enabled_only)
        )

    def unregister(self, skill_id: str) -> SkillDefinition:
        """Remove exactly one registered Skill and return its immutable definition."""
        normalized = validate_skill_id(skill_id)
        try:
            return self._definitions.pop(normalized)
        except KeyError as error:
            raise SkillNotFoundError(f"skill id not found: {normalized}") from error

    def clear(self) -> None:
        self._definitions.clear()

    def __len__(self) -> int:
        return len(self._definitions)


def validate_skill_id(value: str) -> str:
    return _identifier(value, "skill_id")


def _target_type(value: SkillExecutionTargetType | str) -> SkillExecutionTargetType:
    if isinstance(value, SkillExecutionTargetType):
        return value
    if isinstance(value, str):
        try:
            return SkillExecutionTargetType(value.strip().lower())
        except ValueError as error:
            raise InvalidSkillDefinitionError("execution_target_type is invalid.") from error
    raise InvalidSkillDefinitionError("execution_target_type is invalid.")


def _agent_type(value: AgentType | str) -> AgentType:
    if isinstance(value, AgentType):
        return value
    if isinstance(value, str):
        try:
            return AgentType(value.strip().lower())
        except ValueError as error:
            raise InvalidSkillDefinitionError("agent type is invalid.") from error
    raise InvalidSkillDefinitionError("agent type is invalid.")


def _agent_type_tuple(values: Iterable[AgentType | str], field_name: str) -> tuple[AgentType, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidSkillDefinitionError(f"{field_name} must be an iterable.")
    normalized = tuple(dict.fromkeys(_agent_type(value) for value in values))
    if len(normalized) > MAX_SKILL_ITEMS:
        raise InvalidSkillDefinitionError(f"{field_name} has too many items.")
    return normalized


def _identifier_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidSkillDefinitionError(f"{field_name} must be an iterable.")
    normalized = tuple(dict.fromkeys(_identifier(value, field_name) for value in values))
    if len(normalized) > MAX_SKILL_ITEMS:
        raise InvalidSkillDefinitionError(f"{field_name} has too many items.")
    return normalized


def _skill_field_type(value: str) -> str:
    if not isinstance(value, str):
        raise InvalidSkillDefinitionError("skill field type must be a string.")
    normalized = value.strip().lower()
    if normalized not in _SKILL_FIELD_TYPE_NAMES:
        raise InvalidSkillDefinitionError("skill field type is invalid.")
    return normalized


def _skill_field_tuple(
    values: Iterable[SkillFieldDefinition],
    field_name: str,
) -> tuple[SkillFieldDefinition, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidSkillDefinitionError(f"{field_name} must be an iterable.")
    normalized = tuple(values)
    if len(normalized) > MAX_SKILL_ITEMS:
        raise InvalidSkillDefinitionError(f"{field_name} has too many items.")
    if not all(isinstance(value, SkillFieldDefinition) for value in normalized):
        raise InvalidSkillDefinitionError(f"{field_name} must contain skill fields.")
    names = tuple(value.name for value in normalized)
    if len(set(names)) != len(names):
        raise InvalidSkillDefinitionError(f"{field_name} contains duplicate names.")
    return normalized


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidSkillDefinitionError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if normalized != value:
        raise InvalidSkillDefinitionError(f"{field_name} cannot contain surrounding whitespace.")
    if not _IDENTIFIER_PATTERN.fullmatch(normalized):
        raise InvalidSkillDefinitionError(f"{field_name} contains unsupported characters.")
    if _is_sensitive_key(normalized):
        raise InvalidSkillDefinitionError(f"{field_name} cannot contain sensitive content.")
    return normalized


def _text(value: str, field_name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidSkillDefinitionError(f"{field_name} must be a non-empty string.")
    normalized = " ".join(value.strip().split())
    if len(normalized) > limit:
        raise InvalidSkillDefinitionError(f"{field_name} is too long.")
    return normalized


def _version(value: str) -> str:
    if not isinstance(value, str) or not _VERSION_PATTERN.fullmatch(value.strip()):
        raise InvalidSkillDefinitionError("version is invalid.")
    return value.strip()


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidSkillDefinitionError("metadata must be a mapping.")
    if len(metadata) > MAX_SKILL_ITEMS:
        raise InvalidSkillDefinitionError("metadata has too many items.")
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        normalized = _identifier(key, "metadata key")
        if _is_sensitive_key(normalized):
            raise InvalidSkillDefinitionError("metadata contains a sensitive key.")
        safe[normalized] = _safe_value(value)
    return safe


def _safe_value(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidSkillDefinitionError("metadata floats must be finite.")
        return value
    raise InvalidSkillDefinitionError("metadata values must be primitive safe values.")


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)

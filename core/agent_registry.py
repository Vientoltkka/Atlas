"""Declarative registry for specialized Atlas agents."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
import re
from typing import Any, Iterator


MAX_AGENT_ITEMS = 64
MAX_AGENT_METADATA_ITEMS = 32
MAX_AGENT_TEXT_LENGTH = 240
MAX_AGENT_DESCRIPTION_LENGTH = 1_000
MAX_AGENT_LIMIT_VALUE = 1_000
_AGENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RESERVED_AGENT_IDS = frozenset({"__class__", "__dict__", "__mro__", "__subclasses__"})


class AgentRegistryError(RuntimeError):
    """Base error for specialized agent registry operations."""


class InvalidAgentDefinitionError(AgentRegistryError):
    """Raised when an agent definition or policy is malformed."""


class AgentAlreadyRegisteredError(AgentRegistryError):
    """Raised when registering a duplicate agent id."""


class AgentNotFoundError(AgentRegistryError):
    """Raised when an agent id cannot be found."""


class AgentType(str, Enum):
    """Closed categories for specialized Atlas agents."""

    PROJECT_ANALYSIS = "project_analysis"
    CODING = "coding"
    EXECUTION = "execution"
    ARCHITECTURE = "architecture"
    MEMORY = "memory"
    VOICE = "voice"
    GENERAL = "general"


@dataclass(frozen=True, slots=True)
class AgentPermissions:
    """Static permissions for a specialized agent definition."""

    can_read_project: bool = True
    can_write_files: bool = False
    can_execute_tools: bool = False
    can_modify_memory: bool = False
    can_use_network: bool = False
    requires_confirmation: bool = True

    def __post_init__(self) -> None:
        _validate_bool_fields(self)


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """Static non-execution limits for future bounded agent use."""

    max_steps: int = 1
    max_tool_calls: int = 0
    max_context_items: int = 16
    max_memory_items: int = 0
    max_replans: int = 0

    def __post_init__(self) -> None:
        for field_name in ("max_steps", "max_context_items"):
            _validate_positive_int(getattr(self, field_name), field_name)
        for field_name in ("max_tool_calls", "max_memory_items", "max_replans"):
            _validate_non_negative_int(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class AgentCapabilities:
    """Declarative capability and tool metadata for agent discovery."""

    capabilities: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", _normalize_identifier_tuple(self.capabilities, "capabilities"))
        object.__setattr__(self, "tools", _normalize_identifier_tuple(self.tools, "tools"))
        object.__setattr__(self, "tags", _normalize_identifier_tuple(self.tags, "tags"))


@dataclass(frozen=True, slots=True)
class AgentContextPolicy:
    """Policy describing what context may be assembled for an agent."""

    include_project_context: bool = True
    include_conversation_context: bool = False
    include_runtime_context: bool = False
    allow_user_input: bool = False
    allow_shared_context: bool = True
    allow_tool_results: bool = False
    allow_workflow_results: bool = False
    max_context_items: int = 16
    max_context_depth: int = 4
    max_string_length: int = 1_000
    max_sequence_items: int = 32
    max_mapping_items: int = 32
    max_total_items: int = 256
    allowed_context_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_bool_fields(
            self,
            exclude=(
                "max_context_items",
                "max_context_depth",
                "max_string_length",
                "max_sequence_items",
                "max_mapping_items",
                "max_total_items",
                "allowed_context_keys",
            ),
        )
        for field_name in (
            "max_context_items",
            "max_context_depth",
            "max_string_length",
            "max_sequence_items",
            "max_mapping_items",
            "max_total_items",
        ):
            _validate_positive_int(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "allowed_context_keys",
            _normalize_identifier_tuple(self.allowed_context_keys, "allowed_context_keys"),
        )


@dataclass(frozen=True, slots=True)
class AgentMemoryPolicy:
    """Policy describing whether an agent may read or write memory scopes."""

    can_read_memory: bool = False
    can_write_memory: bool = False
    memory_scopes: tuple[str, ...] = ()
    max_memory_items: int = 0

    def __post_init__(self) -> None:
        _validate_bool_fields(self, exclude=("memory_scopes", "max_memory_items"))
        _validate_non_negative_int(self.max_memory_items, "max_memory_items")
        object.__setattr__(self, "memory_scopes", _normalize_identifier_tuple(self.memory_scopes, "memory_scopes"))
        if self.can_write_memory and not self.can_read_memory:
            raise InvalidAgentDefinitionError("can_write_memory requires can_read_memory.")


@dataclass(frozen=True, slots=True)
class AgentSecurityPolicy:
    """Security restrictions for a specialized agent definition."""

    allow_network: bool = False
    allow_file_write: bool = False
    allowed_tools: tuple[str, ...] = ()
    blocked_tools: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    require_confirmation_for_writes: bool = True

    def __post_init__(self) -> None:
        _validate_bool_fields(self, exclude=("allowed_tools", "blocked_tools", "allowed_paths"))
        object.__setattr__(self, "allowed_tools", _normalize_identifier_tuple(self.allowed_tools, "allowed_tools"))
        object.__setattr__(self, "blocked_tools", _normalize_identifier_tuple(self.blocked_tools, "blocked_tools"))
        overlap = set(self.allowed_tools).intersection(self.blocked_tools)
        if overlap:
            raise InvalidAgentDefinitionError("allowed_tools and blocked_tools cannot overlap.")
        object.__setattr__(self, "allowed_paths", _normalize_path_tuple(self.allowed_paths, "allowed_paths"))
        if self.allow_file_write and not self.require_confirmation_for_writes:
            raise InvalidAgentDefinitionError("file writes require confirmation.")


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Immutable declarative definition for one specialized Atlas agent."""

    agent_id: str
    agent_type: AgentType
    name: str
    description: str
    permissions: AgentPermissions = field(default_factory=AgentPermissions)
    limits: AgentLimits = field(default_factory=AgentLimits)
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    context_policy: AgentContextPolicy = field(default_factory=AgentContextPolicy)
    memory_policy: AgentMemoryPolicy = field(default_factory=AgentMemoryPolicy)
    security_policy: AgentSecurityPolicy = field(default_factory=AgentSecurityPolicy)
    enabled: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        object.__setattr__(self, "agent_type", _agent_type(self.agent_type))
        object.__setattr__(self, "name", _normalize_text(self.name, "name", MAX_AGENT_TEXT_LENGTH))
        object.__setattr__(
            self,
            "description",
            _normalize_text(self.description, "description", MAX_AGENT_DESCRIPTION_LENGTH),
        )
        _validate_type(self.permissions, AgentPermissions, "permissions")
        _validate_type(self.limits, AgentLimits, "limits")
        _validate_type(self.capabilities, AgentCapabilities, "capabilities")
        _validate_type(self.context_policy, AgentContextPolicy, "context_policy")
        _validate_type(self.memory_policy, AgentMemoryPolicy, "memory_policy")
        _validate_type(self.security_policy, AgentSecurityPolicy, "security_policy")
        if not isinstance(self.enabled, bool):
            raise InvalidAgentDefinitionError("enabled must be a bool.")
        _validate_policy_consistency(self)
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))

    @property
    def id(self) -> str:
        """Return the stable agent id."""

        return self.agent_id


class AgentRegistry:
    """In-memory registry for immutable specialized agent definitions."""

    def __init__(
        self,
        definitions: Iterable[AgentDefinition] = (),
    ) -> None:
        self._definitions: OrderedDict[str, AgentDefinition] = OrderedDict()
        for definition in definitions:
            self.register(definition)

    def register(
        self,
        definition: AgentDefinition,
        *,
        replace: bool = False,
    ) -> AgentDefinition:
        """Register an agent definition by id."""

        if not isinstance(definition, AgentDefinition):
            raise InvalidAgentDefinitionError("definition must be AgentDefinition.")
        if definition.agent_id in self._definitions and not replace:
            raise AgentAlreadyRegisteredError(f"agent id already registered: {definition.agent_id}")
        self._definitions[definition.agent_id] = definition
        return definition

    def get(
        self,
        agent_id: str,
    ) -> AgentDefinition:
        """Return a registered agent definition by id."""

        normalized = validate_agent_id(agent_id)
        try:
            return self._definitions[normalized]
        except KeyError as error:
            raise AgentNotFoundError(f"agent id not found: {normalized}") from error

    def contains(
        self,
        agent_id: str,
    ) -> bool:
        """Return whether an agent id is registered."""

        return validate_agent_id(agent_id) in self._definitions

    def find_by_capability(
        self,
        capability: str,
        *,
        enabled_only: bool = True,
    ) -> tuple[AgentDefinition, ...]:
        """Return agents declaring one capability."""

        normalized = _normalize_identifier(capability, "capability")
        return tuple(
            definition
            for definition in self._definitions.values()
            if normalized in definition.capabilities.capabilities
            and (definition.enabled or not enabled_only)
        )

    def find_by_type(
        self,
        agent_type: AgentType | str,
        *,
        enabled_only: bool = True,
    ) -> tuple[AgentDefinition, ...]:
        """Return agents with the requested type."""

        normalized = _agent_type(agent_type)
        return tuple(
            definition
            for definition in self._definitions.values()
            if definition.agent_type is normalized
            and (definition.enabled or not enabled_only)
        )

    def list_agents(
        self,
        *,
        enabled_only: bool = False,
    ) -> tuple[AgentDefinition, ...]:
        """Return all registered agent definitions in registration order."""

        return tuple(
            definition
            for definition in self._definitions.values()
            if definition.enabled or not enabled_only
        )

    def clear(self) -> None:
        """Remove all registered definitions."""

        self._definitions.clear()

    def __len__(self) -> int:
        return len(self._definitions)

    def __iter__(self) -> Iterator[AgentDefinition]:
        return iter(self.list_agents())


def validate_agent_id(
    value: str,
) -> str:
    """Validate and return a stable specialized-agent id."""

    normalized = _normalize_identifier(value, "agent_id")
    if normalized in _RESERVED_AGENT_IDS:
        raise InvalidAgentDefinitionError("agent_id is reserved.")
    return normalized


def _validate_policy_consistency(
    definition: AgentDefinition,
) -> None:
    permissions = definition.permissions
    security = definition.security_policy
    memory = definition.memory_policy
    limits = definition.limits
    if security.allow_network and not permissions.can_use_network:
        raise InvalidAgentDefinitionError("allow_network requires can_use_network permission.")
    if security.allow_file_write and not permissions.can_write_files:
        raise InvalidAgentDefinitionError("allow_file_write requires can_write_files permission.")
    if memory.can_read_memory and not permissions.can_modify_memory:
        raise InvalidAgentDefinitionError("memory access requires can_modify_memory permission.")
    if limits.max_tool_calls > 0 and not permissions.can_execute_tools:
        raise InvalidAgentDefinitionError("max_tool_calls requires can_execute_tools permission.")


def _agent_type(
    value: AgentType | str,
) -> AgentType:
    if isinstance(value, AgentType):
        return value
    if isinstance(value, str):
        try:
            return AgentType(value.strip().lower())
        except ValueError as error:
            raise InvalidAgentDefinitionError("invalid agent_type.") from error
    raise InvalidAgentDefinitionError("agent_type must be AgentType or str.")


def _validate_type(
    value: object,
    expected_type: type,
    field_name: str,
) -> None:
    if not isinstance(value, expected_type):
        raise InvalidAgentDefinitionError(f"{field_name} must be {expected_type.__name__}.")


def _validate_bool_fields(
    value: object,
    *,
    exclude: tuple[str, ...] = (),
) -> None:
    for field_name in getattr(value, "__dataclass_fields__", ()):
        if field_name in exclude:
            continue
        if type(getattr(value, field_name)) is not bool:
            raise InvalidAgentDefinitionError(f"{field_name} must be a bool.")


def _validate_positive_int(
    value: int,
    field_name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidAgentDefinitionError(f"{field_name} must be an integer.")
    if value <= 0 or value > MAX_AGENT_LIMIT_VALUE:
        raise InvalidAgentDefinitionError(f"{field_name} must be between 1 and {MAX_AGENT_LIMIT_VALUE}.")


def _validate_non_negative_int(
    value: int,
    field_name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidAgentDefinitionError(f"{field_name} must be an integer.")
    if value < 0 or value > MAX_AGENT_LIMIT_VALUE:
        raise InvalidAgentDefinitionError(f"{field_name} must be between 0 and {MAX_AGENT_LIMIT_VALUE}.")


def _normalize_identifier_tuple(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentDefinitionError(f"{field_name} must be an iterable of strings.")
    normalized = tuple(dict.fromkeys(_normalize_identifier(value, field_name) for value in values))
    if len(normalized) > MAX_AGENT_ITEMS:
        raise InvalidAgentDefinitionError(f"{field_name} exceeds the item limit.")
    return normalized


def _normalize_identifier(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise InvalidAgentDefinitionError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise InvalidAgentDefinitionError(f"{field_name} cannot be empty.")
    if normalized != value:
        raise InvalidAgentDefinitionError(f"{field_name} cannot contain surrounding whitespace.")
    if not _IDENTIFIER_PATTERN.fullmatch(normalized):
        raise InvalidAgentDefinitionError(f"{field_name} contains unsupported characters.")
    return normalized


def _normalize_path_tuple(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentDefinitionError(f"{field_name} must be an iterable of strings.")
    normalized = tuple(dict.fromkeys(_normalize_path(value, field_name) for value in values))
    if len(normalized) > MAX_AGENT_ITEMS:
        raise InvalidAgentDefinitionError(f"{field_name} exceeds the item limit.")
    return normalized


def _normalize_path(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise InvalidAgentDefinitionError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise InvalidAgentDefinitionError(f"{field_name} cannot contain empty paths.")
    if any(ord(character) < 32 for character in normalized):
        raise InvalidAgentDefinitionError(f"{field_name} cannot contain control characters.")
    return normalized


def _normalize_text(
    value: str,
    field_name: str,
    max_length: int,
) -> str:
    if not isinstance(value, str):
        raise InvalidAgentDefinitionError(f"{field_name} must be a string.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise InvalidAgentDefinitionError(f"{field_name} cannot be empty.")
    if len(normalized) > max_length:
        raise InvalidAgentDefinitionError(f"{field_name} exceeds the length limit.")
    return normalized


def _safe_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidAgentDefinitionError("metadata must be a mapping.")
    if len(metadata) > MAX_AGENT_METADATA_ITEMS:
        raise InvalidAgentDefinitionError("metadata has too many items.")
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        safe[_normalize_identifier(key, "metadata key")] = _safe_value(value)
    return safe


def _safe_value(
    value: object,
) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise InvalidAgentDefinitionError("metadata floats must be finite.")
        return value
    raise InvalidAgentDefinitionError("metadata values must be primitive safe values.")

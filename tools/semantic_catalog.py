"""Semantic catalog for registered Atlas tools."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from types import MappingProxyType
from typing import Any, Mapping
import unicodedata

from tools.argument_schema import ArgumentSchemaRegistry
from tools.intent_selector import ToolIntent, ToolSelector
from tools.registry import ToolDescriptor, ToolNotRegisteredError, ToolRegistry


RISK_LEVELS: tuple[str, ...] = ("none", "low", "medium", "high", "critical")
_CONFIRMATION_RISK_LEVELS = {"medium", "high", "critical"}
_SECRET_PATTERNS = ("api_key", "apikey", "token", "secret", "password", "credential")


@dataclass(frozen=True, slots=True)
class SemanticToolDescriptor:
    """Operational meaning for one registered tool."""

    name: str
    description: str
    capabilities: tuple[str, ...]
    supported_intents: tuple[str, ...]
    input_description: str
    required_arguments: tuple[str, ...]
    optional_arguments: tuple[str, ...]
    output_description: str
    output_fields: tuple[str, ...]
    dangerous: bool
    risk_level: str
    risk_reasons: tuple[str, ...]
    requires_confirmation: bool
    preconditions: tuple[str, ...]
    limitations: tuple[str, ...]
    negative_examples: tuple[str, ...]
    compatible_tools: tuple[str, ...]
    tags: tuple[str, ...]
    positive_examples: tuple[str, ...] = ()
    category: str = "general"
    version: str = "1"
    technical_arguments: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation without executable objects."""
        return {
            "category": self.category,
            "compatible_tools": list(self.compatible_tools),
            "capabilities": list(self.capabilities),
            "dangerous": self.dangerous,
            "description": self.description,
            "input_description": self.input_description,
            "limitations": list(self.limitations),
            "name": self.name,
            "negative_examples": list(self.negative_examples),
            "optional_arguments": list(self.optional_arguments),
            "output_description": self.output_description,
            "output_fields": list(self.output_fields),
            "positive_examples": list(self.positive_examples),
            "preconditions": list(self.preconditions),
            "required_arguments": list(self.required_arguments),
            "requires_confirmation": self.requires_confirmation,
            "risk_level": self.risk_level,
            "risk_reasons": list(self.risk_reasons),
            "supported_intents": list(self.supported_intents),
            "tags": list(self.tags),
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class SemanticCatalogValidationResult:
    """Validation result for a semantic tool catalog."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tool_count: int = 0
    validated_tools: list[str] = field(default_factory=list)


class SemanticToolCatalog:
    """Deterministic semantic view over the real ToolRegistry."""

    def __init__(
        self,
        descriptors: Mapping[str, SemanticToolDescriptor],
    ) -> None:
        self._descriptors = MappingProxyType(dict(sorted(descriptors.items())))

    @classmethod
    def build_from_registry(
        cls,
        registry: ToolRegistry,
        *,
        tool_selector: ToolSelector | None = None,
        schema_registry: ArgumentSchemaRegistry | None = None,
    ) -> "SemanticToolCatalog":
        """Build a semantic catalog without executing registered tools."""
        descriptors: dict[str, SemanticToolDescriptor] = {}
        intent_to_tool = _intent_to_tool_map(tool_selector)

        for name in registry.list():
            descriptor = registry.descriptor(name)
            metadata = _semantic_metadata(descriptor.tool)
            intents = tuple(
                intent
                for intent, tool_name in intent_to_tool.items()
                if tool_name == name
            )
            descriptors[name] = _build_semantic_descriptor(
                descriptor,
                metadata,
                intents,
                schema_registry,
            )

        return cls(descriptors)

    def get(
        self,
        name: str,
    ) -> SemanticToolDescriptor:
        """Return one semantic descriptor by tool name."""
        return self._descriptors[name]

    def list_all(self) -> tuple[SemanticToolDescriptor, ...]:
        """Return all descriptors in deterministic order."""
        return tuple(self._descriptors[name] for name in sorted(self._descriptors))

    def find_by_capability(
        self,
        capability: str,
    ) -> tuple[SemanticToolDescriptor, ...]:
        """Return tools that declare one exact normalized capability."""
        normalized = _normalize_identifier(capability)
        return tuple(
            descriptor
            for descriptor in self.list_all()
            if normalized in descriptor.capabilities
        )

    def find_by_intent(
        self,
        intent: str,
    ) -> tuple[SemanticToolDescriptor, ...]:
        """Return tools with an exact normalized supported intent."""
        normalized = _normalize_phrase(intent)
        return tuple(
            descriptor
            for descriptor in self.list_all()
            if normalized in {_normalize_phrase(item) for item in descriptor.supported_intents}
        )

    def validate(self) -> SemanticCatalogValidationResult:
        """Validate descriptor consistency without mutating the catalog."""
        errors: list[str] = []
        warnings: list[str] = []
        names = [descriptor.name for descriptor in self.list_all()]
        name_set = set(names)

        if len(names) != len(name_set):
            errors.append("Semantic catalog contains duplicate tool names.")

        for descriptor in self.list_all():
            _validate_descriptor(descriptor, name_set, errors, warnings)

        return SemanticCatalogValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            tool_count=len(names),
            validated_tools=sorted(names),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serializable catalog representation."""
        return {
            "tools": [
                descriptor.to_dict()
                for descriptor in self.list_all()
            ]
        }

    def to_json(self) -> str:
        """Return deterministic JSON without executable objects."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def summary_text(self) -> str:
        """Return a compact controlled summary for future manuals or prompts."""
        lines: list[str] = []
        for descriptor in self.list_all():
            lines.append(
                f"{descriptor.name}: "
                f"capabilities={','.join(descriptor.capabilities)}; "
                f"risk={descriptor.risk_level}; "
                f"confirmation={str(descriptor.requires_confirmation).lower()}"
            )
        return "\n".join(lines)


def _build_semantic_descriptor(
    descriptor: ToolDescriptor,
    metadata: Mapping[str, Any],
    intents: tuple[str, ...],
    schema_registry: ArgumentSchemaRegistry | None,
) -> SemanticToolDescriptor:
    required_arguments, optional_arguments = _arguments_for_tool(
        descriptor,
        intents,
        schema_registry,
    )
    technical_arguments = tuple(dict.fromkeys(required_arguments + optional_arguments))
    dangerous = bool(metadata.get("dangerous", descriptor.dangerous))
    risk_level = str(
        metadata.get(
            "risk_level",
            _default_risk_level(descriptor.name, dangerous, descriptor.requires_confirmation),
        )
    )
    requires_confirmation = bool(
        descriptor.requires_confirmation
        or dangerous
        or metadata.get("requires_confirmation", False)
        or risk_level in _CONFIRMATION_RISK_LEVELS
    )

    return SemanticToolDescriptor(
        name=descriptor.name,
        description=descriptor.description,
        capabilities=_metadata_tuple(
            metadata,
            "capabilities",
            default=(_normalize_identifier(descriptor.name),),
            normalize=_normalize_identifier,
        ),
        supported_intents=_metadata_tuple(
            metadata,
            "supported_intents",
            default=_default_supported_intents(descriptor.name, intents),
            normalize=_clean_text,
        ),
        input_description=str(
            metadata.get(
                "input_description",
                _default_input_description(required_arguments, optional_arguments),
            )
        ),
        required_arguments=_metadata_tuple(
            metadata,
            "required_arguments",
            default=required_arguments,
            normalize=_normalize_identifier,
        ),
        optional_arguments=_metadata_tuple(
            metadata,
            "optional_arguments",
            default=optional_arguments,
            normalize=_normalize_identifier,
        ),
        output_description=str(
            metadata.get(
                "output_description",
                descriptor.output_description or "Output contract is not explicitly declared.",
            )
        ),
        output_fields=_metadata_tuple(
            metadata,
            "output_fields",
            default=(),
            normalize=_normalize_identifier,
        ),
        dangerous=dangerous,
        risk_level=risk_level,
        risk_reasons=_metadata_tuple(
            metadata,
            "risk_reasons",
            default=_default_risk_reasons(descriptor.name, dangerous, requires_confirmation),
            normalize=_clean_text,
        ),
        requires_confirmation=requires_confirmation,
        preconditions=_metadata_tuple(
            metadata,
            "preconditions",
            default=_default_preconditions(required_arguments),
            normalize=_clean_text,
        ),
        limitations=_metadata_tuple(
            metadata,
            "limitations",
            default=("semantic metadata is incomplete",),
            normalize=_clean_text,
        ),
        negative_examples=_metadata_tuple(
            metadata,
            "negative_examples",
            default=("explain the concept without using local tools",),
            normalize=_clean_text,
        ),
        compatible_tools=_metadata_tuple(
            metadata,
            "compatible_tools",
            default=(),
            normalize=str,
        ),
        tags=_metadata_tuple(
            metadata,
            "tags",
            default=_default_tags(descriptor.name, risk_level),
            normalize=_normalize_identifier,
        ),
        positive_examples=_metadata_tuple(
            metadata,
            "positive_examples",
            default=(),
            normalize=_clean_text,
        ),
        category=str(metadata.get("category", _default_category(descriptor.name))),
        version=str(metadata.get("version", "1")),
        technical_arguments=technical_arguments,
    )


def _validate_descriptor(
    descriptor: SemanticToolDescriptor,
    registered_tools: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not descriptor.name:
        errors.append("Semantic descriptor has empty tool name.")
    if not descriptor.description.strip():
        errors.append(f"Tool '{descriptor.name}' has empty description.")
    if not descriptor.capabilities:
        errors.append(f"Tool '{descriptor.name}' must declare at least one capability.")
    if any(not _is_valid_identifier(item) for item in descriptor.capabilities):
        errors.append(f"Tool '{descriptor.name}' has malformed capability.")
    if descriptor.risk_level not in RISK_LEVELS:
        errors.append(f"Tool '{descriptor.name}' has invalid risk level '{descriptor.risk_level}'.")
    if descriptor.dangerous and not descriptor.requires_confirmation:
        errors.append(f"Tool '{descriptor.name}' is dangerous but does not require confirmation.")
    if descriptor.risk_level in _CONFIRMATION_RISK_LEVELS and not descriptor.requires_confirmation:
        errors.append(f"Tool '{descriptor.name}' has risk level '{descriptor.risk_level}' without confirmation.")
    if not descriptor.supported_intents:
        warnings.append(f"Tool '{descriptor.name}' has no supported intents.")
    if "semantic metadata is incomplete" in descriptor.limitations:
        warnings.append(f"Tool '{descriptor.name}' uses conservative semantic defaults.")
    for argument in descriptor.required_arguments + descriptor.optional_arguments:
        if not argument or not _is_valid_identifier(argument):
            errors.append(f"Tool '{descriptor.name}' has malformed argument '{argument}'.")
        if descriptor.technical_arguments and argument not in descriptor.technical_arguments:
            errors.append(f"Tool '{descriptor.name}' declares unsupported argument '{argument}'.")
        if not descriptor.technical_arguments and argument:
            errors.append(f"Tool '{descriptor.name}' declares argument '{argument}' without a technical schema.")
    for compatible in descriptor.compatible_tools:
        if compatible not in registered_tools:
            errors.append(f"Tool '{descriptor.name}' references unknown compatible tool '{compatible}'.")
        if compatible == descriptor.name:
            errors.append(f"Tool '{descriptor.name}' cannot be compatible with itself.")
    if descriptor.output_fields and descriptor.output_description == "Output contract is not explicitly declared.":
        errors.append(f"Tool '{descriptor.name}' declares output fields without an output contract.")
    if _contains_secret(descriptor.to_dict()):
        errors.append(f"Tool '{descriptor.name}' semantic metadata contains a secret-like value.")


def _semantic_metadata(tool: Any) -> Mapping[str, Any]:
    metadata_provider = getattr(tool, "semantic_metadata", None)
    if metadata_provider is None:
        return {}
    metadata = metadata_provider()
    if not isinstance(metadata, Mapping):
        return {}
    return dict(metadata)


def _intent_to_tool_map(
    tool_selector: ToolSelector | None,
) -> dict[str, str]:
    if tool_selector is None:
        return {}

    mappings: dict[str, str] = {}
    for intent in tool_selector.supported_intents():
        try:
            mappings[intent] = tool_selector.select(ToolIntent(intent)).tool_name
        except ToolNotRegisteredError:
            continue
    return dict(sorted(mappings.items()))


def _arguments_for_tool(
    descriptor: ToolDescriptor,
    intents: tuple[str, ...],
    schema_registry: ArgumentSchemaRegistry | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if schema_registry is None:
        return descriptor.required_arguments, descriptor.optional_arguments

    required: list[str] = []
    optional: list[str] = []
    for intent in intents:
        if not schema_registry.exists(intent):
            continue
        schema = schema_registry.get(intent)
        for field in schema.fields:
            target = required if field.required else optional
            if field.name not in required and field.name not in optional:
                target.append(field.name)

    return tuple(required), tuple(optional)


def _metadata_tuple(
    metadata: Mapping[str, Any],
    key: str,
    *,
    default: tuple[str, ...],
    normalize: Any,
) -> tuple[str, ...]:
    raw = metadata.get(key, default)
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, list | tuple):
        values = tuple(item for item in raw if isinstance(item, str))
    else:
        values = default

    normalized: list[str] = []
    for value in values:
        item = normalize(value)
        if item and item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _default_supported_intents(
    tool_name: str,
    intents: tuple[str, ...],
) -> tuple[str, ...]:
    if intents:
        return tuple(_intent_phrase(intent) for intent in intents)
    return (f"use {tool_name.replace('_', ' ').replace('.', ' ')}",)


def _intent_phrase(intent: str) -> str:
    phrases = {
        "file.read": "read a local file",
        "file.write": "create or update a text file",
        "directory.list": "list files in a directory",
        "project.tree": "list python files in a project",
        "desktop.application.open": "open an installed application",
        "desktop.file.open": "open a local file",
        "desktop.text.type": "type text into a window",
        "desktop.hotkey.press": "send a keyboard shortcut",
        "desktop.windows.list": "list visible desktop windows",
    }
    return phrases.get(intent, intent.replace(".", " "))


def _default_input_description(
    required: tuple[str, ...],
    optional: tuple[str, ...],
) -> str:
    parts: list[str] = []
    if required:
        parts.append("Required arguments: " + ", ".join(required) + ".")
    if optional:
        parts.append("Optional arguments: " + ", ".join(optional) + ".")
    return " ".join(parts) if parts else "No structured arguments are explicitly declared."


def _default_preconditions(
    required_arguments: tuple[str, ...],
) -> tuple[str, ...]:
    if not required_arguments:
        return ()
    return tuple(f"{argument} must be provided" for argument in required_arguments)


def _default_risk_level(
    name: str,
    dangerous: bool,
    requires_confirmation: bool,
) -> str:
    if "terminate" in name or "close" in name or "clear" in name:
        return "high"
    if dangerous or requires_confirmation or "write" in name or "type" in name or "hotkey" in name:
        return "medium"
    if "read" in name or "list" in name or "get" in name or "is_" in name:
        return "low"
    return "low"


def _default_risk_reasons(
    name: str,
    dangerous: bool,
    requires_confirmation: bool,
) -> tuple[str, ...]:
    if dangerous or requires_confirmation:
        return ("tool can modify local state or user interface",)
    if "terminate" in name or "close" in name:
        return ("tool can interrupt running applications or windows",)
    return ()


def _default_tags(
    name: str,
    risk_level: str,
) -> tuple[str, ...]:
    tags = [_default_category(name), f"risk_{risk_level}"]
    return tuple(_normalize_identifier(tag) for tag in tags)


def _default_category(name: str) -> str:
    if name.startswith("desktop."):
        return "desktop"
    if name.startswith("project"):
        return "project"
    if "file" in name or "directory" in name:
        return "filesystem"
    return "general"


def _normalize_identifier(value: str) -> str:
    normalized = _normalize_phrase(value)
    return re.sub(r"[^a-z0-9_]+", "_", normalized).strip("_")


def _normalize_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(re.sub(r"[^\w\s./-]", " ", without_accents).split())


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _is_valid_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?", value))


def _contains_secret(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _looks_secret_like(str(key)) or _contains_secret(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_secret(item) for item in value)
    if isinstance(value, str):
        return _looks_secret_like(value)
    return False


def _looks_secret_like(value: str) -> bool:
    lowered = value.lower()
    return any(pattern in lowered for pattern in _SECRET_PATTERNS)

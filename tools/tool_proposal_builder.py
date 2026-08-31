"""Build structured tool proposals from natural-language requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Any, Mapping
import unicodedata

from tools.argument_schema import (
    ArgumentSchema,
    ArgumentSchemaNotRegisteredError,
    ArgumentValidationError,
    ArgumentValidator,
    ArgumentSchemaRegistry,
)
from tools.calendar.calendar_request_parser import extract_calendar_arguments
from tools.execution_decision import ExecutionDecision, ExecutionMode
from tools.intent_selector import ToolIntent, ToolIntentNotSupportedError, ToolSelector
from tools.registry import ToolNotRegisteredError, ToolRegistry


class ToolProposalStatus(str, Enum):
    """Lifecycle states for a proposed single-tool intent."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"


class ToolProposalError(ValueError):
    """Raised when a proposal cannot safely become an executable intent."""


@dataclass(frozen=True, slots=True)
class StructuredToolProposal:
    """Structured, validated proposal before any Atlas tool execution."""

    tool_name: str | None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    missing_arguments: tuple[str, ...] = ()
    ambiguous_arguments: tuple[str, ...] = ()
    validation_errors: tuple[str, ...] = ()
    source_text: str = ""
    status: ToolProposalStatus = ToolProposalStatus.UNSUPPORTED

    def __post_init__(self) -> None:
        confidence = min(max(float(self.confidence), 0.0), 1.0)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(
                {
                    name: _freeze_argument_value(value)
                    for name, value in self.arguments.items()
                }
            ),
        )

    @property
    def executable(self) -> bool:
        """Return whether this proposal can be converted into a ToolIntent."""
        return self.status is ToolProposalStatus.COMPLETE

    def to_tool_intent(
        self,
        selector: ToolSelector,
        validator: ArgumentValidator,
    ) -> ToolIntent:
        """Return a revalidated ToolIntent only when the proposal is complete."""
        if not self.executable:
            raise ToolProposalError(
                "Only COMPLETE tool proposals can be converted to ToolIntent."
            )

        if self.tool_name is None:
            raise ToolProposalError("Complete proposals require a tool_name.")

        intent = ToolIntent(
            action=self.tool_name,
            arguments=_thaw_arguments(self.arguments),
        )
        validator.validate(selector.select(intent))

        return intent


@dataclass(frozen=True, slots=True)
class _ExtractionResult:
    arguments: Mapping[str, Any]
    ambiguous_arguments: tuple[str, ...] = ()


class ToolProposalBuilder:
    """Convert a SINGLE_TOOL decision into a safe structured proposal."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        selector: ToolSelector,
        schema_registry: ArgumentSchemaRegistry,
        validator: ArgumentValidator,
    ) -> None:
        self._tool_registry = tool_registry
        self._selector = selector
        self._schema_registry = schema_registry
        self._validator = validator

    def build(
        self,
        source_text: str,
        decision: ExecutionDecision,
        candidate_tools: tuple[str, ...] | None = None,
    ) -> StructuredToolProposal:
        """Build one structured proposal without executing any tool."""
        candidates = tuple(candidate_tools or decision.candidate_tools)

        if decision.mode is not ExecutionMode.SINGLE_TOOL:
            return self._unsupported(
                None,
                source_text,
                f"Execution decision mode {decision.mode.value} is not SINGLE_TOOL.",
                decision.confidence,
            )

        if len(candidates) != 1:
            return self._unsupported(
                None,
                source_text,
                "SINGLE_TOOL proposals require exactly one candidate tool intent.",
                decision.confidence,
            )

        intent_action = candidates[0]

        if not self._selector.supports(intent_action):
            return self._unsupported(
                intent_action,
                source_text,
                f"Candidate tool intent '{intent_action}' is not registered.",
                decision.confidence,
            )

        try:
            selection = self._selector.select(ToolIntent(intent_action))
        except (ToolIntentNotSupportedError, ToolNotRegisteredError) as error:
            return self._unsupported(
                intent_action,
                source_text,
                str(error),
                decision.confidence,
            )

        try:
            schema = self._schema_registry.get(intent_action)
        except ArgumentSchemaNotRegisteredError as error:
            return self._unsupported(
                intent_action,
                source_text,
                str(error),
                decision.confidence,
            )

        extraction = self._extract_arguments(source_text, intent_action, schema)
        arguments = dict(extraction.arguments)
        ambiguous = extraction.ambiguous_arguments
        missing = self._missing_required_arguments(schema, arguments, ambiguous)

        if ambiguous:
            return StructuredToolProposal(
                tool_name=intent_action,
                arguments=arguments,
                confidence=min(decision.confidence, 0.55),
                reason=f"{decision.reason} Argument extraction is ambiguous.",
                missing_arguments=missing,
                ambiguous_arguments=ambiguous,
                validation_errors=(),
                source_text=source_text,
                status=ToolProposalStatus.AMBIGUOUS,
            )

        try:
            validation = self._validator.validate(
                self._selector.select(ToolIntent(intent_action, arguments))
            )
        except ArgumentValidationError as error:
            validation_errors = (f"{error.field}: {error.reason}",)
            status = (
                ToolProposalStatus.INCOMPLETE
                if error.reason == "required argument is missing"
                else ToolProposalStatus.UNSUPPORTED
            )
            missing = self._merge_unique(missing, (error.field,)) if status is ToolProposalStatus.INCOMPLETE else missing
            return StructuredToolProposal(
                tool_name=intent_action,
                arguments=arguments,
                confidence=min(decision.confidence, 0.65),
                reason=f"{decision.reason} Argument validation did not pass.",
                missing_arguments=missing,
                ambiguous_arguments=(),
                validation_errors=validation_errors,
                source_text=source_text,
                status=status,
            )

        return StructuredToolProposal(
            tool_name=intent_action,
            arguments=dict(validation.validated_arguments),
            confidence=decision.confidence,
            reason=f"{decision.reason} Extracted arguments satisfy the registered schema.",
            missing_arguments=(),
            ambiguous_arguments=(),
            validation_errors=(),
            source_text=source_text,
            status=ToolProposalStatus.COMPLETE,
        )

    def _unsupported(
        self,
        tool_name: str | None,
        source_text: str,
        reason: str,
        confidence: float,
    ) -> StructuredToolProposal:
        return StructuredToolProposal(
            tool_name=tool_name,
            arguments={},
            confidence=min(confidence, 0.35),
            reason=reason,
            source_text=source_text,
            status=ToolProposalStatus.UNSUPPORTED,
        )

    def to_tool_intent(
        self,
        proposal: StructuredToolProposal,
    ) -> ToolIntent:
        """Revalidate a complete proposal before returning a ToolIntent."""
        return proposal.to_tool_intent(self._selector, self._validator)

    def _extract_arguments(
        self,
        source_text: str,
        intent_action: str,
        schema: ArgumentSchema,
    ) -> _ExtractionResult:
        normalized = _normalize(source_text)
        field_names = {field.name for field in schema.fields}

        if intent_action == "file.read":
            return _ExtractionResult(_filter_fields({"path": _extract_path(source_text, normalized)}, field_names))

        if intent_action == "directory.list":
            if _has_ambiguous_directory_target(normalized):
                return _ExtractionResult({}, ("path",))
            return _ExtractionResult(_filter_fields({"path": _extract_directory_path(source_text, normalized)}, field_names))

        if intent_action == "calendar.events.list":
            return _ExtractionResult(
                _filter_fields(extract_calendar_arguments(source_text), field_names)
            )

        if intent_action == "file.write":
            path = _extract_path(source_text, normalized)
            content = _extract_write_content(source_text, normalized, path)
            arguments = _filter_fields({"path": path, "content": content}, field_names)
            ambiguous = []
            if _has_ambiguous_file_target(normalized):
                arguments.pop("path", None)
                ambiguous.append("path")
            if _has_ambiguous_content(normalized):
                arguments.pop("content", None)
                ambiguous.append("content")
            return _ExtractionResult(arguments, tuple(ambiguous))

        if intent_action == "desktop.application.open":
            application = _extract_application(source_text, normalized)
            return _ExtractionResult(_filter_fields({"application": application}, field_names))

        if intent_action == "desktop.file.open":
            return _ExtractionResult(_filter_fields({"path": _extract_path(source_text, normalized)}, field_names))

        if intent_action == "desktop.hotkey.press":
            return _ExtractionResult(
                _filter_fields(
                    {
                        "keys": _extract_hotkey(normalized),
                        "window_title": _extract_window_title(source_text),
                    },
                    field_names,
                )
            )

        if intent_action == "desktop.text.type":
            text = _extract_type_text(source_text, normalized)
            return _ExtractionResult(
                _filter_fields(
                    {
                        "text": text,
                        "window_title": _extract_type_window_title(source_text),
                    },
                    field_names,
                )
            )

        if intent_action == "desktop.clipboard.copy":
            return _ExtractionResult(
                _filter_fields({"text": _extract_clipboard_copy_text(source_text)}, field_names)
            )

        if intent_action == "desktop.clipboard.paste":
            return _ExtractionResult({})

        return _ExtractionResult({})

    def _missing_required_arguments(
        self,
        schema: ArgumentSchema,
        arguments: Mapping[str, Any],
        ambiguous_arguments: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            field.name
            for field in schema.fields
            if field.required
            and field.name not in arguments
            and field.name not in ambiguous_arguments
        )

    def _merge_unique(
        self,
        left: tuple[str, ...],
        right: tuple[str, ...],
    ) -> tuple[str, ...]:
        merged: list[str] = []
        for item in left + right:
            if item not in merged:
                merged.append(item)
        return tuple(merged)


def _filter_fields(
    arguments: Mapping[str, Any | None],
    field_names: set[str],
) -> dict[str, Any]:
    return {
        name: value
        for name, value in arguments.items()
        if name in field_names and value is not None
    }


def _freeze_argument_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                nested_name: _freeze_argument_value(nested_value)
                for nested_name, nested_value in value.items()
            }
        )

    if isinstance(value, list):
        return tuple(_freeze_argument_value(item) for item in value)

    return value


def _thaw_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: _thaw_argument_value(value)
        for name, value in arguments.items()
    }


def _thaw_argument_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            nested_name: _thaw_argument_value(nested_value)
            for nested_name, nested_value in value.items()
        }

    if isinstance(value, tuple):
        return [_thaw_argument_value(item) for item in value]

    return value


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    without_punctuation = re.sub(r"[^\w\s./+:-]", " ", without_accents)
    return " ".join(without_punctuation.split())


def _extract_path(source_text: str, normalized: str) -> str | None:
    quoted = _extract_quoted(source_text)
    if quoted and _looks_like_path(quoted):
        return quoted

    after_path_keyword = re.search(
        r"\b(?:archivo|fichero|ruta)\s+(?P<path>(?!este\b|ese\b|un\b|una\b)(?:[A-Za-z]:[\\/])?(?:[\w .-]+[\\/])*[\w .-]+\.(?:md|txt|py|json|csv|yaml|yml|toml))\b",
        source_text,
        flags=re.IGNORECASE,
    )
    if after_path_keyword:
        return after_path_keyword.group("path").strip()

    extension_match = re.search(
        r"(?P<path>(?:[A-Za-z]:[\\/])?(?:[\w.-]+[\\/])*[\w.-]+\.(?:md|txt|py|json|csv|yaml|yml|toml))\b",
        source_text,
        flags=re.IGNORECASE,
    )
    if extension_match:
        return extension_match.group("path")

    return None


def _extract_directory_path(source_text: str, normalized: str) -> str | None:
    quoted = _extract_quoted(source_text)
    if quoted:
        return quoted

    directory_match = re.search(
        r"\b(?:carpeta|directorio|ruta)\s+(?P<path>(?!esta\b|este\b|un\b|una\b)(?:[A-Za-z]:[\\/])?[\w./\\-]+)",
        source_text,
        flags=re.IGNORECASE,
    )
    if directory_match:
        return directory_match.group("path")

    return _extract_path(source_text, normalized)


def _extract_write_content(
    source_text: str,
    normalized: str,
    path: str | None,
) -> str | None:
    quoted = _extract_quoted(source_text)
    if quoted and quoted != path and not _looks_like_path(quoted):
        return quoted

    match = re.search(
        r"\b(?:escribe|guarda|copia)\s+(?P<content>.+?)\s+\ben\b",
        source_text,
        flags=re.IGNORECASE,
    )
    if match:
        content = match.group("content").strip()
        if content and not _is_ambiguous_placeholder(_normalize(content)):
            return content

    match = re.search(
        r"\b(?:contenido|texto)\s+(?P<content>.+?)\s+\ben\b",
        source_text,
        flags=re.IGNORECASE,
    )
    if match:
        content = match.group("content").strip()
        if content and not _is_ambiguous_placeholder(_normalize(content)):
            return content

    if path is None:
        match = re.search(
            r"\b(?:escribe)\s+(?P<content>.+)$",
            source_text,
            flags=re.IGNORECASE,
        )
        if match:
            content = match.group("content").strip()
            if content and not _is_ambiguous_placeholder(_normalize(content)):
                return content

    return None


def _extract_application(source_text: str, normalized: str) -> str | None:
    aliases = {
        "vs code": "VS Code",
        "vscode": "VS Code",
        "visual studio code": "VS Code",
        "bloc de notas": "notepad",
        "notepad": "notepad",
    }
    for alias, value in aliases.items():
        if alias in normalized:
            return value

    quoted = _extract_quoted(source_text)
    if quoted:
        return quoted

    match = re.search(r"\b(?:abre|abrir)\s+(?P<application>[\w .-]+)$", normalized)
    if match:
        application = match.group("application").strip()
        if application and not _looks_like_path(application):
            return application

    return None


def _extract_hotkey(normalized: str) -> list[str] | None:
    match = re.search(
        r"\b(?P<keys>(?:ctrl|control|alt|shift|mayus|win|windows|cmd|meta)(?:\s*[+]\s*|\s+mas\s+|\s+)(?:[\w]+)(?:(?:\s*[+]\s*|\s+mas\s+|\s+)(?:[\w]+))*)\b",
        normalized,
    )
    if not match:
        return None

    keys = re.split(r"\s*(?:[+]|\bmas\b)\s*|\s+", match.group("keys"))
    normalized_keys = [_normalize_key(key) for key in keys if key.strip()]

    return normalized_keys or None


def _extract_window_title(source_text: str) -> str | None:
    quoted = _extract_quoted(source_text)
    if quoted:
        return quoted

    match = re.search(
        r"\b(?:en|ventana)\s+(?P<title>[\w .-]+)$",
        source_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    title = match.group("title").strip()
    return title or None


def _extract_type_text(source_text: str, normalized: str) -> str | None:
    quoted = _extract_quoted(source_text)
    if quoted:
        return quoted

    match = re.search(
        r"\b(?:escribe|teclea)\s+(?P<text>.+?)\s+\ben\s+(?:la\s+)?ventana\b",
        source_text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group("text").strip() or None

    match = re.search(r"\b(?:escribe|teclea)\s+(?P<text>.+)$", normalized)
    if match:
        text = match.group("text").strip()
        if text and not _is_ambiguous_placeholder(text):
            return text

    return None


def _extract_clipboard_copy_text(source_text: str) -> str | None:
    match = re.match(
        r"^\s*copia(?:\s+este\s+texto)?\s*(?::\s*|\s+)(?P<text>.*)$",
        source_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    return match.group("text") or None


def _extract_type_window_title(source_text: str) -> str | None:
    match = re.search(
        r"\ben\s+(?:la\s+)?ventana\s+(?:de\s+)?(?P<title>[\w .-]+)$",
        source_text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group("title").strip() or None

    match = re.search(
        r"\b(?:en|ventana)\s+(?P<title>[\w .-]+)$",
        source_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    title = match.group("title").strip()
    return title or None

def _extract_quoted(source_text: str) -> str | None:
    match = re.search(r"[\"'“”‘’](?P<value>.+?)[\"'“”‘’]", source_text)
    if not match:
        return None

    value = match.group("value").strip()
    return value or None


def _looks_like_path(value: str) -> bool:
    return bool(
        re.search(r"\.[A-Za-z0-9]{1,8}\b", value)
        or "/" in value
        or "\\" in value
    )


def _has_ambiguous_file_target(normalized: str) -> bool:
    return bool(
        re.search(r"\b(?:un|una|este|esta|ese|esa)\s+(?:archivo|fichero)\b", normalized)
    )


def _has_ambiguous_directory_target(normalized: str) -> bool:
    return bool(
        re.search(r"\b(?:un|una|este|esta|ese|esa)\s+(?:carpeta|directorio)\b", normalized)
    )


def _has_ambiguous_content(normalized: str) -> bool:
    return bool(
        re.search(r"\b(?:algo|cualquier cosa|lo que sea)\b", normalized)
    )


def _is_ambiguous_placeholder(value: str) -> bool:
    return value.strip().lower() in {
        "algo",
        "cualquier cosa",
        "lo que sea",
        "texto",
        "contenido",
    }


def _normalize_key(key: str) -> str:
    aliases = {
        "control": "ctrl",
        "mayus": "shift",
        "windows": "win",
        "cmd": "win",
        "meta": "win",
        "escape": "esc",
        "intro": "enter",
        "entrar": "enter",
        "espacio": "space",
    }
    normalized = key.strip().lower()
    return aliases.get(normalized, normalized)

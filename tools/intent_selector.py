"""Deterministic selection of registered tools from structured intents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping
import unicodedata

from tools.registry import ToolDescriptor, ToolRegistry, ToolNotRegisteredError

if TYPE_CHECKING:
    from tools.semantic_catalog import SemanticToolCatalog, SemanticToolDescriptor


CAPABILITY_EXACT_SCORE = 100.0
INTENT_EXACT_SCORE = 80.0
INTENT_PARTIAL_SCORE = 40.0
TAG_EXACT_SCORE = 25.0
POSITIVE_EXAMPLE_SCORE = 20.0
ARGUMENT_HINT_SCORE = 10.0
INCOMPLETE_METADATA_PENALTY = 15.0
NEGATIVE_EXAMPLE_PENALTY = 100.0
MINIMUM_SELECTION_SCORE = 45.0
AMBIGUITY_MARGIN = 12.0
MAXIMUM_CANDIDATES = 5
MINIMUM_QUERY_LENGTH = 2


class ToolSelectionErrorCode(str, Enum):
    """Stable semantic tool-selection result codes."""

    EMPTY_QUERY = "EMPTY_QUERY"
    NO_TOOL_MATCH = "NO_TOOL_MATCH"
    AMBIGUOUS_TOOL_SELECTION = "AMBIGUOUS_TOOL_SELECTION"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    CATALOG_INVALID = "CATALOG_INVALID"
    SELECTOR_INTERNAL_ERROR = "SELECTOR_INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class ToolCandidate:
    """One ranked semantic tool candidate."""

    tool_name: str
    score: float
    matched_capabilities: tuple[str, ...] = ()
    matched_intents: tuple[str, ...] = ()
    matched_examples: tuple[str, ...] = ()
    negative_matches: tuple[str, ...] = ()
    risk_level: str = "low"
    requires_confirmation: bool = False
    explanation: str = ""


@dataclass(frozen=True, slots=True)
class ToolSelectionResult:
    """Structured result of semantic tool selection without execution."""

    success: bool
    query: str
    normalized_query: str
    candidates: tuple[ToolCandidate, ...] = ()
    selected_tool: str | None = None
    confidence: float = 0.0
    ambiguous: bool = False
    requires_clarification: bool = False
    reasons: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class _CandidateScore:
    descriptor: SemanticToolDescriptor
    score: float
    matched_capabilities: tuple[str, ...]
    matched_intents: tuple[str, ...]
    matched_examples: tuple[str, ...]
    negative_matches: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolIntent:
    """Structured request for selecting a tool without executing it."""

    action: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    target: str | None = None

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("Tool intent action cannot be empty.")

        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )


@dataclass(frozen=True, slots=True)
class ToolSelection:
    """Result of resolving one structured intent to one registered tool."""

    intent: ToolIntent
    tool_name: str
    descriptor: ToolDescriptor
    arguments: Mapping[str, Any]
    executed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )


class ToolIntentAlreadyRegisteredError(ValueError):
    """Raised when an intent action is mapped twice."""


class ToolIntentNotSupportedError(RuntimeError):
    """Raised when an intent action has no registered mapping."""


class ToolIntentRegistry:
    """Central source of truth for intent-to-tool mappings."""

    def __init__(self) -> None:
        self._mappings: dict[str, str] = {}

    def register(
        self,
        action: str,
        tool_name: str,
    ) -> None:
        """Register one stable intent action to one tool identifier."""
        if not action:
            raise ValueError("Tool intent action cannot be empty.")

        if not tool_name:
            raise ValueError("Tool name cannot be empty.")

        if action in self._mappings:
            raise ToolIntentAlreadyRegisteredError(
                f"Tool intent '{action}' is already registered."
            )

        self._mappings[action] = tool_name

    def supports(
        self,
        action: str,
    ) -> bool:
        """Return whether an action has an explicit tool mapping."""
        return action in self._mappings

    def resolve(
        self,
        action: str,
    ) -> str:
        """Return the mapped tool identifier for an action."""
        try:
            return self._mappings[action]
        except KeyError as error:
            raise ToolIntentNotSupportedError(
                f"Tool intent '{action}' is not supported."
            ) from error

    def list(self) -> tuple[str, ...]:
        """Return supported intent actions."""
        return tuple(sorted(self._mappings.keys()))

    @property
    def mappings(self) -> Mapping[str, str]:
        """Return a read-only view of intent-to-tool mappings."""
        return MappingProxyType(self._mappings)


class ToolSelector:
    """Select registered tools from structured intents without executing them."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        intent_registry: ToolIntentRegistry,
    ) -> None:
        self._tool_registry = tool_registry
        self._intent_registry = intent_registry

    def supports(
        self,
        action: str,
    ) -> bool:
        """Return whether an intent action can be selected."""
        return self._intent_registry.supports(action)

    def supported_intents(self) -> tuple[str, ...]:
        """Return all supported intent actions."""
        return self._intent_registry.list()

    def select(
        self,
        intent: ToolIntent,
    ) -> ToolSelection:
        """Resolve an intent to a registered tool descriptor without execution."""
        tool_name = self._intent_registry.resolve(intent.action)

        try:
            descriptor = self._tool_registry.descriptor(tool_name)
        except ToolNotRegisteredError as error:
            raise ToolNotRegisteredError(
                f"Tool intent '{intent.action}' maps to missing tool '{tool_name}'."
            ) from error

        return ToolSelection(
            intent=intent,
            tool_name=tool_name,
            descriptor=descriptor,
            arguments=intent.arguments,
            executed=False,
        )

    def select_from_catalog(
        self,
        query: str,
        catalog: SemanticToolCatalog,
        *,
        top_k: int = MAXIMUM_CANDIDATES,
    ) -> ToolSelectionResult:
        """Select semantic candidates from a catalog without execution."""
        return select(query, catalog, top_k=top_k)


def select(
    query: str,
    catalog: SemanticToolCatalog,
    *,
    top_k: int = MAXIMUM_CANDIDATES,
) -> ToolSelectionResult:
    """Rank semantic tool candidates for a user query without execution."""
    normalized_query = _normalize_selection_text(query)
    if len(normalized_query) < MINIMUM_QUERY_LENGTH:
        return ToolSelectionResult(
            success=False,
            query=query,
            normalized_query=normalized_query,
            selected_tool=None,
            confidence=0.0,
            ambiguous=False,
            requires_clarification=True,
            reasons=("Query is empty or too short for safe tool selection.",),
            errors=("Tool selection query cannot be empty.",),
            error_code=ToolSelectionErrorCode.EMPTY_QUERY.value,
        )

    validation = catalog.validate()
    if not validation.is_valid:
        return ToolSelectionResult(
            success=False,
            query=query,
            normalized_query=normalized_query,
            selected_tool=None,
            confidence=0.0,
            ambiguous=False,
            requires_clarification=True,
            reasons=("Semantic tool catalog is invalid.",),
            errors=tuple(validation.errors),
            warnings=tuple(validation.warnings),
            error_code=ToolSelectionErrorCode.CATALOG_INVALID.value,
        )

    scores = tuple(
        score
        for score in (
            _score_descriptor(normalized_query, descriptor)
            for descriptor in catalog.list_all()
        )
        if score.score > 0 or score.negative_matches
    )
    ranked_scores = tuple(
        sorted(
            scores,
            key=lambda item: (-item.score, item.descriptor.name),
        )
    )
    selected_scores = ranked_scores[: max(1, min(top_k, MAXIMUM_CANDIDATES))]
    candidates = tuple(_candidate_from_score(item) for item in selected_scores)
    warnings = tuple(validation.warnings) + tuple(
        warning
        for score in selected_scores
        for warning in score.warnings
    )

    weak_close_match = (
        len(selected_scores) > 1
        and selected_scores[0].score - selected_scores[1].score <= AMBIGUITY_MARGIN
        and selected_scores[0].score > 0
    )
    if candidates and (_is_generic_or_ambiguous(normalized_query) or weak_close_match):
        return ToolSelectionResult(
            success=True,
            query=query,
            normalized_query=normalized_query,
            candidates=candidates,
            selected_tool=None,
            confidence=_confidence(candidates[0].score),
            ambiguous=True,
            requires_clarification=True,
            reasons=("Tool selection is ambiguous; no tool was selected automatically.",),
            warnings=warnings,
            error_code=ToolSelectionErrorCode.AMBIGUOUS_TOOL_SELECTION.value,
        )

    if not candidates or candidates[0].score < MINIMUM_SELECTION_SCORE:
        return ToolSelectionResult(
            success=True,
            query=query,
            normalized_query=normalized_query,
            candidates=candidates,
            selected_tool=None,
            confidence=0.0,
            ambiguous=False,
            requires_clarification=False,
            reasons=("No candidate exceeded the minimum selection score.",),
            warnings=warnings,
            error_code=ToolSelectionErrorCode.NO_TOOL_MATCH.value,
        )

    top_score = selected_scores[0]
    second = selected_scores[1] if len(selected_scores) > 1 else None
    ambiguous = (
        _is_generic_or_ambiguous(normalized_query)
        or (second is not None and top_score.score - second.score <= AMBIGUITY_MARGIN)
    )
    missing_required = bool(top_score.missing_arguments)
    requires_clarification = ambiguous or missing_required
    selected_tool = None if requires_clarification else top_score.descriptor.name
    reasons = [_selection_reason(top_score)]
    if ambiguous:
        reasons.append("Tool selection is ambiguous; no tool was selected automatically.")
    if missing_required:
        reasons.append(
            "Missing required argument hints: "
            + ", ".join(top_score.missing_arguments)
            + "."
        )

    error_code: str | None = None
    if ambiguous:
        error_code = ToolSelectionErrorCode.AMBIGUOUS_TOOL_SELECTION.value
    elif missing_required:
        error_code = ToolSelectionErrorCode.INSUFFICIENT_INFORMATION.value

    return ToolSelectionResult(
        success=True,
        query=query,
        normalized_query=normalized_query,
        candidates=candidates,
        selected_tool=selected_tool,
        confidence=_confidence(top_score.score),
        ambiguous=ambiguous,
        requires_clarification=requires_clarification,
        reasons=tuple(reasons),
        warnings=warnings,
        error_code=error_code,
    )


def rank_candidates(
    query: str,
    catalog: SemanticToolCatalog,
    *,
    top_k: int = MAXIMUM_CANDIDATES,
) -> tuple[ToolCandidate, ...]:
    """Return ranked candidates only, without selecting a winner."""
    return select(query, catalog, top_k=top_k).candidates


def _score_descriptor(
    normalized_query: str,
    descriptor: SemanticToolDescriptor,
) -> _CandidateScore:
    score = 0.0
    matched_capabilities: list[str] = []
    matched_intents: list[str] = []
    matched_examples: list[str] = []
    negative_matches: list[str] = []
    warnings: list[str] = []

    query_tokens = _tokens(normalized_query)
    for capability in descriptor.capabilities:
        normalized_capability = _normalize_identifier(capability)
        capability_tokens = tuple(part for part in normalized_capability.split("_") if part)
        if normalized_query == normalized_capability or normalized_capability in query_tokens:
            score += CAPABILITY_EXACT_SCORE
            matched_capabilities.append(capability)
        elif capability_tokens and set(capability_tokens).issubset(query_tokens):
            score += CAPABILITY_EXACT_SCORE
            matched_capabilities.append(capability)

    for intent in descriptor.supported_intents:
        normalized_intent = _normalize_selection_text(intent)
        if normalized_query == normalized_intent:
            score += INTENT_EXACT_SCORE
            matched_intents.append(intent)
        elif _safe_phrase_overlap(normalized_query, normalized_intent):
            score += INTENT_PARTIAL_SCORE
            matched_intents.append(intent)

    for tag in descriptor.tags:
        normalized_tag = _normalize_identifier(tag)
        if normalized_tag in query_tokens:
            score += TAG_EXACT_SCORE

    for example in descriptor.positive_examples:
        normalized_example = _normalize_selection_text(example)
        if normalized_query == normalized_example or _safe_phrase_overlap(normalized_query, normalized_example):
            score += POSITIVE_EXAMPLE_SCORE
            matched_examples.append(example)

    for example in descriptor.negative_examples:
        normalized_example = _normalize_selection_text(example)
        if normalized_query == normalized_example or _safe_phrase_overlap(normalized_query, normalized_example):
            score -= NEGATIVE_EXAMPLE_PENALTY
            negative_matches.append(example)

    present_arguments, missing_arguments = _argument_hints(normalized_query, descriptor)
    if present_arguments:
        score += ARGUMENT_HINT_SCORE * len(present_arguments)

    if "semantic metadata is incomplete" in descriptor.limitations:
        score -= INCOMPLETE_METADATA_PENALTY
        warnings.append(f"Tool '{descriptor.name}' has incomplete semantic metadata.")

    if _clear_incompatibility(normalized_query, descriptor):
        score = 0.0
        warnings.append(f"Tool '{descriptor.name}' was excluded by a deterministic safety rule.")

    return _CandidateScore(
        descriptor=descriptor,
        score=max(score, 0.0),
        matched_capabilities=tuple(matched_capabilities),
        matched_intents=tuple(matched_intents),
        matched_examples=tuple(matched_examples),
        negative_matches=tuple(negative_matches),
        warnings=tuple(warnings),
        missing_arguments=tuple(missing_arguments),
    )


def _candidate_from_score(
    score: _CandidateScore,
) -> ToolCandidate:
    return ToolCandidate(
        tool_name=score.descriptor.name,
        score=score.score,
        matched_capabilities=score.matched_capabilities,
        matched_intents=score.matched_intents,
        matched_examples=score.matched_examples,
        negative_matches=score.negative_matches,
        risk_level=score.descriptor.risk_level,
        requires_confirmation=score.descriptor.requires_confirmation,
        explanation=_candidate_explanation(score),
    )


def _candidate_explanation(
    score: _CandidateScore,
) -> str:
    parts = [f"Candidate {score.descriptor.name} scored {score.score:.1f}."]
    if score.matched_capabilities:
        parts.append("Matched capabilities: " + ", ".join(score.matched_capabilities) + ".")
    if score.matched_intents:
        parts.append("Matched intents: " + ", ".join(score.matched_intents) + ".")
    if score.matched_examples:
        parts.append("Matched positive examples: " + ", ".join(score.matched_examples) + ".")
    if score.negative_matches:
        parts.append("Penalized by negative examples: " + ", ".join(score.negative_matches) + ".")
    if score.missing_arguments:
        parts.append("Missing required argument hints: " + ", ".join(score.missing_arguments) + ".")
    parts.append(
        f"Risk level is {score.descriptor.risk_level}; "
        f"confirmation required is {str(score.descriptor.requires_confirmation).lower()}."
    )
    return " ".join(parts)


def _selection_reason(
    score: _CandidateScore,
) -> str:
    if score.matched_capabilities:
        return f"Top candidate '{score.descriptor.name}' matched capability signals."
    if score.matched_intents:
        return f"Top candidate '{score.descriptor.name}' matched intent signals."
    if score.matched_examples:
        return f"Top candidate '{score.descriptor.name}' matched positive examples."
    return f"Top candidate '{score.descriptor.name}' exceeded the deterministic threshold."


def _argument_hints(
    normalized_query: str,
    descriptor: SemanticToolDescriptor,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    present: list[str] = []
    missing: list[str] = []
    for argument in descriptor.required_arguments:
        if _argument_present(argument, normalized_query):
            present.append(argument)
        else:
            missing.append(argument)
    return tuple(present), tuple(missing)


def _argument_present(
    argument: str,
    normalized_query: str,
) -> bool:
    if argument == "path":
        return bool(re.search(r"(?:[a-z]:[\\/])?[\w./\\-]+\.[a-z0-9]{1,8}\b|(?:[a-z]:[\\/])?[\w./\\-]+[\\/][\w./\\-]+", normalized_query))
    if argument == "content":
        return bool(re.search(r"\b(?:contenido|texto|con)\b", normalized_query))
    if argument == "pid":
        return bool(re.search(r"\bpid\s*\d+\b|\b\d{2,}\b", normalized_query))
    if argument in {"query", "application", "title", "window_title", "text"}:
        return len(_tokens(normalized_query)) > 2
    if argument == "keys":
        return bool(re.search(r"\b(?:ctrl|control|alt|shift|win|windows)\b", normalized_query))
    return argument in _tokens(normalized_query)


def _clear_incompatibility(
    normalized_query: str,
    descriptor: SemanticToolDescriptor,
) -> bool:
    tokens = _tokens(normalized_query)
    if normalized_query.startswith(("que es ", "que son ", "explica ", "explicame ")):
        return any(
            capability in descriptor.capabilities
            for capability in ("read_file", "write_file", "terminate_process")
        )
    if "historia" in tokens and "write_file" in descriptor.capabilities and not _looks_like_disk_write(normalized_query):
        return True
    if "proceso" in tokens or "procesos" in tokens:
        if "terminate_process" in descriptor.capabilities and not ({"termina", "terminar", "mata", "cerrar", "cierra"} & tokens):
            return True
    return False


def _looks_like_disk_write(
    normalized_query: str,
) -> bool:
    tokens = _tokens(normalized_query)
    return bool(
        {"archivo", "fichero", "guarda", "guardar", "copia"} & tokens
        or re.search(r"\.[a-z0-9]{1,8}\b", normalized_query)
    )


def _is_generic_or_ambiguous(
    normalized_query: str,
) -> bool:
    return normalized_query in {
        "abre eso",
        "hazlo",
        "mira el archivo",
        "gestiona el proceso",
        "gestiona el archivo",
        "gestiona eso",
    }


def _safe_phrase_overlap(
    normalized_query: str,
    normalized_phrase: str,
) -> bool:
    query_tokens = _tokens(normalized_query)
    phrase_tokens = _tokens(normalized_phrase)
    if not query_tokens or not phrase_tokens:
        return False
    meaningful_phrase_tokens = tuple(
        token for token in phrase_tokens if token not in _STOP_WORDS and len(token) > 2
    )
    meaningful_query_tokens = {
        token for token in query_tokens if token not in _STOP_WORDS and len(token) > 2
    }
    if not meaningful_phrase_tokens or not meaningful_query_tokens:
        return False
    overlap = meaningful_query_tokens.intersection(meaningful_phrase_tokens)
    return len(overlap) >= min(2, len(meaningful_phrase_tokens))


def _confidence(
    score: float,
) -> float:
    return min(max(score / 120.0, 0.0), 1.0)


def _normalize_selection_text(
    text: str,
) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    without_punctuation = re.sub(r"[^\w\s./\\:-]", " ", without_accents)
    return " ".join(without_punctuation.split())


def _normalize_identifier(
    value: str,
) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", _normalize_selection_text(value)).strip("_")


def _tokens(
    normalized_text: str,
) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9_]+", normalized_text))


_STOP_WORDS = {
    "a",
    "al",
    "an",
    "and",
    "de",
    "del",
    "el",
    "en",
    "es",
    "in",
    "into",
    "la",
    "las",
    "lo",
    "los",
    "of",
    "or",
    "the",
    "to",
    "un",
    "una",
}

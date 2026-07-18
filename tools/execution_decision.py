"""Deterministic execution-mode decisions for Atlas tool requests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping
import re
import unicodedata


class ExecutionMode(str, Enum):
    """High-level processing mode for one user request."""

    DIRECT_RESPONSE = "DIRECT_RESPONSE"
    SINGLE_TOOL = "SINGLE_TOOL"
    TOOL_CHAIN = "TOOL_CHAIN"


@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    """Structured decision without selection, arguments or execution."""

    mode: ExecutionMode
    reason: str
    confidence: float
    candidate_tools: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        confidence = min(max(float(self.confidence), 0.0), 1.0)
        object.__setattr__(self, "confidence", confidence)

        if self.metadata is not None:
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(dict(self.metadata)),
            )


@dataclass(frozen=True, slots=True)
class _CapabilityMatch:
    capability: str
    tool: str
    position: int


class ExecutionDecisionEngine:
    """Classify user requests as direct, single-tool or tool-chain work."""

    def __init__(
        self,
        supported_tool_intents: tuple[str, ...],
    ) -> None:
        self._supported_tool_intents = tuple(sorted(set(supported_tool_intents)))

    def decide(
        self,
        prompt: str,
    ) -> ExecutionDecision:
        """Return a structured execution-mode decision without side effects."""
        normalized = _normalize(prompt)

        if not normalized:
            return ExecutionDecision(
                mode=ExecutionMode.DIRECT_RESPONSE,
                reason="Empty request; no tool capability can be selected safely.",
                confidence=0.4,
                candidate_tools=(),
                required_capabilities=(),
            )

        matches = self._detect_capabilities(normalized)
        candidates = self._registered_tools(matches)
        capabilities = tuple(match.capability for match in matches)

        if self._is_general_or_conversational(normalized) and not matches:
            return ExecutionDecision(
                mode=ExecutionMode.DIRECT_RESPONSE,
                reason="The request is conversational or asks for general knowledge.",
                confidence=0.9,
                candidate_tools=(),
                required_capabilities=("direct_response",),
            )

        if not matches:
            return ExecutionDecision(
                mode=ExecutionMode.DIRECT_RESPONSE,
                reason="No registered tool capability matches the requested action.",
                confidence=0.65,
                candidate_tools=(),
                required_capabilities=(),
            )

        if len(candidates) == 1:
            return ExecutionDecision(
                mode=ExecutionMode.SINGLE_TOOL,
                reason="The request maps to one registered tool capability.",
                confidence=0.82,
                candidate_tools=candidates,
                required_capabilities=capabilities,
            )

        return ExecutionDecision(
            mode=ExecutionMode.TOOL_CHAIN,
            reason="The request contains multiple concrete registered tool capabilities.",
            confidence=0.78,
            candidate_tools=candidates,
            required_capabilities=capabilities,
        )

    def _detect_capabilities(
        self,
        text: str,
    ) -> tuple[_CapabilityMatch, ...]:
        matches: list[_CapabilityMatch] = []

        for capability, tool, patterns in _CAPABILITY_PATTERNS:
            positions = [
                match.start()
                for pattern in patterns
                for match in re.finditer(pattern, text)
            ]

            if positions and tool in self._supported_tool_intents:
                matches.append(
                    _CapabilityMatch(
                        capability=capability,
                        tool=tool,
                        position=min(positions),
                    )
                )

        matches.sort(key=lambda item: item.position)
        return tuple(_dedupe_matches(matches))

    def _registered_tools(
        self,
        matches: tuple[_CapabilityMatch, ...],
    ) -> tuple[str, ...]:
        tools: list[str] = []

        for match in matches:
            if match.tool not in tools and match.tool in self._supported_tool_intents:
                tools.append(match.tool)

        return tuple(tools)

    def _is_general_or_conversational(
        self,
        text: str,
    ) -> bool:
        if text in {"hola", "buenas", "buenos dias", "buenas tardes"}:
            return True

        if text.startswith(("hola ", "explicame ", "explica ", "que es ", "que son ")):
            return True

        return any(
            phrase in text
            for phrase in (
                "como estas",
                "clean architecture",
                "git y github",
            )
        )


_CAPABILITY_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "read_file",
        "file.read",
        (
            r"\blee(?:r)?\b.*\barchivo\b",
            r"\blee(?:r)?\b.*\b[\w.-]+\.(?:md|txt|py|json|csv)\b",
            r"\bmuestra\b.*\b[\w.-]+\.(?:md|txt|py|json|csv)\b",
        ),
    ),
    (
        "list_directory",
        "directory.list",
        (
            r"\blista\b.*\b(?:archivos|carpeta|directorio)\b",
            r"\blistar\b.*\b(?:archivos|carpeta|directorio)\b",
        ),
    ),
    (
        "write_file",
        "file.write",
        (
            r"\bescribe\b.*\b(?:archivo|\.txt|\.md|\.py|\.json)\b",
            r"\bcopia\b.*\b(?:contenido|en)\b.*\b[\w.-]+\.(?:md|txt|py|json|csv)\b",
            r"\bguarda\b.*\b[\w.-]+\.(?:md|txt|py|json|csv)\b",
        ),
    ),
    (
        "open_application",
        "desktop.application.open",
        (
            r"\babre\b.*\b(?:vs code|vscode|visual studio code|bloc de notas|notepad)\b",
            r"\babrir\b.*\b(?:vs code|vscode|visual studio code|bloc de notas|notepad)\b",
        ),
    ),
    (
        "open_file",
        "desktop.file.open",
        (
            r"\babre\b.*\b[\w.-]+\.(?:md|txt|py|json|csv)\b",
            r"\babrir\b.*\b[\w.-]+\.(?:md|txt|py|json|csv)\b",
        ),
    ),
    (
        "type_text",
        "desktop.text.type",
        (
            r"\bescribe\b.*\b(?:linea|texto)\b",
            r"\bteclea\b",
        ),
    ),
    (
        "press_hotkey",
        "desktop.hotkey.press",
        (
            r"\batajo\b",
            r"\bpulsa\b.*\b(?:ctrl|alt|shift)\b",
        ),
    ),
)


def _dedupe_matches(
    matches: list[_CapabilityMatch],
) -> tuple[_CapabilityMatch, ...]:
    seen_tools: set[str] = set()
    result: list[_CapabilityMatch] = []

    for match in matches:
        if match.tool in seen_tools:
            continue

        seen_tools.add(match.tool)
        result.append(match)

    return tuple(result)


def _normalize(
    text: str,
) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    without_punctuation = re.sub(r"[^\w\s./-]", " ", without_accents)
    return " ".join(without_punctuation.split())

"""Conservative conversational detection for explicit prolonged-autonomy orders.

Only explicit orders ("trabaja durante N minutos", "trabaja en segundo plano",
"continúa trabajando hasta terminar") start background autonomy. Normal
conversation is never converted into autonomous work automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

DEFAULT_MAX_DURATION_SECONDS = 900.0
DEFAULT_MAX_TASK_EXECUTIONS = 24
_MAX_ACCEPTED_DURATION_SECONDS = 4 * 60 * 60.0


@dataclass(frozen=True, slots=True)
class AutonomousGoalRequest:
    """An explicit user order to work autonomously on one objective."""

    objective: str
    max_duration_seconds: float
    max_task_executions: int = DEFAULT_MAX_TASK_EXECUTIONS
    duration_requested: bool = False


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )


def _normalize_loose(text: str) -> str:
    """Accent- and punctuation-free text, for status/cancel phrase matching."""
    normalized = re.sub(r"[^\w\s]", " ", _normalize(text))
    return " ".join(normalized.split())


def parse_requested_minutes(text: str) -> "int | None":
    normalized = _normalize(text)
    match = re.search(r"\b(?:durante\s+|por\s+)?(\d+)\s*minutos?\b", normalized)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(?:durante\s+|por\s+)?(\d+)\s*horas?\b", normalized)
    if match:
        return int(match.group(1)) * 60
    return None


_UNTIL_DONE = r"hasta\s+(?:terminar|acabar)(?:\s+o\s+(?:bloquearte|bloquear))?"
_AUTONOMY_PATTERNS = (
    re.compile(
        r"\btrabaj\w+\s+(?:durante\s+\d+\s*(?:minutos?|horas?)\s+)?"
        r"en\s+(?:segundo\s+plano|background)\b"
        r"(?:\s+(?:en|para|sobre)\s+)?[:\s]*(.*)$"
    ),
    re.compile(
        r"\btrabaj\w+\s+(?:durante|por)\s+\d+\s*(?:minutos?|horas?)\b"
        r"(?:\s+en\s+(?:este\s+objetivo|segundo\s+plano|background))?"
        r"\s*[:\-]?\s*(.*)$"
    ),
    re.compile(
        r"\bcontinu\w+\s+trabajando\s+(?:en\s+)?(?:este\s+)?objetivo\s+"
        + _UNTIL_DONE + r"\s*[:\-]?\s*(.*)$"
    ),
    re.compile(
        r"\bsigue\s+trabajando\s+(?:en\s+(?:este\s+)?objetivo\s+|"
        r"en\s+segundo\s+plano\s+)?" + _UNTIL_DONE + r"\s*[:\-]?\s*(.*)$"
    ),
)

# Explicit, conservative structured-plan entry: an explicit order to plan
# AND execute one objective in the background. Normal conversation never
# triggers this; the objective text after the ':' is parsed by the planner.
# The objective may span several typed lines, so the capture group must
# match any character, not just one line.
_STRUCTURED_PLAN_PATTERNS = (
    re.compile(
        r"\bplanific\w+\s+y\s+ejecut\w+\s+este\s+objetivo\s*[:\-]\s*([\s\S]+)$"
    ),
    re.compile(
        r"\btrabaj\w+\s+en\s+este\s+objetivo\s*:\s*([\s\S]+)$"
    ),
)

_UNTIL_DONE_PHRASE = re.compile(r"hasta\s+(?:terminar|acabar)")

_STATUS_PHRASES = (
    "estado",
    "como va el objetivo",
    "como va el trabajo",
    "como va el trabajo en segundo plano",
    "como va el trabajo autonomo",
    "estado del objetivo",
    "estado del trabajo",
    "estado del trabajo en segundo plano",
    "estado del trabajo autonomo",
    "estado del objetivo autonomo",
    "progreso del objetivo",
    "progreso del trabajo",
    "que tal va el objetivo",
    "que tal va el trabajo",
    "sigues trabajando",
    "estas trabajando",
)

_CANCEL_PHRASES = (
    "deten el trabajo",
    "detener el trabajo",
    "para el trabajo",
    "parar el trabajo",
    "deten el objetivo",
    "detener el objetivo",
    "cancela el objetivo",
    "cancelar el objetivo",
    "cancela el trabajo",
    "cancelar el trabajo",
)


def detect_autonomous_goal(prompt: str) -> "AutonomousGoalRequest | None":
    """Return a bounded autonomy request only for explicit orders.

    An empty objective means "continue the existing objective" and must be
    resolved against the orchestrator's active background goal.
    """
    normalized = _normalize(prompt)
    for pattern in _AUTONOMY_PATTERNS:
        match = pattern.search(normalized)
        if match is None:
            continue
        objective = (match.group(1) or "").strip(" :-.,")
        minutes = parse_requested_minutes(normalized)
        if minutes is not None:
            max_duration = min(minutes * 60.0, _MAX_ACCEPTED_DURATION_SECONDS)
            duration_requested = True
        else:
            max_duration = DEFAULT_MAX_DURATION_SECONDS
            duration_requested = False
        return AutonomousGoalRequest(
            objective=objective,
            max_duration_seconds=max_duration,
            max_task_executions=DEFAULT_MAX_TASK_EXECUTIONS,
            duration_requested=duration_requested,
        )
    return None


def detect_structured_plan_objective(prompt: str) -> "str | None":
    """Return the objective of an explicit 'plan and execute' order, if any."""
    normalized = _normalize(prompt)
    if _UNTIL_DONE_PHRASE.search(normalized):
        # "hasta terminar" keeps the existing prolonged-autonomy routing.
        return None
    for pattern in _STRUCTURED_PLAN_PATTERNS:
        match = pattern.search(normalized)
        if match is None:
            continue
        objective = (match.group(1) or "").strip(" :-.,")
        if not objective:
            return None
        original_tail = prompt.partition(":")[2].strip()
        if original_tail:
            # Recover the original casing lost by accent normalization.
            objective = original_tail
        return objective
    return None


def is_background_goal_status_query(prompt: str) -> bool:
    normalized = _normalize_loose(prompt)
    return any(
        normalized == phrase or normalized.startswith(phrase + " ")
        for phrase in _STATUS_PHRASES
    )


def is_background_goal_cancel_request(prompt: str) -> bool:
    normalized = _normalize_loose(prompt)
    return any(phrase in normalized for phrase in _CANCEL_PHRASES)

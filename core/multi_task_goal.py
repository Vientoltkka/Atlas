"""Conservative detection of multi-task objectives for the async scheduler.

Detection is intentionally narrow: only well-understood read →
(accumulate/transform) → write objectives are converted into scheduler
tasks. Every other prompt stays on the existing conversational flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

_WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"\b[A-Za-z]:[\\/]\S")

_QUESTION_START_PATTERN = re.compile(
    r"^(que|cual|cuales|como|cuando|donde|quien|puedes|podrias|quieres|"
    r"me puedes|dime si|es posible)\b"
)

_NEGATED_ACTION_PATTERN = re.compile(
    r"\bno\s+(?:leas|leer|guardes|guardar|escribas|escribir)\b"
)

_SOURCE_PATTERN = re.compile(
    r"\b(?:leeme|leer|lee|read)\s+"
    r"(?:el\s+|la\s+|los\s+|las\s+|the\s+|my\s+)?"
    r"(?:archivo\s+|fichero\s+|file\s+)?"
    r"(?P<path>[A-Za-z0-9_\-][\w.\-/\\]*\.\w{1,6})"
)

_TARGET_PATTERN = re.compile(
    r"\b(?:guarda(?:me|lo|la|nos|selo|sela)?|escribe(?:lo|la|me)?|"
    r"guardar|escribir|save|write)\s+"
    r"(?:el\s+|la\s+|los\s+|las\s+|the\s+|my\s+)?"
    r"(?:contenido\s+|resumen\s+|resultado\s+|texto\s+|informe\s+|"
    r"reporte\s+|data\s+)?"
    r"(?:en|in|to|into|dentro\s+de)\s+"
    r"(?:el\s+|la\s+|the\s+)?(?:archivo\s+|fichero\s+|file\s+)?"
    r"(?P<path>[A-Za-z0-9_\-][\w.\-/\\]*\.\w{1,6})"
)

_CONNECTOR_PATTERN = re.compile(
    r"(,|;|\by\b|\be\b|\bluego\b|\bdespues\b|\bposteriormente\b|"
    r"\btras\b|\bthen\b|\band\b)"
)

_SUMMARY_PATTERN = re.compile(r"resum|summar")


def normalize_prompt_text(text: str) -> str:
    """Lowercase and strip accents while keeping punctuation and paths."""
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return without_accents.lower()


@dataclass(frozen=True)
class MultiTaskGoal:
    """A detected multi-task objective with scheduler-ready task specs."""

    description: str
    tasks: list[dict[str, object]] = field(default_factory=list)


def _path_is_plausible(path: str) -> bool:
    return 1 < len(path) <= 120 and ":" not in path


def _restore_path_case(path: str, original: str) -> str:
    """Recover the original path casing lost by prompt normalization."""
    match = re.search(re.escape(path), original, re.IGNORECASE)
    return match.group(0) if match else path


def _tool_task(
    task_id: str,
    description: str,
    tool: str,
    arguments: dict[str, object],
    dependencies: tuple[str, ...] = (),
    *,
    content_task: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "tool",
        "tool": tool,
        "arguments": arguments,
    }
    if content_task is not None:
        payload["content_task"] = content_task
    return {
        "task_id": task_id,
        "description": description,
        "dependencies": list(dependencies),
        "requires_approval": False,
        "payload": payload,
    }


def detect_multi_task_goal(prompt: str) -> MultiTaskGoal | None:
    """Detect a conservative read → summarize? → write objective.

    Returns ``None`` for anything else so the prompt keeps using the
    existing conversational / single-tool / structured routing.
    """
    normalized = normalize_prompt_text(prompt)
    if not normalized.strip() or "?" in normalized:
        return None
    if _QUESTION_START_PATTERN.match(normalized):
        return None
    if _NEGATED_ACTION_PATTERN.search(normalized):
        return None
    if _WINDOWS_ABSOLUTE_PATH_PATTERN.search(normalized):
        # Windows absolute paths keep the existing supervised flows.
        return None

    source_matches = list(_SOURCE_PATTERN.finditer(normalized))
    target_match = _TARGET_PATTERN.search(normalized)
    if not source_matches or target_match is None:
        return None

    first_source = source_matches[0]
    if first_source.start() >= target_match.start():
        return None
    between = normalized[first_source.end() : target_match.start()]
    if not _CONNECTOR_PATTERN.search(between):
        return None

    source_path = _restore_path_case(first_source.group("path"), prompt)
    target_path = _restore_path_case(target_match.group("path"), prompt)
    if not _path_is_plausible(source_path) or not _path_is_plausible(target_path):
        return None
    if source_path.casefold() == target_path.casefold():
        return None

    tasks: list[dict[str, object]] = [
        _tool_task(
            "read_source",
            f"Leer {source_path}",
            "read_file",
            {"path": source_path},
        )
    ]
    content_task_id = "read_source"
    if _SUMMARY_PATTERN.search(normalized):
        tasks.append(
            {
                "task_id": "summarize_source",
                "description": "Resumir el contenido leído",
                "dependencies": ["read_source"],
                "requires_approval": False,
                "payload": {
                    "kind": "transform",
                    "instruction": (
                        "Resume el siguiente contenido de forma breve y fiel. "
                        "Responde solo con el resumen:\n\n{input}"
                    ),
                    "input_task": "read_source",
                },
            }
        )
        content_task_id = "summarize_source"

    tasks.append(
        _tool_task(
            "write_target",
            f"Guardar el resultado en {target_path}",
            "write_file",
            {"path": target_path},
            (content_task_id,),
            content_task=content_task_id,
        )
    )

    extra_reads = 0
    for match in source_matches[1:]:
        extra_path = _restore_path_case(match.group("path"), prompt)
        if (
            not _path_is_plausible(extra_path)
            or extra_path.casefold() == target_path.casefold()
            or extra_path.casefold() == source_path.casefold()
        ):
            continue
        extra_reads += 1
        tasks.append(
            _tool_task(
                f"read_extra_{extra_reads}",
                f"Leer {extra_path}",
                "read_file",
                {"path": extra_path},
            )
        )

    return MultiTaskGoal(description=prompt.strip(), tasks=tasks)

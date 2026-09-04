"""Conservative detection of multi-task objectives for the async scheduler.

Detection is intentionally narrow: only well-understood read →
(transform?) → write objectives plus fully independent safe read bundles
are converted into scheduler tasks. Every other prompt stays on the
existing conversational flow.
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
    r"\b(?:guarda(?:me|lo|la|los|las|nos|selo|sela)?|escribe(?:lo|la|me)?|"
    r"guardar|escribir|save|write)\s+"
    r"(?:el\s+|la\s+|los\s+|las\s+|the\s+|my\s+)?"
    r"(?:contenido\s+|resumen\s+|resultado\s+|texto\s+|informe\s+|"
    r"reporte\s+|data\s+|diferencias?\s+|puntos\s+clave\s+|sintesis\s+)?"
    r"(?:en|in|to|into|dentro\s+de)\s+"
    r"(?:el\s+|la\s+|the\s+)?(?:archivo\s+|fichero\s+|file\s+)?"
    r"(?P<path>[A-Za-z0-9_\-][\w.\-/\\]*\.\w{1,6})"
)

_BARE_PATH_PATTERN = re.compile(
    r"(?<![\w.\-/\\])"
    r"(?P<path>[A-Za-z0-9_\-][\w.\-/\\]*\.\w{1,6})"
    r"(?![\w.\-/\\])"
)

_CONNECTOR_PATTERN = re.compile(
    r"(,|;|\by\b|\be\b|\bluego\b|\bdespues\b|\bposteriormente\b|"
    r"\btras\b|\bthen\b|\band\b)"
)

_SUMMARY_PATTERN = re.compile(r"resum|summar")

# Search sources: "busca (información) (sobre) X" with the query captured up to
# the first transform or write verb so the rest of the goal parses as usual.
_SEARCH_PATTERN = re.compile(
    r"\bbusc(?:a|ar)\b\s+"
    r"(?:en\s+(?:internet|la\s+web|la\s+red)\s+)?"
    r"(?:informacion\s+|info\s+|noticias\s+)?"
    r"(?:sobre\s+|acerca\s+de\s+|de\s+|del\s+)?"
    r"(?P<query>.+?)"
    r"(?=\s*(?:\bresum|\bsummar|\bcompar|\borden|\bcorrig|\bextrae"
    r"|\bguarda|\bguardar|\bescribe|\bescribir|\bsave\b|\bwrite\b"
    r"|\band\b|\bthen\b|$))"
)

# File-like tokens inside a search query mean the prompt is really about local
# files, not the web; those stay on the existing read flows.
_FILE_EXTENSION_PATTERN = re.compile(
    r"\.(?:txt|md|json|csv|log|py|docx?|xlsx?|pdf|cfg|ini|ya?ml)\b"
)

_QUERY_SPLIT_PATTERN = re.compile(r"\s*(?:,|;|\by\b|\be\b)\s*")

# Deterministic transform verbs, checked in order; the first match wins so a
# prompt never gets two competing transform interpretations.
_TRANSFORMATION_PATTERNS: tuple[tuple[re.Pattern[str], str, str, str], ...] = (
    (
        re.compile(r"\bcompara(?:r|ndo|los|las|lo|la)?\b|\bdiferencias?\b"),
        "transform_content",
        "Compara los siguientes contenidos y describe sus diferencias de forma "
        "clara. Responde solo con el resultado:\n\n{input}",
        "Comparar el contenido leído",
    ),
    (
        re.compile(r"\bordena(?:r|ndo|los|las|lo|la)?\b|\bsorted?\b"),
        "transform_content",
        "Ordena el siguiente texto de forma coherente y legible. Responde solo "
        "con el texto ordenado:\n\n{input}",
        "Ordenar el contenido leído",
    ),
    (
        re.compile(r"\bcorrige(?:r|ndo|los|las|lo|la)?\b|\bcorregir\b"),
        "transform_content",
        "Corrige la redacción del siguiente texto sin cambiar su significado. "
        "Responde solo con el texto corregido:\n\n{input}",
        "Corregir la redacción del contenido leído",
    ),
    (
        re.compile(r"\bextrae(?:r|ndo|los|las|lo|la)?\b|\bpuntos?\s+clave\b"),
        "transform_content",
        "Extrae los puntos clave del siguiente contenido. Responde solo con "
        "los puntos clave:\n\n{input}",
        "Extraer los puntos clave del contenido leído",
    ),
    (
        _SUMMARY_PATTERN,
        "summarize_source",
        "Resume el siguiente contenido de forma breve y fiel. "
        "Responde solo con el resumen:\n\n{input}",
        "Resumir el contenido leído",
    ),
)


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


def _spans_overlap(
    span: tuple[int, int],
    spans: list[tuple[int, int]],
) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in spans)


def _collect_reads(
    normalized: str,
    prompt: str,
    source_matches: list[re.Match[str]],
    target_match: re.Match[str] | None,
) -> list[tuple[re.Match[str], str]]:
    """Gather verb-driven reads plus standalone path mentions, in order."""
    excluded: list[tuple[int, int]] = [
        (match.start(), match.end()) for match in source_matches
    ]
    if target_match is not None:
        excluded.append((target_match.start(), target_match.end()))

    candidates: list[tuple[re.Match[str], str]] = [
        (match, _restore_path_case(match.group("path"), prompt))
        for match in source_matches
    ]
    for match in _BARE_PATH_PATTERN.finditer(normalized):
        if _spans_overlap((match.start(), match.end()), excluded):
            continue
        candidates.append((match, _restore_path_case(match.group("path"), prompt)))

    target_path = (
        _restore_path_case(target_match.group("path"), prompt)
        if target_match is not None
        else None
    )
    reads: list[tuple[re.Match[str], str]] = []
    seen: set[str] = set()
    for match, path in sorted(candidates, key=lambda item: item[0].start()):
        if not _path_is_plausible(path):
            continue
        folded = path.casefold()
        if folded in seen:
            continue
        if target_path is not None and folded == target_path.casefold():
            continue
        seen.add(folded)
        reads.append((match, path))
    return reads


def _build_transform_task(
    normalized: str,
    content_tasks: list[tuple[int, str]],
) -> dict[str, object] | None:
    """Build one transform task over the content sources preceding the verb."""
    if not content_tasks:
        return None
    for pattern, task_id, instruction, description in _TRANSFORMATION_PATTERNS:
        match = pattern.search(normalized)
        if match is None:
            continue
        input_ids = [
            content_id for end, content_id in content_tasks if end <= match.start()
        ]
        if not input_ids:
            input_ids = [content_tasks[0][1]]
        payload: dict[str, object] = {
            "kind": "transform",
            "instruction": instruction,
        }
        if len(input_ids) == 1:
            payload["input_task"] = input_ids[0]
        else:
            payload["input_tasks"] = list(input_ids)
        return {
            "task_id": task_id,
            "description": description,
            "dependencies": list(input_ids),
            "requires_approval": False,
            "payload": payload,
        }
    return None


def _split_search_queries(raw_query: str, prompt: str) -> list[str]:
    """Split the captured search text into one or several clean queries."""
    cleaned = " ".join(raw_query.strip().strip(",;").split())
    if not cleaned or len(cleaned) > 120:
        return []
    queries: list[str] = []
    for part in _QUERY_SPLIT_PATTERN.split(cleaned):
        if _FILE_EXTENSION_PATTERN.search(part):
            return []
        if not part:
            continue
        query = _restore_path_case(part, prompt).strip()
        if not query or ":" in query or len(query) > 120:
            return []
        queries.append(query)
    return queries


def _build_search_goal(
    prompt: str,
    normalized: str,
    search_match: re.Match[str],
) -> MultiTaskGoal | None:
    """Build a web_search → transform → write goal from one search phrase.

    Conservative by design: prompts without an explicit write target, without
    an explicit transform verb, or that also mention local file reads stay on
    the existing conversational / single-tool routing.
    """
    if _SOURCE_PATTERN.search(normalized):
        return None
    target_match = _TARGET_PATTERN.search(normalized)
    if target_match is None:
        return None
    if not _CONNECTOR_PATTERN.search(
        normalized[search_match.end() : target_match.start()]
    ):
        return None
    queries = _split_search_queries(search_match.group("query"), prompt)
    if not queries:
        return None

    tasks: list[dict[str, object]] = []
    content_tasks: list[tuple[int, str]] = []
    for index, query in enumerate(queries):
        task_id = "web_search" if index == 0 else f"web_search_extra_{index}"
        tasks.append(
            _tool_task(
                task_id,
                f"Buscar {query}",
                "web_search",
                {"query": query},
            )
        )
        content_tasks.append((search_match.end(), task_id))

    transform = _build_transform_task(normalized, content_tasks)
    if transform is None:
        return None
    tasks.append(transform)

    content_task_id = str(transform["task_id"])
    target_path = _restore_path_case(target_match.group("path"), prompt)
    if not _path_is_plausible(target_path):
        return None
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
    return MultiTaskGoal(description=prompt.strip(), tasks=tasks)


def detect_multi_task_goal(prompt: str) -> MultiTaskGoal | None:
    """Detect a conservative read → transform? → write objective.

    Also detects bundles of two or more fully independent safe reads with no
    write target. Returns ``None`` for anything else so the prompt keeps
    using the existing conversational / single-tool / structured routing.
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

    search_match = _SEARCH_PATTERN.search(normalized)
    if search_match is not None:
        return _build_search_goal(prompt, normalized, search_match)

    source_matches = list(_SOURCE_PATTERN.finditer(normalized))
    if not source_matches:
        return None
    target_match = _TARGET_PATTERN.search(normalized)

    reads = _collect_reads(normalized, prompt, source_matches, target_match)
    if not reads:
        return None
    first_read = reads[0][0]

    tasks: list[dict[str, object]] = []
    content_tasks: list[tuple[int, str]] = []
    for index, (match, path) in enumerate(reads):
        task_id = "read_source" if index == 0 else f"read_extra_{index}"
        tasks.append(
            _tool_task(task_id, f"Leer {path}", "read_file", {"path": path})
        )
        content_tasks.append((match.end(), task_id))

    transform = _build_transform_task(normalized, content_tasks)

    if target_match is None:
        # Independent safe reads only: a transform without an output sink or a
        # single read stays on the existing conversational flow.
        if transform is not None or len(tasks) < 2:
            return None
        if not _CONNECTOR_PATTERN.search(
            normalized[first_read.end() : reads[-1][0].start()]
        ):
            return None
        return MultiTaskGoal(description=prompt.strip(), tasks=tasks)

    target_path = _restore_path_case(target_match.group("path"), prompt)
    if not _path_is_plausible(target_path):
        return None
    if first_read.start() >= target_match.start():
        return None
    if not _CONNECTOR_PATTERN.search(
        normalized[first_read.end() : target_match.start()]
    ):
        return None
    if any(path.casefold() == target_path.casefold() for _, path in reads):
        return None

    content_task_id = (
        str(transform["task_id"]) if transform is not None else content_tasks[0][1]
    )
    if transform is not None:
        tasks.append(transform)
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

    return MultiTaskGoal(description=prompt.strip(), tasks=tasks)

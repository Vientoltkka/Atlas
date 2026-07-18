"""Multi-turn clarification helpers for execution conversation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from tools.execution_coordinator import (
    ExecutionCoordinationResult,
    ExecutionCoordinationStatus,
)
from tools.execution_decision import ExecutionDecision, ExecutionMode
from tools.tool_chain_proposal_builder import StructuredToolChainProposal
from tools.tool_proposal_builder import StructuredToolProposal


_CANCEL_CLARIFICATION_RESPONSES = {
    "cancelar",
    "cancela",
    "cancelalo",
    "cancelala",
    "olvidalo",
    "olvidalo.",
    "olvídalo",
    "salir de esta operacion",
    "salir de esta operación",
}


@dataclass(frozen=True, slots=True)
class PendingClarification:
    """One incomplete or ambiguous operation waiting for user-provided fields."""

    original_text: str
    mode: ExecutionMode
    proposal: StructuredToolProposal | StructuredToolChainProposal
    candidate_tools: tuple[str, ...]
    missing_information: tuple[str, ...]
    ambiguous_information: tuple[str, ...]
    requested_fields: tuple[str, ...]


class ClarificationResolver:
    """Resolve one clarification answer against one pending operation."""

    def resolve(
        self,
        pending: PendingClarification,
        response: str,
    ) -> str | None:
        """Return a rebuilt request, or None when the answer is unusable."""
        if pending.mode is ExecutionMode.TOOL_CHAIN:
            return self._resolve_chain(pending, response)

        if isinstance(pending.proposal, StructuredToolProposal):
            return self._resolve_single(pending.proposal, response)

        return None

    def looks_like_new_order(
        self,
        response: str,
        pending: PendingClarification,
    ) -> bool:
        """Return whether the answer is likely a separate command."""
        normalized = _normalize(response)
        if not normalized:
            return False

        if pending.mode is ExecutionMode.TOOL_CHAIN and " y " in f" {normalized} ":
            return False

        return bool(
            re.search(
                r"\b(?:borra|elimina|abre|explicame|explica|corrige|renombra)\b",
                normalized,
            )
        )

    def _resolve_single(
        self,
        proposal: StructuredToolProposal,
        response: str,
    ) -> str | None:
        tool_name = proposal.tool_name
        arguments = dict(proposal.arguments)
        requested = merge_fields(proposal.missing_arguments, proposal.ambiguous_arguments)

        if tool_name == "file.read":
            path = extract_path_answer(response)
            if path is None:
                return None
            return f"Lee el archivo {path}"

        if tool_name == "directory.list":
            path = extract_directory_answer(response)
            if path is None:
                return None
            return f"Lista la carpeta {path}"

        if tool_name == "file.write":
            path = arguments.get("path")
            content = arguments.get("content")

            if "path" in requested:
                path = extract_path_answer(response)

            if "content" in requested:
                content = extract_content_answer(response, str(path) if path else None)

            if path is not None and content is None:
                return f"Escribe en {path}"

            if path is None or content is None:
                return None

            return f"Escribe {content} en {path}"

        if tool_name == "desktop.hotkey.press":
            keys = arguments.get("keys")
            title = clean_answer(response)
            if not keys or not title:
                return None

            key_text = "+".join(str(key) for key in keys)
            return f"Pulsa {key_text} en {title}"

        return None

    def _resolve_chain(
        self,
        pending: PendingClarification,
        response: str,
    ) -> str | None:
        if contains_tool_phrase(response):
            return response

        if not isinstance(pending.proposal, StructuredToolChainProposal):
            return None

        text = pending.original_text
        for field in pending.requested_fields:
            if field.endswith(".path"):
                path = extract_path_answer(response) or extract_directory_answer(response)
                if path is None:
                    return None

                if field.startswith("read."):
                    text = text.replace("este archivo", path)
                    text = text.replace("este fichero", path)
                    text = text.replace("otro archivo", path)
                    text = text.replace("otro", path)
                    continue

                if field.startswith("write."):
                    text = text.replace("otro archivo", path)
                    text = text.replace("otro fichero", path)
                    text = text.replace("otro", path)
                    if path not in text:
                        text = f"{text} en {path}"

        return text


class ClarificationQuestionPresenter:
    """Create natural questions for pending clarification fields."""

    def present(
        self,
        pending: PendingClarification | None,
    ) -> str:
        if pending is None:
            return "Necesito mas informacion para continuar."

        fields = pending.requested_fields
        proposal = pending.proposal

        if isinstance(proposal, StructuredToolProposal):
            return self._present_single(proposal.tool_name, fields)

        if isinstance(proposal, StructuredToolChainProposal):
            return self._present_chain(fields)

        return generic_clarification_question(fields)

    def _present_single(
        self,
        tool_name: str | None,
        fields: tuple[str, ...],
    ) -> str:
        field_set = set(fields)

        if tool_name == "file.read" and "path" in field_set:
            return "Que archivo quieres leer?"

        if tool_name == "directory.list" and "path" in field_set:
            return "Que carpeta quieres listar?"

        if tool_name == "file.write":
            if {"path", "content"}.issubset(field_set):
                return "Que contenido quieres escribir y en que archivo?"
            if "path" in field_set:
                return "En que archivo quieres escribir?"
            if "content" in field_set:
                return "Que contenido quieres escribir?"

        if tool_name == "desktop.hotkey.press" and "window_title" in field_set:
            return "En que ventana quieres pulsar el atajo?"

        return generic_clarification_question(fields)

    def _present_chain(
        self,
        fields: tuple[str, ...],
    ) -> str:
        field_set = set(fields)

        if "read.path" in field_set and "write.path" in field_set:
            return "Que archivo quieres leer y en que archivo quieres guardarlo?"

        if "read.path" in field_set:
            return "Que archivo quieres leer?"

        if "write.path" in field_set:
            return "En que archivo quieres guardar el resultado?"

        return generic_clarification_question(fields)


def synthetic_result(
    status: ExecutionCoordinationStatus,
    mode: ExecutionMode,
    message: str,
    proposal: StructuredToolProposal | StructuredToolChainProposal | None,
    *,
    candidate_tools: tuple[str, ...] = (),
    missing_information: tuple[str, ...] = (),
    ambiguous_information: tuple[str, ...] = (),
) -> ExecutionCoordinationResult:
    """Build an inert coordination result for session-control responses."""
    return ExecutionCoordinationResult(
        status=status,
        mode=mode,
        decision=ExecutionDecision(
            mode=mode,
            reason=message,
            confidence=1.0,
            candidate_tools=candidate_tools,
        ),
        proposal=proposal,
        execution_result=None,
        message=message,
        missing_information=missing_information,
        ambiguous_information=ambiguous_information,
        executed=False,
    )


def merge_fields(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    """Merge field names without changing their first-seen order."""
    merged: list[str] = []
    for field in left + right:
        if field not in merged:
            merged.append(field)
    return tuple(merged)


def is_cancel_clarification(
    text: str,
) -> bool:
    """Return whether text cancels a pending clarification."""
    return _normalize(text) in {_normalize(item) for item in _CANCEL_CLARIFICATION_RESPONSES}


def contains_tool_phrase(
    text: str,
) -> bool:
    """Return whether text resembles a complete supported tool request."""
    return bool(
        re.search(
            r"\b(?:lee|leer|lista|listar|escribe|guarda|guardalo|guardala|copia|pulsa)\b",
            _normalize(text),
        )
    )


def extract_path_answer(
    text: str,
) -> str | None:
    """Extract a file path from a clarification answer."""
    cleaned = clean_answer(text)
    after_in = re.search(
        r"\ben\s+(?P<path>(?:[A-Za-z]:[\\/])?(?:[\w .-]+[\\/])*[\w .-]+\.(?:md|txt|py|json|csv|yaml|yml|toml))\b",
        text,
        flags=re.IGNORECASE,
    )
    if after_in:
        return after_in.group("path").strip()

    match = re.search(
        r"(?P<path>(?:[A-Za-z]:[\\/])?(?:[\w .-]+[\\/])*[\w .-]+\.(?:md|txt|py|json|csv|yaml|yml|toml))\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group("path").strip()

    return None


def extract_directory_answer(
    text: str,
) -> str | None:
    """Extract a directory path from a clarification answer."""
    cleaned = clean_answer(text)
    match = re.search(
        r"\b(?:carpeta|directorio|ruta)\s+(?P<path>(?:[A-Za-z]:[\\/])?[\w .\\/:-]+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group("path").strip()

    if _looks_like_plain_directory(cleaned):
        return cleaned

    return None


def extract_content_answer(
    text: str,
    path: str | None,
) -> str | None:
    """Extract write content from a clarification answer."""
    cleaned = clean_answer(text)

    if path and path in cleaned:
        content = cleaned.replace(path, "").strip(" ,.")
        content = re.sub(r"\b(?:en|archivo|fichero)\b", "", content, flags=re.IGNORECASE)
        content = " ".join(content.split())
        return content or None

    match = re.search(
        r"(?P<content>.+?)\s+\ben\s+(?:[A-Za-z]:[\\/])?(?:[\w .-]+[\\/])*[\w .-]+\.(?:md|txt|py|json|csv|yaml|yml|toml)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        content = match.group("content").strip(" ,.")
        return content or None

    if extract_path_answer(cleaned) is not None:
        return None

    if not re.search(r"[\w]", cleaned):
        return None

    return cleaned or None


def clean_answer(
    text: str,
) -> str:
    """Trim wrapper words from a short clarification answer."""
    cleaned = text.strip().strip("\"'")
    cleaned = re.sub(
        r"^\s*(?:en|archivo|fichero|carpeta|directorio)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def generic_clarification_question(
    fields: tuple[str, ...],
) -> str:
    """Return a generic clarification question."""
    if not fields:
        return "Necesito mas informacion para continuar."

    return "Necesito que aclares: " + ", ".join(fields) + "."


def _looks_like_plain_directory(
    text: str,
) -> bool:
    if not text or re.search(r"\s", text):
        return False

    return bool(re.fullmatch(r"(?:[A-Za-z]:[\\/])?[\w./\\:-]+", text))


def _normalize(
    text: str,
) -> str:
    return " ".join(text.strip().lower().split())

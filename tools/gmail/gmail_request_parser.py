"""Extract structured Gmail arguments from natural-language requests."""

from __future__ import annotations

from typing import Any
import re

_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def extract_gmail_arguments(source_text: str, intent_action: str) -> dict[str, Any]:
    """Extract schema arguments for one Gmail tool intent."""
    if intent_action == "gmail.messages.list":
        return _extract_list_arguments(source_text)
    if intent_action == "gmail.messages.read":
        return _extract_read_arguments(source_text)
    if intent_action == "gmail.messages.send":
        return _extract_send_arguments(source_text)
    return {}


def _extract_list_arguments(source_text: str) -> dict[str, Any]:
    normalized = _normalize(source_text)
    match = re.search(r"\b(?:ultimos?|primeros?)\s+(\d{1,2})\b", normalized)
    if match:
        try:
            return {"max_results": int(match.group(1))}
        except ValueError:
            return {}
    return {}


def _extract_read_arguments(source_text: str) -> dict[str, Any]:
    normalized = _normalize(source_text)

    id_match = re.search(r"\b(?:id|uid)\s*:?\s*(\d{1,15})\b", normalized)
    if id_match:
        return {"message_id": id_match.group(1)}

    address_match = _EMAIL_PATTERN.search(normalized)
    if address_match:
        return {"sender": address_match.group(0)}

    sender_match = re.search(
        r"\b(?:correos?|emails?|mensajes?)\s+de\s+(?P<sender>[\w .-]{2,60})$",
        normalized,
    )
    if sender_match:
        sender = sender_match.group("sender").strip()
        if sender and sender not in {"correo", "email", "mensaje"}:
            return {"sender": sender}

    return {}


def _extract_send_arguments(source_text: str) -> dict[str, Any]:
    normalized = _normalize(source_text)
    arguments: dict[str, Any] = {}

    address_match = _EMAIL_PATTERN.search(normalized)
    if address_match:
        arguments["to"] = address_match.group(0)

    quoted = re.findall(r"[\"'“”‘’](?P<value>[^\"'“”‘’]+)[\"'“”‘’]", source_text)

    body_split = re.split(
        r"\s+y\s+(?:el\s+)?cuerpo\b|\s+cuerpo\s+|\s+diciendo\s+|\s+que\s+diga\s+",
        source_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    subject_segment = body_split[0]
    body_segment = body_split[1] if len(body_split) > 1 else None

    subject_match = re.search(
        r"\basunto\s*(?::|que diga|de|el)?\s*(?P<subject>[^.,;]+)",
        subject_segment,
        flags=re.IGNORECASE,
    )
    if subject_match:
        arguments["subject"] = subject_match.group("subject").strip(" .,-")
    elif quoted:
        arguments["subject"] = quoted[0].strip()

    if body_segment:
        body = body_segment.strip().strip(" .,-")
        if body:
            arguments["body"] = body
    elif quoted and len(quoted) > 1:
        arguments["body"] = quoted[-1].strip()

    return {name: value for name, value in arguments.items() if value}


def _normalize(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return normalized

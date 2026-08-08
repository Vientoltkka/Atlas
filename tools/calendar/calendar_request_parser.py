"""Extract and validate normalized arguments for calendar event searches."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from tools.calendar.calendar_list_events_tool import MAX_RESULTS_LIMIT


_RFC3339_TEXT = (
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
)
_RFC3339_PATTERN = re.compile(rf"^{_RFC3339_TEXT}$")
_LABELED_TIMESTAMP_PATTERN = re.compile(
    rf"\b(time_min|time_max)\s*[:=]\s*[\"']?({_RFC3339_TEXT})[\"']?",
    re.IGNORECASE,
)
_TIMESTAMP_PATTERN = re.compile(_RFC3339_TEXT, re.IGNORECASE)
_MAX_RESULTS_PATTERN = re.compile(
    r"\b(?:max_results|m[aá]ximo|m[aá]ximo\s+de|l[ií]mite|limit)"
    r"\s*[:=]?\s*(\d+)\b",
    re.IGNORECASE,
)


def extract_calendar_arguments(source_text: str) -> dict[str, Any]:
    """Extract only the supported calendar search arguments from text."""
    arguments: dict[str, Any] = {
        name.casefold(): value
        for name, value in _LABELED_TIMESTAMP_PATTERN.findall(source_text)
    }

    timestamps = _TIMESTAMP_PATTERN.findall(source_text)
    if "time_min" not in arguments and timestamps:
        arguments["time_min"] = timestamps[0]
    if "time_max" not in arguments and len(timestamps) > 1:
        arguments["time_max"] = timestamps[1]

    max_results = _MAX_RESULTS_PATTERN.search(source_text)
    if max_results is not None:
        arguments["max_results"] = int(max_results.group(1))

    return arguments


def require_rfc3339_timestamp(value: Any) -> None:
    """Require one timezone-aware RFC3339 timestamp."""
    if not isinstance(value, str) or _RFC3339_PATTERN.fullmatch(value) is None:
        raise ValueError("value must be an RFC3339 timestamp with timezone")
    normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("value must be an RFC3339 timestamp with timezone")


def require_calendar_max_results(value: Any) -> None:
    """Keep calendar result limits within the adapter's supported range."""
    if type(value) is not int or not 1 <= value <= MAX_RESULTS_LIMIT:
        raise ValueError(f"value must be between 1 and {MAX_RESULTS_LIMIT}")

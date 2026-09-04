"""Extract and validate normalized arguments for calendar event searches."""

from __future__ import annotations

from datetime import datetime, time, timedelta
import re
from typing import Any
import unicodedata

from tools.calendar.calendar_list_events_tool import MAX_RESULTS_LIMIT

DEFAULT_EVENT_DURATION_MINUTES = 60

_WEEKDAY_NAMES = {
    "lunes": 0,
    "martes": 1,
    "miercoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sabado": 5,
    "domingo": 6,
}


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


def extract_calendar_arguments(
    source_text: str,
    *,
    current_time: datetime | None = None,
) -> dict[str, Any]:
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

    if "time_min" not in arguments and "time_max" not in arguments:
        natural_range = _natural_calendar_range(source_text, current_time)
        if natural_range is not None:
            arguments.update(natural_range)

    return arguments


def _natural_calendar_range(
    source_text: str,
    current_time: datetime | None,
) -> dict[str, str] | None:
    normalized = _normalize(source_text)
    expression = next(
        (
            value
            for value in ("esta semana", "manana", "hoy")
            if re.search(rf"\b{value}\b", normalized)
        ),
        None,
    )
    if expression is None:
        expression = "hoy"

    now = current_time or datetime.now().astimezone()
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("current_time must include timezone information")

    today_start = datetime.combine(now.date(), time.min, tzinfo=now.tzinfo)
    if expression == "hoy":
        range_start = today_start
        range_end = today_start + timedelta(days=1)
    elif expression == "manana":
        range_start = today_start + timedelta(days=1)
        range_end = range_start + timedelta(days=1)
    else:
        range_start = today_start - timedelta(days=today_start.weekday())
        range_end = range_start + timedelta(days=7)

    return {
        "time_min": range_start.isoformat(timespec="seconds"),
        "time_max": range_end.isoformat(timespec="seconds"),
    }


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_accents.split())


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


def extract_calendar_create_arguments(
    source_text: str,
    *,
    current_time: datetime | None = None,
) -> dict[str, Any]:
    """Extract title, start and end for one event creation request."""
    now = current_time or datetime.now().astimezone()
    normalized = _normalize(source_text)

    title = _extract_event_title(source_text, normalized)
    start = _extract_event_start(normalized, now)
    if start is None:
        return {}

    end = _extract_event_end(normalized, now, start)
    return {
        "title": title,
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": end.isoformat(timespec="seconds"),
    }


def _extract_event_title(source_text: str, normalized: str) -> str | None:
    quoted = re.search(
        r"[\"'“”‘’](?P<value>[^\"'“”‘’]+)[\"'“”‘’]",
        source_text,
    )
    if quoted:
        return quoted.group("value").strip()

    create_match = re.search(
        r"\b(?:crea|crear|apunta|apuntar|agenda|agendar|programa|programar)\b"
        r"\s+(?P<rest>.+)$",
        normalized,
    )
    if create_match:
        rest = create_match.group("rest")
        date_marker = re.search(
            r"\b(?:pasado\s+manana|manana|hoy|lunes|martes|miercoles|jueves"
            r"|viernes|sabado|domingo|a\s+las|de\s+las)\b",
            rest,
        )
        title = rest[: date_marker.start()] if date_marker else rest
        title = re.sub(
            r"^(?:una?\s+|el\s+|la\s+|los\s+|las\s+|un\s+)",
            "",
            title.strip(),
        )
        title = re.sub(
            r"\s+(?:el|la|los|las|un|una|para|este|esta|proximo|proxima|siguiente)$",
            "",
            title.strip(),
        )
        title = title.strip()
        if title:
            return title[:1].upper() + title[1:]

    return None


def _extract_event_start(
    normalized: str,
    now: datetime,
) -> datetime | None:
    hour_match = re.search(r"\ba\s+las?\s+(\d{1,2})(?::(\d{2}))?", normalized)
    if hour_match is None:
        return None
    hour = int(hour_match.group(1))
    minute = int(hour_match.group(2) or 0)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None

    day_offset = _extract_event_day_offset(normalized, now)
    start = datetime.combine(
        now.date() + timedelta(days=day_offset),
        time(hour=hour, minute=minute),
        tzinfo=now.tzinfo,
    )
    return start


def _extract_event_day_offset(normalized: str, now: datetime) -> int:
    if re.search(r"\bpasado\s+manana\b", normalized):
        return 2
    if re.search(r"\bmanana\b", normalized):
        return 1
    if re.search(r"\bhoy\b", normalized):
        return 0
    for name, weekday in _WEEKDAY_NAMES.items():
        if re.search(rf"\b(?:el\s+|proximo\s+|siguiente\s+)?{name}\b", normalized):
            offset = (weekday - now.weekday()) % 7
            if offset == 0:
                offset = 7
            return offset
    return 0


def _extract_event_end(
    normalized: str,
    now: datetime,
    start: datetime,
) -> datetime:
    end_match = re.search(r"\bhasta\s+las?\s+(\d{1,2})(?::(\d{2}))?", normalized)
    if end_match:
        hour = int(end_match.group(1))
        minute = int(end_match.group(2) or 0)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            end = datetime.combine(
                start.date(),
                time(hour=hour, minute=minute),
                tzinfo=now.tzinfo,
            )
            if end <= start:
                end += timedelta(days=1)
            return end
    return start + timedelta(minutes=DEFAULT_EVENT_DURATION_MINUTES)

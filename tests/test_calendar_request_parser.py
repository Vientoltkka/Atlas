from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tools.calendar.calendar_request_parser import extract_calendar_arguments


_CURRENT_TIME = datetime(2026, 8, 12, 14, 30, tzinfo=timezone(timedelta(hours=1)))


@pytest.mark.parametrize(
    ("prompt", "time_min", "time_max"),
    [
        ("Qué tengo hoy", "2026-08-12T00:00:00+01:00", "2026-08-13T00:00:00+01:00"),
        ("Qué tengo mañana", "2026-08-13T00:00:00+01:00", "2026-08-14T00:00:00+01:00"),
        ("Qué tengo esta semana", "2026-08-10T00:00:00+01:00", "2026-08-17T00:00:00+01:00"),
    ],
)
def test_natural_calendar_ranges_are_local_rfc3339(
    prompt: str,
    time_min: str,
    time_max: str,
) -> None:
    arguments = extract_calendar_arguments(prompt, current_time=_CURRENT_TIME)

    assert arguments["time_min"] == time_min
    assert arguments["time_max"] == time_max
    assert datetime.fromisoformat(arguments["time_max"]) > datetime.fromisoformat(
        arguments["time_min"]
    )

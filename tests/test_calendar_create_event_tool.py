from __future__ import annotations

from datetime import datetime

import pytest

from bootstrap.bootstrap import Bootstrap
from tools.calendar.calendar_create_event_tool import (
    CodexAppServerCalendarCreateAdapter,
    normalize_calendar_create_result,
    validate_calendar_event_request,
)
from tools.calendar.calendar_request_parser import (
    extract_calendar_create_arguments,
)
from tools.calendar.calendar_list_events_tool import (
    CodexAppServerResponseError,
)


def _reference_now() -> datetime:
    return datetime(2026, 9, 4, 12, 0).astimezone()


def test_calendar_event_request_validation() -> None:
    title, start, end = validate_calendar_event_request(
        "Reunion",
        "2026-09-05T10:00:00+01:00",
        "2026-09-05T11:00:00+01:00",
    )
    assert (title, start, end) == (
        "Reunion",
        "2026-09-05T10:00:00+01:00",
        "2026-09-05T11:00:00+01:00",
    )

    with pytest.raises(ValueError):
        validate_calendar_event_request("", "2026-09-05T10:00:00+01:00", "2026-09-05T11:00:00+01:00")

    with pytest.raises(ValueError):
        validate_calendar_event_request(
            "Reunion",
            "2026-09-05T11:00:00+01:00",
            "2026-09-05T10:00:00+01:00",
        )

    with pytest.raises(ValueError):
        validate_calendar_event_request(
            "Reunion",
            "2026-09-05T10:00:00",
            "2026-09-05T11:00:00+01:00",
        )


def test_calendar_create_argument_extraction() -> None:
    now = _reference_now()
    arguments = extract_calendar_create_arguments(
        "Crea una reunion manana a las 10",
        current_time=now,
    )
    assert arguments["title"] == "Reunion"
    assert arguments["start_time"].startswith("2026-09-05T10:00:00")

    arguments = extract_calendar_create_arguments(
        "Apunta entrenamiento el viernes a las 18",
        current_time=now,
    )
    assert arguments["title"] == "Entrenamiento"
    assert arguments["start_time"].startswith("2026-09-11T18:00:00")

    arguments = extract_calendar_create_arguments(
        'Crea "Review Atlas" el lunes a las 9:30 hasta las 11',
        current_time=now,
    )
    assert arguments["title"] == "Review Atlas"
    assert arguments["start_time"].startswith("2026-09-07T09:30:00")
    assert arguments["end_time"].startswith("2026-09-07T11:00:00")

    arguments = extract_calendar_create_arguments(
        "Crea una reunion",
        current_time=now,
    )
    assert "start_time" not in arguments


def test_calendar_create_tool_requires_confirmation_and_permission() -> None:
    from tools.calendar.calendar_create_event_tool import CalendarCreateEventTool

    tool = CalendarCreateEventTool()
    assert tool.requires_confirmation is True
    assert tool.required_permissions == ("calendar.write",)


def test_normalize_calendar_create_result_and_errors() -> None:
    normalized = normalize_calendar_create_result(
        {
            "structuredContent": {
                "id": "evt_1",
                "summary": "Reunion",
                "start": "2026-09-05T10:00:00+01:00",
                "end": "2026-09-05T11:00:00+01:00",
            }
        }
    )
    assert "evt_1" in normalized["summary"]
    assert "Reunion" in normalized["summary"]

    with pytest.raises(CodexAppServerResponseError) as error:
        normalize_calendar_create_result(
            {
                "isError": True,
                "structuredContent": {
                    "error": "Google Calendar create_event failed: HTTP 400"
                },
            }
        )
    assert "HTTP 400" in str(error.value)


def _configure_runtime(monkeypatch, tmp_path) -> None:
    for variable in (
        "ATLAS_HYBRID_PLANNING_ENABLED",
        "ATLAS_STRUCTURED_PLAN_EXECUTION_ENABLED",
        "ATLAS_EXECUTION_PERSISTENCE_ENABLED",
        "ATLAS_STRUCTURED_PLAN_PROVIDER_ENABLED",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ATLAS_EXECUTION_HISTORY_PATH", str(tmp_path / "sessions"))
    monkeypatch.setenv(
        "ATLAS_EXECUTION_STATE_PATH",
        str(tmp_path / "execution_state.json"),
    )


def test_calendar_create_stays_pending_without_confirmation(
    monkeypatch,
    tmp_path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)

    def forbidden_create(self, **_kwargs):
        raise AssertionError("calendar write must wait for explicit confirmation")

    monkeypatch.setattr(
        CodexAppServerCalendarCreateAdapter,
        "create_event",
        forbidden_create,
    )
    orchestrator = Bootstrap.build()

    pending = orchestrator.process_prompt(
        "Crea una reunion manana a las 10",
        confirm=lambda _prompt: "",
    )

    assert "pendiente de confirmacion" in pending
    assert "calendar_create_event" in pending


def test_calendar_create_confirmed_once_executes_once(
    monkeypatch,
    tmp_path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    calls: list[dict[str, str]] = []

    def fake_create(self, *, title, start_time, end_time):
        calls.append(
            {
                "title": title,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        return {
            "summary": (
                "id: evt_1 | summary: Reunion | "
                f"start: {start_time} | end: {end_time}"
            )
        }

    monkeypatch.setattr(
        CodexAppServerCalendarCreateAdapter,
        "create_event",
        fake_create,
    )
    orchestrator = Bootstrap.build()
    prompt = "Crea una reunion manana a las 10"

    pending = orchestrator.process_prompt(prompt, confirm=lambda _p: "")
    confirmed = orchestrator.process_prompt("confirmo", confirm=lambda _p: "")
    repeated = orchestrator.process_prompt("confirmo", confirm=lambda _p: "")

    assert "pendiente de confirmacion" in pending
    assert "Reunion" in confirmed
    assert len(calls) == 1
    assert calls[0]["title"] == "Reunion"
    assert confirmed != repeated


def test_calendar_create_cancel_never_executes(
    monkeypatch,
    tmp_path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)

    def forbidden_create(self, **_kwargs):
        raise AssertionError("cancelled calendar write must not execute")

    monkeypatch.setattr(
        CodexAppServerCalendarCreateAdapter,
        "create_event",
        forbidden_create,
    )
    orchestrator = Bootstrap.build()

    pending = orchestrator.process_prompt(
        "Apunta entrenamiento el viernes a las 18",
        confirm=lambda _p: "",
    )
    cancelled = orchestrator.process_prompt("cancela", confirm=lambda _p: "")

    assert "pendiente de confirmacion" in pending
    assert "cancelado" in cancelled.lower() or "cancelada" in cancelled.lower()

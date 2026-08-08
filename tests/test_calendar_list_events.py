from __future__ import annotations

from io import StringIO
import json
from threading import Event
from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from tools.calendar.calendar_list_events_tool import (
    CALENDAR_LIST_EVENTS_ARGUMENTS_SCHEMA,
    MAX_RESULTS_LIMIT,
    CalendarListEventsTool,
    CodexAppServerCalendarAdapter,
    CodexAppServerResponseError,
    CodexAppServerTimeoutError,
    CodexCliNotFoundError,
    normalize_calendar_search_result,
    validate_calendar_request,
)
from tools.executor import ToolExecutor
from tools.tool_context import ToolContext
from tools.tool_schema import ToolSchemaErrorCode, ToolSchemaValidationException


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.closed = False

    def write(self, value: str) -> int:
        self.writes.append(value)
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _BlockingStdout:
    def __init__(self) -> None:
        self._stopped = Event()

    def __iter__(self) -> _BlockingStdout:
        return self

    def __next__(self) -> str:
        self._stopped.wait()
        raise StopIteration

    def stop(self) -> None:
        self._stopped.set()


class _FakeProcess:
    def __init__(self, messages: list[dict[str, object]] | None = None) -> None:
        data = "".join(json.dumps(message) + "\n" for message in (messages or []))
        self.stdin = _FakeStdin()
        self.stdout: Any = StringIO(data)
        self.returncode: int | None = None
        self.pid = 987654
        self.waited = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        self.returncode = 0
        stop = getattr(self.stdout, "stop", None)
        if callable(stop):
            stop()
        return 0

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class _BlockingProcess(_FakeProcess):
    def __init__(self) -> None:
        super().__init__()
        self.stdout = _BlockingStdout()


def _success_messages(payload: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"id": 1, "result": {"userAgent": "codex"}},
        {"id": 2, "result": {"thread": {"id": "thr_test"}}},
        {
            "id": 3,
            "result": {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "isError": False,
                "structuredContent": None,
            },
        },
    ]


def _adapter(
    process: _FakeProcess,
    *,
    timeout: float = 1.0,
) -> CodexAppServerCalendarAdapter:
    return CodexAppServerCalendarAdapter(
        timeout_seconds=timeout,
        cli_locator=lambda _: r"C:\Users\test\AppData\Roaming\npm\codex.cmd",
        process_factory=lambda *args, **kwargs: process,  # type: ignore[arg-type]
    )


def test_normalizes_response_with_events() -> None:
    result = normalize_calendar_search_result(
        {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "events": [
                                {
                                    "id": "evt_1",
                                    "summary": "Planning",
                                    "start": "2026-08-09T10:00:00+01:00",
                                    "end": "2026-08-09T10:30:00+01:00",
                                    "ignored": "not exposed",
                                }
                            ]
                        }
                    ),
                }
            ],
            "isError": False,
        }
    )

    assert result == {
        "events": [
            {
                "id": "evt_1",
                "summary": "Planning",
                "start": "2026-08-09T10:00:00+01:00",
                "end": "2026-08-09T10:30:00+01:00",
            }
        ]
    }


def test_normalizes_empty_response() -> None:
    assert normalize_calendar_search_result(
        {
            "content": [{"type": "text", "text": '{"events":[]}'}],
            "isError": False,
        }
    ) == {"events": []}


def test_structured_app_server_error_is_raised_and_process_is_closed() -> None:
    process = _FakeProcess(
        [
            {"id": 1, "result": {"userAgent": "codex"}},
            {"id": 2, "result": {"thread": {"id": "thr_test"}}},
            {"id": 3, "error": {"code": -32000, "message": "failed"}},
        ]
    )

    with pytest.raises(CodexAppServerResponseError) as captured:
        _adapter(process).list_events(
            time_min="2026-08-09T00:00:00+01:00",
            time_max="2026-08-10T00:00:00+01:00",
        )

    assert captured.value.code == -32000
    assert process.stdin.closed is True
    assert process.waited is True


def test_timeout_closes_process_without_leaving_it_running() -> None:
    process = _BlockingProcess()

    with pytest.raises(CodexAppServerTimeoutError):
        _adapter(process, timeout=0.01).list_events(
            time_min="2026-08-09T00:00:00+01:00",
            time_max="2026-08-10T00:00:00+01:00",
        )

    assert process.stdin.closed is True
    assert process.waited is True
    assert process.poll() is not None


def test_cli_not_available_fails_before_starting_process() -> None:
    def forbidden_process(*args: object, **kwargs: object) -> _FakeProcess:
        raise AssertionError("process must not start")

    adapter = CodexAppServerCalendarAdapter(
        cli_locator=lambda _: None,
        process_factory=forbidden_process,  # type: ignore[arg-type]
    )

    with pytest.raises(CodexCliNotFoundError, match="not available on PATH"):
        adapter.list_events(
            time_min="2026-08-09T00:00:00+01:00",
            time_max="2026-08-10T00:00:00+01:00",
        )


@pytest.mark.parametrize(
    ("time_min", "time_max"),
    [
        ("invalid", "2026-08-10T00:00:00+01:00"),
        ("2026-08-09T00:00:00", "2026-08-10T00:00:00+01:00"),
        ("2026-08-10T00:00:00+01:00", "2026-08-09T00:00:00+01:00"),
        ("2026-08-09T00:00:00+01:00", "2026-08-09T00:00:00+01:00"),
    ],
)
def test_validates_time_range(time_min: str, time_max: str) -> None:
    with pytest.raises(ValueError):
        validate_calendar_request(time_min, time_max, 5)


def test_max_results_is_bounded_by_registered_schema() -> None:
    validation = CALENDAR_LIST_EVENTS_ARGUMENTS_SCHEMA.validate(
        "calendar_list_events",
        {
            "time_min": "2026-08-09T00:00:00+01:00",
            "time_max": "2026-08-10T00:00:00+01:00",
            "max_results": MAX_RESULTS_LIMIT + 1,
        },
    )

    assert validation.is_valid is False
    assert validation.errors[0].error_code == ToolSchemaErrorCode.ABOVE_MAXIMUM.value


def test_tool_executor_rejects_excessive_max_results_before_starting_app_server() -> None:
    registry = Bootstrap.build_tool_registry()

    with pytest.raises(ToolSchemaValidationException):
        ToolExecutor(registry).execute(
            "calendar_list_events",
            arguments={
                "time_min": "2026-08-09T00:00:00+01:00",
                "time_max": "2026-08-10T00:00:00+01:00",
                "max_results": MAX_RESULTS_LIMIT + 1,
            },
        )


def test_tool_uses_only_read_only_google_calendar_search() -> None:
    process = _FakeProcess(
        _success_messages(
            {
                "events": [
                    {
                        "id": "evt_1",
                        "summary": "Review",
                        "start": "2026-08-09T12:00:00+01:00",
                        "end": "2026-08-09T12:30:00+01:00",
                    }
                ]
            }
        )
    )
    tool = CalendarListEventsTool(_adapter(process))
    result = tool.execute(
        ToolContext(
            parameters={
                "time_min": "2026-08-09T00:00:00+01:00",
                "time_max": "2026-08-10T00:00:00+01:00",
                "max_results": 3,
            }
        )
    )

    sent = [json.loads(message) for message in process.stdin.writes]
    tool_call = next(message for message in sent if message.get("id") == 3)

    assert result["events"][0]["id"] == "evt_1"
    assert tool_call["params"]["server"] == "codex_apps"
    assert tool_call["params"]["tool"] == "google_calendar.search"
    assert tool_call["params"]["arguments"]["max_results"] == 3
    assert "create" not in json.dumps(tool_call).lower()
    assert "update" not in json.dumps(tool_call).lower()
    assert "delete" not in json.dumps(tool_call).lower()


def test_bootstrap_registers_calendar_list_events_as_safe_tool() -> None:
    registry = Bootstrap.build_tool_registry()

    descriptor = registry.descriptor("calendar_list_events")

    assert descriptor.requires_confirmation is False
    assert descriptor.arguments_schema is CALENDAR_LIST_EVENTS_ARGUMENTS_SCHEMA

"""Create Google Calendar events through the official Codex App Server."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
from threading import Thread
import time
from typing import TextIO

from tools.base_tool import BaseTool
from tools.calendar.calendar_list_events_tool import (
    CodexAppServerResponseError,
    _close_process,
    _END_OF_STREAM,
    _is_packaged_windows_app_binary,
    _read_stdout,
    _thread_id,
)
from tools.tool_context import ToolContext
from tools.tool_schema import ToolArgumentsSchema, ToolParameterSchema

DEFAULT_DURATION_MINUTES = 60
DEFAULT_TIMEOUT_SECONDS = 60.0


def validate_calendar_event_request(
    title: object,
    start_time: object,
    end_time: object,
) -> tuple[str, str, str]:
    """Validate the normalized event creation request."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string.")
    start = _parse_rfc3339("start_time", start_time)
    end = _parse_rfc3339("end_time", end_time)
    if start >= end:
        raise ValueError("end_time must be later than start_time.")
    assert isinstance(start_time, str)
    assert isinstance(end_time, str)
    return title.strip(), start_time, end_time


def normalize_calendar_create_result(
    result: Mapping[str, object],
) -> dict[str, str]:
    """Normalize a successful google_calendar.create_event tool result."""
    if result.get("isError") is True:
        raise CodexAppServerResponseError(detail=_connector_error_text(result))

    text_parts: list[str] = []
    structured = result.get("structuredContent")
    if isinstance(structured, Mapping):
        for key in ("id", "summary", "start", "end"):
            value = structured.get(key)
            if isinstance(value, str):
                text_parts.append(f"{key}: {value}")
        if text_parts:
            return {"summary": " | ".join(text_parts)}

    content = result.get("content", [])
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str):
                text_parts.append(text)

    if not text_parts:
        raise CodexAppServerResponseError()
    return {"summary": " | ".join(text_parts)}


def _connector_error_text(result: Mapping[str, object]) -> str | None:
    structured = result.get("structuredContent")
    if isinstance(structured, Mapping) and isinstance(structured.get("error"), str):
        return structured["error"]
    return None


class CodexAppServerCalendarCreateAdapter:
    """Fixed adapter for google_calendar.create_event."""

    def __init__(
        self,
        *,
        command: str = "codex",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        cwd: str | Path | None = None,
        cli_locator: Callable[[str], str | None] = shutil.which,
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        self._command = command
        self._timeout_seconds = float(timeout_seconds)
        self._cwd = Path(cwd) if cwd is not None else Path.cwd()
        self._cli_locator = cli_locator
        self._process_factory = process_factory

    def create_event(
        self,
        *,
        title: object,
        start_time: object,
        end_time: object,
    ) -> dict[str, str]:
        """Create one calendar event and normalize its result."""
        normalized_title, normalized_start, normalized_end = (
            validate_calendar_event_request(title, start_time, end_time)
        )
        cli_path = self._resolve_cli()
        process = self._start_process(cli_path)
        messages: "queue.Queue[object]" = queue.Queue()
        reader = Thread(
            target=_read_stdout,
            args=(process.stdout, messages),
            daemon=True,
            name="atlas-codex-app-server-create-reader",
        )
        reader.start()
        deadline = time.monotonic() + self._timeout_seconds

        try:
            self._send(
                process,
                {
                    "method": "initialize",
                    "id": 1,
                    "params": {
                        "clientInfo": {
                            "name": "atlas_calendar_create_adapter",
                            "title": "Atlas calendar create adapter",
                            "version": "1.0",
                        },
                        "capabilities": {"experimentalApi": True},
                    },
                },
            )
            _wait_response(process, messages, 1, deadline)
            self._send(process, {"method": "initialized", "params": {}})
            self._send(
                process,
                {
                    "method": "thread/start",
                    "id": 2,
                    "params": {
                        "cwd": str(self._cwd.resolve()),
                        "ephemeral": True,
                        "sandbox": "read-only",
                    },
                },
            )
            thread_response = _wait_response(process, messages, 2, deadline)
            thread_id = _thread_id(thread_response)
            self._send(
                process,
                {
                    "method": "mcpServer/tool/call",
                    "id": 3,
                    "params": {
                        "server": "codex_apps",
                        "threadId": thread_id,
                        "tool": "google_calendar.create_event",
                        "arguments": {
                            "calendar_id": "primary",
                            "title": normalized_title,
                            "start_time": normalized_start,
                            "end_time": normalized_end,
                            "attendees": [],
                        },
                    },
                },
            )
            tool_response = _wait_response(process, messages, 3, deadline)
            result = tool_response.get("result")
            if not isinstance(result, Mapping):
                raise CodexAppServerResponseError()
            return normalize_calendar_create_result(result)
        finally:
            _close_process(process, self._timeout_seconds)
            reader.join(timeout=0.5)

    def _resolve_cli(self) -> str:
        path = self._cli_locator(self._command)
        if path is None or _is_packaged_windows_app_binary(path):
            raise RuntimeError(
                "Official Codex CLI is not available on PATH."
            )
        return path

    def _start_process(self, cli_path: str) -> subprocess.Popen[str]:
        try:
            return self._process_factory(
                [cli_path, "app-server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(self._cwd),
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    if os.name == "nt"
                    else 0
                ),
            )
        except OSError as error:
            raise RuntimeError(
                "Unable to start Codex App Server."
            ) from error

    @staticmethod
    def _send(
        process: subprocess.Popen[str],
        message: Mapping[str, object],
    ) -> None:
        if process.stdin is None:
            raise RuntimeError("Codex App Server stdin is unavailable.")
        try:
            process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(
                "Codex App Server closed its input unexpectedly."
            ) from error


def _wait_response(
    process: subprocess.Popen[str],
    messages: "queue.Queue[object]",
    request_id: int,
    deadline: float,
) -> Mapping[str, object]:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Codex App Server request timed out.")
        try:
            raw = messages.get(timeout=remaining)
        except queue.Empty as error:
            raise RuntimeError(
                "Codex App Server request timed out."
            ) from error

        if raw is _END_OF_STREAM:
            code = process.poll()
            raise RuntimeError(
                f"Codex App Server stopped before responding (exit {code})."
            )
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Codex App Server returned invalid JSON."
            ) from error
        if not isinstance(message, Mapping) or message.get("id") != request_id:
            continue
        error_payload = message.get("error")
        if isinstance(error_payload, Mapping):
            raise CodexAppServerResponseError(error_payload.get("code"))
        return message


class CalendarCreateEventTool(BaseTool):
    """Atlas tool exposing confirmed calendar event creation."""

    def __init__(
        self,
        adapter: CodexAppServerCalendarCreateAdapter | None = None,
    ) -> None:
        self._adapter = adapter or CodexAppServerCalendarCreateAdapter()

    @property
    def name(self) -> str:
        return "calendar_create_event"

    @property
    def description(self) -> str:
        return "Create one primary-calendar event with title, start and end. Requires confirmation (writes calendar)."

    @property
    def requires_confirmation(self) -> bool:
        return True

    @property
    def required_permissions(self) -> tuple[str, ...]:
        return ("calendar.write",)

    def semantic_metadata(self) -> dict[str, object]:
        return {
            "capabilities": ["calendar_create_event"],
            "supported_intents": ["create one calendar event"],
            "input_description": "Requires title, RFC3339 start_time and end_time.",
            "output_description": "Human-readable created event confirmation.",
            "risk_level": "high",
            "limitations": ["requires confirmation", "primary calendar only", "no recurrence or guests"],
            "tags": ["calendar", "google-calendar", "write"],
            "category": "calendar",
        }

    def execute(self, context: ToolContext) -> str:
        title = context.parameters.get("title")
        start_time = context.parameters.get("start_time")
        end_time = context.parameters.get("end_time")
        result = self._adapter.create_event(
            title=title,
            start_time=start_time,
            end_time=end_time,
        )
        return str(result.get("summary") or "Evento creado.")


def _parse_rfc3339(name: str, value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty RFC3339 string.")
    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{name} must be a valid RFC3339 timestamp.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone offset.")
    return parsed


CALENDAR_CREATE_EVENT_ARGUMENTS_SCHEMA = ToolArgumentsSchema(
    parameters=(
        ToolParameterSchema("title", str, required=True),
        ToolParameterSchema("start_time", str, required=True),
        ToolParameterSchema("end_time", str, required=True),
    ),
)

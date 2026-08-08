"""Read Google Calendar events through the official Codex App Server."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
from threading import Thread
import time
from typing import TextIO

from tools.base_tool import BaseTool
from tools.tool_context import ToolContext
from tools.tool_schema import ToolArgumentsSchema, ToolParameterSchema


DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_RESULTS_LIMIT = 20
_END_OF_STREAM = object()


class CalendarAdapterError(RuntimeError):
    """Base error for the Codex App Server calendar boundary."""


class CodexCliNotFoundError(CalendarAdapterError):
    """Raised when the official Codex CLI cannot be resolved."""


class CodexAppServerTimeoutError(CalendarAdapterError):
    """Raised when App Server does not answer within the configured timeout."""


class CodexAppServerResponseError(CalendarAdapterError):
    """Raised for a JSON-RPC or connector error returned by App Server."""

    def __init__(self, code: object | None = None) -> None:
        self.code = code
        suffix = f" (code {code})" if code is not None else ""
        super().__init__(f"Codex App Server returned an error{suffix}.")


CALENDAR_LIST_EVENTS_ARGUMENTS_SCHEMA = ToolArgumentsSchema(
    parameters=(
        ToolParameterSchema("time_min", str, required=True),
        ToolParameterSchema("time_max", str, required=True),
        ToolParameterSchema(
            "max_results",
            int,
            default=5,
            minimum=1,
            maximum=MAX_RESULTS_LIMIT,
        ),
    ),
)


def validate_calendar_request(
    time_min: object,
    time_max: object,
    max_results: object,
) -> tuple[str, str, int]:
    """Validate the normalized calendar request."""
    start = _parse_rfc3339("time_min", time_min)
    end = _parse_rfc3339("time_max", time_max)

    if start >= end:
        raise ValueError("time_min must be earlier than time_max.")
    if type(max_results) is not int:
        raise ValueError("max_results must be an integer.")
    if not 1 <= max_results <= MAX_RESULTS_LIMIT:
        raise ValueError(
            f"max_results must be between 1 and {MAX_RESULTS_LIMIT}."
        )

    assert isinstance(time_min, str)
    assert isinstance(time_max, str)
    return time_min, time_max, max_results


def normalize_calendar_search_result(
    result: Mapping[str, object],
) -> dict[str, list[dict[str, str]]]:
    """Normalize a successful google_calendar.search tool result."""
    if result.get("isError") is True:
        raise CodexAppServerResponseError()

    payloads: list[object] = []
    structured = result.get("structuredContent")
    if structured is not None:
        payloads.append(structured)

    content = result.get("content", [])
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "text":
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            try:
                payloads.append(json.loads(text))
            except json.JSONDecodeError:
                continue

    raw_events: list[object] | None = None
    for payload in payloads:
        raw_events = _find_events(payload)
        if raw_events is not None:
            break

    if raw_events is None:
        raise CodexAppServerResponseError()

    events: list[dict[str, str]] = []
    for event in raw_events:
        if not isinstance(event, Mapping):
            raise CodexAppServerResponseError()
        events.append(
            {
                "id": _normalized_text(event.get("id")),
                "summary": _normalized_text(
                    event.get("summary", event.get("title"))
                ),
                "start": _normalized_text(event.get("start")),
                "end": _normalized_text(event.get("end")),
            }
        )

    return {"events": events}


class CodexAppServerCalendarAdapter:
    """Fixed read-only adapter for google_calendar.search."""

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

    def list_events(
        self,
        *,
        time_min: object,
        time_max: object,
        max_results: object = 5,
    ) -> dict[str, list[dict[str, str]]]:
        """Run one read-only calendar search and normalize its result."""
        normalized_min, normalized_max, normalized_limit = validate_calendar_request(
            time_min,
            time_max,
            max_results,
        )
        cli_path = self._resolve_cli()
        process = self._start_process(cli_path)
        messages: Queue[object] = Queue()
        reader = Thread(
            target=_read_stdout,
            args=(process.stdout, messages),
            daemon=True,
            name="atlas-codex-app-server-reader",
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
                            "name": "atlas_calendar_adapter",
                            "title": "Atlas calendar adapter",
                            "version": "1.0",
                        },
                        "capabilities": {"experimentalApi": True},
                    },
                },
            )
            self._response(process, messages, 1, deadline)
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
            thread_response = self._response(process, messages, 2, deadline)
            thread_id = _thread_id(thread_response)
            self._send(
                process,
                {
                    "method": "mcpServer/tool/call",
                    "id": 3,
                    "params": {
                        "server": "codex_apps",
                        "threadId": thread_id,
                        "tool": "google_calendar.search",
                        "arguments": {
                            "query": "",
                            "time_min": normalized_min,
                            "time_max": normalized_max,
                            "max_results": normalized_limit,
                            "calendar_id": "primary",
                        },
                    },
                },
            )
            tool_response = self._response(process, messages, 3, deadline)
            result = tool_response.get("result")
            if not isinstance(result, Mapping):
                raise CodexAppServerResponseError()
            return normalize_calendar_search_result(result)
        finally:
            _close_process(process, self._timeout_seconds)
            reader.join(timeout=0.5)

    def _resolve_cli(self) -> str:
        path = self._cli_locator(self._command)
        if path is None or _is_packaged_windows_app_binary(path):
            raise CodexCliNotFoundError(
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
            raise CalendarAdapterError(
                "Unable to start Codex App Server."
            ) from error

    @staticmethod
    def _send(
        process: subprocess.Popen[str],
        message: Mapping[str, object],
    ) -> None:
        if process.stdin is None:
            raise CalendarAdapterError("Codex App Server stdin is unavailable.")
        try:
            process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise CalendarAdapterError(
                "Codex App Server closed its input unexpectedly."
            ) from error

    @staticmethod
    def _response(
        process: subprocess.Popen[str],
        messages: Queue[object],
        request_id: int,
        deadline: float,
    ) -> Mapping[str, object]:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodexAppServerTimeoutError(
                    "Codex App Server request timed out."
                )
            try:
                raw = messages.get(timeout=remaining)
            except Empty as error:
                raise CodexAppServerTimeoutError(
                    "Codex App Server request timed out."
                ) from error

            if raw is _END_OF_STREAM:
                code = process.poll()
                raise CalendarAdapterError(
                    f"Codex App Server stopped before responding (exit {code})."
                )
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError as error:
                raise CalendarAdapterError(
                    "Codex App Server returned invalid JSON."
                ) from error
            if not isinstance(message, Mapping) or message.get("id") != request_id:
                continue
            error_payload = message.get("error")
            if isinstance(error_payload, Mapping):
                raise CodexAppServerResponseError(error_payload.get("code"))
            return message


class CalendarListEventsTool(BaseTool):
    """Atlas tool exposing one normalized read-only calendar operation."""

    def __init__(
        self,
        adapter: CodexAppServerCalendarAdapter | None = None,
    ) -> None:
        self._adapter = adapter or CodexAppServerCalendarAdapter()

    @property
    def name(self) -> str:
        return "calendar_list_events"

    @property
    def description(self) -> str:
        return "List Google Calendar events in a bounded time range (read-only)."

    def semantic_metadata(self) -> dict[str, object]:
        """Return semantic metadata for catalog generation."""
        return {
            "capabilities": ["calendar_list_events"],
            "supported_intents": ["list calendar events in a time range"],
            "input_description": "Requires RFC3339 time_min/time_max and an optional limit.",
            "output_description": "Normalized event id, summary, start, and end fields.",
            "risk_level": "low",
            "limitations": ["read-only", "maximum 20 events", "primary calendar only"],
            "tags": ["calendar", "google-calendar", "read"],
            "category": "calendar",
        }

    def execute(
        self,
        context: ToolContext,
    ) -> dict[str, list[dict[str, str]]]:
        return self._adapter.list_events(
            time_min=context.parameters.get("time_min"),
            time_max=context.parameters.get("time_max"),
            max_results=context.parameters.get("max_results", 5),
        )


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


def _thread_id(response: Mapping[str, object]) -> str:
    result = response.get("result")
    thread = result.get("thread") if isinstance(result, Mapping) else None
    thread_id = thread.get("id") if isinstance(thread, Mapping) else None
    if not isinstance(thread_id, str) or not thread_id:
        raise CodexAppServerResponseError()
    return thread_id


def _find_events(value: object) -> list[object] | None:
    if isinstance(value, Mapping):
        events = value.get("events")
        if isinstance(events, list):
            return events
        for nested in value.values():
            found = _find_events(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_events(nested)
            if found is not None:
                return found
    return None


def _normalized_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _read_stdout(stream: TextIO | None, messages: Queue[object]) -> None:
    if stream is None:
        messages.put(_END_OF_STREAM)
        return
    try:
        for line in stream:
            messages.put(line)
    finally:
        messages.put(_END_OF_STREAM)


def _close_process(
    process: subprocess.Popen[str],
    request_timeout: float,
) -> None:
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    if process.poll() is not None:
        return

    shutdown_timeout = min(2.0, max(0.1, request_timeout))
    try:
        process.wait(timeout=shutdown_timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass

    if process.poll() is None:
        try:
            process.kill()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _is_packaged_windows_app_binary(path: str) -> bool:
    normalized = path.replace("/", "\\").lower()
    return (
        "\\windowsapps\\openai.codex_" in normalized
        and "\\app\\resources\\codex" in normalized
    )

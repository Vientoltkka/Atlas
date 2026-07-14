"""Condition-based waits for desktop workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

from tools.executor import ToolExecutor
from tools.tool_context import ToolContext


@dataclass(frozen=True)
class WaitResult:
    """Controlled result for one wait condition."""

    completed: bool
    elapsed_time: float
    timeout: float
    condition: str
    description: str
    timed_out: bool = False
    error: str | None = None


class WaitEngine:
    """Wait for real desktop and filesystem conditions with bounded polling."""

    def __init__(
        self,
        executor: ToolExecutor,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._executor = executor
        self._monotonic = monotonic or time.monotonic
        self._sleeper = sleeper or time.sleep

    def wait_process(
        self,
        name: str,
        *,
        timeout: float,
        poll_interval: float,
        exists: bool = True,
        cancelled: Callable[[], bool] | None = None,
    ) -> WaitResult:
        """Wait until a process exists or disappears."""
        condition = "process_exists" if exists else "process_absent"
        target = self._require_text(name, "nombre de proceso")
        description = self._description(condition, target)

        return self._wait(
            condition,
            description,
            timeout,
            poll_interval,
            lambda: bool(self._list_processes(target)) is exists,
            cancelled,
        )

    def wait_window(
        self,
        title: str,
        *,
        timeout: float,
        poll_interval: float,
        exists: bool = True,
        cancelled: Callable[[], bool] | None = None,
    ) -> WaitResult:
        """Wait until a window exists or disappears."""
        condition = "window_exists" if exists else "window_absent"
        target = self._require_text(title, "titulo de ventana")
        description = self._description(condition, target)

        return self._wait(
            condition,
            description,
            timeout,
            poll_interval,
            lambda: bool(self._list_windows(target)) is exists,
            cancelled,
        )

    def wait_active_window(
        self,
        title: str,
        *,
        timeout: float,
        poll_interval: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> WaitResult:
        """Wait until the foreground window title matches."""
        target = self._require_text(title, "titulo de ventana")
        condition = "active_window"
        description = self._description(condition, target)

        return self._wait(
            condition,
            description,
            timeout,
            poll_interval,
            lambda: self._active_window_matches(target),
            cancelled,
        )

    def wait_file(
        self,
        path: str | Path,
        *,
        timeout: float,
        poll_interval: float,
        exists: bool = True,
        cancelled: Callable[[], bool] | None = None,
    ) -> WaitResult:
        """Wait until a filesystem path exists or disappears."""
        condition = "file_exists" if exists else "file_absent"
        target = Path(self._require_text(str(path), "ruta de archivo"))
        description = self._description(condition, str(target))

        return self._wait(
            condition,
            description,
            timeout,
            poll_interval,
            lambda: target.exists() is exists,
            cancelled,
        )

    def wait_application(
        self,
        application: str,
        *,
        timeout: float,
        poll_interval: float,
        opened: bool = True,
        cancelled: Callable[[], bool] | None = None,
    ) -> WaitResult:
        """Wait until an application is open or closed."""
        return self.wait_process(
            application,
            timeout=timeout,
            poll_interval=poll_interval,
            exists=opened,
            cancelled=cancelled,
        )

    def _wait(
        self,
        condition: str,
        description: str,
        timeout: float,
        poll_interval: float,
        predicate: Callable[[], bool],
        cancelled: Callable[[], bool] | None,
    ) -> WaitResult:
        """Poll a predicate until it is true or timeout expires."""
        timeout = self._validate_seconds(timeout, "timeout")
        poll_interval = self._validate_seconds(poll_interval, "poll_interval")
        start = self._monotonic()
        last_error: str | None = None

        while True:
            now = self._monotonic()
            elapsed = max(0.0, now - start)

            if elapsed >= timeout:
                return WaitResult(
                    completed=False,
                    elapsed_time=elapsed,
                    timeout=timeout,
                    condition=condition,
                    description=description,
                    timed_out=True,
                    error=last_error or "Timeout agotado.",
                )

            if cancelled is not None and cancelled():
                return WaitResult(
                    completed=False,
                    elapsed_time=elapsed,
                    timeout=timeout,
                    condition=condition,
                    description=description,
                    error="Espera cancelada.",
                )

            try:
                if predicate():
                    return WaitResult(
                        completed=True,
                        elapsed_time=elapsed,
                        timeout=timeout,
                        condition=condition,
                        description=description,
                    )
            except Exception as exc:
                return WaitResult(
                    completed=False,
                    elapsed_time=elapsed,
                    timeout=timeout,
                    condition=condition,
                    description=description,
                    error=str(exc),
                )

            remaining = timeout - elapsed
            if remaining <= 0:
                continue

            self._sleeper(min(poll_interval, remaining))

    def _list_processes(
        self,
        query: str,
    ) -> list[dict[str, object]]:
        result = self._executor.execute(
            "desktop.list_processes",
            ToolContext(parameters={"query": query}),
        )

        if not isinstance(result, list):
            raise RuntimeError("Respuesta de procesos invalida.")

        return result

    def _list_windows(
        self,
        title: str,
    ) -> list[dict[str, object]]:
        result = self._executor.execute(
            "desktop.list_windows",
            ToolContext(parameters={"title": title}),
        )

        if not isinstance(result, list):
            raise RuntimeError("Respuesta de ventanas invalida.")

        return result

    def _active_window_matches(
        self,
        title: str,
    ) -> bool:
        result = self._executor.execute(
            "desktop.get_foreground_window",
            ToolContext(),
        )

        if not isinstance(result, dict):
            raise RuntimeError("Respuesta de ventana activa invalida.")

        active_title = str(result.get("title", "")).lower()
        return title.lower() in active_title

    def _require_text(
        self,
        value: str,
        label: str,
    ) -> str:
        cleaned = value.strip().strip('"')

        if not cleaned:
            raise ValueError(f"Falta {label}.")

        return cleaned

    def _validate_seconds(
        self,
        value: float,
        label: str,
    ) -> float:
        if not isinstance(value, int | float) or value <= 0:
            raise ValueError(f"{label} debe ser mayor que cero.")

        return float(value)

    def _description(
        self,
        condition: str,
        target: str,
    ) -> str:
        labels = {
            "process_exists": "Esperando proceso",
            "process_absent": "Esperando desaparicion de proceso",
            "window_exists": "Esperando ventana",
            "window_absent": "Esperando desaparicion de ventana",
            "active_window": "Esperando ventana activa",
            "file_exists": "Esperando archivo",
            "file_absent": "Esperando desaparicion de archivo",
        }

        return f"{labels[condition]}: {target}"

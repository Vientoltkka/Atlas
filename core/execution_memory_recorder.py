"""Opt-in recording of durable successful execution summaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memory.operational import (
    MemoryCategory,
    MemoryEntry,
    MemoryPolicy,
    contains_secret_material,
)


@dataclass(frozen=True, slots=True)
class ExecutionMemoryEvent:
    event_type: str
    request_id: str | None
    memory_id: str | None
    category: MemoryCategory | None
    operation: str
    selected_count: int
    content_length: int
    truncated: bool
    reason_code: str
    timestamp: datetime


class ExecutionMemoryRecorder:
    """Store only explicitly enabled, bounded, successful execution summaries."""

    def __init__(
        self,
        memory: object | None,
        *,
        policy: MemoryPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        max_summary_characters: int = 500,
    ) -> None:
        if max_summary_characters < 1:
            raise ValueError("max_summary_characters must be positive.")
        self._memory = memory
        self._policy = policy or getattr(memory, "policy", None) or MemoryPolicy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._max_summary_characters = max_summary_characters
        self._events: list[ExecutionMemoryEvent] = []

    @property
    def events(self) -> tuple[ExecutionMemoryEvent, ...]:
        return tuple(self._events)

    def record(
        self,
        result: object,
        *,
        request: object | None = None,
    ) -> MemoryEntry | None:
        request_id = getattr(result, "request_id", None)
        if not self._policy.automatic_write_enabled:
            self._skip(request_id, "automatic_write_disabled")
            return None
        if not self._policy.allow_execution_result_storage:
            self._skip(request_id, "execution_result_storage_disabled")
            return None
        status = getattr(getattr(result, "status", None), "value", None)
        if status != "completed":
            self._skip(request_id, "execution_not_completed")
            return None
        route = getattr(getattr(result, "route", None), "value", None)
        if route != "autonomous_execution":
            self._skip(request_id, "route_not_durable")
            return None
        store_entry = getattr(self._memory, "store_entry", None)
        if not callable(store_entry):
            self._skip(request_id, "memory_not_configured")
            return None
        summary = _safe_result_summary(result)
        if not summary:
            self._skip(request_id, "no_durable_summary")
            return None
        if contains_secret_material(summary):
            self._skip(request_id, "sensitive_result")
            return None
        truncated = len(summary) > self._max_summary_characters
        summary = summary[: self._max_summary_characters]
        entry = store_entry(
            summary,
            category=MemoryCategory.EXECUTION_RESULT,
            source_request_id=request_id,
            user_id=getattr(request, "user_id", None),
            conversation_id=getattr(request, "conversation_id", None),
            importance=0.6,
            tags=("execution-result",),
            sensitive=False,
            metadata={"automatic": True},
        )
        self._events.append(
            ExecutionMemoryEvent(
                event_type="execution_memory_recorded",
                request_id=request_id,
                memory_id=entry.memory_id,
                category=entry.category,
                operation="record",
                selected_count=1,
                content_length=len(summary),
                truncated=truncated,
                reason_code="durable_completed_result",
                timestamp=self._clock(),
            )
        )
        return entry

    def _skip(self, request_id: str | None, reason_code: str) -> None:
        self._events.append(
            ExecutionMemoryEvent(
                event_type="execution_memory_record_skipped",
                request_id=request_id,
                memory_id=None,
                category=None,
                operation="record",
                selected_count=0,
                content_length=0,
                truncated=False,
                reason_code=reason_code,
                timestamp=self._clock(),
            )
        )


def _safe_result_summary(result: object) -> str:
    output = getattr(result, "output", None)
    if isinstance(output, str):
        return " ".join(output.split())
    if isinstance(output, Mapping):
        for key in ("summary", "result", "message"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
    return ""

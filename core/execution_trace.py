"""In-memory execution trace primitives for Atlas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TraceStatus(str, Enum):
    """Closed statuses for an execution trace."""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TraceEventStatus(str, Enum):
    """Closed statuses for individual trace events."""

    STARTED = "STARTED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One structured observability event inside an execution trace."""

    timestamp: datetime
    component: str
    action: str
    status: str
    duration_ms: int | None = None
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {status.value for status in TraceEventStatus}:
            raise ValueError(f"Invalid trace event status: {self.status}")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("Trace event duration cannot be negative.")


@dataclass(slots=True)
class ExecutionTrace:
    """Mutable in-memory trace for one execution."""

    execution_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime = field(default_factory=_utc_now)
    finished_at: datetime | None = None
    status: str = TraceStatus.RUNNING.value
    events: list[TraceEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in {status.value for status in TraceStatus}:
            raise ValueError(f"Invalid execution trace status: {self.status}")

    def add_event(
        self,
        *,
        component: str,
        action: str,
        status: str,
        duration_ms: int | None = None,
        details: dict[str, object] | None = None,
        timestamp: datetime | None = None,
    ) -> TraceEvent:
        """Append one structured event and return it."""
        event = TraceEvent(
            timestamp=timestamp or _utc_now(),
            component=component,
            action=action,
            status=status,
            duration_ms=duration_ms,
            details={} if details is None else dict(details),
        )
        self.events.append(event)
        return event

    def finish(
        self,
        status: str,
        *,
        finished_at: datetime | None = None,
    ) -> None:
        """Mark the trace as finished with a terminal status."""
        if status not in {
            TraceStatus.SUCCESS.value,
            TraceStatus.FAILED.value,
            TraceStatus.CANCELLED.value,
        }:
            raise ValueError(f"Invalid terminal trace status: {status}")
        self.status = status
        self.finished_at = finished_at or _utc_now()

    def duration(self) -> int:
        """Return the trace duration in milliseconds."""
        end = self.finished_at or _utc_now()
        return max(0, int((end - self.started_at).total_seconds() * 1000))

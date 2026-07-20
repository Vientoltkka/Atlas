"""Bounded in-memory history for recent structured executions."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Iterator

from core.execution_metrics import ExecutionMetrics, ExecutionMetricsCalculator
from core.execution_trace import ExecutionTrace, TraceStatus

if TYPE_CHECKING:
    from core.execution_plan_executor import PlanExecutionResult


@dataclass(frozen=True, slots=True)
class ExecutionHistoryEntry:
    """Immutable entry stored in the in-memory execution history."""

    execution_id: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    trace: ExecutionTrace
    metrics: ExecutionMetrics
    result: PlanExecutionResult | None = None


class ExecutionHistory:
    """Configurable bounded history of recent executions kept only in memory."""

    def __init__(
        self,
        max_entries: int = 100,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("ExecutionHistory max_entries must be greater than zero.")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, ExecutionHistoryEntry] = OrderedDict()

    @property
    def max_entries(self) -> int:
        """Return the configured history capacity."""
        return self._max_entries

    def add(
        self,
        result: PlanExecutionResult,
    ) -> ExecutionHistoryEntry:
        """Store one completed execution result and return its history entry."""
        trace = result.trace
        if trace is None:
            raise ValueError("Execution result must include trace to be stored.")

        metrics = result.metrics or ExecutionMetricsCalculator().calculate(trace)
        entry = ExecutionHistoryEntry(
            execution_id=trace.execution_id,
            status=trace.status,
            started_at=trace.started_at,
            finished_at=trace.finished_at,
            trace=trace,
            metrics=metrics,
            result=result,
        )

        if entry.execution_id in self._entries:
            del self._entries[entry.execution_id]
        self._entries[entry.execution_id] = entry
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return entry

    def latest(self) -> ExecutionHistoryEntry | None:
        """Return the most recently stored entry, if any."""
        if not self._entries:
            return None
        return next(reversed(self._entries.values()))

    def recent(
        self,
        limit: int,
    ) -> tuple[ExecutionHistoryEntry, ...]:
        """Return up to limit entries from newest to oldest."""
        if limit < 0:
            raise ValueError("Recent history limit cannot be negative.")
        if limit == 0:
            return ()
        return tuple(reversed(tuple(self._entries.values())))[0:limit]

    def get(
        self,
        execution_id: str,
    ) -> ExecutionHistoryEntry | None:
        """Return the entry for execution_id, if present."""
        return self._entries.get(execution_id)

    def count(self) -> int:
        """Return the number of entries currently stored."""
        return len(self._entries)

    def count_successful(self) -> int:
        """Return the number of successful stored executions."""
        return sum(
            1
            for entry in self._entries.values()
            if entry.status == TraceStatus.SUCCESS.value
        )

    def count_failed(self) -> int:
        """Return the number of failed stored executions."""
        return sum(
            1
            for entry in self._entries.values()
            if entry.status == TraceStatus.FAILED.value
        )

    def slowest(self) -> ExecutionHistoryEntry | None:
        """Return the slowest entry, choosing the newest one on ties."""
        slowest_entry: ExecutionHistoryEntry | None = None
        for entry in self._entries.values():
            if (
                slowest_entry is None
                or entry.metrics.total_duration_ms
                >= slowest_entry.metrics.total_duration_ms
            ):
                slowest_entry = entry
        return slowest_entry

    def clear(self) -> None:
        """Remove all entries from memory."""
        self._entries.clear()

    def __len__(self) -> int:
        return self.count()

    def __iter__(self) -> Iterator[ExecutionHistoryEntry]:
        return iter(tuple(self._entries.values()))

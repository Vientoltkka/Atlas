"""Execution histories for results and persisted supervised sessions."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterator, Protocol

from core.execution_metrics import ExecutionMetrics, ExecutionMetricsCalculator
from core.execution_report import (
    ExecutionReportGenerator,
    OperationalExecutionReport,
    OperationalExecutionStatus,
)
from core.execution_session_persistence import ExecutionSessionRepository
from core.execution_session_persistence import ExecutionPersistenceError
from core.execution_supervisor import (
    ExecutionSession,
    ExecutionState,
    StepExecutionState,
    summarize_execution_session,
)
from core.execution_trace import ExecutionTrace, TraceStatus

if TYPE_CHECKING:
    from core.execution_plan_executor import PlanExecutionResult


class ExecutionHistorySink(Protocol):
    """Minimal sink accepted by executors that record final execution results."""

    def add(
        self,
        result: PlanExecutionResult,
    ) -> "ExecutionHistoryEntry":
        """Store one completed execution result and return its history entry."""


@dataclass(frozen=True, slots=True)
class ExecutionHistoryEntry:
    """Immutable entry stored in the bounded in-memory execution history."""

    execution_id: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    trace: ExecutionTrace
    metrics: ExecutionMetrics
    result: PlanExecutionResult | None = None


class ExecutionHistory:
    """Configurable bounded history of recent execution results."""

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

    def add_entry(
        self,
        entry: ExecutionHistoryEntry,
    ) -> ExecutionHistoryEntry:
        """Store a validated history entry without requiring a live result."""
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


_SUCCESS_STATUSES = frozenset(
    {
        OperationalExecutionStatus.COMPLETED,
        OperationalExecutionStatus.COMPLETED_WITH_RECOVERY,
    }
)
_FAILED_STEP_STATES = frozenset(
    {
        StepExecutionState.FAILED,
        StepExecutionState.BLOCKED,
        StepExecutionState.INTERRUPTED,
    }
)


class ExecutionSessionSource(Protocol):
    """Minimal live-session source required by the history service."""

    def list_sessions(
        self,
        *,
        state: ExecutionState | None = None,
        limit: int | None = None,
        newest_first: bool = True,
    ) -> tuple[ExecutionSession, ...]:
        """Return supervised execution sessions."""


@dataclass(frozen=True, slots=True)
class ExecutionHistoryRecord:
    """One immutable, fully derived historical execution projection."""

    id: str
    date: datetime
    objective: str
    duration_seconds: float
    final_result: OperationalExecutionStatus
    progress_percent: float
    executed_step_ids: tuple[str, ...]
    failed_step_ids: tuple[str, ...]
    omitted_step_ids: tuple[str, ...]
    replanned_step_ids: tuple[str, ...]
    retry_count: int
    failure_reason: str | None
    required_actions: tuple[str, ...]
    operational_report: OperationalExecutionReport
    recovery_types: tuple[str, ...]
    state: ExecutionState
    tool_names: tuple[str, ...] = ()
    tools_by_step: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_names", tuple(self.tool_names))
        object.__setattr__(
            self,
            "tools_by_step",
            MappingProxyType(dict(self.tools_by_step)),
        )


@dataclass(frozen=True, slots=True)
class ExecutionHistoryStatistics:
    """Deterministic aggregate statistics rebuilt from history records."""

    total_executions: int
    successful_executions: int
    failed_executions: int
    cancelled_executions: int
    success_frequency: float
    failure_frequency: float
    average_duration_seconds: float
    average_retry_count: float
    frequently_failed_steps: Mapping[str, int] = field(default_factory=dict)
    normally_omitted_steps: Mapping[str, int] = field(default_factory=dict)
    recovery_types: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frequently_failed_steps",
            MappingProxyType(dict(self.frequently_failed_steps)),
        )
        object.__setattr__(
            self,
            "normally_omitted_steps",
            MappingProxyType(dict(self.normally_omitted_steps)),
        )
        object.__setattr__(
            self,
            "recovery_types",
            MappingProxyType(dict(self.recovery_types)),
        )


class ExecutionSessionHistory:
    """Query terminal executions without changing planning or future decisions."""

    def __init__(
        self,
        *,
        session_source: ExecutionSessionSource | None = None,
        session_repository: ExecutionSessionRepository | None = None,
        report_generator: ExecutionReportGenerator | None = None,
    ) -> None:
        if session_source is None and session_repository is None:
            raise ValueError(
                "session_source or session_repository must be provided."
            )
        self._session_source = session_source
        self._session_repository = session_repository
        self._report_generator = report_generator or ExecutionReportGenerator()

    def latest_execution(self) -> ExecutionHistoryRecord | None:
        """Return the most recent terminal execution, if any."""
        records = self.latest_executions(1)
        return records[0] if records else None

    def latest_executions(self, limit: int) -> tuple[ExecutionHistoryRecord, ...]:
        """Return the latest N terminal executions."""
        _validate_limit(limit)
        return self._records()[:limit]

    def execution_by_id(self, session_id: str) -> ExecutionHistoryRecord | None:
        """Return one terminal execution by id without changing its state."""
        if not isinstance(session_id, str):
            return None
        normalized_id = session_id.strip()
        if not normalized_id:
            return None
        for session in self._sessions():
            if session.session_id == normalized_id and session.is_terminal:
                return self._record(session)
        return None

    def executions_by_objective(
        self,
        objective: str,
    ) -> tuple[ExecutionHistoryRecord, ...]:
        """Return executions whose normalized objective matches exactly."""
        normalized = _normalize_objective(objective)
        return tuple(
            record
            for record in self._records()
            if _normalize_objective(record.objective) == normalized
        )

    def failed_executions(self) -> tuple[ExecutionHistoryRecord, ...]:
        """Return executions that ended in the FAILED lifecycle state."""
        return tuple(
            record
            for record in self._records()
            if record.state is ExecutionState.FAILED
        )

    def successful_executions(self) -> tuple[ExecutionHistoryRecord, ...]:
        """Return executions with a completed operational result."""
        return tuple(
            record
            for record in self._records()
            if record.final_result in _SUCCESS_STATUSES
        )

    def executions_with_recovery(self) -> tuple[ExecutionHistoryRecord, ...]:
        """Return executions that used retries or controlled replanning."""
        return tuple(
            record for record in self._records() if record.recovery_types
        )

    def statistics(self) -> ExecutionHistoryStatistics:
        """Aggregate all known terminal executions."""
        records = self._records()
        total = len(records)
        successes = sum(
            record.final_result in _SUCCESS_STATUSES for record in records
        )
        failures = sum(
            record.state is ExecutionState.FAILED for record in records
        )
        cancelled = sum(
            record.state is ExecutionState.CANCELLED for record in records
        )
        return ExecutionHistoryStatistics(
            total_executions=total,
            successful_executions=successes,
            failed_executions=failures,
            cancelled_executions=cancelled,
            success_frequency=_ratio(successes, total),
            failure_frequency=_ratio(failures, total),
            average_duration_seconds=_average(
                record.duration_seconds for record in records
            ),
            average_retry_count=_average(
                float(record.retry_count) for record in records
            ),
            frequently_failed_steps=_ordered_counts(
                step_id
                for record in records
                for step_id in record.failed_step_ids
            ),
            normally_omitted_steps=_ordered_counts(
                step_id
                for record in records
                for step_id in record.omitted_step_ids
            ),
            recovery_types=_ordered_counts(
                recovery_type
                for record in records
                for recovery_type in record.recovery_types
            ),
        )

    def _records(self) -> tuple[ExecutionHistoryRecord, ...]:
        sessions = self._sessions()
        records = tuple(
            self._record(session)
            for session in sessions
            if session.is_terminal
        )
        return tuple(
            sorted(
                records,
                key=lambda record: (record.date, record.id),
                reverse=True,
            )
        )

    def _sessions(self) -> tuple[ExecutionSession, ...]:
        by_id: dict[str, ExecutionSession] = {}
        if self._session_repository is not None:
            for session_id in self._session_repository.list():
                try:
                    snapshot = self._session_repository.load(session_id)
                except ExecutionPersistenceError:
                    continue
                if snapshot is not None:
                    by_id[session_id] = snapshot.to_session()
        if self._session_source is not None:
            for session in self._session_source.list_sessions():
                by_id[session.session_id] = session
        return tuple(by_id.values())

    def _record(self, session: ExecutionSession) -> ExecutionHistoryRecord:
        summary = summarize_execution_session(session)
        report = self._report_generator.generate(session, summary)
        snapshots = tuple(session.step_states.values())
        executed = tuple(
            snapshot.step_id
            for snapshot in snapshots
            if snapshot.attempt_count > 0
            and snapshot.state is not StepExecutionState.SKIPPED
        )
        failed = tuple(
            snapshot.step_id
            for snapshot in snapshots
            if snapshot.state in _FAILED_STEP_STATES
        )
        omitted = tuple(
            snapshot.step_id
            for snapshot in snapshots
            if snapshot.state is StepExecutionState.SKIPPED
        )
        replanned = _replanned_step_ids(session)
        recovery_types = _recovery_types(session, summary.retry_count)
        tools_by_step = {
            step.id: step.tool
            for plan in (session.original_plan, session.active_plan)
            for step in plan.ordered_steps
            if step.tool is not None
        }
        tool_names = _unique(
            tuple(session.original_plan.required_tools)
            + tuple(session.active_plan.required_tools)
            + tuple(tools_by_step.values())
        )
        failure_reason = session.last_error
        if failure_reason is None and summary.errors:
            failure_reason = next(iter(summary.errors.values()))
        return ExecutionHistoryRecord(
            id=session.session_id,
            date=session.started_at,
            objective=report.objective,
            duration_seconds=report.duration_seconds,
            final_result=report.status,
            progress_percent=report.progress_percent,
            executed_step_ids=executed,
            failed_step_ids=failed,
            omitted_step_ids=omitted,
            replanned_step_ids=replanned,
            retry_count=report.retry_count,
            failure_reason=failure_reason,
            required_actions=report.pending_user_actions,
            operational_report=report,
            recovery_types=recovery_types,
            state=session.state,
            tool_names=tool_names,
            tools_by_step=tools_by_step,
        )


def _replanned_step_ids(session: ExecutionSession) -> tuple[str, ...]:
    step_ids: list[str] = []
    for record in session.replan_history:
        if record.failed_step is not None:
            step_ids.append(record.failed_step)
        if record.replacement_step_ids:
            step_ids.extend(record.replacement_step_ids)
            continue
        previous_ids = {step.id for step in record.previous_plan.ordered_steps}
        step_ids.extend(
            step.id
            for step in record.revised_plan.ordered_steps
            if step.id not in previous_ids
        )
    return _unique(step_ids)


def _recovery_types(
    session: ExecutionSession,
    retry_count: int,
) -> tuple[str, ...]:
    recovery_types: list[str] = []
    if retry_count > 0:
        recovery_types.append("retry")
    recovery_types.extend(
        f"replan:{record.reason.value}" for record in session.replan_history
    )
    return _unique(recovery_types)


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer.")
    if limit < 0:
        raise ValueError("limit cannot be negative.")


def _normalize_objective(objective: str) -> str:
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("objective must be a non-empty string.")
    return " ".join(objective.casefold().split())


def _average(values: Iterable[float]) -> float:
    items = tuple(values)
    if not items:
        return 0.0
    return round(sum(items) / len(items), 3)


def _ratio(value: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(value / total, 4)


def _ordered_counts(values: Iterable[str]) -> Mapping[str, int]:
    counts = Counter(values)
    return {
        key: count
        for key, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))

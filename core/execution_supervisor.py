"""In-memory lifecycle supervision for Atlas executions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any

from core.structured_plan_replanner import ReplanRecord


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionState(str, Enum):
    """Closed lifecycle states for one supervised execution."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_CONFIRMATION = "waiting_confirmation"
    REPLANNING = "replanning"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class StepExecutionState(str, Enum):
    """Closed supervised states for one execution step."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    SUCCESS = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    INTERRUPTED = "interrupted"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


_TERMINAL_STATES = frozenset(
    {
        ExecutionState.FAILED,
        ExecutionState.COMPLETED,
        ExecutionState.CANCELLED,
    }
)
_ACTIVE_STATES = frozenset(
    {
        ExecutionState.PENDING,
        ExecutionState.RUNNING,
        ExecutionState.WAITING_CONFIRMATION,
        ExecutionState.REPLANNING,
        ExecutionState.INTERRUPTED,
    }
)


class ExecutionSupervisorError(RuntimeError):
    """Base exception raised by the execution supervisor."""


class ExecutionSessionNotFoundError(ExecutionSupervisorError):
    """Raised when a supervised execution session does not exist."""


class InvalidExecutionTransitionError(ExecutionSupervisorError):
    """Raised when a lifecycle transition is not allowed."""


class ExecutionSessionAlreadyExistsError(ExecutionSupervisorError):
    """Raised when a restored session would overwrite a live session."""


@dataclass(frozen=True, slots=True)
class ExecutionSupervisorEvent:
    """Structured in-memory event emitted by the execution supervisor."""

    session_id: str
    event_type: str
    state: ExecutionState
    timestamp: datetime
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True, slots=True)
class StepExecutionSnapshot:
    """Immutable state snapshot for one supervised execution step."""

    step_id: str
    state: StepExecutionState
    dependency_ids: tuple[str, ...] = ()
    error: str | None = None
    ready_since: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempt_count: int = 0
    max_attempts: int = 1
    is_critical: bool = False

    def __post_init__(self) -> None:
        if not self.step_id.strip():
            raise ValueError("step_id must be a non-empty string.")
        if not isinstance(self.state, StepExecutionState):
            raise TypeError("state must be a StepExecutionState.")
        if self.attempt_count < 0:
            raise ValueError("attempt_count cannot be negative.")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be greater than zero.")
        if self.finished_at is not None and self.started_at is None:
            raise ValueError("finished_at requires started_at.")
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at cannot be earlier than started_at.")
        object.__setattr__(self, "dependency_ids", tuple(self.dependency_ids))


@dataclass(frozen=True, slots=True)
class ExecutionSession:
    """Current in-memory lifecycle snapshot for one execution."""

    session_id: str
    plan: Any
    state: ExecutionState
    current_step: str | None
    started_at: datetime
    finished_at: datetime | None = None
    last_error: str | None = None
    results: Mapping[str, object] = field(default_factory=dict)
    events: tuple[ExecutionSupervisorEvent, ...] = ()
    original_plan: Any | None = None
    active_plan: Any | None = None
    replan_count: int = 0
    replan_history: tuple[ReplanRecord, ...] = ()
    step_states: Mapping[str, StepExecutionSnapshot] = field(default_factory=dict)
    active_batch_id: str | None = None
    active_step_ids: tuple[str, ...] = ()
    last_batch_result: Any | None = None
    batch_history: tuple[Any, ...] = ()
    last_priority_decision: Any | None = None
    priority_history: tuple[Any, ...] = ()
    max_priority_history_entries: int = 100
    execution_budget: Any | None = None
    budget_usage: Any | None = None
    last_resource_decision: Any | None = None
    resource_decision_history: tuple[Any, ...] = ()
    selected_resources_by_step: Mapping[str, str] = field(default_factory=dict)
    max_resource_decision_history_entries: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if not isinstance(self.state, ExecutionState):
            raise TypeError("state must be an ExecutionState.")
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
        object.__setattr__(
            self,
            "step_states",
            MappingProxyType(dict(self.step_states)),
        )
        object.__setattr__(
            self,
            "original_plan",
            self.plan if self.original_plan is None else self.original_plan,
        )
        object.__setattr__(
            self,
            "active_plan",
            self.plan if self.active_plan is None else self.active_plan,
        )
        object.__setattr__(self, "replan_history", tuple(self.replan_history))
        object.__setattr__(self, "active_step_ids", tuple(self.active_step_ids))
        object.__setattr__(self, "batch_history", tuple(self.batch_history))
        object.__setattr__(self, "priority_history", tuple(self.priority_history))
        object.__setattr__(
            self,
            "resource_decision_history",
            tuple(self.resource_decision_history),
        )
        object.__setattr__(
            self,
            "selected_resources_by_step",
            MappingProxyType(dict(self.selected_resources_by_step)),
        )
        if self.max_priority_history_entries < 1:
            raise ValueError("max_priority_history_entries must be greater than zero.")
        if self.max_resource_decision_history_entries < 1:
            raise ValueError("max_resource_decision_history_entries must be greater than zero.")
        if self.replan_count < 0:
            raise ValueError("replan_count cannot be negative.")
        if self.replan_count != len(self.replan_history):
            raise ValueError("replan_count must match replan_history length.")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be earlier than started_at.")

    @property
    def is_terminal(self) -> bool:
        """Return whether the session is in a terminal state."""
        return self.state in _TERMINAL_STATES

    @property
    def progress(self) -> float:
        """Return deterministic completion progress in the inclusive range 0..1."""
        if not self.step_states:
            return 1.0 if self.is_terminal else 0.0
        terminal = {
            StepExecutionState.COMPLETED,
            StepExecutionState.FAILED,
            StepExecutionState.BLOCKED,
            StepExecutionState.INTERRUPTED,
            StepExecutionState.SKIPPED,
            StepExecutionState.CANCELLED,
        }
        completed = sum(
            1 for snapshot in self.step_states.values() if snapshot.state in terminal
        )
        return completed / len(self.step_states)


@dataclass(frozen=True, slots=True)
class ExecutionOverview:
    """Immutable aggregate snapshot of known supervised executions."""

    total_sessions: int
    pending_sessions: int
    running_sessions: int
    waiting_confirmation_sessions: int
    replanning_sessions: int
    interrupted_sessions: int
    completed_sessions: int
    failed_sessions: int
    cancelled_sessions: int
    active_sessions: int
    terminal_sessions: int
    latest_session_id: str | None
    generated_at: datetime

    def __post_init__(self) -> None:
        counters = (
            self.total_sessions,
            self.pending_sessions,
            self.running_sessions,
            self.waiting_confirmation_sessions,
            self.replanning_sessions,
            self.interrupted_sessions,
            self.completed_sessions,
            self.failed_sessions,
            self.cancelled_sessions,
            self.active_sessions,
            self.terminal_sessions,
        )
        if any(counter < 0 for counter in counters):
            raise ValueError("Execution overview counters cannot be negative.")

        state_total = (
            self.pending_sessions
            + self.running_sessions
            + self.waiting_confirmation_sessions
            + self.replanning_sessions
            + self.interrupted_sessions
            + self.completed_sessions
            + self.failed_sessions
            + self.cancelled_sessions
        )
        if self.total_sessions != state_total:
            raise ValueError("Execution overview total_sessions invariant failed.")

        active_total = (
            self.pending_sessions
            + self.running_sessions
            + self.waiting_confirmation_sessions
            + self.replanning_sessions
            + self.interrupted_sessions
        )
        if self.active_sessions != active_total:
            raise ValueError("Execution overview active_sessions invariant failed.")

        terminal_total = (
            self.completed_sessions
            + self.failed_sessions
            + self.cancelled_sessions
        )
        if self.terminal_sessions != terminal_total:
            raise ValueError("Execution overview terminal_sessions invariant failed.")


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    """Final or live aggregate for one supervised execution."""

    session_id: str
    state: ExecutionState
    total_steps: int
    pending_steps: int
    running_steps: int
    successful_steps: int
    failed_steps: int
    retrying_steps: int
    cancelled_steps: int
    skipped_steps: int
    progress: float
    retry_count: int
    started_at: datetime
    finished_at: datetime | None
    duration_seconds: float
    errors: Mapping[str, str] = field(default_factory=dict)
    critical_failure_step: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be between zero and one.")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative.")
        object.__setattr__(self, "errors", MappingProxyType(dict(self.errors)))


class ExecutionSupervisor:
    """Supervise execution lifecycle transitions without executing tools."""

    _ALLOWED_TRANSITIONS: Mapping[ExecutionState, frozenset[ExecutionState]] = {
        ExecutionState.PENDING: frozenset(
            {
                ExecutionState.RUNNING,
                ExecutionState.INTERRUPTED,
            }
        ),
        ExecutionState.RUNNING: frozenset(
            {
                ExecutionState.WAITING_CONFIRMATION,
                ExecutionState.REPLANNING,
                ExecutionState.INTERRUPTED,
                ExecutionState.FAILED,
                ExecutionState.COMPLETED,
                ExecutionState.CANCELLED,
            }
        ),
        ExecutionState.WAITING_CONFIRMATION: frozenset(
            {
                ExecutionState.RUNNING,
                ExecutionState.CANCELLED,
            }
        ),
        ExecutionState.INTERRUPTED: frozenset(
            {
                ExecutionState.RUNNING,
                ExecutionState.WAITING_CONFIRMATION,
                ExecutionState.FAILED,
                ExecutionState.CANCELLED,
            }
        ),
        ExecutionState.REPLANNING: frozenset(
            {
                ExecutionState.RUNNING,
                ExecutionState.INTERRUPTED,
                ExecutionState.FAILED,
                ExecutionState.CANCELLED,
            }
        ),
        ExecutionState.FAILED: frozenset({ExecutionState.REPLANNING}),
        ExecutionState.COMPLETED: frozenset(),
        ExecutionState.CANCELLED: frozenset(),
    }

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        session_id_prefix: str = "execution.session",
        session_repository: Any | None = None,
    ) -> None:
        if not session_id_prefix.strip():
            raise ValueError("session_id_prefix must be a non-empty string.")
        self._clock = clock or _utc_now
        self._session_id_prefix = session_id_prefix
        self._counter = 0
        self._sessions: dict[str, ExecutionSession] = {}
        self._events: list[ExecutionSupervisorEvent] = []
        self._lock = RLock()
        self._session_repository = session_repository

    def start(self, plan: Any) -> ExecutionSession:
        """Create a pending supervised session for an execution plan."""
        with self._lock:
            self._counter += 1
            session_id = f"{self._session_id_prefix}.{self._counter:06d}"
            started_at = self._clock()
            session = ExecutionSession(
                session_id=session_id,
                plan=plan,
                state=ExecutionState.PENDING,
                current_step=None,
                started_at=started_at,
                step_states=self._initial_step_states(plan),
            )
            session = self._with_event(
                session,
                "execution_started",
                timestamp=started_at,
            )
            self._sessions[session_id] = session
        self._persist_session(session)
        return session

    def get_session(self, session_id: str) -> ExecutionSession:
        """Return the current immutable snapshot for a session."""
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise ExecutionSessionNotFoundError(
                    f"Execution session '{session_id}' was not found."
                ) from exc

    def get_overview(self) -> ExecutionOverview:
        """Return an immutable aggregate snapshot of all known sessions."""
        with self._lock:
            sessions = tuple(self._sessions.values())
        counts = {state: 0 for state in ExecutionState}
        for session in sessions:
            counts[session.state] += 1

        latest = self._latest_session(sessions)
        return ExecutionOverview(
            total_sessions=len(sessions),
            pending_sessions=counts[ExecutionState.PENDING],
            running_sessions=counts[ExecutionState.RUNNING],
            waiting_confirmation_sessions=counts[
                ExecutionState.WAITING_CONFIRMATION
            ],
            replanning_sessions=counts[ExecutionState.REPLANNING],
            interrupted_sessions=counts[ExecutionState.INTERRUPTED],
            completed_sessions=counts[ExecutionState.COMPLETED],
            failed_sessions=counts[ExecutionState.FAILED],
            cancelled_sessions=counts[ExecutionState.CANCELLED],
            active_sessions=sum(counts[state] for state in _ACTIVE_STATES),
            terminal_sessions=sum(counts[state] for state in _TERMINAL_STATES),
            latest_session_id=latest.session_id if latest is not None else None,
            generated_at=self._clock(),
        )

    def list_sessions(
        self,
        *,
        state: ExecutionState | None = None,
        limit: int | None = None,
        newest_first: bool = True,
    ) -> tuple[ExecutionSession, ...]:
        """Return immutable session snapshots, newest first by default.

        ``limit=0`` is explicit and returns an empty tuple.
        """
        if state is not None and not isinstance(state, ExecutionState):
            raise TypeError("state must be an ExecutionState or None.")
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool):
                raise TypeError("limit must be an integer or None.")
            if limit < 0:
                raise ValueError("limit cannot be negative.")

        with self._lock:
            sessions = tuple(self._sessions.values())
        if state is not None:
            sessions = tuple(session for session in sessions if session.state is state)

        ordered = tuple(
            sorted(
                sessions,
                key=self._session_sort_key,
                reverse=newest_first,
            )
        )
        if limit is None:
            return ordered
        return ordered[:limit]

    def mark_running(
        self,
        session_id: str,
        *,
        current_step: str | None = None,
    ) -> ExecutionSession:
        """Move a session into RUNNING state."""
        return self._transition(
            session_id,
            ExecutionState.RUNNING,
            event_type="execution_running",
            current_step=current_step,
        )

    def mark_waiting_confirmation(
        self,
        session_id: str,
        *,
        current_step: str | None = None,
    ) -> ExecutionSession:
        """Move a running session into WAITING_CONFIRMATION state."""
        return self._transition(
            session_id,
            ExecutionState.WAITING_CONFIRMATION,
            event_type="execution_waiting_confirmation",
            current_step=current_step,
        )

    def mark_step_ready(
        self,
        session_id: str,
        step_id: str,
        *,
        dependency_ids: tuple[str, ...] = (),
    ) -> ExecutionSession:
        """Record one step as ready without executing it."""
        return self._mark_step_state(
            session_id,
            step_id,
            StepExecutionState.READY,
            event_type="step_ready",
            dependency_ids=dependency_ids,
        )

    def mark_step_started(
        self,
        session_id: str,
        step_id: str,
        *,
        dependency_ids: tuple[str, ...] = (),
    ) -> ExecutionSession:
        """Record one step as running."""
        return self._mark_step_state(
            session_id,
            step_id,
            StepExecutionState.RUNNING,
            event_type="step_started",
            dependency_ids=dependency_ids,
            current_step=step_id,
        )

    def mark_step_retrying(
        self,
        session_id: str,
        step_id: str,
        *,
        attempt_number: int,
        max_attempts: int,
        dependency_ids: tuple[str, ...] = (),
        error: object | None = None,
    ) -> ExecutionSession:
        """Record an automatic retry already scheduled by the execution engine."""
        if attempt_number < 2:
            raise ValueError("attempt_number must be at least two for a retry.")
        if max_attempts < attempt_number:
            raise ValueError("max_attempts cannot be lower than attempt_number.")
        return self._mark_step_state(
            session_id,
            step_id,
            StepExecutionState.RETRYING,
            event_type="step_retrying",
            dependency_ids=dependency_ids,
            current_step=step_id,
            error=self._error_message(error) if error is not None else None,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
        )

    def mark_step_completed(
        self,
        session_id: str,
        step_id: str,
        *,
        dependency_ids: tuple[str, ...] = (),
    ) -> ExecutionSession:
        """Record one step as completed."""
        return self._mark_step_state(
            session_id,
            step_id,
            StepExecutionState.COMPLETED,
            event_type="step_completed",
            dependency_ids=dependency_ids,
            current_step=step_id,
        )

    def mark_step_failed(
        self,
        session_id: str,
        step_id: str,
        error: object,
        *,
        dependency_ids: tuple[str, ...] = (),
    ) -> ExecutionSession:
        """Record one step as failed."""
        return self._mark_step_state(
            session_id,
            step_id,
            StepExecutionState.FAILED,
            event_type="step_failed",
            dependency_ids=dependency_ids,
            current_step=step_id,
            error=self._error_message(error),
        )

    def mark_step_blocked(
        self,
        session_id: str,
        step_id: str,
        *,
        dependency_ids: tuple[str, ...] = (),
        error: object | None = None,
    ) -> ExecutionSession:
        """Record one step as blocked by dependencies."""
        return self._mark_step_state(
            session_id,
            step_id,
            StepExecutionState.BLOCKED,
            event_type="step_blocked",
            dependency_ids=dependency_ids,
            error=self._error_message(error) if error is not None else None,
        )

    def mark_step_cancelled(
        self,
        session_id: str,
        step_id: str,
        *,
        dependency_ids: tuple[str, ...] = (),
        error: object | None = None,
    ) -> ExecutionSession:
        """Record one step as cancelled."""
        return self._mark_step_state(
            session_id,
            step_id,
            StepExecutionState.CANCELLED,
            event_type="step_cancelled",
            dependency_ids=dependency_ids,
            error=self._error_message(error) if error is not None else None,
        )

    def mark_step_interrupted(
        self,
        session_id: str,
        step_id: str,
        *,
        dependency_ids: tuple[str, ...] = (),
        error: object | None = None,
    ) -> ExecutionSession:
        """Record one ambiguous step after process recovery."""
        return self._mark_step_state(
            session_id,
            step_id,
            StepExecutionState.INTERRUPTED,
            event_type="step_interrupted",
            dependency_ids=dependency_ids,
            error=self._error_message(error) if error is not None else None,
        )

    def mark_step_skipped(
        self,
        session_id: str,
        step_id: str,
        *,
        dependency_ids: tuple[str, ...] = (),
    ) -> ExecutionSession:
        """Record one conditionally omitted step."""
        return self._mark_step_state(
            session_id,
            step_id,
            StepExecutionState.SKIPPED,
            event_type="step_skipped",
            dependency_ids=dependency_ids,
        )

    def get_summary(self, session_id: str) -> ExecutionSummary:
        """Return a deterministic live or final summary for one session."""
        session = self.get_session(session_id)
        snapshots = tuple(session.step_states.values())
        counts = {state: 0 for state in StepExecutionState}
        for snapshot in snapshots:
            counts[snapshot.state] += 1
        end = session.finished_at or self._clock()
        errors = {
            snapshot.step_id: snapshot.error
            for snapshot in snapshots
            if snapshot.error is not None
        }
        critical_failure = next(
            (
                snapshot.step_id
                for snapshot in snapshots
                if snapshot.is_critical
                and snapshot.state is StepExecutionState.FAILED
            ),
            None,
        )
        return ExecutionSummary(
            session_id=session.session_id,
            state=session.state,
            total_steps=len(snapshots),
            pending_steps=counts[StepExecutionState.PENDING]
            + counts[StepExecutionState.READY],
            running_steps=counts[StepExecutionState.RUNNING],
            successful_steps=counts[StepExecutionState.COMPLETED],
            failed_steps=counts[StepExecutionState.FAILED]
            + counts[StepExecutionState.BLOCKED]
            + counts[StepExecutionState.INTERRUPTED],
            retrying_steps=counts[StepExecutionState.RETRYING],
            cancelled_steps=counts[StepExecutionState.CANCELLED],
            skipped_steps=counts[StepExecutionState.SKIPPED],
            progress=session.progress,
            retry_count=sum(
                max(0, snapshot.attempt_count - 1) for snapshot in snapshots
            ),
            started_at=session.started_at,
            finished_at=session.finished_at,
            duration_seconds=max(0.0, (end - session.started_at).total_seconds()),
            errors=errors,
            critical_failure_step=critical_failure,
        )

    def generate_summary(self, session_id: str) -> ExecutionSummary:
        """Compatibility-friendly verb for callers generating a final report."""
        return self.get_summary(session_id)

    def mark_interrupted(
        self,
        session_id: str,
        *,
        current_step: str | None = None,
        error: object | None = None,
    ) -> ExecutionSession:
        """Move an active restored session into INTERRUPTED state."""
        details = {"error": self._error_message(error)} if error is not None else None
        return self._transition(
            session_id,
            ExecutionState.INTERRUPTED,
            event_type="execution_interrupted_detected",
            current_step=current_step,
            details=details,
        )

    def restore_session(self, session: ExecutionSession) -> ExecutionSession:
        """Register a validated restored session without normal execution events."""
        if not isinstance(session, ExecutionSession):
            raise TypeError("session must be an ExecutionSession.")
        with self._lock:
            if session.session_id in self._sessions:
                raise ExecutionSessionAlreadyExistsError(
                    f"Execution session '{session.session_id}' already exists."
                )
            restored = self._with_event(
                session,
                "execution_session_restored",
                details={
                    "restored_state": session.state.value,
                    "schema_version": 1,
                },
            )
            self._sessions[session.session_id] = restored
        self._persist_session(restored)
        return restored

    def record_dependency_graph_validated(
        self,
        session_id: str,
        *,
        step_count: int,
        dependency_count: int,
    ) -> ExecutionSession:
        """Record a validated dependency graph event."""
        return self._record_graph_event(
            session_id,
            "dependency_graph_validated",
            step_count=step_count,
            dependency_count=dependency_count,
        )

    def record_dependency_graph_rejected(
        self,
        session_id: str,
        *,
        error: object,
    ) -> ExecutionSession:
        """Record a rejected dependency graph event."""
        return self._record_graph_event(
            session_id,
            "dependency_graph_rejected",
            error=self._error_message(error),
        )

    def record_execution_blocked(
        self,
        session_id: str,
        *,
        error: object,
    ) -> ExecutionSession:
        """Record an execution blocked event."""
        return self._record_graph_event(
            session_id,
            "execution_blocked",
            error=self._error_message(error),
        )

    def record_execution_batch_created(
        self,
        session_id: str,
        batch: Any,
    ) -> ExecutionSession:
        """Record that a deterministic execution batch was created."""
        step_ids = tuple(getattr(batch, "step_ids", ()))
        details = {
            "batch_id": getattr(batch, "batch_id", None),
            "step_ids": list(step_ids),
            "concurrency_limit": getattr(batch, "concurrency_limit", None),
        }
        return self._record_batch_event(
            session_id,
            "execution_batch_created",
            active_batch_id=getattr(batch, "batch_id", None),
            active_step_ids=step_ids,
            **details,
        )

    def mark_execution_batch_started(
        self,
        session_id: str,
        batch: Any,
    ) -> ExecutionSession:
        """Record that a selected execution batch started."""
        step_ids = tuple(getattr(batch, "step_ids", ()))
        return self._record_batch_event(
            session_id,
            "execution_batch_started",
            active_batch_id=getattr(batch, "batch_id", None),
            active_step_ids=step_ids,
            batch_id=getattr(batch, "batch_id", None),
            step_ids=list(step_ids),
        )

    def record_execution_batch_result(
        self,
        session_id: str,
        result: Any,
    ) -> ExecutionSession:
        """Record the terminal result of one execution batch."""
        details = {
            "batch_id": getattr(result, "batch_id", None),
            "completed_step_ids": list(getattr(result, "completed_step_ids", ())),
            "failed_step_ids": list(getattr(result, "failed_step_ids", ())),
            "cancelled_step_ids": list(getattr(result, "cancelled_step_ids", ())),
            "fail_fast_triggered": bool(getattr(result, "fail_fast_triggered", False)),
        }
        return self._record_batch_event(
            session_id,
            "execution_batch_result_recorded",
            active_batch_id=None,
            active_step_ids=(),
            last_batch_result=result,
            append_batch_result=result,
            **details,
        )

    def record_resource_conflict_detected(
        self,
        session_id: str,
        *,
        step_id: str,
        resource_keys: tuple[str, ...],
    ) -> ExecutionSession:
        """Record a resource conflict that kept a step out of a batch."""
        return self._record_graph_event(
            session_id,
            "execution_resource_conflict_detected",
            step_id=step_id,
            resource_keys=list(resource_keys),
        )

    def record_concurrency_limit_applied(
        self,
        session_id: str,
        *,
        max_concurrency: int,
        selected_step_count: int,
    ) -> ExecutionSession:
        """Record that the configured concurrency bound was applied."""
        return self._record_graph_event(
            session_id,
            "execution_concurrency_limit_applied",
            max_concurrency=max_concurrency,
            selected_step_count=selected_step_count,
        )

    def record_fail_fast_triggered(
        self,
        session_id: str,
        *,
        batch_id: str,
    ) -> ExecutionSession:
        """Record that batch fail-fast stopped pending work."""
        return self._record_graph_event(
            session_id,
            "execution_fail_fast_triggered",
            batch_id=batch_id,
        )

    def record_priority_decision(
        self,
        session_id: str,
        decision: Any,
    ) -> ExecutionSession:
        """Record a ready-step priority decision without computing it."""
        ordered_step_ids = tuple(getattr(decision, "ordered_step_ids", ()))
        selected_step_ids = tuple(getattr(decision, "selected_step_ids", ()))
        scores = tuple(getattr(decision, "scores", ()))
        details = {
            "step_ids": list(ordered_step_ids),
            "selected_step_ids": list(selected_step_ids),
            "policy_name": getattr(decision, "policy_name", None),
            "score": (
                getattr(scores[0], "final_score", None)
                if scores
                else None
            ),
            "reason": getattr(decision, "rationale_summary", None),
        }
        event_type = (
            "priority_policy_disabled"
            if getattr(decision, "rationale_summary", None) == "priority policy disabled"
            else "ready_steps_prioritized"
        )
        with self._lock:
            session = self.get_session(session_id)
            history = session.priority_history + (decision,)
            history = history[-session.max_priority_history_entries :]
            updated = replace(
                session,
                last_priority_decision=decision,
                priority_history=history,
            )
            updated = self._with_event(updated, event_type, details=details)
            if selected_step_ids:
                updated = self._with_event(
                    updated,
                    "priority_step_selected"
                    if len(selected_step_ids) == 1
                    else "priority_batch_selected",
                    details=details,
                )
            if getattr(decision, "tie_breaker_used", None):
                updated = self._with_event(
                    updated,
                    "priority_tie_resolved",
                    details={
                        "step_ids": list(ordered_step_ids),
                        "selected_step_ids": list(selected_step_ids),
                        "policy_name": getattr(decision, "policy_name", None),
                        "reason": getattr(decision, "tie_breaker_used", None),
                    },
                )
            self._sessions[session_id] = updated
        self._persist_session(updated)
        return updated

    def record_resource_decision(
        self,
        session_id: str,
        decision: Any,
        *,
        budget_usage: Any | None = None,
    ) -> ExecutionSession:
        """Record a resource selection decision without computing it."""
        step_id = getattr(decision, "step_id", None)
        resource_id = getattr(decision, "selected_resource_id", None)
        provider_id = getattr(decision, "provider_id", None)
        details = {
            "step_id": step_id,
            "resource_id": resource_id,
            "provider_id": provider_id,
            "decision_reason": getattr(getattr(decision, "reason", None), "value", None),
            "estimated_cost": getattr(decision, "estimated_cost", None),
            "remaining_budget": getattr(budget_usage, "remaining_cost", None),
        }
        with self._lock:
            session = self.get_session(session_id)
            history = session.resource_decision_history + (decision,)
            history = history[-session.max_resource_decision_history_entries :]
            selected = dict(session.selected_resources_by_step)
            if isinstance(step_id, str) and isinstance(resource_id, str):
                selected[step_id] = resource_id
            updated = replace(
                session,
                last_resource_decision=decision,
                resource_decision_history=history,
                selected_resources_by_step=selected,
                budget_usage=budget_usage
                if budget_usage is not None
                else session.budget_usage,
            )
            event_type = (
                "resource_selection_failed"
                if resource_id is None
                else "resource_selected"
            )
            updated = self._with_event(updated, event_type, details=details)
            if resource_id is not None:
                updated = self._with_event(
                    updated,
                    "model_selected_for_step",
                    details=details,
                )
            if getattr(decision, "degradation_applied", False):
                updated = self._with_event(
                    updated,
                    "resource_degradation_applied",
                    details=details,
                )
            if getattr(decision, "tie_breaker_used", None):
                updated = self._with_event(
                    updated,
                    "resource_tie_resolved",
                    details=details,
                )
            self._sessions[session_id] = updated
        self._persist_session(updated)
        return updated

    def record_budget_usage(
        self,
        session_id: str,
        budget_usage: Any,
        *,
        event_type: str = "execution_budget_consumed",
    ) -> ExecutionSession:
        """Record a budget usage snapshot."""
        with self._lock:
            session = self.get_session(session_id)
            updated = replace(session, budget_usage=budget_usage)
            updated = self._with_event(
                updated,
                event_type,
                details={
                    "estimated_cost": getattr(budget_usage, "estimated_cost", None),
                    "actual_cost": getattr(budget_usage, "actual_cost", None),
                    "estimated_tokens": getattr(budget_usage, "estimated_tokens", None),
                    "actual_tokens": getattr(budget_usage, "actual_tokens", None),
                    "remaining_budget": getattr(budget_usage, "remaining_cost", None),
                },
            )
            self._sessions[session_id] = updated
        self._persist_session(updated)
        return updated

    def mark_completed(
        self,
        session_id: str,
        *,
        results: Mapping[str, object] | None = None,
        current_step: str | None = None,
    ) -> ExecutionSession:
        """Move a running session into COMPLETED state."""
        return self._transition(
            session_id,
            ExecutionState.COMPLETED,
            event_type="execution_completed",
            current_step=current_step,
            results=results,
            finished_at=self._clock(),
        )

    def mark_replanning(
        self,
        session_id: str,
        *,
        attempt_number: int,
        current_step: str | None = None,
        reason: str | None = None,
    ) -> ExecutionSession:
        """Move a failed or running session into REPLANNING state."""
        if attempt_number < 1:
            raise ValueError("attempt_number must be greater than zero.")
        return self._transition(
            session_id,
            ExecutionState.REPLANNING,
            event_type="replan_started",
            current_step=current_step,
            finished_at=None,
            details={
                "attempt_number": attempt_number,
                "reason": reason or "recoverable_failure",
            },
        )

    def record_replan(
        self,
        session_id: str,
        record: ReplanRecord,
    ) -> ExecutionSession:
        """Attach one accepted replan record and make its plan active."""
        if not isinstance(record, ReplanRecord):
            raise TypeError("record must be a ReplanRecord.")
        with self._lock:
            session = self.get_session(session_id)
            if session.state is not ExecutionState.REPLANNING:
                raise InvalidExecutionTransitionError(
                    "Replan records can only be added while session "
                    f"'{session_id}' is replanning."
                )
            updated = replace(
                session,
                plan=record.revised_plan,
                active_plan=record.revised_plan,
                replan_count=session.replan_count + 1,
                replan_history=session.replan_history + (record,),
            )
            updated = self._with_event(
                updated,
                "replan_produced",
                details={
                    "attempt_number": record.attempt_number,
                    "failed_step": record.failed_step,
                    "reason": record.reason.value,
                    "previous_plan": f"attempt.{record.attempt_number - 1}",
                    "revised_plan": f"attempt.{record.attempt_number}",
                },
            )
            self._sessions[session_id] = updated
        self._persist_session(updated)
        return updated

    def record_replan_event(
        self,
        session_id: str,
        event_type: str,
        *,
        attempt_number: int,
        failed_step: str | None = None,
        reason: str | None = None,
        error: str | None = None,
    ) -> ExecutionSession:
        """Append a structured replanning event without changing state."""
        if attempt_number < 1:
            raise ValueError("attempt_number must be greater than zero.")
        with self._lock:
            session = self.get_session(session_id)
            details: dict[str, object] = {"attempt_number": attempt_number}
            if failed_step is not None:
                details["failed_step"] = failed_step
            if reason is not None:
                details["reason"] = reason
            if error is not None:
                details["error"] = error
            updated = self._with_event(session, event_type, details=details)
            self._sessions[session_id] = updated
        self._persist_session(updated)
        return updated

    def mark_failed(
        self,
        session_id: str,
        error: object,
        *,
        current_step: str | None = None,
        results: Mapping[str, object] | None = None,
    ) -> ExecutionSession:
        """Move a running session into FAILED state and store the error."""
        error_message = self._error_message(error)
        return self._transition(
            session_id,
            ExecutionState.FAILED,
            event_type="execution_failed",
            current_step=current_step,
            last_error=error_message,
            results=results,
            finished_at=self._clock(),
            details={"error": error_message},
        )

    def mark_cancelled(
        self,
        session_id: str,
        *,
        error: object | None = None,
        current_step: str | None = None,
        results: Mapping[str, object] | None = None,
    ) -> ExecutionSession:
        """Move a running or waiting session into CANCELLED state."""
        error_message = self._error_message(error) if error is not None else None
        details = {"error": error_message} if error_message is not None else None
        return self._transition(
            session_id,
            ExecutionState.CANCELLED,
            event_type="execution_cancelled",
            current_step=current_step,
            last_error=error_message,
            results=results,
            finished_at=self._clock(),
            details=details,
        )

    @property
    def events(self) -> tuple[ExecutionSupervisorEvent, ...]:
        """Return structured lifecycle events emitted by the supervisor."""
        with self._lock:
            return tuple(self._events)

    def _transition(
        self,
        session_id: str,
        target_state: ExecutionState,
        *,
        event_type: str,
        current_step: str | None = None,
        last_error: str | None = None,
        results: Mapping[str, object] | None = None,
        finished_at: datetime | None = None,
        details: Mapping[str, object] | None = None,
    ) -> ExecutionSession:
        with self._lock:
            session = self.get_session(session_id)
            allowed = self._ALLOWED_TRANSITIONS[session.state]
            if target_state not in allowed:
                raise InvalidExecutionTransitionError(
                    "Invalid execution transition "
                    f"{session.state.value} -> {target_state.value} "
                    f"for session '{session_id}'."
                )

            updated_results: Mapping[str, object] = (
                session.results if results is None else results
            )
            updated = replace(
                session,
                state=target_state,
                current_step=current_step
                if current_step is not None
                else session.current_step,
                finished_at=finished_at,
                last_error=last_error if last_error is not None else session.last_error,
                results=updated_results,
            )
            updated = self._with_event(
                updated,
                event_type,
                details=details,
            )
            self._sessions[session_id] = updated
        self._persist_session(updated)
        return updated

    def _with_event(
        self,
        session: ExecutionSession,
        event_type: str,
        *,
        timestamp: datetime | None = None,
        details: Mapping[str, object] | None = None,
    ) -> ExecutionSession:
        event = ExecutionSupervisorEvent(
            session_id=session.session_id,
            event_type=event_type,
            state=session.state,
            timestamp=timestamp or self._clock(),
            details={} if details is None else details,
        )
        self._events.append(event)
        return replace(session, events=session.events + (event,))

    def _mark_step_state(
        self,
        session_id: str,
        step_id: str,
        state: StepExecutionState,
        *,
        event_type: str,
        dependency_ids: tuple[str, ...] = (),
        current_step: str | None = None,
        error: str | None = None,
        attempt_number: int | None = None,
        max_attempts: int | None = None,
    ) -> ExecutionSession:
        with self._lock:
            session = self.get_session(session_id)
            previous = session.step_states.get(step_id)
            timestamp = self._clock()
            started_at = previous.started_at if previous is not None else None
            finished_at = previous.finished_at if previous is not None else None
            attempts = previous.attempt_count if previous is not None else 0
            configured_attempts = previous.max_attempts if previous is not None else 1
            if state is StepExecutionState.RUNNING:
                started_at = started_at or timestamp
                finished_at = None
                attempts = max(1, attempts)
            elif state is StepExecutionState.RETRYING:
                started_at = started_at or timestamp
                finished_at = None
                attempts = attempt_number or max(2, attempts + 1)
                configured_attempts = max_attempts or max(configured_attempts, attempts)
            elif state in {
                StepExecutionState.COMPLETED,
                StepExecutionState.FAILED,
                StepExecutionState.BLOCKED,
                StepExecutionState.INTERRUPTED,
                StepExecutionState.SKIPPED,
                StepExecutionState.CANCELLED,
            }:
                started_at = started_at or timestamp
                finished_at = timestamp
            snapshot = StepExecutionSnapshot(
                step_id=step_id,
                state=state,
                dependency_ids=dependency_ids,
                error=error,
                ready_since=(
                    previous.ready_since
                    if previous is not None
                    else None
                ),
                started_at=started_at,
                finished_at=finished_at,
                attempt_count=attempts,
                max_attempts=configured_attempts,
                is_critical=previous.is_critical if previous is not None else False,
            )
            if state is StepExecutionState.READY and snapshot.ready_since is None:
                snapshot = replace(snapshot, ready_since=timestamp)
            step_states = dict(session.step_states)
            step_states[step_id] = snapshot
            critical_failure = (
                state is StepExecutionState.FAILED and snapshot.is_critical
            )
            if critical_failure:
                for pending_id, pending in tuple(step_states.items()):
                    if pending_id == step_id or pending.state not in {
                        StepExecutionState.PENDING,
                        StepExecutionState.READY,
                        StepExecutionState.RETRYING,
                    }:
                        continue
                    step_states[pending_id] = replace(
                        pending,
                        state=StepExecutionState.CANCELLED,
                        started_at=pending.started_at or timestamp,
                        finished_at=timestamp,
                        error=f"cancelled after critical step '{step_id}' failed",
                    )
            updated = replace(
                session,
                state=ExecutionState.CANCELLED if critical_failure else session.state,
                current_step=current_step
                if current_step is not None
                else session.current_step,
                finished_at=timestamp if critical_failure else session.finished_at,
                last_error=error if critical_failure else session.last_error,
                step_states=step_states,
            )
            details: dict[str, object] = {
                "step_id": step_id,
                "state": state.value,
                "dependency_ids": list(dependency_ids),
            }
            if error is not None:
                details["error"] = error
            if attempt_number is not None:
                details["attempt_number"] = attempt_number
            if max_attempts is not None:
                details["max_attempts"] = max_attempts
            updated = self._with_event(
                updated,
                event_type,
                timestamp=timestamp,
                details=details,
            )
            if critical_failure:
                updated = self._with_event(
                    updated,
                    "execution_cancelled_critical_step",
                    timestamp=timestamp,
                    details={"step_id": step_id, "error": error or "critical failure"},
                )
            self._sessions[session_id] = updated
        self._persist_session(updated)
        return updated

    def _record_graph_event(
        self,
        session_id: str,
        event_type: str,
        **details: object,
    ) -> ExecutionSession:
        with self._lock:
            session = self.get_session(session_id)
            updated = self._with_event(session, event_type, details=details)
            self._sessions[session_id] = updated
        self._persist_session(updated)
        return updated

    def _record_batch_event(
        self,
        session_id: str,
        event_type: str,
        *,
        active_batch_id: str | None,
        active_step_ids: tuple[str, ...],
        last_batch_result: Any | None = None,
        append_batch_result: Any | None = None,
        **details: object,
    ) -> ExecutionSession:
        with self._lock:
            session = self.get_session(session_id)
            history = session.batch_history
            if append_batch_result is not None:
                history = history + (append_batch_result,)
            updated = replace(
                session,
                active_batch_id=active_batch_id,
                active_step_ids=active_step_ids,
                last_batch_result=(
                    session.last_batch_result
                    if last_batch_result is None
                    else last_batch_result
                ),
                batch_history=history,
            )
            updated = self._with_event(updated, event_type, details=details)
            self._sessions[session_id] = updated
        self._persist_session(updated)
        return updated

    def _initial_step_states(
        self,
        plan: Any,
    ) -> Mapping[str, StepExecutionSnapshot]:
        steps = getattr(plan, "ordered_steps", ())
        snapshots: dict[str, StepExecutionSnapshot] = {}
        for step in steps:
            step_id = getattr(step, "id", None)
            if not isinstance(step_id, str) or not step_id.strip():
                continue
            snapshots[step_id] = StepExecutionSnapshot(
                step_id=step_id,
                state=StepExecutionState.PENDING,
                dependency_ids=tuple(getattr(step, "depends_on", ())),
                max_attempts=getattr(
                    getattr(step, "retry_policy", None),
                    "max_attempts",
                    1,
                ),
                is_critical=getattr(step, "criticality", 0) > 0,
            )
        return snapshots

    @staticmethod
    def _error_message(error: object) -> str:
        message = str(error).strip()
        return message or type(error).__name__

    @staticmethod
    def _session_sort_key(session: ExecutionSession) -> tuple[datetime, str]:
        return (session.started_at, session.session_id)

    def _latest_session(
        self,
        sessions: tuple[ExecutionSession, ...],
    ) -> ExecutionSession | None:
        if not sessions:
            return None
        return max(sessions, key=self._session_sort_key)

    def _persist_session(self, session: ExecutionSession) -> None:
        if self._session_repository is None:
            return
        from core.execution_session_persistence import ExecutionSessionSnapshot

        self._session_repository.save(ExecutionSessionSnapshot.from_session(session))

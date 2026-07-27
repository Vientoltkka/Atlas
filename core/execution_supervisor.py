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
    COMPLETED = "completed"
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

    def __post_init__(self) -> None:
        if not self.step_id.strip():
            raise ValueError("step_id must be a non-empty string.")
        if not isinstance(self.state, StepExecutionState):
            raise TypeError("state must be a StepExecutionState.")
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
    ) -> ExecutionSession:
        with self._lock:
            session = self.get_session(session_id)
            snapshot = StepExecutionSnapshot(
                step_id=step_id,
                state=state,
                dependency_ids=dependency_ids,
                error=error,
            )
            step_states = dict(session.step_states)
            step_states[step_id] = snapshot
            updated = replace(
                session,
                current_step=current_step
                if current_step is not None
                else session.current_step,
                step_states=step_states,
            )
            details: dict[str, object] = {
                "step_id": step_id,
                "state": state.value,
                "dependency_ids": list(dependency_ids),
            }
            if error is not None:
                details["error"] = error
            updated = self._with_event(updated, event_type, details=details)
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

"""In-memory lifecycle supervision for Atlas executions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionState(str, Enum):
    """Closed lifecycle states for one supervised execution."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_CONFIRMATION = "waiting_confirmation"
    FAILED = "failed"
    COMPLETED = "completed"
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
    }
)


class ExecutionSupervisorError(RuntimeError):
    """Base exception raised by the execution supervisor."""


class ExecutionSessionNotFoundError(ExecutionSupervisorError):
    """Raised when a supervised execution session does not exist."""


class InvalidExecutionTransitionError(ExecutionSupervisorError):
    """Raised when a lifecycle transition is not allowed."""


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

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if not isinstance(self.state, ExecutionState):
            raise TypeError("state must be an ExecutionState.")
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
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
        ExecutionState.PENDING: frozenset({ExecutionState.RUNNING}),
        ExecutionState.RUNNING: frozenset(
            {
                ExecutionState.WAITING_CONFIRMATION,
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
        ExecutionState.FAILED: frozenset(),
        ExecutionState.COMPLETED: frozenset(),
        ExecutionState.CANCELLED: frozenset(),
    }

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        session_id_prefix: str = "execution.session",
    ) -> None:
        if not session_id_prefix.strip():
            raise ValueError("session_id_prefix must be a non-empty string.")
        self._clock = clock or _utc_now
        self._session_id_prefix = session_id_prefix
        self._counter = 0
        self._sessions: dict[str, ExecutionSession] = {}
        self._events: list[ExecutionSupervisorEvent] = []

    def start(self, plan: Any) -> ExecutionSession:
        """Create a pending supervised session for an execution plan."""
        self._counter += 1
        session_id = f"{self._session_id_prefix}.{self._counter:06d}"
        started_at = self._clock()
        session = ExecutionSession(
            session_id=session_id,
            plan=plan,
            state=ExecutionState.PENDING,
            current_step=None,
            started_at=started_at,
        )
        session = self._with_event(
            session,
            "execution_started",
            timestamp=started_at,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ExecutionSession:
        """Return the current immutable snapshot for a session."""
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise ExecutionSessionNotFoundError(
                f"Execution session '{session_id}' was not found."
            ) from exc

    def get_overview(self) -> ExecutionOverview:
        """Return an immutable aggregate snapshot of all known sessions."""
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

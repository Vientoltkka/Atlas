"""Versioned persistence and conservative recovery for execution sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
from threading import Lock
from types import MappingProxyType
from typing import Any, Protocol

from core.concurrent_step_executor import (
    ConcurrentStepResult,
    ExecutionBatchResult,
)
from core.execution_dependency_resolver import ExecutionDependencyResolver
from core.execution_priority import (
    MAX_PRIORITY_HISTORY_ENTRIES,
    PriorityDecision,
    PriorityScore,
)
from core.execution_resources import (
    ExecutionBudget,
    ExecutionBudgetUsage,
    OptimizationGoal,
    ResourceScore,
    ResourceSelectionDecision,
    ResourceSelectionReason,
)
from core.execution_plan_topology import ExecutionPlanTopologyError
from core.execution_plan_validator import ExecutionPlanValidator
from core.execution_supervisor import (
    ExecutionSession,
    ExecutionState,
    ExecutionSupervisor,
    ExecutionSupervisorEvent,
    StepExecutionSnapshot,
    StepExecutionState,
)
from core.planner import ExecutionPlan
from core.resumable_execution_store import (
    ResumableExecutionStoreError,
    _dict_to_plan,
    _plan_to_dict,
)
from core.structured_plan_replanner import ReplanReason, ReplanRecord


SCHEMA_VERSION = 1
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_FILE_BYTES = 2_000_000
_TERMINAL_STATES = {
    ExecutionState.COMPLETED,
    ExecutionState.FAILED,
    ExecutionState.CANCELLED,
}
_ACTIVE_RESTART_STATES = {
    ExecutionState.PENDING,
    ExecutionState.RUNNING,
    ExecutionState.REPLANNING,
    ExecutionState.INTERRUPTED,
}
_SENSITIVE_KEYS = ("secret", "token", "password", "credential", "api_key")


class ExecutionPersistenceError(RuntimeError):
    """Base exception raised by execution session persistence."""


class ExecutionSerializationError(ExecutionPersistenceError):
    """Raised when a session cannot be serialized safely."""


class ExecutionSnapshotCorruptedError(ExecutionPersistenceError):
    """Raised when a persisted snapshot is malformed."""


class UnsupportedExecutionSnapshotVersion(ExecutionPersistenceError):
    """Raised when a snapshot version is unknown."""


class RecoveryDecisionType(str, Enum):
    """Closed recovery decisions for restored sessions."""

    RESUME_AUTOMATICALLY = "resume_automatically"
    REQUIRE_CONFIRMATION = "require_confirmation"
    REQUIRE_MANUAL_REVIEW = "require_manual_review"
    CANNOT_RESUME = "cannot_resume"
    ALREADY_TERMINAL = "already_terminal"


@dataclass(frozen=True, slots=True)
class PersistedError:
    """Safe, typed error representation for persistence."""

    error_type: str
    error_code: str | None
    message: str
    step_id: str | None
    recoverable: bool
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class PersistedStepResult:
    """Safe persisted representation of a step result."""

    step_id: str
    status: str
    value_type: str
    serializable_value: Any | None
    summary: str
    produced_at: datetime
    fully_restorable: bool = True


@dataclass(frozen=True, slots=True)
class ExecutionSessionSnapshot:
    """Immutable versioned snapshot for one supervised execution session."""

    schema_version: int
    session_id: str
    state: ExecutionState
    original_plan: ExecutionPlan
    active_plan: ExecutionPlan
    current_step: str | None
    started_at: datetime
    finished_at: datetime | None
    last_error: PersistedError | None
    results: MappingProxyType
    step_states: MappingProxyType
    replan_count: int
    replan_history: tuple[ReplanRecord, ...]
    active_batch_id: str | None
    active_step_ids: tuple[str, ...]
    batch_history: tuple[ExecutionBatchResult, ...]
    last_priority_decision: PriorityDecision | None
    priority_history: tuple[PriorityDecision, ...]
    execution_budget: ExecutionBudget | None
    budget_usage: ExecutionBudgetUsage | None
    last_resource_decision: ResourceSelectionDecision | None
    resource_decision_history: tuple[ResourceSelectionDecision, ...]
    selected_resources_by_step: MappingProxyType
    created_at: datetime
    updated_at: datetime
    recovery_metadata: MappingProxyType
    events: tuple[ExecutionSupervisorEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise UnsupportedExecutionSnapshotVersion(
                f"Unsupported execution snapshot version: {self.schema_version}."
            )
        _validate_session_id(self.session_id)
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
        object.__setattr__(
            self,
            "step_states",
            MappingProxyType(dict(self.step_states)),
        )
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
        object.__setattr__(
            self,
            "recovery_metadata",
            MappingProxyType(dict(self.recovery_metadata)),
        )
        object.__setattr__(self, "events", tuple(self.events))
        if self.replan_count != len(self.replan_history):
            raise ExecutionSnapshotCorruptedError(
                "replan_count must match replan_history length."
            )

    @property
    def completed_step_ids(self) -> tuple[str, ...]:
        return _step_ids_in_state(self.step_states, StepExecutionState.COMPLETED)

    @property
    def failed_step_ids(self) -> tuple[str, ...]:
        return _step_ids_in_state(self.step_states, StepExecutionState.FAILED)

    @property
    def blocked_step_ids(self) -> tuple[str, ...]:
        return _step_ids_in_state(self.step_states, StepExecutionState.BLOCKED)

    @property
    def cancelled_step_ids(self) -> tuple[str, ...]:
        return _step_ids_in_state(self.step_states, StepExecutionState.CANCELLED)

    @classmethod
    def from_session(
        cls,
        session: ExecutionSession,
        *,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        recovery_metadata: dict[str, object] | None = None,
    ) -> "ExecutionSessionSnapshot":
        now = updated_at or _utc_now()
        return cls(
            schema_version=SCHEMA_VERSION,
            session_id=session.session_id,
            state=session.state,
            original_plan=session.original_plan,
            active_plan=session.active_plan,
            current_step=session.current_step,
            started_at=session.started_at,
            finished_at=session.finished_at,
            last_error=_persisted_error(session.last_error, session.current_step),
            results=_safe_results(session.results),
            step_states=MappingProxyType(dict(session.step_states)),
            replan_count=session.replan_count,
            replan_history=session.replan_history,
            active_batch_id=session.active_batch_id,
            active_step_ids=session.active_step_ids,
            batch_history=tuple(
                item
                for item in session.batch_history
                if isinstance(item, ExecutionBatchResult)
            ),
            last_priority_decision=(
                session.last_priority_decision
                if isinstance(session.last_priority_decision, PriorityDecision)
                else None
            ),
            priority_history=tuple(
                item
                for item in session.priority_history
                if isinstance(item, PriorityDecision)
            ),
            execution_budget=(
                session.execution_budget
                if isinstance(session.execution_budget, ExecutionBudget)
                else None
            ),
            budget_usage=(
                session.budget_usage
                if isinstance(session.budget_usage, ExecutionBudgetUsage)
                else None
            ),
            last_resource_decision=(
                session.last_resource_decision
                if isinstance(session.last_resource_decision, ResourceSelectionDecision)
                else None
            ),
            resource_decision_history=tuple(
                item
                for item in session.resource_decision_history
                if isinstance(item, ResourceSelectionDecision)
            ),
            selected_resources_by_step=MappingProxyType(
                dict(session.selected_resources_by_step)
            ),
            created_at=created_at or session.started_at,
            updated_at=now,
            recovery_metadata=recovery_metadata or {},
            events=session.events,
        )

    def to_session(self) -> ExecutionSession:
        state = self.state
        step_states = dict(self.step_states)
        if state in {ExecutionState.RUNNING, ExecutionState.REPLANNING}:
            state = ExecutionState.INTERRUPTED
            step_states = {
                step_id: (
                    StepExecutionSnapshot(
                        step_id=snapshot.step_id,
                        state=StepExecutionState.INTERRUPTED,
                        dependency_ids=snapshot.dependency_ids,
                        error=snapshot.error or "interrupted during recovery",
                        ready_since=snapshot.ready_since,
                    )
                    if snapshot.state is StepExecutionState.RUNNING
                    else snapshot
                )
                for step_id, snapshot in step_states.items()
            }
        return ExecutionSession(
            session_id=self.session_id,
            plan=self.active_plan,
            state=state,
            current_step=self.current_step,
            started_at=self.started_at,
            finished_at=self.finished_at,
            last_error=self.last_error.message if self.last_error is not None else None,
            results={key: value.serializable_value for key, value in self.results.items()},
            original_plan=self.original_plan,
            active_plan=self.active_plan,
            replan_count=self.replan_count,
            replan_history=self.replan_history,
            step_states=step_states,
            active_batch_id=self.active_batch_id,
            active_step_ids=self.active_step_ids,
            batch_history=self.batch_history,
            last_priority_decision=self.last_priority_decision,
            priority_history=self.priority_history,
            max_priority_history_entries=MAX_PRIORITY_HISTORY_ENTRIES,
            execution_budget=self.execution_budget,
            budget_usage=self.budget_usage,
            last_resource_decision=self.last_resource_decision,
            resource_decision_history=self.resource_decision_history,
            selected_resources_by_step=self.selected_resources_by_step,
        )


class ExecutionSessionRepository(Protocol):
    """Persistence contract for supervised execution sessions."""

    def save(self, snapshot: ExecutionSessionSnapshot) -> None:
        """Persist one session snapshot."""

    def load(self, session_id: str) -> ExecutionSessionSnapshot | None:
        """Load one session snapshot if present."""

    def list(self) -> tuple[str, ...]:
        """Return persisted session identifiers."""

    def delete(self, session_id: str) -> None:
        """Delete one persisted session snapshot."""

    def exists(self, session_id: str) -> bool:
        """Return whether a persisted snapshot exists."""


class FileExecutionSessionRepository:
    """JSON-file repository with one atomically written file per session."""

    def __init__(
        self,
        root: Path,
        *,
        max_file_bytes: int = _MAX_FILE_BYTES,
    ) -> None:
        self._root = Path(root)
        self._max_file_bytes = max_file_bytes
        self._locks: dict[str, Lock] = {}
        self._locks_guard = Lock()

    @property
    def root(self) -> Path:
        """Return the configured repository root."""
        return self._root

    def save(self, snapshot: ExecutionSessionSnapshot) -> None:
        _validate_session_id(snapshot.session_id)
        payload = snapshot_to_dict(snapshot)
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ExecutionSerializationError(
                "Execution session snapshot is not JSON serializable."
            ) from error
        if len(encoded.encode("utf-8")) > self._max_file_bytes:
            raise ExecutionSerializationError(
                "Execution session snapshot exceeds the maximum allowed size."
            )

        path = self._path_for(snapshot.session_id)
        lock = self._lock_for(snapshot.session_id)
        with lock:
            temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
            except OSError as error:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise ExecutionPersistenceError(
                    "Could not save execution session snapshot."
                ) from error

    def load(self, session_id: str) -> ExecutionSessionSnapshot | None:
        path = self._path_for(session_id)
        if not path.exists():
            return None
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ExecutionPersistenceError(
                "Could not inspect execution session snapshot."
            ) from error
        if size > self._max_file_bytes:
            raise ExecutionSnapshotCorruptedError(
                "Execution session snapshot exceeds the maximum allowed size."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ExecutionSnapshotCorruptedError(
                "Execution session snapshot is not valid JSON."
            ) from error
        except OSError as error:
            raise ExecutionPersistenceError(
                "Could not load execution session snapshot."
            ) from error
        return snapshot_from_dict(payload)

    def list(self) -> tuple[str, ...]:
        if not self._root.exists():
            return ()
        session_ids = []
        for path in self._root.glob("*.json"):
            session_id = path.stem
            try:
                _validate_session_id(session_id)
            except ExecutionSnapshotCorruptedError:
                continue
            session_ids.append(session_id)
        return tuple(sorted(session_ids))

    def delete(self, session_id: str) -> None:
        path = self._path_for(session_id)
        if not path.exists():
            return
        try:
            path.unlink()
        except OSError as error:
            raise ExecutionPersistenceError(
                "Could not delete execution session snapshot."
            ) from error

    def exists(self, session_id: str) -> bool:
        return self._path_for(session_id).exists()

    def _path_for(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        return self._root / f"{session_id}.json"

    def _lock_for(self, session_id: str) -> Lock:
        with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = Lock()
                self._locks[session_id] = lock
            return lock


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Typed decision returned by recovery policy."""

    decision: RecoveryDecisionType
    session_id: str
    reason: str
    ambiguous_step_ids: tuple[str, ...] = ()


class ExecutionRecoveryPolicy:
    """Conservative policy for deciding whether a snapshot may resume."""

    def evaluate(self, snapshot: ExecutionSessionSnapshot) -> RecoveryDecision:
        if snapshot.state in _TERMINAL_STATES:
            return RecoveryDecision(
                RecoveryDecisionType.ALREADY_TERMINAL,
                snapshot.session_id,
                "session is terminal",
            )
        if snapshot.state is ExecutionState.WAITING_CONFIRMATION:
            return RecoveryDecision(
                RecoveryDecisionType.REQUIRE_CONFIRMATION,
                snapshot.session_id,
                "pending confirmation must be preserved",
            )
        running_steps = _step_ids_in_state(snapshot.step_states, StepExecutionState.RUNNING)
        interrupted_steps = _step_ids_in_state(
            snapshot.step_states,
            StepExecutionState.INTERRUPTED,
        )
        ambiguous = running_steps + interrupted_steps
        if snapshot.active_batch_id or snapshot.active_step_ids:
            return RecoveryDecision(
                RecoveryDecisionType.REQUIRE_MANUAL_REVIEW,
                snapshot.session_id,
                "active batch is ambiguous after restart",
                ambiguous_step_ids=tuple(dict.fromkeys(snapshot.active_step_ids + ambiguous)),
            )
        if ambiguous:
            return RecoveryDecision(
                RecoveryDecisionType.REQUIRE_MANUAL_REVIEW,
                snapshot.session_id,
                "running step is ambiguous after restart",
                ambiguous_step_ids=ambiguous,
            )
        if not _all_pending_steps_recovery_safe(snapshot):
            return RecoveryDecision(
                RecoveryDecisionType.REQUIRE_MANUAL_REVIEW,
                snapshot.session_id,
                "pending steps lack explicit recovery safety metadata",
            )
        return RecoveryDecision(
            RecoveryDecisionType.RESUME_AUTOMATICALLY,
            snapshot.session_id,
            "pending work is explicitly recovery safe",
        )


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """Report produced by loading and classifying persisted sessions."""

    discovered_session_ids: tuple[str, ...]
    restored_session_ids: tuple[str, ...]
    interrupted_session_ids: tuple[str, ...]
    terminal_session_ids: tuple[str, ...]
    corrupted_session_ids: tuple[str, ...]
    unsupported_session_ids: tuple[str, ...]
    decisions: MappingProxyType
    errors: MappingProxyType
    generated_at: datetime


class ExecutionRecoveryService:
    """Load, validate, classify and restore persisted sessions without executing."""

    def __init__(
        self,
        repository: ExecutionSessionRepository,
        supervisor: ExecutionSupervisor,
        *,
        policy: ExecutionRecoveryPolicy | None = None,
        validator: ExecutionPlanValidator | None = None,
        dependency_resolver: ExecutionDependencyResolver | None = None,
        clock: Any | None = None,
    ) -> None:
        self._repository = repository
        self._supervisor = supervisor
        self._policy = policy or ExecutionRecoveryPolicy()
        self._validator = validator or ExecutionPlanValidator()
        self._dependency_resolver = dependency_resolver or ExecutionDependencyResolver()
        self._clock = clock or _utc_now
        self._decisions: dict[str, RecoveryDecision] = {}

    def recover(self) -> RecoveryReport:
        discovered = self._repository.list()
        restored: list[str] = []
        interrupted: list[str] = []
        terminal: list[str] = []
        corrupted: list[str] = []
        unsupported: list[str] = []
        decisions: dict[str, RecoveryDecision] = {}
        errors: dict[str, str] = {}

        for session_id in discovered:
            try:
                snapshot = self._repository.load(session_id)
                if snapshot is None:
                    continue
                self._validate_snapshot(snapshot)
                restored_session = self._supervisor.restore_session(snapshot.to_session())
                restored.append(restored_session.session_id)
                decision = self._policy.evaluate(snapshot)
                decisions[session_id] = decision
                self._decisions[session_id] = decision
                if restored_session.state is ExecutionState.INTERRUPTED:
                    interrupted.append(session_id)
                if restored_session.state in _TERMINAL_STATES:
                    terminal.append(session_id)
            except UnsupportedExecutionSnapshotVersion as error:
                unsupported.append(session_id)
                errors[session_id] = str(error)
            except (
                ExecutionSnapshotCorruptedError,
                ExecutionPersistenceError,
                ResumableExecutionStoreError,
                ValueError,
                TypeError,
                ExecutionPlanTopologyError,
            ) as error:
                corrupted.append(session_id)
                errors[session_id] = str(error)

        return RecoveryReport(
            discovered_session_ids=discovered,
            restored_session_ids=tuple(restored),
            interrupted_session_ids=tuple(interrupted),
            terminal_session_ids=tuple(terminal),
            corrupted_session_ids=tuple(corrupted),
            unsupported_session_ids=tuple(unsupported),
            decisions=MappingProxyType(decisions),
            errors=MappingProxyType(errors),
            generated_at=self._clock(),
        )

    def decision_for(self, session_id: str) -> RecoveryDecision | None:
        return self._decisions.get(session_id)

    def _validate_snapshot(self, snapshot: ExecutionSessionSnapshot) -> None:
        validation = self._validator.validate(snapshot.active_plan)
        if not validation.is_valid:
            raise ExecutionSnapshotCorruptedError("active_plan is invalid.")
        self._dependency_resolver.resolve(
            snapshot.active_plan,
            completed_step_ids=snapshot.completed_step_ids,
            failed_step_ids=snapshot.failed_step_ids,
        )
        step_ids = {step.id for step in snapshot.active_plan.ordered_steps}
        if snapshot.current_step is not None and snapshot.current_step not in step_ids:
            raise ExecutionSnapshotCorruptedError("current_step is not in active_plan.")
        if not set(snapshot.step_states).issubset(step_ids):
            raise ExecutionSnapshotCorruptedError("step_states contain unknown step IDs.")
        if not set(snapshot.active_step_ids).issubset(step_ids):
            raise ExecutionSnapshotCorruptedError("active_step_ids contain unknown step IDs.")


def snapshot_to_dict(snapshot: ExecutionSessionSnapshot) -> dict[str, Any]:
    return {
        "schema_version": snapshot.schema_version,
        "session_id": snapshot.session_id,
        "state": snapshot.state.value,
        "original_plan": _plan_to_dict(snapshot.original_plan),
        "active_plan": _plan_to_dict(snapshot.active_plan),
        "current_step": snapshot.current_step,
        "started_at": _datetime_to_json(snapshot.started_at),
        "finished_at": _datetime_to_json(snapshot.finished_at),
        "last_error": _error_to_json(snapshot.last_error),
        "results": {
            key: _step_result_to_json(value)
            for key, value in snapshot.results.items()
        },
        "step_states": {
            key: _step_state_to_json(value)
            for key, value in snapshot.step_states.items()
        },
        "replan_count": snapshot.replan_count,
        "replan_history": [_replan_to_json(record) for record in snapshot.replan_history],
        "active_batch_id": snapshot.active_batch_id,
        "active_step_ids": list(snapshot.active_step_ids),
        "batch_history": [_batch_result_to_json(item) for item in snapshot.batch_history],
        "last_priority_decision": _priority_decision_to_json(
            snapshot.last_priority_decision
        ),
        "priority_history": [
            _priority_decision_to_json(item)
            for item in snapshot.priority_history
        ],
        "execution_budget": _budget_to_json(snapshot.execution_budget),
        "budget_usage": _budget_usage_to_json(snapshot.budget_usage),
        "last_resource_decision": _resource_decision_to_json(
            snapshot.last_resource_decision
        ),
        "resource_decision_history": [
            _resource_decision_to_json(item)
            for item in snapshot.resource_decision_history
        ],
        "selected_resources_by_step": dict(snapshot.selected_resources_by_step),
        "created_at": _datetime_to_json(snapshot.created_at),
        "updated_at": _datetime_to_json(snapshot.updated_at),
        "recovery_metadata": _safe_mapping(snapshot.recovery_metadata),
        "events": [_event_to_json(event) for event in snapshot.events],
        "completed_step_ids": list(snapshot.completed_step_ids),
        "failed_step_ids": list(snapshot.failed_step_ids),
        "blocked_step_ids": list(snapshot.blocked_step_ids),
        "cancelled_step_ids": list(snapshot.cancelled_step_ids),
    }


def snapshot_from_dict(payload: Any) -> ExecutionSessionSnapshot:
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Execution snapshot root must be an object.")
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise UnsupportedExecutionSnapshotVersion(
            f"Unsupported execution snapshot version: {schema_version}."
        )
    session_id = _required_str(payload, "session_id")
    state = _enum_value(ExecutionState, _required_str(payload, "state"), "state")
    step_states = {
        step_id: _step_state_from_json(item)
        for step_id, item in _required_dict(payload, "step_states").items()
    }
    return ExecutionSessionSnapshot(
        schema_version=schema_version,
        session_id=session_id,
        state=state,
        original_plan=_dict_to_plan(_required_dict(payload, "original_plan")),
        active_plan=_dict_to_plan(_required_dict(payload, "active_plan")),
        current_step=_optional_str(payload, "current_step"),
        started_at=_datetime_from_json(_required_str(payload, "started_at"), "started_at"),
        finished_at=_optional_datetime(payload, "finished_at"),
        last_error=_error_from_json(payload.get("last_error")),
        results=MappingProxyType(
            {
                key: _step_result_from_json(value)
                for key, value in _required_dict(payload, "results").items()
            }
        ),
        step_states=MappingProxyType(step_states),
        replan_count=_required_int(payload, "replan_count"),
        replan_history=tuple(
            _replan_from_json(item)
            for item in _required_list(payload, "replan_history")
        ),
        active_batch_id=_optional_str(payload, "active_batch_id"),
        active_step_ids=_str_tuple(payload, "active_step_ids"),
        batch_history=tuple(
            _batch_result_from_json(item)
            for item in _required_list(payload, "batch_history")
        ),
        last_priority_decision=_priority_decision_from_json(
            payload.get("last_priority_decision")
        ),
        priority_history=tuple(
            decision
            for decision in (
                _priority_decision_from_json(item)
                for item in payload.get("priority_history", [])
            )
            if decision is not None
        ),
        execution_budget=_budget_from_json(payload.get("execution_budget")),
        budget_usage=_budget_usage_from_json(payload.get("budget_usage")),
        last_resource_decision=_resource_decision_from_json(
            payload.get("last_resource_decision")
        ),
        resource_decision_history=tuple(
            decision
            for decision in (
                _resource_decision_from_json(item)
                for item in payload.get("resource_decision_history", [])
            )
            if decision is not None
        ),
        selected_resources_by_step=MappingProxyType(
            {
                str(key): str(value)
                for key, value in payload.get("selected_resources_by_step", {}).items()
                if isinstance(key, str) and isinstance(value, str)
            }
        ),
        created_at=_datetime_from_json(_required_str(payload, "created_at"), "created_at"),
        updated_at=_datetime_from_json(_required_str(payload, "updated_at"), "updated_at"),
        recovery_metadata=MappingProxyType(
            _safe_mapping(_required_dict(payload, "recovery_metadata"))
        ),
        events=tuple(_event_from_json(item) for item in payload.get("events", [])),
    )


def _all_pending_steps_recovery_safe(snapshot: ExecutionSessionSnapshot) -> bool:
    completed = set(snapshot.completed_step_ids)
    for step in snapshot.active_plan.ordered_steps:
        if step.id in completed:
            continue
        if not (
            getattr(step, "idempotent", False)
            and getattr(step, "recovery_safe", False)
            and getattr(step, "side_effect_free", False)
        ):
            return False
    return True


def _safe_results(results: Any) -> MappingProxyType:
    return MappingProxyType(
        {
            str(step_id): _safe_step_result(str(step_id), value)
            for step_id, value in dict(results).items()
        }
    )


def _safe_step_result(step_id: str, value: object) -> PersistedStepResult:
    produced_at = _utc_now()
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return PersistedStepResult(
            step_id=step_id,
            status="partial",
            value_type=type(value).__name__,
            serializable_value=None,
            summary=f"non-serializable result of type {type(value).__name__}",
            produced_at=produced_at,
            fully_restorable=False,
        )
    return PersistedStepResult(
        step_id=step_id,
        status="completed",
        value_type=type(value).__name__,
        serializable_value=value,
        summary=_safe_summary(value),
        produced_at=produced_at,
        fully_restorable=True,
    )


def _safe_summary(value: object) -> str:
    if value is None or isinstance(value, (str, int, float, bool)):
        return str(value)[:120]
    if isinstance(value, dict):
        return f"object with {len(value)} keys"
    if isinstance(value, (list, tuple)):
        return f"array with {len(value)} items"
    return type(value).__name__


def _persisted_error(message: str | None, step_id: str | None) -> PersistedError | None:
    if message is None:
        return None
    return PersistedError(
        error_type="ExecutionError",
        error_code=None,
        message=message[:500],
        step_id=step_id,
        recoverable=False,
        timestamp=_utc_now(),
    )


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise ExecutionSnapshotCorruptedError("Invalid execution session_id.")


def _step_ids_in_state(
    step_states: Any,
    state: StepExecutionState,
) -> tuple[str, ...]:
    return tuple(
        step_id
        for step_id, snapshot in step_states.items()
        if snapshot.state is state
    )


def _safe_mapping(mapping: Any) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in dict(mapping).items():
        if not isinstance(key, str) or _is_sensitive_key(key):
            continue
        if _json_safe(value):
            result[key] = value
    return result


def _json_safe(value: object) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return False
    return True


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SENSITIVE_KEYS)


def _step_state_to_json(snapshot: StepExecutionSnapshot) -> dict[str, Any]:
    return {
        "step_id": snapshot.step_id,
        "state": snapshot.state.value,
        "dependency_ids": list(snapshot.dependency_ids),
        "error": snapshot.error,
        "ready_since": _datetime_to_json(snapshot.ready_since),
    }


def _step_state_from_json(payload: Any) -> StepExecutionSnapshot:
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Step state snapshot must be an object.")
    return StepExecutionSnapshot(
        step_id=_required_str(payload, "step_id"),
        state=_enum_value(
            StepExecutionState,
            _required_str(payload, "state"),
            "step state",
        ),
        dependency_ids=_str_tuple(payload, "dependency_ids"),
        error=_optional_str(payload, "error"),
        ready_since=_optional_datetime(payload, "ready_since"),
    )


def _priority_decision_to_json(
    decision: PriorityDecision | None,
) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "ordered_step_ids": list(decision.ordered_step_ids),
        "scores": [_priority_score_to_json(score) for score in decision.scores],
        "selected_step_ids": list(decision.selected_step_ids),
        "policy_name": decision.policy_name,
        "generated_at": _datetime_to_json(decision.generated_at),
        "tie_breaker_used": decision.tie_breaker_used,
        "rationale_summary": decision.rationale_summary,
    }


def _priority_decision_from_json(payload: Any) -> PriorityDecision | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Priority decision must be an object.")
    return PriorityDecision(
        ordered_step_ids=_str_tuple(payload, "ordered_step_ids"),
        scores=tuple(
            _priority_score_from_json(item)
            for item in _required_list(payload, "scores")
        ),
        selected_step_ids=_str_tuple(payload, "selected_step_ids"),
        policy_name=_required_str(payload, "policy_name"),
        generated_at=_datetime_from_json(
            _required_str(payload, "generated_at"),
            "generated_at",
        ),
        tie_breaker_used=_optional_str(payload, "tie_breaker_used"),
        rationale_summary=_required_str(payload, "rationale_summary"),
    )


def _priority_score_to_json(score: PriorityScore) -> dict[str, Any]:
    return {
        "step_id": score.step_id,
        "declared_priority": score.declared_priority,
        "urgency_score": score.urgency_score,
        "criticality_score": score.criticality_score,
        "deadline_score": score.deadline_score,
        "dependency_impact_score": score.dependency_impact_score,
        "age_score": score.age_score,
        "cost_penalty": score.cost_penalty,
        "duration_penalty": score.duration_penalty,
        "risk_penalty": score.risk_penalty,
        "final_score": score.final_score,
    }


def _priority_score_from_json(payload: Any) -> PriorityScore:
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Priority score must be an object.")
    return PriorityScore(
        step_id=_required_str(payload, "step_id"),
        declared_priority=_required_int(payload, "declared_priority"),
        urgency_score=_required_number(payload, "urgency_score"),
        criticality_score=_required_number(payload, "criticality_score"),
        deadline_score=_required_number(payload, "deadline_score"),
        dependency_impact_score=_required_number(payload, "dependency_impact_score"),
        age_score=_required_number(payload, "age_score"),
        cost_penalty=_required_number(payload, "cost_penalty"),
        duration_penalty=_required_number(payload, "duration_penalty"),
        risk_penalty=_required_number(payload, "risk_penalty"),
        final_score=_required_number(payload, "final_score"),
    )


def _budget_to_json(budget: ExecutionBudget | None) -> dict[str, Any] | None:
    if budget is None:
        return None
    return {
        "max_total_cost": budget.max_total_cost,
        "max_tokens": budget.max_tokens,
        "max_duration_seconds": budget.max_duration_seconds,
        "max_remote_calls": budget.max_remote_calls,
        "max_model_calls": budget.max_model_calls,
        "max_tool_calls": budget.max_tool_calls,
        "max_replans": budget.max_replans,
        "reserved_cost": budget.reserved_cost,
        "reserved_tokens": budget.reserved_tokens,
        "currency_code": budget.currency_code,
        "hard_limit": budget.hard_limit,
    }


def _budget_from_json(payload: Any) -> ExecutionBudget | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Execution budget must be an object.")
    return ExecutionBudget(
        max_total_cost=payload.get("max_total_cost"),
        max_tokens=payload.get("max_tokens"),
        max_duration_seconds=payload.get("max_duration_seconds"),
        max_remote_calls=payload.get("max_remote_calls"),
        max_model_calls=payload.get("max_model_calls"),
        max_tool_calls=payload.get("max_tool_calls"),
        max_replans=payload.get("max_replans"),
        reserved_cost=float(payload.get("reserved_cost", 0.0)),
        reserved_tokens=int(payload.get("reserved_tokens", 0)),
        currency_code=_required_str(payload, "currency_code"),
        hard_limit=_required_bool(payload, "hard_limit"),
    )


def _budget_usage_to_json(usage: ExecutionBudgetUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "estimated_cost": usage.estimated_cost,
        "actual_cost": usage.actual_cost,
        "estimated_tokens": usage.estimated_tokens,
        "actual_tokens": usage.actual_tokens,
        "elapsed_duration": usage.elapsed_duration,
        "remote_calls": usage.remote_calls,
        "model_calls": usage.model_calls,
        "tool_calls": usage.tool_calls,
        "remaining_cost": usage.remaining_cost,
        "remaining_tokens": usage.remaining_tokens,
        "exhausted_limits": list(usage.exhausted_limits),
    }


def _budget_usage_from_json(payload: Any) -> ExecutionBudgetUsage | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Execution budget usage must be an object.")
    return ExecutionBudgetUsage(
        estimated_cost=_required_number(payload, "estimated_cost"),
        actual_cost=payload.get("actual_cost"),
        estimated_tokens=_required_int(payload, "estimated_tokens"),
        actual_tokens=payload.get("actual_tokens"),
        elapsed_duration=_required_number(payload, "elapsed_duration"),
        remote_calls=_required_int(payload, "remote_calls"),
        model_calls=_required_int(payload, "model_calls"),
        tool_calls=_required_int(payload, "tool_calls"),
        remaining_cost=payload.get("remaining_cost"),
        remaining_tokens=payload.get("remaining_tokens"),
        exhausted_limits=_str_tuple(payload, "exhausted_limits"),
    )


def _resource_decision_to_json(
    decision: ResourceSelectionDecision | None,
) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "step_id": decision.step_id,
        "selected_resource_id": decision.selected_resource_id,
        "provider_id": decision.provider_id,
        "scores": [_resource_score_to_json(score) for score in decision.scores],
        "rejected_candidate_ids": list(decision.rejected_candidate_ids),
        "reason": decision.reason.value,
        "optimization_goal": decision.optimization_goal.value,
        "degradation_applied": decision.degradation_applied,
        "estimated_cost": decision.estimated_cost,
        "estimated_tokens": decision.estimated_tokens,
        "budget_snapshot": _budget_usage_to_json(decision.budget_snapshot),
        "tie_breaker_used": decision.tie_breaker_used,
    }


def _resource_decision_from_json(payload: Any) -> ResourceSelectionDecision | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Resource decision must be an object.")
    return ResourceSelectionDecision(
        step_id=_required_str(payload, "step_id"),
        selected_resource_id=_optional_str(payload, "selected_resource_id"),
        provider_id=_optional_str(payload, "provider_id"),
        scores=tuple(
            _resource_score_from_json(item)
            for item in _required_list(payload, "scores")
        ),
        rejected_candidate_ids=_str_tuple(payload, "rejected_candidate_ids"),
        reason=ResourceSelectionReason(_required_str(payload, "reason")),
        optimization_goal=OptimizationGoal(_required_str(payload, "optimization_goal")),
        degradation_applied=_required_bool(payload, "degradation_applied"),
        estimated_cost=payload.get("estimated_cost"),
        estimated_tokens=payload.get("estimated_tokens"),
        budget_snapshot=_budget_usage_from_json(payload.get("budget_snapshot")),
        tie_breaker_used=_optional_str(payload, "tie_breaker_used"),
    )


def _resource_score_to_json(score: ResourceScore) -> dict[str, Any]:
    return {
        "resource_id": score.resource_id,
        "capability_score": score.capability_score,
        "quality_score": score.quality_score,
        "cost_score": score.cost_score,
        "latency_score": score.latency_score,
        "reliability_score": score.reliability_score,
        "privacy_score": score.privacy_score,
        "locality_score": score.locality_score,
        "availability_penalty": score.availability_penalty,
        "budget_penalty": score.budget_penalty,
        "final_score": score.final_score,
    }


def _resource_score_from_json(payload: Any) -> ResourceScore:
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Resource score must be an object.")
    return ResourceScore(
        resource_id=_required_str(payload, "resource_id"),
        capability_score=_required_number(payload, "capability_score"),
        quality_score=_required_number(payload, "quality_score"),
        cost_score=_required_number(payload, "cost_score"),
        latency_score=_required_number(payload, "latency_score"),
        reliability_score=_required_number(payload, "reliability_score"),
        privacy_score=_required_number(payload, "privacy_score"),
        locality_score=_required_number(payload, "locality_score"),
        availability_penalty=_required_number(payload, "availability_penalty"),
        budget_penalty=_required_number(payload, "budget_penalty"),
        final_score=_required_number(payload, "final_score"),
    )


def _step_result_to_json(result: PersistedStepResult) -> dict[str, Any]:
    return {
        "step_id": result.step_id,
        "status": result.status,
        "value_type": result.value_type,
        "serializable_value": result.serializable_value,
        "summary": result.summary,
        "produced_at": _datetime_to_json(result.produced_at),
        "fully_restorable": result.fully_restorable,
    }


def _step_result_from_json(payload: Any) -> PersistedStepResult:
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Persisted step result must be an object.")
    return PersistedStepResult(
        step_id=_required_str(payload, "step_id"),
        status=_required_str(payload, "status"),
        value_type=_required_str(payload, "value_type"),
        serializable_value=payload.get("serializable_value"),
        summary=_required_str(payload, "summary"),
        produced_at=_datetime_from_json(
            _required_str(payload, "produced_at"),
            "produced_at",
        ),
        fully_restorable=_required_bool(payload, "fully_restorable"),
    )


def _batch_result_to_json(result: ExecutionBatchResult) -> dict[str, Any]:
    return {
        "batch_id": result.batch_id,
        "step_results": [_concurrent_result_to_json(item) for item in result.step_results],
        "completed_step_ids": list(result.completed_step_ids),
        "failed_step_ids": list(result.failed_step_ids),
        "cancelled_step_ids": list(result.cancelled_step_ids),
        "duration_ms": result.duration_ms,
        "fail_fast_triggered": result.fail_fast_triggered,
    }


def _batch_result_from_json(payload: Any) -> ExecutionBatchResult:
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Batch result must be an object.")
    return ExecutionBatchResult(
        batch_id=_required_str(payload, "batch_id"),
        step_results=tuple(
            _concurrent_result_from_json(item)
            for item in _required_list(payload, "step_results")
        ),
        completed_step_ids=_str_tuple(payload, "completed_step_ids"),
        failed_step_ids=_str_tuple(payload, "failed_step_ids"),
        cancelled_step_ids=_str_tuple(payload, "cancelled_step_ids"),
        duration_ms=_required_int(payload, "duration_ms"),
        fail_fast_triggered=_required_bool(payload, "fail_fast_triggered"),
    )


def _concurrent_result_to_json(result: ConcurrentStepResult) -> dict[str, Any]:
    persisted = _safe_step_result(result.step_id, result.result)
    return {
        "step_id": result.step_id,
        "status": result.status,
        "result": _step_result_to_json(persisted),
        "error": result.error,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
    }


def _concurrent_result_from_json(payload: Any) -> ConcurrentStepResult:
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Concurrent step result must be an object.")
    result = _step_result_from_json(_required_dict(payload, "result"))
    return ConcurrentStepResult(
        step_id=_required_str(payload, "step_id"),
        status=_required_str(payload, "status"),
        result=result.serializable_value,
        error=_optional_str(payload, "error"),
        started_at=_optional_str(payload, "started_at"),
        finished_at=_optional_str(payload, "finished_at"),
    )


def _replan_to_json(record: ReplanRecord) -> dict[str, Any]:
    return {
        "attempt_number": record.attempt_number,
        "previous_plan": _plan_to_dict(record.previous_plan),
        "revised_plan": _plan_to_dict(record.revised_plan),
        "reason": record.reason.value,
        "failed_step": record.failed_step,
        "error": record.error[:500],
        "created_at": _datetime_to_json(record.created_at),
    }


def _replan_from_json(payload: Any) -> ReplanRecord:
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Replan record must be an object.")
    return ReplanRecord(
        attempt_number=_required_int(payload, "attempt_number"),
        previous_plan=_dict_to_plan(_required_dict(payload, "previous_plan")),
        revised_plan=_dict_to_plan(_required_dict(payload, "revised_plan")),
        reason=_enum_value(
            ReplanReason,
            _required_str(payload, "reason"),
            "replan reason",
        ),
        failed_step=_optional_str(payload, "failed_step"),
        error=_required_str(payload, "error"),
        created_at=_datetime_from_json(
            _required_str(payload, "created_at"),
            "created_at",
        ),
    )


def _event_to_json(event: ExecutionSupervisorEvent) -> dict[str, Any]:
    return {
        "session_id": event.session_id,
        "event_type": event.event_type,
        "state": event.state.value,
        "timestamp": _datetime_to_json(event.timestamp),
        "details": _safe_mapping(event.details),
    }


def _event_from_json(payload: Any) -> ExecutionSupervisorEvent:
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Execution event must be an object.")
    return ExecutionSupervisorEvent(
        session_id=_required_str(payload, "session_id"),
        event_type=_required_str(payload, "event_type"),
        state=_enum_value(ExecutionState, _required_str(payload, "state"), "event state"),
        timestamp=_datetime_from_json(_required_str(payload, "timestamp"), "timestamp"),
        details=_safe_mapping(_required_dict(payload, "details")),
    )


def _error_to_json(error: PersistedError | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "error_type": error.error_type,
        "error_code": error.error_code,
        "message": error.message,
        "step_id": error.step_id,
        "recoverable": error.recoverable,
        "timestamp": _datetime_to_json(error.timestamp),
    }


def _error_from_json(payload: Any) -> PersistedError | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ExecutionSnapshotCorruptedError("Persisted error must be an object.")
    return PersistedError(
        error_type=_required_str(payload, "error_type"),
        error_code=_optional_str(payload, "error_code"),
        message=_required_str(payload, "message"),
        step_id=_optional_str(payload, "step_id"),
        recoverable=_required_bool(payload, "recoverable"),
        timestamp=_datetime_from_json(_required_str(payload, "timestamp"), "timestamp"),
    )


def _datetime_to_json(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _optional_datetime(payload: dict[str, Any], key: str) -> datetime | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ExecutionSnapshotCorruptedError(f"{key} must be a timestamp or null.")
    return _datetime_from_json(value, key)


def _datetime_from_json(value: str, key: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ExecutionSnapshotCorruptedError(f"{key} is not a valid timestamp.") from error
    if parsed.tzinfo is None:
        raise ExecutionSnapshotCorruptedError(f"{key} must include timezone.")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _enum_value(enum_type: Any, raw: str, label: str) -> Any:
    try:
        return enum_type(raw)
    except ValueError as error:
        raise ExecutionSnapshotCorruptedError(f"Unknown {label}: {raw}.") from error


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ExecutionSnapshotCorruptedError(f"{key} must be an object.")
    return value


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ExecutionSnapshotCorruptedError(f"{key} must be a list.")
    return value


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ExecutionSnapshotCorruptedError(f"{key} must be a string.")
    return value


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ExecutionSnapshotCorruptedError(f"{key} must be a string or null.")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutionSnapshotCorruptedError(f"{key} must be an integer.")
    return value


def _required_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecutionSnapshotCorruptedError(f"{key} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ExecutionSnapshotCorruptedError(f"{key} must be finite.")
    return result


def _required_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise ExecutionSnapshotCorruptedError(f"{key} must be a boolean.")
    return value


def _str_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ExecutionSnapshotCorruptedError(f"{key} must be a list of strings.")
    return tuple(value)

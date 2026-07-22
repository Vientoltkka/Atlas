"""Live execution context for one Atlas execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType, ModuleType
from typing import Any, Mapping
from uuid import uuid4


_MISSING = object()


class ExecutionStepState(str, Enum):
    """Closed step states tracked by ExecutionContext."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ExecutionContextError(ValueError):
    """Base error for execution context failures."""


class InvalidExecutionContextValueError(ExecutionContextError):
    """Raised when a context value is unsafe or structurally invalid."""


class ExecutionStepStateTransitionError(ExecutionContextError):
    """Raised when a step state transition is invalid."""


class ExecutionResultNotFoundError(ExecutionContextError):
    """Raised when a required result is absent."""


class ExecutionContextRestoreError(ExecutionContextError):
    """Raised when a context snapshot cannot be restored."""


class ExecutionContextMismatchError(ExecutionContextError):
    """Raised when a context does not match the expected execution."""


@dataclass(frozen=True, slots=True)
class ExecutionContextSnapshot:
    """Immutable persistible snapshot of live execution context state."""

    execution_id: str
    results_by_step_id: Mapping[str, object] = field(default_factory=dict)
    step_states: Mapping[str, str] = field(default_factory=dict)
    current_step_id: str | None = None
    current_attempt: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_execution_id(self.execution_id)
        results = _copy_result_mapping(self.results_by_step_id, "snapshot.results")
        states = _normalize_step_states(self.step_states)
        if self.current_step_id is not None:
            _validate_step_id(self.current_step_id, "snapshot.current_step_id")
            if self.current_step_id not in states:
                raise ExecutionContextRestoreError(
                    f"execution_id={self.execution_id} step_id={self.current_step_id} "
                    "operation=snapshot reason=current step is unknown"
                )
        if self.current_attempt is not None:
            _validate_attempt(self.current_attempt, self.execution_id, "snapshot")
        metadata = _copy_result_mapping(self.metadata, "snapshot.metadata")
        object.__setattr__(self, "results_by_step_id", MappingProxyType(results))
        object.__setattr__(self, "step_states", MappingProxyType(states))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


class ExecutionContext:
    """Mutable state for one concrete execution."""

    def __init__(
        self,
        execution_id: str | None = None,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        active_id = execution_id or str(uuid4())
        _validate_execution_id(active_id)
        self._execution_id = active_id
        self._results_by_step_id: dict[str, object] = {}
        self._step_states: dict[str, str] = {}
        self._current_step_id: str | None = None
        self._current_attempt: int | None = None
        self._metadata = _copy_result_mapping(metadata or {}, "metadata")

    @property
    def execution_id(self) -> str:
        return self._execution_id

    @property
    def current_step_id(self) -> str | None:
        return self._current_step_id

    @property
    def current_attempt(self) -> int | None:
        return self._current_attempt

    @property
    def completed_step_ids(self) -> tuple[str, ...]:
        return tuple(
            step_id
            for step_id, state in self._step_states.items()
            if state == ExecutionStepState.SUCCESS.value
        )

    @property
    def failed_step_ids(self) -> tuple[str, ...]:
        return tuple(
            step_id
            for step_id, state in self._step_states.items()
            if state == ExecutionStepState.FAILED.value
        )

    @property
    def cancelled_step_ids(self) -> tuple[str, ...]:
        return tuple(
            step_id
            for step_id, state in self._step_states.items()
            if state == ExecutionStepState.CANCELLED.value
        )

    @property
    def step_states(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._step_states))

    def state_for_step(
        self,
        step_id: str,
    ) -> str:
        _validate_step_id(step_id, "state_for_step")
        return self._step_states.get(step_id, ExecutionStepState.PENDING.value)

    def set_result(
        self,
        step_id: str,
        value: object,
    ) -> None:
        _validate_step_id(step_id, "set_result")
        self._results_by_step_id[step_id] = _copy_result_value(
            value,
            f"result[{step_id}]",
        )

    def get_result(
        self,
        step_id: str,
        default: object = None,
    ) -> object:
        _validate_step_id(step_id, "get_result")
        if step_id not in self._results_by_step_id:
            return default
        return _copy_result_value(
            self._results_by_step_id[step_id],
            f"result[{step_id}]",
        )

    def require_result(
        self,
        step_id: str,
    ) -> object:
        _validate_step_id(step_id, "require_result")
        if step_id not in self._results_by_step_id:
            raise ExecutionResultNotFoundError(
                f"execution_id={self.execution_id} step_id={step_id} "
                "operation=require_result reason=result not found"
            )
        return self.get_result(step_id)

    def has_result(
        self,
        step_id: str,
    ) -> bool:
        _validate_step_id(step_id, "has_result")
        return step_id in self._results_by_step_id

    def results_snapshot(self) -> dict[str, object]:
        return _copy_result_mapping(
            self._results_by_step_id,
            "results_snapshot",
        )

    def mark_step_started(
        self,
        step_id: str,
        attempt: int,
    ) -> tuple[str, str]:
        _validate_step_id(step_id, "mark_step_started")
        _validate_attempt(attempt, self.execution_id, step_id)
        previous = self.state_for_step(step_id)
        if previous not in {
            ExecutionStepState.PENDING.value,
            ExecutionStepState.FAILED.value,
        }:
            raise ExecutionStepStateTransitionError(
                f"execution_id={self.execution_id} step_id={step_id} "
                f"operation=mark_step_started reason=cannot transition from {previous} to RUNNING"
            )
        self._step_states[step_id] = ExecutionStepState.RUNNING.value
        self._current_step_id = step_id
        self._current_attempt = attempt
        return previous, ExecutionStepState.RUNNING.value

    def mark_step_succeeded(
        self,
        step_id: str,
        result: object,
    ) -> tuple[str, str]:
        self._require_running(step_id, "mark_step_succeeded")
        previous = self.state_for_step(step_id)
        self.set_result(step_id, result)
        self._step_states[step_id] = ExecutionStepState.SUCCESS.value
        self._clear_current(step_id)
        return previous, ExecutionStepState.SUCCESS.value

    def mark_step_failed(
        self,
        step_id: str,
    ) -> tuple[str, str]:
        self._require_running(step_id, "mark_step_failed")
        previous = self.state_for_step(step_id)
        self._step_states[step_id] = ExecutionStepState.FAILED.value
        self._clear_current(step_id)
        return previous, ExecutionStepState.FAILED.value

    def mark_step_cancelled(
        self,
        step_id: str,
    ) -> tuple[str, str]:
        self._require_running(step_id, "mark_step_cancelled")
        previous = self.state_for_step(step_id)
        self._step_states[step_id] = ExecutionStepState.CANCELLED.value
        self._clear_current(step_id)
        return previous, ExecutionStepState.CANCELLED.value

    def snapshot(self) -> ExecutionContextSnapshot:
        return ExecutionContextSnapshot(
            execution_id=self.execution_id,
            results_by_step_id=self.results_snapshot(),
            step_states=dict(self._step_states),
            current_step_id=self.current_step_id,
            current_attempt=self.current_attempt,
            metadata=dict(self._metadata),
        )

    @classmethod
    def restore(
        cls,
        snapshot: ExecutionContextSnapshot,
    ) -> "ExecutionContext":
        context = cls(snapshot.execution_id, metadata=snapshot.metadata)
        context._results_by_step_id = _copy_result_mapping(
            snapshot.results_by_step_id,
            "restore.results",
        )
        context._step_states = _normalize_step_states(snapshot.step_states)
        context._current_step_id = snapshot.current_step_id
        context._current_attempt = snapshot.current_attempt
        return context

    def _require_running(
        self,
        step_id: str,
        operation: str,
    ) -> None:
        _validate_step_id(step_id, operation)
        previous = self.state_for_step(step_id)
        if previous != ExecutionStepState.RUNNING.value:
            raise ExecutionStepStateTransitionError(
                f"execution_id={self.execution_id} step_id={step_id} "
                f"operation={operation} reason=cannot transition from {previous}"
            )

    def _clear_current(
        self,
        step_id: str,
    ) -> None:
        if self._current_step_id == step_id:
            self._current_step_id = None
            self._current_attempt = None


def _validate_execution_id(
    execution_id: str,
) -> None:
    if not isinstance(execution_id, str) or not execution_id.strip():
        raise ExecutionContextRestoreError(
            "execution_id=<invalid> operation=validate reason=execution_id must be non-empty"
        )


def _validate_step_id(
    step_id: str,
    operation: str,
) -> None:
    if not isinstance(step_id, str) or not step_id.strip():
        raise InvalidExecutionContextValueError(
            f"execution_id=<unknown> step_id=<invalid> operation={operation} "
            "reason=step_id must be non-empty"
        )


def _validate_attempt(
    attempt: int,
    execution_id: str,
    step_id: str,
) -> None:
    if type(attempt) is not int or attempt < 1:
        raise InvalidExecutionContextValueError(
            f"execution_id={execution_id} step_id={step_id} "
            "operation=mark_step_started reason=attempt must be an integer >= 1"
        )


def _normalize_step_states(
    states: Mapping[str, object],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    valid_states = {state.value for state in ExecutionStepState}
    for step_id, state in states.items():
        _validate_step_id(step_id, "step_states")
        if not isinstance(state, str) or state not in valid_states:
            raise ExecutionContextRestoreError(
                f"execution_id=<unknown> step_id={step_id} "
                "operation=restore reason=invalid step state"
            )
        normalized[step_id] = state
    return normalized


def _copy_result_mapping(
    values: Mapping[str, object],
    path: str,
) -> dict[str, object]:
    copied: dict[str, object] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise InvalidExecutionContextValueError(
                f"execution_id=<unknown> step_id=<invalid> operation={path} "
                "reason=result keys must be non-empty strings"
            )
        copied[key] = _copy_result_value(value, f"{path}.{key}")
    return copied


def _copy_result_value(
    value: object,
    path: str,
) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidExecutionContextValueError(
                f"execution_id=<unknown> step_id=<unknown> operation={path} "
                "reason=non-finite float is not allowed"
            )
        return value

    if isinstance(value, Mapping):
        return {
            key: _copy_result_value(item, f"{path}.{key}")
            for key, item in value.items()
            if _validate_mapping_key(key, path)
        }

    if isinstance(value, (list, tuple)):
        return [
            _copy_result_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if isinstance(value, ModuleType):
        type_name = "module"
    elif callable(value):
        type_name = "callable"
    elif isinstance(value, type):
        type_name = "class"
    else:
        type_name = type(value).__name__

    raise InvalidExecutionContextValueError(
        f"execution_id=<unknown> step_id=<unknown> operation={path} "
        f"reason=unsupported result type {type_name}"
    )


def _validate_mapping_key(
    key: object,
    path: str,
) -> bool:
    if not isinstance(key, str) or not key:
        raise InvalidExecutionContextValueError(
            f"execution_id=<unknown> step_id=<unknown> operation={path} "
            "reason=mapping keys must be non-empty strings"
        )
    return True

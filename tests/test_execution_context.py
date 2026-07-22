from __future__ import annotations

import inspect
import math
from types import MappingProxyType

import pytest

from core.execution_context import (
    ExecutionContext,
    ExecutionContextSnapshot,
    ExecutionResultNotFoundError,
    ExecutionVariableNotFoundError,
    ExecutionStepState,
    ExecutionStepStateTransitionError,
    InvalidExecutionVariableValueError,
    InvalidExecutionContextValueError,
)
from core.execution_variable_reference import InvalidExecutionVariableNameError


def test_context_generates_id_and_starts_empty() -> None:
    context = ExecutionContext()

    assert context.execution_id
    assert context.current_step_id is None
    assert context.current_attempt is None
    assert context.results_snapshot() == {}
    assert context.variables_snapshot() == {}
    assert context.state_for_step("step_1") == ExecutionStepState.PENDING.value


def test_context_accepts_initial_variables_and_copies_them_defensively() -> None:
    initial = {"workspace_path": "C:/AI/Atlas", "config": {"limit": 20}}

    context = ExecutionContext("exec-1", initial_variables=initial)
    initial["config"]["limit"] = 99  # type: ignore[index]

    assert context.require_variable("workspace_path") == "C:/AI/Atlas"
    assert context.require_variable("config") == {"limit": 20}


def test_results_distinguish_none_from_missing_and_are_defensive_copies() -> None:
    context = ExecutionContext("exec-1")
    payload = {"items": [{"name": "alpha"}]}

    context.set_result("step_1", None)
    context.set_result("step_2", payload)
    payload["items"][0]["name"] = "changed"

    assert context.has_result("step_1") is True
    assert context.require_result("step_1") is None
    assert context.get_result("missing", "fallback") == "fallback"
    assert context.require_result("step_2") == {"items": [{"name": "alpha"}]}

    snapshot = context.results_snapshot()
    snapshot["step_2"]["items"][0]["name"] = "mutated"  # type: ignore[index]
    assert context.require_result("step_2") == {"items": [{"name": "alpha"}]}

    with pytest.raises(ExecutionResultNotFoundError):
        context.require_result("missing")


def test_result_values_reject_unsafe_objects_and_non_finite_floats() -> None:
    context = ExecutionContext("exec-1")

    for value in (object(), lambda: None, math.nan, math.inf):
        with pytest.raises(InvalidExecutionContextValueError):
            context.set_result("step_1", value)

    with pytest.raises(InvalidExecutionContextValueError):
        context.set_result("step_1", {"": "bad"})


def test_variable_api_replaces_deletes_and_distinguishes_none_from_missing() -> None:
    context = ExecutionContext("exec-1")
    payload = {"items": [{"name": "alpha"}]}

    context.set_variable("workspace_path", "C:/AI/Atlas")
    context.set_variable("workspace_path", "C:/AI/Atlas/work")
    context.set_variable("_temporary", None)
    context.set_variable("result_1", payload)
    payload["items"][0]["name"] = "changed"

    assert context.has_variable("workspace_path") is True
    assert context.require_variable("workspace_path") == "C:/AI/Atlas/work"
    assert context.has_variable("_temporary") is True
    assert context.require_variable("_temporary") is None
    assert context.get_variable("missing", "fallback") == "fallback"
    assert context.require_variable("result_1") == {"items": [{"name": "alpha"}]}
    assert context.delete_variable("workspace_path") is True
    assert context.delete_variable("workspace_path") is False
    assert context.has_variable("workspace_path") is False

    with pytest.raises(ExecutionVariableNotFoundError):
        context.require_variable("missing")


def test_variables_are_defensive_copies_and_contexts_are_isolated() -> None:
    first = ExecutionContext("exec-1")
    second = ExecutionContext("exec-2")
    first.set_variable("config", {"items": [{"name": "alpha"}]})
    second.set_variable("config", {"items": [{"name": "beta"}]})

    returned = first.require_variable("config")
    returned["items"][0]["name"] = "mutated"  # type: ignore[index]
    snapshot = first.variables_snapshot()
    snapshot["config"]["items"][0]["name"] = "snapshot"  # type: ignore[index]

    assert first.require_variable("config") == {"items": [{"name": "alpha"}]}
    assert second.require_variable("config") == {"items": [{"name": "beta"}]}


@pytest.mark.parametrize(
    "name",
    ["", "workspace path", "a.b", "1value", "$secret", "__class__"],
)
def test_variable_names_reject_unsafe_forms(name: str) -> None:
    context = ExecutionContext("exec-1")

    with pytest.raises(InvalidExecutionVariableNameError):
        context.set_variable(name, "value")


def test_variable_values_reject_unsafe_objects_and_non_finite_floats() -> None:
    context = ExecutionContext("exec-1")

    for value in (object(), lambda: None, math.nan, math.inf):
        with pytest.raises(InvalidExecutionVariableValueError):
            context.set_variable("config", value)


def test_step_state_transitions_are_explicit_and_retryable_after_failure() -> None:
    context = ExecutionContext("exec-1")

    assert context.mark_step_started("step_1", 1) == (
        ExecutionStepState.PENDING.value,
        ExecutionStepState.RUNNING.value,
    )
    assert context.current_step_id == "step_1"
    assert context.current_attempt == 1
    assert context.mark_step_failed("step_1") == (
        ExecutionStepState.RUNNING.value,
        ExecutionStepState.FAILED.value,
    )
    assert context.mark_step_started("step_1", 2) == (
        ExecutionStepState.FAILED.value,
        ExecutionStepState.RUNNING.value,
    )
    context.mark_step_succeeded("step_1", {"ok": True})

    assert context.completed_step_ids == ("step_1",)
    assert context.current_step_id is None
    assert context.current_attempt is None

    with pytest.raises(ExecutionStepStateTransitionError):
        context.mark_step_started("step_1", 3)


def test_invalid_attempts_and_terminal_transitions_are_rejected() -> None:
    context = ExecutionContext("exec-1")

    for attempt in (0, True):
        with pytest.raises(InvalidExecutionContextValueError):
            context.mark_step_started("step_1", attempt)  # type: ignore[arg-type]

    context.mark_step_started("step_2", 1)
    context.mark_step_cancelled("step_2")

    with pytest.raises(ExecutionStepStateTransitionError):
        context.mark_step_started("step_2", 2)


def test_snapshot_is_immutable_and_restore_preserves_state() -> None:
    context = ExecutionContext("exec-1", metadata={"source": "test"})
    context.mark_step_started("step_1", 1)
    context.mark_step_succeeded("step_1", {"items": [1, 2]})

    snapshot = context.snapshot()

    assert isinstance(snapshot.results_by_step_id, MappingProxyType)
    assert isinstance(snapshot.variables, MappingProxyType)
    assert snapshot.results_by_step_id == {"step_1": {"items": [1, 2]}}
    with pytest.raises(TypeError):
        snapshot.results_by_step_id["step_2"] = "bad"  # type: ignore[index]

    restored = ExecutionContext.restore(snapshot)
    assert restored.execution_id == "exec-1"
    assert restored.completed_step_ids == ("step_1",)
    assert restored.require_result("step_1") == {"items": [1, 2]}


def test_snapshot_includes_variables_and_restore_preserves_them() -> None:
    context = ExecutionContext(
        "exec-1",
        initial_variables={"workspace_path": "C:/AI/Atlas", "unicode": "Víctor"},
    )

    snapshot = context.snapshot()
    restored = ExecutionContext.restore(snapshot)

    assert snapshot.variables == {
        "workspace_path": "C:/AI/Atlas",
        "unicode": "Víctor",
    }
    assert restored.require_variable("workspace_path") == "C:/AI/Atlas"
    assert restored.require_variable("unicode") == "Víctor"


def test_snapshot_rejects_unknown_current_step() -> None:
    with pytest.raises(ValueError):
        ExecutionContextSnapshot(
            execution_id="exec-1",
            step_states={"step_1": ExecutionStepState.SUCCESS.value},
            current_step_id="missing",
        )


def test_context_source_has_no_dynamic_execution_or_service_dependencies() -> None:
    source = inspect.getsource(ExecutionContext)

    assert "eval(" not in source
    assert "exec(" not in source
    assert "os.environ" not in source
    assert "ToolRegistry" not in source
    assert "ToolExecutor" not in source

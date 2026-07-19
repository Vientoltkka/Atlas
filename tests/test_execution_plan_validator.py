from __future__ import annotations

from dataclasses import replace

from core.execution_plan_validator import (
    ExecutionPlanValidator,
    PlanValidationResult,
)
from core.planner import ExecutionPlan, ExecutionStep, Planner


def _valid_plan() -> ExecutionPlan:
    return Planner().create_execution_plan(
        "Lee README.md y copia su contenido en resumen.txt"
    )


def _step(
    step_id: str,
    tool: str = "read_file",
    dependencies: tuple[str, ...] = (),
    status: str = "pending",
) -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        description=f"Step {step_id}.",
        tool=tool,
        dependencies=dependencies,
        status=status,
    )


def _validate(plan: ExecutionPlan) -> PlanValidationResult:
    return ExecutionPlanValidator().validate(plan)


def test_valid_plan_returns_structured_valid_result() -> None:
    result = _validate(_valid_plan())

    assert result == PlanValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        requires_confirmation=True,
        status="valid",
    )


def test_empty_goal_invalidates_plan() -> None:
    result = _validate(replace(_valid_plan(), goal="   "))

    assert result.is_valid is False
    assert "Plan goal cannot be empty." in result.errors


def test_plan_without_steps_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(),
        estimated_steps=0,
        required_tools=(),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Plan must contain at least one step." in result.errors


def test_duplicate_step_ids_are_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step("step_1", tool="write_file", dependencies=("step_1",)),
        ),
        required_tools=("read_file", "write_file"),
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Duplicate step id: step_1." in result.errors


def test_unknown_dependency_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step("step_2", tool="write_file", dependencies=("missing",)),
        ),
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_2' depends on unknown step 'missing'." in result.errors


def test_self_dependency_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step("step_2", tool="write_file", dependencies=("step_2",)),
        ),
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_2' cannot depend on itself." in result.errors


def test_circular_dependency_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1", dependencies=("step_2",)),
            _step("step_2", tool="write_file", dependencies=("step_1",)),
        ),
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert any(error.startswith("Circular dependency detected:") for error in result.errors)


def test_invalid_dependency_order_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1", tool="write_file", dependencies=("step_2",)),
            _step("step_2", tool="read_file"),
        ),
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_1' depends on 'step_2' before it is executable." in result.errors


def test_incorrect_estimated_steps_is_invalid() -> None:
    result = _validate(replace(_valid_plan(), estimated_steps=99))

    assert result.is_valid is False
    assert (
        "Plan estimated_steps must match the real number of ordered steps."
        in result.errors
    )


def test_undeclared_tool_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step("step_2", tool="write_file", dependencies=("step_1",)),
        ),
        required_tools=("read_file",),
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_2' uses undeclared tool 'write_file'." in result.errors


def test_dangerous_plan_without_confirmation_is_invalid() -> None:
    plan = replace(_valid_plan(), requires_confirmation=False)

    result = _validate(plan)

    assert result.is_valid is False
    assert "Dangerous plan cannot be marked as not requiring confirmation." in result.errors


def test_non_blocking_warning_keeps_plan_valid() -> None:
    plan = replace(
        Planner().create_execution_plan("Lee el archivo README.md"),
        required_tools=("read_file", "directory.list"),
    )

    result = _validate(plan)

    assert result.is_valid is True
    assert result.errors == []
    assert (
        "Required tool 'directory.list' is declared but not used by any step."
        in result.warnings
    )


def test_malformed_tool_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(_step("step_1", tool="bad tool"),),
        estimated_steps=1,
        required_tools=("bad tool",),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Malformed required tool: bad tool." in result.errors
    assert "Malformed tool for step 'step_1': bad tool." in result.errors


def test_invalid_initial_statuses_are_invalid() -> None:
    plan = replace(
        _valid_plan(),
        status="running",
        ordered_steps=(
            _step("step_1", status="done"),
            _step("step_2", tool="write_file", dependencies=("step_1",)),
        ),
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Invalid initial plan status: running." in result.errors
    assert "Invalid initial status for step 'step_1': done." in result.errors


def test_validation_does_not_execute_or_modify_plan(tmp_path) -> None:
    target = tmp_path / "resumen.txt"
    plan = Planner().create_execution_plan(f"Escribe hola en {target}")
    before = repr(plan)

    result = _validate(plan)

    assert result.is_valid is True
    assert repr(plan) == before
    assert target.exists() is False

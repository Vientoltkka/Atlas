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
    tool: str | None = "read_file",
    dependencies: tuple[str, ...] = (),
    status: str = "pending",
    arguments: dict | None = None,
) -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        description=f"Step {step_id}.",
        tool=tool,
        dependencies=dependencies,
        status=status,
        arguments={} if arguments is None else arguments,
    )


def _validate(plan: ExecutionPlan) -> PlanValidationResult:
    return ExecutionPlanValidator().validate(plan)


def test_valid_plan_returns_structured_valid_result() -> None:
    result = _validate(_valid_plan())

    assert isinstance(result, PlanValidationResult)
    assert result.is_valid is True
    assert result.errors == []
    assert result.warnings == []
    assert result.requires_confirmation is True
    assert result.status == "valid"
    assert result.plan_signature


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


def test_step_arguments_with_supported_values_are_valid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                arguments={
                    "path": "README.md",
                    "count": 1,
                    "enabled": True,
                    "items": ["a", None],
                    "nested": {"mode": "safe"},
                },
            ),
        ),
        estimated_steps=1,
        required_tools=("read_file",),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is True
    assert result.plan_signature


def test_logical_step_allows_empty_arguments() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(_step("step_1", tool=None),),
        estimated_steps=1,
        required_tools=(),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is True


def test_logical_step_rejects_arguments() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(_step("step_1", tool=None, arguments={"path": "README.md"}),),
        estimated_steps=1,
        required_tools=(),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Logical step 'step_1' cannot declare arguments." in result.errors


def test_argument_keys_must_be_non_empty_strings() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1", arguments={"": "empty", 3: "number"}),
        ),
        estimated_steps=1,
        required_tools=("read_file",),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_1' argument keys cannot be empty." in result.errors
    assert "Step 'step_1' argument keys must be strings." in result.errors


def test_non_serializable_argument_value_is_rejected() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(_step("step_1", arguments={"value": object()}),),
        estimated_steps=1,
        required_tools=("read_file",),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert any(
        error.startswith(
            "Step 'step_1' arguments are not deterministically serializable:"
        )
        for error in result.errors
    )


def test_argument_key_order_does_not_change_plan_signature() -> None:
    base = _valid_plan()
    first = replace(
        base,
        ordered_steps=(
            _step("step_1", arguments={"a": 1, "b": 2}),
        ),
        estimated_steps=1,
        required_tools=("read_file",),
        requires_confirmation=False,
    )
    second = replace(
        base,
        ordered_steps=(
            _step("step_1", arguments={"b": 2, "a": 1}),
        ),
        estimated_steps=1,
        required_tools=("read_file",),
        requires_confirmation=False,
    )

    assert _validate(first).plan_signature == _validate(second).plan_signature


def test_different_arguments_change_plan_signature() -> None:
    base = _valid_plan()
    first = replace(
        base,
        ordered_steps=(_step("step_1", arguments={"path": "a.txt"}),),
        estimated_steps=1,
        required_tools=("read_file",),
        requires_confirmation=False,
    )
    second = replace(
        base,
        ordered_steps=(_step("step_1", arguments={"path": "b.txt"}),),
        estimated_steps=1,
        required_tools=("read_file",),
        requires_confirmation=False,
    )

    assert _validate(first).plan_signature != _validate(second).plan_signature


def test_static_reference_to_dependency_is_valid_and_signed_as_original() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1", arguments={"path": "README.md"}),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={
                    "content": {"$ref": "steps.step_1.output.content"},
                },
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is True
    assert result.plan_signature


def test_reference_object_with_extra_keys_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={
                    "content": {
                        "$ref": "steps.step_1.output.content",
                        "default": "",
                    },
                },
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_2' reference objects must contain only '$ref'." in result.errors


def test_reference_value_must_be_non_empty_string() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={"content": {"$ref": ""}},
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_2' reference value must be a non-empty string." in result.errors


def test_invalid_reference_syntax_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={"content": {"$ref": "step.step_1.output.content"}},
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert (
        "Step 'step_2' has invalid reference syntax: step.step_1.output.content."
        in result.errors
    )


def test_reference_to_unknown_step_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={"content": {"$ref": "steps.missing.output.content"}},
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_2' references unknown step 'missing'." in result.errors


def test_reference_to_self_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                arguments={"path": {"$ref": "steps.step_1.output.path"}},
            ),
        ),
        estimated_steps=1,
        required_tools=("read_file",),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_1' cannot reference itself." in result.errors


def test_reference_to_future_step_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                arguments={"path": {"$ref": "steps.step_2.output.path"}},
            ),
            _step("step_2", tool="write_file", dependencies=("step_1",)),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_1' references non-dependent step 'step_2'." in result.errors


def test_reference_to_non_dependent_step_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step("step_2", tool="read_file"),
            _step(
                "step_3",
                tool="write_file",
                dependencies=("step_2",),
                arguments={"content": {"$ref": "steps.step_1.output.content"}},
            ),
        ),
        estimated_steps=3,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_3' references non-dependent step 'step_1'." in result.errors


def test_reference_to_transitive_dependency_is_valid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step("step_2", tool="read_file", dependencies=("step_1",)),
            _step(
                "step_3",
                tool="write_file",
                dependencies=("step_2",),
                arguments={"content": {"$ref": "steps.step_1.output.content"}},
            ),
        ),
        estimated_steps=3,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is True


def test_reference_blocks_private_and_dunder_path_segments() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={
                    "a": {"$ref": "steps.step_1.output._secret"},
                    "b": {"$ref": "steps.step_1.output.__class__"},
                },
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_2' has unsafe reference path segment: _secret." in result.errors
    assert "Step 'step_2' has unsafe reference path segment: __class__." in result.errors

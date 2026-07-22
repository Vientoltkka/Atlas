from __future__ import annotations

from dataclasses import replace

from core.execution_plan_validator import (
    ExecutionPlanValidator,
    PlanValidationResult,
    plan_signature,
)
from core.execution_condition import (
    AllOfCondition,
    AnyOfCondition,
    ExecutionCondition,
    ExecutionConditionOperator,
    NotCondition,
)
from core.execution_variable_binding import ExecutionVariableBinding
from core.execution_variable_reference import ExecutionVariableReference
from core.parameter_resolver import MAX_TEMPLATE_LENGTH, MAX_TEMPLATE_REFERENCES
from core.planner import ExecutionPlan, ExecutionStep, Planner
from core.step_output_reference import StepOutputReference


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
    output_binding: ExecutionVariableBinding | None = None,
    condition: ExecutionCondition | None = None,
) -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        description=f"Step {step_id}.",
        tool=tool,
        dependencies=dependencies,
        status=status,
        arguments={} if arguments is None else arguments,
        output_binding=output_binding,
        condition=condition,
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


def test_validator_accepts_valid_execution_variable_reference() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                arguments={"path": ExecutionVariableReference("workspace_path")},
            ),
        ),
        required_tools=("read_file",),
        estimated_steps=1,
        detected_risks=(),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is True
    assert result.errors == []


def test_plan_signature_changes_when_step_condition_changes() -> None:
    base = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                condition=ExecutionCondition(True, ExecutionConditionOperator.TRUTHY),
            ),
        ),
        required_tools=("read_file",),
        estimated_steps=1,
        detected_risks=(),
        requires_confirmation=False,
    )
    changed = replace(
        base,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                condition=ExecutionCondition(False, ExecutionConditionOperator.TRUTHY),
            ),
        ),
    )

    assert plan_signature(base) != plan_signature(changed)


def test_validator_accepts_composite_condition() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                condition=AllOfCondition(
                    (
                        ExecutionCondition(True, ExecutionConditionOperator.TRUTHY),
                        NotCondition(ExecutionCondition(False, ExecutionConditionOperator.TRUTHY)),
                    )
                ),
            ),
        ),
        required_tools=("read_file",),
        estimated_steps=1,
        detected_risks=(),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is True
    assert result.errors == []


def test_validator_walks_short_circuitable_composite_nodes_statically() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1", "read_file"),
            _step(
                "step_2",
                "write_file",
                dependencies=("step_1",),
                condition=AllOfCondition(
                    (
                        ExecutionCondition(False, ExecutionConditionOperator.TRUTHY),
                        ExecutionCondition(
                            StepOutputReference("step_3"),
                            ExecutionConditionOperator.EXISTS,
                        ),
                    )
                ),
            ),
            _step("step_3", "read_file"),
        ),
        required_tools=("read_file", "write_file"),
        estimated_steps=3,
        detected_risks=(),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert any("future step 'step_3'" in error for error in result.errors)


def test_validator_rejects_self_reference_inside_composite_condition() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                condition=AnyOfCondition(
                    (
                        ExecutionCondition(True, ExecutionConditionOperator.TRUTHY),
                        ExecutionCondition(
                            StepOutputReference("step_1"),
                            ExecutionConditionOperator.EXISTS,
                        ),
                    )
                ),
            ),
        ),
        required_tools=("read_file",),
        estimated_steps=1,
        detected_risks=(),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert any("cannot reference itself" in error for error in result.errors)


def test_plan_signature_changes_for_composite_type_order_and_child() -> None:
    condition_a = ExecutionCondition("a", ExecutionConditionOperator.EQUALS, "a")
    condition_b = ExecutionCondition("b", ExecutionConditionOperator.EQUALS, "b")
    all_plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                condition=AllOfCondition((condition_a, condition_b)),
            ),
        ),
        required_tools=("read_file",),
        estimated_steps=1,
        detected_risks=(),
        requires_confirmation=False,
    )
    any_plan = replace(
        all_plan,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                condition=AnyOfCondition((condition_a, condition_b)),
            ),
        ),
    )
    reordered_plan = replace(
        all_plan,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                condition=AllOfCondition((condition_b, condition_a)),
            ),
        ),
    )
    changed_child_plan = replace(
        all_plan,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                condition=AllOfCondition(
                    (condition_a, ExecutionCondition("b", ExecutionConditionOperator.EQUALS, "c"))
                ),
            ),
        ),
    )
    same_plan = replace(
        all_plan,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                condition=AllOfCondition((condition_a, condition_b)),
            ),
        ),
    )

    assert plan_signature(all_plan) != plan_signature(any_plan)
    assert plan_signature(all_plan) != plan_signature(reordered_plan)
    assert plan_signature(all_plan) != plan_signature(changed_child_plan)
    assert plan_signature(all_plan) == plan_signature(same_plan)


def test_validator_accepts_valid_output_binding() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                arguments={"path": "README.md"},
                output_binding=ExecutionVariableBinding("workspace_path", ("path",)),
            ),
        ),
        required_tools=("read_file",),
        estimated_steps=1,
        detected_risks=(),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is True
    assert result.errors == []


def test_validator_rejects_invalid_output_binding_structure() -> None:
    try:
        ExecutionVariableBinding("workspace_path", (True,))
    except ValueError as error:
        assert "bool segments" in str(error)
    else:
        raise AssertionError("invalid output binding path must be rejected")


def test_validator_rejects_invalid_execution_variable_reference_structure() -> None:
    try:
        reference = ExecutionVariableReference("workspace_path", (True,))
    except ValueError as error:
        assert "bool segments" in str(error)
    else:
        raise AssertionError("invalid variable reference path must be rejected")


def test_plan_signature_includes_variable_reference_name_and_path_not_value() -> None:
    base = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                arguments={"path": ExecutionVariableReference("workspace_path")},
            ),
        ),
        required_tools=("read_file",),
        estimated_steps=1,
        detected_risks=(),
        requires_confirmation=False,
    )
    changed_name = replace(
        base,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                arguments={"path": ExecutionVariableReference("other_path")},
            ),
        ),
    )
    changed_path = replace(
        base,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                arguments={
                    "path": ExecutionVariableReference("workspace_path", ("root",))
                },
            ),
        ),
    )
    equivalent = replace(
        base,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                arguments={"path": ExecutionVariableReference.from_path("workspace_path")},
            ),
        ),
    )

    assert plan_signature(base) != plan_signature(changed_name)
    assert plan_signature(base) != plan_signature(changed_path)
    assert plan_signature(base) == plan_signature(equivalent)


def test_plan_signature_includes_output_binding_fields() -> None:
    base = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                arguments={"path": "README.md"},
                output_binding=ExecutionVariableBinding("selected_file"),
            ),
        ),
        required_tools=("read_file",),
        estimated_steps=1,
        detected_risks=(),
        requires_confirmation=False,
    )
    changed_name = replace(
        base,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                arguments={"path": "README.md"},
                output_binding=ExecutionVariableBinding("other_file"),
            ),
        ),
    )
    changed_path = replace(
        base,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                arguments={"path": "README.md"},
                output_binding=ExecutionVariableBinding("selected_file", ("path",)),
            ),
        ),
    )
    changed_overwrite = replace(
        base,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                arguments={"path": "README.md"},
                output_binding=ExecutionVariableBinding(
                    "selected_file",
                    overwrite=False,
                ),
            ),
        ),
    )
    equivalent = replace(
        base,
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                arguments={"path": "README.md"},
                output_binding=ExecutionVariableBinding.from_path("selected_file"),
            ),
        ),
    )

    assert plan_signature(base) != plan_signature(changed_name)
    assert plan_signature(base) != plan_signature(changed_path)
    assert plan_signature(base) != plan_signature(changed_overwrite)
    assert plan_signature(base) == plan_signature(equivalent)


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
    step = _step("step_1")
    object.__setattr__(step, "arguments", {"": "empty", 3: "number"})
    plan = replace(
        _valid_plan(),
        ordered_steps=(step,),
        estimated_steps=1,
        required_tools=("read_file",),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_1' argument keys cannot be empty." in result.errors
    assert "Step 'step_1' argument keys must be strings." in result.errors


def test_non_serializable_argument_value_is_rejected() -> None:
    step = _step("step_1")
    object.__setattr__(step, "arguments", {"value": object()})
    plan = replace(
        _valid_plan(),
        ordered_steps=(step,),
        estimated_steps=1,
        required_tools=("read_file",),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert any(
        error.startswith("Step 'step_1' arguments are invalid:")
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


def test_template_to_dependency_is_valid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1", arguments={"path": "README.md"}),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={
                    "content": {
                        "$template": "Archivo: {{steps.step_1.output.path}}"
                    },
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


def test_template_object_with_extra_keys_is_invalid() -> None:
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
                        "$template": "{{steps.step_1.output.path}}",
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
    assert "Step 'step_2' template objects must contain only '$template'." in result.errors


def test_template_value_must_be_string_static_validation() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={"content": {"$template": 123}},
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_2' template value must be a string." in result.errors


def test_template_invalid_braces_are_invalid() -> None:
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
                        "$template": "Archivo: {{steps.step_1.output.path"
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
    assert "Step 'step_2' has invalid template brace syntax." in result.errors


def test_template_invalid_reference_syntax_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={
                    "content": {"$template": "{{step.step_1.output.path}}"},
                },
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert (
        "Step 'step_2' has invalid template reference syntax: step.step_1.output.path."
        in result.errors
    )


def test_template_rejects_unsupported_expression() -> None:
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
                        "$template": "{{steps.step_1.output.path.upper()}}"
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
    assert (
        "Step 'step_2' has unsupported template expression: steps.step_1.output.path.upper()."
        in result.errors
    )


def test_template_reference_to_unknown_step_is_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={
                    "content": {"$template": "{{steps.missing.output.path}}"},
                },
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_2' references unknown step 'missing'." in result.errors


def test_template_reference_to_self_future_and_non_dependent_steps_are_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step(
                "step_1",
                arguments={"path": {"$template": "{{steps.step_1.output.path}}"}},
            ),
            _step("step_2", tool="read_file"),
            _step(
                "step_3",
                tool="write_file",
                dependencies=("step_2",),
                arguments={
                    "a": {"$template": "{{steps.step_4.output.path}}"},
                    "b": {"$template": "{{steps.step_1.output.path}}"},
                },
            ),
            _step("step_4", tool="read_file", dependencies=("step_3",)),
        ),
        estimated_steps=4,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_1' cannot reference itself." in result.errors
    assert "Step 'step_3' references non-dependent step 'step_4'." in result.errors
    assert "Step 'step_3' references non-dependent step 'step_1'." in result.errors


def test_template_blocks_private_and_sensitive_segments_static_validation() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={
                    "a": {"$template": "{{steps.step_1.output._secret}}"},
                    "b": {"$template": "{{steps.step_1.output.password}}"},
                    "c": {"$template": "{{steps.step_1.output.token}}"},
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
    assert "Step 'step_2' has unsafe reference path segment: password." in result.errors
    assert "Step 'step_2' has unsafe reference path segment: token." in result.errors


def test_template_limits_are_validated_statically() -> None:
    too_many_refs = "".join(
        "{{steps.step_1.output.path}}"
        for _ in range(MAX_TEMPLATE_REFERENCES + 1)
    )
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1"),
            _step(
                "step_2",
                tool="write_file",
                dependencies=("step_1",),
                arguments={
                    "a": {"$template": "a" * (MAX_TEMPLATE_LENGTH + 1)},
                    "b": {"$template": too_many_refs},
                },
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert (
        "Step 'step_2' template exceeds the maximum supported length."
        in result.errors
    )
    assert (
        "Step 'step_2' template exceeds the maximum number of references."
        in result.errors
    )


def test_template_escaped_braces_do_not_create_references() -> None:
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
                        "$template": "{{{{steps.missing.output.path}}}}"
                    },
                },
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is True


def test_step_output_reference_to_previous_step_requires_dependency() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("read", arguments={"path": "README.md"}),
            _step(
                "write",
                tool="write_file",
                arguments={"content": StepOutputReference("read")},
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        requires_confirmation=True,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert any("ImplicitStepDependencyError" in error for error in result.errors)


def test_step_output_reference_to_unknown_self_and_future_steps_are_invalid() -> None:
    plan = replace(
        _valid_plan(),
        ordered_steps=(
            _step("step_1", arguments={"a": StepOutputReference("missing")}),
            _step("step_2", arguments={"b": StepOutputReference("step_2")}),
            _step("step_3", arguments={"c": StepOutputReference("step_4")}),
            _step("step_4"),
        ),
        estimated_steps=4,
        required_tools=("read_file",),
        requires_confirmation=False,
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert "Step 'step_1' references unknown step 'missing'." in result.errors
    assert "Step 'step_2' cannot reference itself." in result.errors
    assert "Step 'step_3' references future step 'step_4'." in result.errors


def test_step_output_reference_path_changes_plan_signature() -> None:
    base = _valid_plan()
    first = replace(
        base,
        ordered_steps=(
            _step("read"),
            _step(
                "use",
                dependencies=("read",),
                arguments={"value": StepOutputReference("read", ("a",))},
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file",),
        requires_confirmation=False,
    )
    second = replace(
        first,
        ordered_steps=(
            _step("read"),
            _step(
                "use",
                dependencies=("read",),
                arguments={"value": StepOutputReference("read", ("b",))},
            ),
        ),
    )
    third = replace(
        first,
        ordered_steps=(
            _step("other"),
            _step(
                "use",
                dependencies=("other",),
                arguments={"value": StepOutputReference("other", ("a",))},
            ),
        ),
    )

    assert plan_signature(first) != plan_signature(second)
    assert plan_signature(first) != plan_signature(third)
    assert _validate(first).plan_signature == plan_signature(first)

from __future__ import annotations

from dataclasses import replace

from core.execution_plan_executor import (
    ExecutionPlanExecutor,
    PlanExecutionResult,
    PlanExecutionStatus,
)
from core.execution_plan_output import ExecutionPlanOutput
from core.execution_plan_validator import ExecutionPlanValidator
from core.execution_variable_binding import ExecutionVariableBinding
from core.goal_verifier import (
    GoalVerificationReason,
    GoalVerifier,
    OutputValidatorKind,
)
from core.planner import ExecutionPlan, ExecutionStep
from core.step_output_reference import StepOutputReference
from tools.base_tool import BaseTool
from tools.filesystem.list_directory_tool import ListDirectoryTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class StaticTool(BaseTool):
    def __init__(self, output: object) -> None:
        self._output = output

    @property
    def name(self) -> str:
        return "static_tool"

    @property
    def description(self) -> str:
        return "Return static output."

    def execute(self, context: ToolContext) -> object:
        del context
        return self._output


def _plan(
    *,
    output: object | None = None,
    required_outputs: tuple[str, ...] = (),
    output_validators: dict[str, tuple[str, ...]] | None = None,
) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Verify deterministic goal.",
        ordered_steps=(ExecutionStep("step_1", "Return output.", "static_tool"),),
        estimated_steps=1,
        required_tools=("static_tool",),
        detected_risks=(),
        requires_confirmation=False,
        output=output,
        required_outputs=required_outputs,
        output_validators={} if output_validators is None else output_validators,
    )


def _result(
    *,
    status: str = PlanExecutionStatus.COMPLETED.value,
    success: bool = True,
    output: object | None = None,
    error_code: str | None = None,
) -> PlanExecutionResult:
    return PlanExecutionResult(
        plan_status=status,
        success=success,
        completed=status == PlanExecutionStatus.COMPLETED.value and success,
        cancelled=status == PlanExecutionStatus.CANCELLED.value,
        failed=not success and status == PlanExecutionStatus.FAILED.value,
        blocked=status == PlanExecutionStatus.BLOCKED.value,
        error_code=error_code,
        output=output,
    )


def test_goal_satisfied_when_required_output_exists() -> None:
    plan = _plan(required_outputs=("directory_path", "entries"))
    result = GoalVerifier().verify(
        plan,
        _result(output={"directory_path": ".", "entries": ["README.md"]}),
    )

    assert result.satisfied is True
    assert result.reason is GoalVerificationReason.SUCCESS
    assert result.verified_outputs == ("directory_path", "entries")


def test_missing_required_output_fails_goal() -> None:
    plan = _plan(required_outputs=("directory_path", "entries"))
    result = GoalVerifier().verify(plan, _result(output={"directory_path": "."}))

    assert result.satisfied is False
    assert result.reason is GoalVerificationReason.MISSING_REQUIRED_OUTPUTS
    assert result.missing_outputs == ("entries",)


def test_null_output_fails_not_null_validator() -> None:
    plan = _plan(
        output_validators={"entries": (OutputValidatorKind.NOT_NULL.value,)},
    )
    result = GoalVerifier().verify(plan, _result(output={"entries": None}))

    assert result.reason is GoalVerificationReason.OUTPUT_VALIDATION_FAILED
    assert result.missing_outputs == ("entries",)


def test_empty_string_fails_non_empty_string_validator() -> None:
    plan = _plan(
        output_validators={"name": (OutputValidatorKind.NON_EMPTY_STRING.value,)},
    )
    result = GoalVerifier().verify(plan, _result(output={"name": ""}))

    assert result.reason is GoalVerificationReason.OUTPUT_VALIDATION_FAILED


def test_empty_collection_fails_non_empty_collection_validator() -> None:
    plan = _plan(
        output_validators={"entries": (OutputValidatorKind.NON_EMPTY_COLLECTION.value,)},
    )
    result = GoalVerifier().verify(plan, _result(output={"entries": []}))

    assert result.reason is GoalVerificationReason.OUTPUT_VALIDATION_FAILED


def test_plan_failed_cancelled_and_blocked_are_structured_reasons() -> None:
    plan = _plan()

    failed = GoalVerifier().verify(
        plan,
        _result(status=PlanExecutionStatus.FAILED.value, success=False),
    )
    cancelled = GoalVerifier().verify(
        plan,
        _result(status=PlanExecutionStatus.CANCELLED.value, success=False),
    )
    blocked = GoalVerifier().verify(
        plan,
        _result(status=PlanExecutionStatus.BLOCKED.value, success=False),
    )

    assert failed.reason is GoalVerificationReason.PLAN_FAILED
    assert cancelled.reason is GoalVerificationReason.PLAN_CANCELLED
    assert blocked.reason is GoalVerificationReason.PLAN_BLOCKED


def test_exists_non_empty_and_boolean_validators() -> None:
    plan = _plan(
        output_validators={
            "path": (OutputValidatorKind.EXISTS.value,),
            "entries": (OutputValidatorKind.NON_EMPTY.value,),
            "ok": (OutputValidatorKind.BOOLEAN_TRUE.value,),
        },
    )

    result = GoalVerifier().verify(
        plan,
        _result(output={"path": ".", "entries": ["a"], "ok": True}),
    )

    assert result.satisfied is True
    assert result.verified_outputs == ("path", "entries", "ok")


def test_valid_output_binding_is_verified_from_resolved_plan_output() -> None:
    plan = _plan(
        output={
            "entries": StepOutputReference("step_1"),
        },
        required_outputs=("entries",),
    )
    registry = ToolRegistry()
    registry.register(StaticTool(["README.md"]))

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
    )

    assert result.goal_verification_result is not None
    assert result.goal_verification_result.satisfied is True


def test_invalid_binding_fails_before_goal_satisfaction() -> None:
    plan = replace(
        _plan(required_outputs=("selected",)),
        ordered_steps=(
            ExecutionStep(
                "step_1",
                "Return output.",
                "static_tool",
                output_binding=ExecutionVariableBinding("selected", ("missing",)),
            ),
        ),
        output=ExecutionPlanOutput({"selected": StepOutputReference("step_1")}),
    )
    registry = ToolRegistry()
    registry.register(StaticTool({"value": "ok"}))

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
    )

    assert result.success is False
    assert result.goal_verification_result is not None
    assert result.goal_verification_result.reason is GoalVerificationReason.INVALID_OUTPUT_BINDING


def test_e2e_read_only_tool_satisfies_required_outputs(tmp_path) -> None:
    (tmp_path / "README.md").write_text("atlas", encoding="utf-8")
    plan = ExecutionPlan(
        goal="List a directory.",
        ordered_steps=(
            ExecutionStep(
                "step_1",
                "List directory.",
                "list_directory",
                arguments={"path": str(tmp_path)},
            ),
        ),
        estimated_steps=1,
        required_tools=("list_directory",),
        detected_risks=(),
        requires_confirmation=False,
        output={"entries": StepOutputReference("step_1")},
        required_outputs=("entries",),
        output_validators={"entries": (OutputValidatorKind.NON_EMPTY_COLLECTION.value,)},
    )
    registry = ToolRegistry()
    registry.register(ListDirectoryTool())

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
    )

    assert result.success is True
    assert result.output == {"entries": ["README.md"]}
    assert result.goal_verification_result is not None
    assert result.goal_verification_result.satisfied is True

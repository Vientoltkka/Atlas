from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from core.execution_condition import (
    ExecutionCondition,
    ExecutionConditionEvaluator,
    ExecutionConditionOperator,
)
from core.execution_context import ExecutionContext
from core.execution_metrics import ExecutionMetricsCalculator
from core.execution_plan_executor import (
    ExecutionControl,
    ExecutionErrorCode,
    ExecutionPlanExecutor,
    ResumableExecutionState,
    StepExecutionStatus,
)
from core.execution_plan_output import ExecutionPlanOutput
from core.execution_plan_validator import ExecutionPlanValidator, plan_signature
from core.execution_retry import RetryPolicy
from core.execution_variable_binding import ExecutionVariableBinding
from core.execution_variable_reference import ExecutionVariableReference
from core.planner import ExecutionBranch, ExecutionPlan, ExecutionStep
from core.resumable_execution_store import JsonResumableExecutionStore, ResumableExecutionStoreError
from core.step_output_reference import StepOutputReference
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class RecordingTool(BaseTool):
    def __init__(self, name: str, output: object = None, *, fail: bool = False) -> None:
        self._name = name
        self._output = output if output is not None else name
        self._fail = fail
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Test-only safe recording tool."

    def execute(self, context: ToolContext) -> object:
        del context
        self.calls += 1
        if self._fail:
            raise RuntimeError(f"{self._name} failed")
        return self._output


class FailOnceTool(RecordingTool):
    def execute(self, context: ToolContext) -> object:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient")
        return self._output


class CountingConditionEvaluator(ExecutionConditionEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def evaluate(self, condition, context):  # type: ignore[override]
        self.calls += 1
        return super().evaluate(condition, context)


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _tool_plan(tool_name: str, *, output_name: str = "value") -> ExecutionPlan:
    return ExecutionPlan(
        goal=f"Run {tool_name}.",
        ordered_steps=(ExecutionStep("child", f"Run {tool_name}.", tool_name),),
        estimated_steps=1,
        required_tools=(tool_name,),
        detected_risks=(),
        requires_confirmation=False,
        output=ExecutionPlanOutput({output_name: StepOutputReference("child")}),
    )


def _branch_plan(
    condition_value: object,
    *,
    then_plan: ExecutionPlan | None = None,
    else_plan: ExecutionPlan | None = None,
    output_binding: ExecutionVariableBinding | None = None,
    step_condition: ExecutionCondition | None = None,
    dependencies: tuple[str, ...] = (),
) -> ExecutionPlan:
    branch = ExecutionBranch(
        condition=ExecutionCondition(condition_value, ExecutionConditionOperator.TRUTHY),
        then_plan=then_plan or _tool_plan("then_tool"),
        else_plan=else_plan,
    )
    return ExecutionPlan(
        goal="Run branch.",
        ordered_steps=(
            ExecutionStep(
                "branch",
                "Run explicit branch.",
                None,
                dependencies=dependencies,
                branch=branch,
                output_binding=output_binding,
                condition=step_condition,
            ),
        ),
        estimated_steps=1,
        required_tools=(),
        detected_risks=(),
        requires_confirmation=False,
    )


def _execute(plan: ExecutionPlan, registry: ToolRegistry, **kwargs: Any):
    validator = ExecutionPlanValidator(registry)
    validation = validator.validate(plan)
    assert validation.is_valid, validation.errors
    return ExecutionPlanExecutor(registry, **kwargs).execute(plan, validation)


def test_condition_true_executes_only_then() -> None:
    then = RecordingTool("then_tool", "then")
    otherwise = RecordingTool("else_tool", "else")
    result = _execute(_branch_plan(True, else_plan=_tool_plan("else_tool")), _registry(then, otherwise))

    assert result.success is True
    assert then.calls == 1
    assert otherwise.calls == 0
    assert result.step_results[0].metadata["selected_branch"] == "then"


def test_condition_false_executes_only_else() -> None:
    then = RecordingTool("then_tool", "then")
    otherwise = RecordingTool("else_tool", "else")
    result = _execute(_branch_plan(False, else_plan=_tool_plan("else_tool")), _registry(then, otherwise))

    assert result.success is True
    assert then.calls == 0
    assert otherwise.calls == 1
    assert result.step_results[0].metadata["selected_branch"] == "else"


def test_false_without_else_skips_parent_step() -> None:
    then = RecordingTool("then_tool", "then")
    result = _execute(_branch_plan(False), _registry(then))

    assert result.success is True
    assert result.skipped_steps == ["branch"]
    assert result.step_results[0].status == StepExecutionStatus.SKIPPED.value
    assert then.calls == 0


def test_then_and_else_are_never_both_executed() -> None:
    then = RecordingTool("then_tool", "then")
    otherwise = RecordingTool("else_tool", "else")
    true_result = _execute(_branch_plan(True, else_plan=_tool_plan("else_tool")), _registry(then, otherwise))
    false_result = _execute(_branch_plan(False, else_plan=_tool_plan("else_tool")), _registry(then, otherwise))

    assert true_result.success is True
    assert false_result.success is True
    assert then.calls == 1
    assert otherwise.calls == 1


def test_then_returns_functional_output() -> None:
    result = _execute(_branch_plan(True), _registry(RecordingTool("then_tool", "then-output")))

    assert result.step_results[0].output == {"value": "then-output"}


def test_else_returns_functional_output() -> None:
    result = _execute(
        _branch_plan(False, else_plan=_tool_plan("else_tool")),
        _registry(RecordingTool("then_tool", "then"), RecordingTool("else_tool", "else-output")),
    )

    assert result.step_results[0].output == {"value": "else-output"}


def test_output_binding_receives_selected_output() -> None:
    plan = _branch_plan(
        False,
        else_plan=_tool_plan("else_tool"),
        output_binding=ExecutionVariableBinding("selected"),
    )
    context = ExecutionContext()
    validation = ExecutionPlanValidator(_registry(RecordingTool("else_tool", {"answer": 42}))).validate(plan)
    result = ExecutionPlanExecutor(_registry(RecordingTool("else_tool", {"answer": 42}))).execute(
        plan,
        validation,
        execution_context=context,
    )

    assert result.success is True
    assert context.get_variable("selected") == {"value": {"answer": 42}}


def test_then_failure_fails_parent_step() -> None:
    result = _execute(_branch_plan(True), _registry(RecordingTool("then_tool", fail=True)))

    assert result.success is False
    assert result.failed_step == "branch"
    assert result.step_results[0].error_code == ExecutionErrorCode.SUBPLAN_FAILED.value


def test_else_failure_fails_parent_step() -> None:
    result = _execute(
        _branch_plan(False, else_plan=_tool_plan("else_tool")),
        _registry(RecordingTool("then_tool"), RecordingTool("else_tool", fail=True)),
    )

    assert result.success is False
    assert result.failed_step == "branch"
    assert result.step_results[0].error_code == ExecutionErrorCode.SUBPLAN_FAILED.value


def test_branch_cancellation_propagates() -> None:
    plan = _branch_plan(True)
    registry = _registry(RecordingTool("then_tool"))
    validation = ExecutionPlanValidator(registry).validate(plan)

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        validation,
        control=ExecutionControl(should_cancel=lambda: True),
    )

    assert result.cancelled is True
    assert result.step_results[0].status == StepExecutionStatus.CANCELLED.value


def test_retry_reevaluates_branch_condition() -> None:
    evaluator = CountingConditionEvaluator()
    result = _execute(
        _branch_plan(True),
        _registry(FailOnceTool("then_tool", "ok")),
        condition_evaluator=evaluator,
        retry_policy=RetryPolicy(max_attempts=2),
    )

    assert result.success is True
    assert evaluator.calls == 2
    assert result.step_results[0].metadata["attempt_number"] == 2


def test_dependencies_are_checked_before_branch_condition() -> None:
    evaluator = CountingConditionEvaluator()
    plan = _branch_plan(True, dependencies=("missing",))
    registry = _registry(RecordingTool("then_tool"))
    validation = ExecutionPlanValidator(registry).validate(plan)

    assert validation.is_valid is False
    assert evaluator.calls == 0


def test_step_condition_runs_before_branch_condition() -> None:
    evaluator = CountingConditionEvaluator()
    plan = _branch_plan(
        True,
        step_condition=ExecutionCondition(False, ExecutionConditionOperator.TRUTHY),
    )
    result = _execute(plan, _registry(RecordingTool("then_tool")), condition_evaluator=evaluator)

    assert result.success is True
    assert result.skipped_steps == ["branch"]
    assert evaluator.calls == 1


def test_branch_serialization_and_restore(tmp_path: Path) -> None:
    plan = _branch_plan(True, else_plan=_tool_plan("else_tool"))
    registry = _registry(RecordingTool("then_tool"), RecordingTool("else_tool"))
    validation = ExecutionPlanValidator(registry).validate(plan)
    state = ResumableExecutionState(
        objective="resume branch",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=(),
        pending_step_ids=("branch",),
        failed_step_ids=(),
        interrupted_step_id="branch",
        previous_results={},
        resumable=True,
        execution_context_snapshot=ExecutionContext().snapshot(),
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)
    loaded = store.load()

    loaded_branch = loaded.original_plan.ordered_steps[0].branch
    assert loaded_branch is not None
    assert loaded_branch.then_plan.ordered_steps[0].tool == "then_tool"
    assert loaded_branch.else_plan is not None
    assert loaded_branch.else_plan.ordered_steps[0].tool == "else_tool"


def test_legacy_checkpoints_without_branch_still_load(tmp_path: Path) -> None:
    plan = ExecutionPlan(
        goal="legacy",
        ordered_steps=(ExecutionStep("one", "one", "then_tool"),),
        estimated_steps=1,
        required_tools=("then_tool",),
        detected_risks=(),
        requires_confirmation=False,
    )
    registry = _registry(RecordingTool("then_tool"))
    validation = ExecutionPlanValidator(registry).validate(plan)
    state = ResumableExecutionState(
        objective="legacy",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=(),
        pending_step_ids=("one",),
        failed_step_ids=(),
        interrupted_step_id="one",
        previous_results={},
        resumable=True,
        execution_context_snapshot=ExecutionContext().snapshot(),
    )
    store = JsonResumableExecutionStore(tmp_path / "legacy.json")
    store.save(state)
    payload = Path(tmp_path / "legacy.json").read_text(encoding="utf-8")
    Path(tmp_path / "legacy.json").write_text(payload.replace('"branch": null,', ""), encoding="utf-8")

    loaded = store.load()

    assert loaded.original_plan.ordered_steps[0].branch is None


def test_plan_signature_changes_when_branch_changes() -> None:
    base = _branch_plan(True, else_plan=_tool_plan("else_tool"))
    changed_condition = _branch_plan(False, else_plan=_tool_plan("else_tool"))
    changed_else = _branch_plan(True, else_plan=_tool_plan("other_tool"))

    assert plan_signature(base) != plan_signature(changed_condition)
    assert plan_signature(base) != plan_signature(changed_else)


def test_direct_branch_recursion_is_rejected() -> None:
    placeholder = _branch_plan(True)
    branch = ExecutionBranch(ExecutionCondition(True, ExecutionConditionOperator.TRUTHY), placeholder)
    recursive = ExecutionPlan(
        goal="recursive",
        ordered_steps=(ExecutionStep("branch", "branch", None, branch=branch),),
        estimated_steps=1,
        required_tools=(),
        detected_risks=(),
        requires_confirmation=False,
    )
    assert recursive.ordered_steps[0].branch is not None
    object.__setattr__(recursive.ordered_steps[0].branch, "then_plan", recursive)

    validation = ExecutionPlanValidator(_registry()).validate(recursive)

    assert validation.is_valid is False
    assert any("RecursiveSubplanError" in error for error in validation.errors)


def test_indirect_branch_recursion_is_rejected() -> None:
    inner = _branch_plan(True)
    outer_branch = ExecutionBranch(ExecutionCondition(True, ExecutionConditionOperator.TRUTHY), inner)
    outer = ExecutionPlan(
        goal="outer",
        ordered_steps=(ExecutionStep("outer_branch", "branch", None, branch=outer_branch),),
        estimated_steps=1,
        required_tools=(),
        detected_risks=(),
        requires_confirmation=False,
    )
    inner_branch = ExecutionBranch(ExecutionCondition(True, ExecutionConditionOperator.TRUTHY), outer)
    object.__setattr__(
        inner,
        "ordered_steps",
        (ExecutionStep("inner_branch", "branch", None, branch=inner_branch),),
    )

    validation = ExecutionPlanValidator(_registry()).validate(outer)

    assert validation.is_valid is False
    assert any("RecursiveSubplanError" in error for error in validation.errors)


def test_branch_depth_limit_is_enforced() -> None:
    plan = _tool_plan("then_tool")
    for index in range(ExecutionPlanValidator.MAX_SUBPLAN_DEPTH + 2):
        plan = ExecutionPlan(
            goal=f"level {index}",
            ordered_steps=(
                ExecutionStep(
                    f"branch_{index}",
                    "branch",
                    None,
                    branch=ExecutionBranch(ExecutionCondition(True, ExecutionConditionOperator.TRUTHY), plan),
                ),
            ),
            estimated_steps=1,
            required_tools=(),
            detected_risks=(),
            requires_confirmation=False,
        )

    validation = ExecutionPlanValidator(_registry(RecordingTool("then_tool"))).validate(plan)

    assert validation.is_valid is False
    assert any("SubplanDepthExceededError" in error for error in validation.errors)


def test_observability_and_metrics_are_safe() -> None:
    result = _execute(
        _branch_plan(False, else_plan=_tool_plan("else_tool")),
        _registry(RecordingTool("then_tool", {"secret": "hidden"}), RecordingTool("else_tool", {"token": "hidden"})),
    )

    assert result.trace is not None
    actions = {event.action for event in result.trace.events}
    assert "execution_branch_evaluation_started" in actions
    assert "execution_branch_else_selected" in actions
    assert "execution_branch_succeeded" in actions
    assert "hidden" not in repr(result.trace.events)
    metrics = ExecutionMetricsCalculator().calculate(result.trace)
    assert metrics.branches_evaluated == 1
    assert metrics.else_branches_selected == 1
    assert metrics.then_branches_selected == 0
    assert metrics.branches_failed == 0


def test_branch_e2e_with_real_read_only_list_directory(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    registry = Bootstrap.build_tool_registry()
    plan = ExecutionPlan(
        goal="Real read-only branch.",
        ordered_steps=(
            ExecutionStep(
                "branch",
                "List directory when enabled.",
                None,
                branch=ExecutionBranch(
                    ExecutionCondition(True, ExecutionConditionOperator.TRUTHY),
                    ExecutionPlan(
                        goal="List directory.",
                        ordered_steps=(
                            ExecutionStep(
                                "list_directory",
                                "List directory.",
                                "list_directory",
                                arguments={"path": ExecutionVariableReference("directory_path")},
                            ),
                        ),
                        estimated_steps=1,
                        required_tools=("list_directory",),
                        detected_risks=(),
                        requires_confirmation=False,
                        output=ExecutionPlanOutput({"entries": StepOutputReference("list_directory")}),
                    ),
                ),
                arguments={"directory_path": str(tmp_path)},
            ),
        ),
        estimated_steps=1,
        required_tools=(),
        detected_risks=(),
        requires_confirmation=False,
    )

    result = _execute(plan, registry)

    assert result.success is True
    assert result.step_results[0].output == {"entries": ["README.md", "pkg"]}

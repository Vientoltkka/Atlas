from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from core.execution_condition import ExecutionCondition, ExecutionConditionOperator
from core.execution_context import ExecutionContext
from core.execution_metrics import ExecutionMetricsCalculator
from core.execution_plan_executor import (
    ExecutionControl,
    ExecutionErrorCode,
    ExecutionPlanExecutor,
    LoopTerminationReason,
    ResumableExecutionState,
)
from core.execution_plan_output import ExecutionPlanOutput
from core.execution_plan_validator import ExecutionPlanValidator, plan_signature
from core.execution_variable_binding import ExecutionVariableBinding
from core.execution_variable_reference import ExecutionVariableReference
from core.planner import MAX_LOOP_ITERATIONS, ExecutionBranch, ExecutionLoop, ExecutionPlan, ExecutionStep
from core.resumable_execution_store import JsonResumableExecutionStore
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
        return "Safe test tool."

    def execute(self, context: ToolContext) -> object:
        del context
        self.calls += 1
        if self._fail:
            raise RuntimeError(f"{self._name} failed")
        return self._output


class IncrementTool(RecordingTool):
    def execute(self, context: ToolContext) -> object:
        self.calls += 1
        return int(context.parameters["value"]) + 1


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _body_step(
    tool_name: str = "body_tool",
    *,
    arguments: dict[str, object] | None = None,
    output_binding: ExecutionVariableBinding | None = None,
) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Loop body.",
        ordered_steps=(
            ExecutionStep(
                "body",
                "Run body.",
                tool_name,
                arguments={} if arguments is None else arguments,
                output_binding=output_binding,
            ),
        ),
        estimated_steps=1,
        required_tools=(tool_name,),
        detected_risks=(),
        requires_confirmation=False,
        output=ExecutionPlanOutput({"value": StepOutputReference("body")}),
    )


def _loop_plan(
    condition: ExecutionCondition,
    body_plan: ExecutionPlan,
    *,
    max_iterations: int = 3,
    arguments: dict[str, object] | None = None,
) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Run loop.",
        ordered_steps=(
            ExecutionStep(
                "loop",
                "Run controlled loop.",
                None,
                loop=ExecutionLoop(condition, body_plan, max_iterations),
                arguments={} if arguments is None else arguments,
            ),
        ),
        estimated_steps=1,
        required_tools=(),
        detected_risks=(),
        requires_confirmation=False,
    )


def _execute(
    plan: ExecutionPlan,
    registry: ToolRegistry,
    *,
    execution_context: ExecutionContext | None = None,
    control: ExecutionControl | None = None,
) -> Any:
    validation = ExecutionPlanValidator(registry).validate(plan)
    assert validation.is_valid, validation.errors
    return ExecutionPlanExecutor(registry).execute(
        plan,
        validation,
        execution_context=execution_context,
        control=control,
    )


def test_initial_false_condition_completes_zero_iterations() -> None:
    tool = RecordingTool("body_tool", "unused")
    plan = _loop_plan(
        ExecutionCondition(False, ExecutionConditionOperator.TRUTHY),
        _body_step(),
    )

    result = _execute(plan, _registry(tool))

    assert result.success is True
    assert result.step_results[0].output is None
    assert result.step_results[0].metadata["iterations_completed"] == 0
    assert result.step_results[0].metadata["termination_reason"] == LoopTerminationReason.CONDITION_FALSE.value
    assert tool.calls == 0


def test_one_iteration_returns_last_output() -> None:
    tool = RecordingTool("body_tool", False)
    context = ExecutionContext(initial_variables={"run": True})
    body = _body_step(output_binding=ExecutionVariableBinding("run"))
    plan = _loop_plan(
        ExecutionCondition(ExecutionVariableReference("run"), ExecutionConditionOperator.TRUTHY),
        body,
        max_iterations=2,
    )

    result = _execute(plan, _registry(tool), execution_context=context)

    assert result.success is True
    assert result.step_results[0].output == {"value": False}
    assert tool.calls == 1


def test_multiple_iterations_use_output_binding_to_change_condition() -> None:
    tool = IncrementTool("increment")
    context = ExecutionContext(initial_variables={"count": 0})
    body = _body_step(
        "increment",
        arguments={"value": ExecutionVariableReference("count")},
        output_binding=ExecutionVariableBinding("count"),
    )
    plan = _loop_plan(
        ExecutionCondition(ExecutionVariableReference("count"), ExecutionConditionOperator.LESS_THAN, 3),
        body,
        max_iterations=5,
    )

    result = _execute(plan, _registry(tool), execution_context=context)

    assert result.success is True
    assert tool.calls == 3
    assert context.get_variable("count") == 3
    assert result.step_results[0].metadata["iterations_completed"] == 3
    assert result.step_results[0].output == {"value": 3}


def test_max_iterations_reached_is_structured_failure() -> None:
    tool = RecordingTool("body_tool", "again")
    plan = _loop_plan(
        ExecutionCondition(True, ExecutionConditionOperator.TRUTHY),
        _body_step(),
        max_iterations=2,
    )

    result = _execute(plan, _registry(tool))

    assert result.success is False
    assert result.step_results[0].error_code == ExecutionErrorCode.LOOP_MAX_ITERATIONS_REACHED.value
    assert result.step_results[0].metadata["termination_reason"] == LoopTerminationReason.MAX_ITERATIONS_REACHED.value
    assert tool.calls == 2


def test_body_failed_fails_loop_step() -> None:
    result = _execute(
        _loop_plan(ExecutionCondition(True, ExecutionConditionOperator.TRUTHY), _body_step()),
        _registry(RecordingTool("body_tool", fail=True)),
    )

    assert result.success is False
    assert result.step_results[0].error_code == ExecutionErrorCode.LOOP_BODY_FAILED.value
    assert result.step_results[0].metadata["termination_reason"] == LoopTerminationReason.BODY_FAILED.value


def test_body_cancelled_cancels_loop_step() -> None:
    calls = {"count": 0}

    def cancel_after_parent_check() -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    result = _execute(
        _loop_plan(ExecutionCondition(True, ExecutionConditionOperator.TRUTHY), _body_step()),
        _registry(RecordingTool("body_tool")),
        control=ExecutionControl(should_cancel=cancel_after_parent_check),
    )

    assert result.cancelled is True
    assert result.step_results[0].error_code == ExecutionErrorCode.LOOP_BODY_CANCELLED.value


def test_invalid_condition_fails_structurally_at_execution() -> None:
    result = _execute(
        _loop_plan(
            ExecutionCondition(ExecutionVariableReference("missing"), ExecutionConditionOperator.TRUTHY),
            _body_step(),
        ),
        _registry(RecordingTool("body_tool")),
    )

    assert result.success is False
    assert result.step_results[0].error_code == ExecutionErrorCode.LOOP_CONDITION_FAILED.value
    assert result.step_results[0].metadata["termination_reason"] == (
        LoopTerminationReason.CONDITION_EVALUATION_FAILED.value
    )


def test_loop_nested_inside_branch() -> None:
    tool = RecordingTool("body_tool", False)
    loop_plan = _loop_plan(
        ExecutionCondition(ExecutionVariableReference("run"), ExecutionConditionOperator.TRUTHY),
        _body_step(output_binding=ExecutionVariableBinding("run")),
        max_iterations=2,
    )
    plan = ExecutionPlan(
        goal="Branch with loop.",
        ordered_steps=(
            ExecutionStep(
                "branch",
                "Run branch.",
                None,
                arguments={"run": ExecutionVariableReference("run")},
                branch=ExecutionBranch(
                    ExecutionCondition(True, ExecutionConditionOperator.TRUTHY),
                    loop_plan,
                ),
            ),
        ),
        estimated_steps=1,
        required_tools=(),
        detected_risks=(),
        requires_confirmation=False,
    )

    result = _execute(plan, _registry(tool), execution_context=ExecutionContext(initial_variables={"run": True}))

    assert result.success is True
    assert tool.calls == 1


def test_branch_nested_inside_loop() -> None:
    tool = RecordingTool("body_tool", False)
    branch_body = ExecutionPlan(
        goal="Loop body branch.",
        ordered_steps=(
            ExecutionStep(
                "body_branch",
                "Branch body.",
                None,
                output_binding=ExecutionVariableBinding.from_path("run", ("value",)),
                branch=ExecutionBranch(
                    ExecutionCondition(True, ExecutionConditionOperator.TRUTHY),
                    _body_step(output_binding=ExecutionVariableBinding("run")),
                ),
            ),
        ),
        estimated_steps=1,
        required_tools=(),
        detected_risks=(),
        requires_confirmation=False,
    )
    plan = _loop_plan(
        ExecutionCondition(ExecutionVariableReference("run"), ExecutionConditionOperator.TRUTHY),
        branch_body,
        max_iterations=2,
    )

    result = _execute(plan, _registry(tool), execution_context=ExecutionContext(initial_variables={"run": True}))

    assert result.success is True
    assert tool.calls == 1


def test_resume_does_not_repeat_completed_loop_step() -> None:
    body_tool = RecordingTool("body_tool", "body")
    after_tool = RecordingTool("after_tool", "after")
    plan = ExecutionPlan(
        goal="Resume after loop.",
        ordered_steps=(
            ExecutionStep(
                "loop",
                "Loop already completed.",
                None,
                loop=ExecutionLoop(
                    ExecutionCondition(False, ExecutionConditionOperator.TRUTHY),
                    _body_step(),
                    1,
                ),
            ),
            ExecutionStep("after", "After loop.", "after_tool", dependencies=("loop",)),
        ),
        estimated_steps=2,
        required_tools=("after_tool",),
        detected_risks=(),
        requires_confirmation=False,
    )
    registry = _registry(body_tool, after_tool)
    validation = ExecutionPlanValidator(registry).validate(plan)
    context = ExecutionContext()
    context.mark_step_started("loop", 1)
    context.mark_step_succeeded("loop", None)
    state = ResumableExecutionState(
        objective="resume",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=("loop",),
        pending_step_ids=("after",),
        failed_step_ids=(),
        interrupted_step_id="after",
        previous_results={"loop": None},
        resumable=True,
        execution_context_snapshot=context.snapshot(),
    )

    result = ExecutionPlanExecutor(registry).resume(state)

    assert result.success is True
    assert body_tool.calls == 0
    assert after_tool.calls == 1


def test_loop_serialization_and_legacy_checkpoint_loading(tmp_path: Path) -> None:
    plan = _loop_plan(ExecutionCondition(False, ExecutionConditionOperator.TRUTHY), _body_step())
    registry = _registry(RecordingTool("body_tool"))
    validation = ExecutionPlanValidator(registry).validate(plan)
    state = ResumableExecutionState(
        objective="loop",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=(),
        pending_step_ids=("loop",),
        failed_step_ids=(),
        interrupted_step_id="loop",
        previous_results={},
        resumable=True,
        execution_context_snapshot=ExecutionContext().snapshot(),
    )
    store = JsonResumableExecutionStore(tmp_path / "loop.json")
    store.save(state)
    loaded = store.load()
    assert loaded is not None
    assert loaded.original_plan.ordered_steps[0].loop is not None

    payload = Path(tmp_path / "loop.json").read_text(encoding="utf-8")
    Path(tmp_path / "loop.json").write_text(payload.replace('"loop": null,', ""), encoding="utf-8")
    legacy = store.load()
    assert legacy is not None


def test_plan_signature_changes_for_loop_condition_body_and_limit() -> None:
    base = _loop_plan(ExecutionCondition(False, ExecutionConditionOperator.TRUTHY), _body_step(), max_iterations=2)
    changed_condition = _loop_plan(ExecutionCondition(True, ExecutionConditionOperator.TRUTHY), _body_step(), max_iterations=2)
    changed_body = _loop_plan(ExecutionCondition(False, ExecutionConditionOperator.TRUTHY), _body_step("other_tool"), max_iterations=2)
    changed_limit = _loop_plan(ExecutionCondition(False, ExecutionConditionOperator.TRUTHY), _body_step(), max_iterations=3)

    assert plan_signature(base) != plan_signature(changed_condition)
    assert plan_signature(base) != plan_signature(changed_body)
    assert plan_signature(base) != plan_signature(changed_limit)


def test_loop_xor_and_max_iteration_limits_are_enforced() -> None:
    loop = ExecutionLoop(ExecutionCondition(False, ExecutionConditionOperator.TRUTHY), _body_step(), 1)
    invalid_xor = ExecutionPlan(
        goal="bad",
        ordered_steps=(ExecutionStep("bad", "bad", "body_tool", loop=loop),),
        estimated_steps=1,
        required_tools=("body_tool",),
        detected_risks=(),
        requires_confirmation=False,
    )

    result = ExecutionPlanValidator(_registry(RecordingTool("body_tool"))).validate(invalid_xor)

    assert result.is_valid is False
    assert any("exactly one of tool, subplan, or subplan_ref" in error for error in result.errors)
    with pytest.raises(ValueError):
        ExecutionLoop(ExecutionCondition(False, ExecutionConditionOperator.TRUTHY), _body_step(), MAX_LOOP_ITERATIONS + 1)


def test_loop_observability_and_metrics_are_safe() -> None:
    result = _execute(
        _loop_plan(ExecutionCondition(True, ExecutionConditionOperator.TRUTHY), _body_step(), max_iterations=1),
        _registry(RecordingTool("body_tool", {"secret": "hidden"})),
    )

    assert result.trace is not None
    assert "hidden" not in repr(result.trace.events)
    actions = {event.action for event in result.trace.events}
    assert "execution_loop_started" in actions
    assert "execution_loop_condition_evaluated" in actions
    assert "execution_loop_iteration_succeeded" in actions
    assert "execution_loop_max_iterations_reached" in actions
    metrics = ExecutionMetricsCalculator().calculate(result.trace)
    assert metrics.loops_started == 1
    assert metrics.loops_failed == 1
    assert metrics.loop_iterations_completed == 1
    assert metrics.loops_max_iterations_reached == 1


def test_loop_e2e_with_real_read_only_list_directory(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    registry = Bootstrap.build_tool_registry()
    body = ExecutionPlan(
        goal="List once.",
        ordered_steps=(
            ExecutionStep(
                "list",
                "List directory.",
                "list_directory",
                arguments={"path": ExecutionVariableReference("directory_path")},
                output_binding=ExecutionVariableBinding("entries"),
            ),
        ),
        estimated_steps=1,
        required_tools=("list_directory",),
        detected_risks=(),
        requires_confirmation=False,
        output=ExecutionPlanOutput({"entries": StepOutputReference("list")}),
    )
    plan = _loop_plan(
        ExecutionCondition(ExecutionVariableReference("entries"), ExecutionConditionOperator.NOT_EXISTS),
        body,
        max_iterations=2,
    )

    result = _execute(
        plan,
        registry,
        execution_context=ExecutionContext(initial_variables={"directory_path": str(tmp_path)}),
    )

    assert result.success is True
    assert result.step_results[0].output == {"entries": ["README.md", "pkg"]}

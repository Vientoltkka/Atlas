from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from core.execution_plan_executor import (
    ExecutionPlanExecutor,
    PlanExecutionResult,
    StepExecutionResult,
)
from core.execution_plan_validator import (
    ExecutionPlanValidator,
    PlanValidationResult,
)
from core.planner import ExecutionPlan, ExecutionStep
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(
        self,
        name: str,
        calls: list[str],
        output: Any = "ok",
        *,
        fail: bool = False,
    ) -> None:
        self._name = name
        self._calls = calls
        self._output = output
        self._fail = fail
        self.contexts: list[ToolContext] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Spy tool {self._name}."

    def execute(
        self,
        context: ToolContext,
    ) -> Any:
        self._calls.append(self._name)
        self.contexts.append(context)

        if self._fail:
            raise RuntimeError(f"{self._name} exploded")

        return self._output


def _step(
    step_id: str,
    tool: str | None,
    dependencies: tuple[str, ...] = (),
    status: str = "pending",
) -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        description=f"Execute {step_id}.",
        tool=tool,
        dependencies=dependencies,
        status=status,
    )


def _plan(
    steps: tuple[ExecutionStep, ...],
    *,
    required_tools: tuple[str, ...] | None = None,
    requires_confirmation: bool = False,
    status: str = "planned",
) -> ExecutionPlan:
    tools = required_tools
    if tools is None:
        tools = tuple(
            step.tool
            for step in steps
            if step.tool is not None and step.tool != "direct_response"
        )

    return ExecutionPlan(
        goal="Execute controlled plan.",
        ordered_steps=steps,
        estimated_steps=len(steps),
        required_tools=tools,
        detected_risks=(
            ("confirmation-gated operation",)
            if requires_confirmation
            else ()
        ),
        requires_confirmation=requires_confirmation,
        status=status,
    )


def _registry(
    *tools: BaseTool,
) -> ToolRegistry:
    registry = ToolRegistry()

    for tool in tools:
        registry.register(tool)

    return registry


def _validation(plan: ExecutionPlan) -> PlanValidationResult:
    return ExecutionPlanValidator().validate(plan)


def _manual_valid_result(
    *,
    requires_confirmation: bool = False,
) -> PlanValidationResult:
    return PlanValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        requires_confirmation=requires_confirmation,
        status="valid",
        plan_signature=None,
    )


def test_executes_valid_single_step_plan() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls, output="done")
    plan = _plan((_step("step_1", "safe_tool"),))

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert result == PlanExecutionResult(
        plan_status="completed",
        success=True,
        completed_steps=["step_1"],
        failed_step=None,
        skipped_steps=[],
        step_results=[
            StepExecutionResult(
                step_id="step_1",
                status="completed",
                success=True,
                tool_name="safe_tool",
                output="done",
                error=None,
            )
        ],
        error=None,
        requires_confirmation=False,
        interrupted=False,
    )
    assert calls == ["safe_tool"]


def test_executes_multiple_steps_in_order() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls)
    second = SpyTool("second_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step("step_2", "second_tool", dependencies=("step_1",)),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        _validation(plan),
    )

    assert result.success is True
    assert result.completed_steps == ["step_1", "step_2"]
    assert calls == ["first_tool", "second_tool"]


def test_dependencies_are_respected() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls)
    second = SpyTool("second_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool", status="completed"),
            _step("step_2", "second_tool", dependencies=("step_1",)),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        _manual_valid_result(),
    )

    assert result.success is True
    assert result.completed_steps == ["step_1", "step_2"]
    assert calls == ["second_tool"]


def test_logical_step_with_none_tool_completes_without_tool_call() -> None:
    calls: list[str] = []
    plan = _plan((_step("step_1", None),), required_tools=())

    result = ExecutionPlanExecutor(_registry()).execute(plan, _validation(plan))

    assert result.success is True
    assert result.completed_steps == ["step_1"]
    assert result.step_results[0].tool_name is None
    assert calls == []


def test_invalid_plan_is_rejected_without_execution() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = replace(_plan((_step("step_1", "safe_tool"),)), goal="")
    validation = _validation(plan)

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, validation)

    assert result.success is False
    assert result.plan_status == "rejected"
    assert calls == []


def test_missing_validation_result_is_rejected() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = _plan((_step("step_1", "safe_tool"),))

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, None)

    assert result.success is False
    assert result.plan_status == "rejected"
    assert result.error == "Plan execution requires an explicit PlanValidationResult."
    assert calls == []


def test_mismatched_validation_result_is_rejected_without_execution() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = _plan((_step("step_1", "safe_tool"),))
    other_plan = replace(plan, goal="Different goal.")

    result = ExecutionPlanExecutor(_registry(tool)).execute(
        other_plan,
        _validation(plan),
    )

    assert result.success is False
    assert result.plan_status == "rejected"
    assert result.error == "PlanValidationResult does not match the execution plan."
    assert calls == []


def test_incompatible_plan_status_is_rejected_without_execution() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = _plan((_step("step_1", "safe_tool"),), status="completed")

    result = ExecutionPlanExecutor(_registry(tool)).execute(
        plan,
        _manual_valid_result(),
    )

    assert result.success is False
    assert result.plan_status == "rejected"
    assert result.error == "Plan status 'completed' is not executable."
    assert calls == []


def test_confirmation_required_blocks_without_explicit_grant() -> None:
    calls: list[str] = []
    tool = SpyTool("write_file", calls)
    plan = _plan(
        (_step("step_1", "write_file"),),
        requires_confirmation=True,
    )

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert result.success is False
    assert result.plan_status == "blocked_confirmation"
    assert result.requires_confirmation is True
    assert calls == []


def test_confirmation_grant_allows_execution() -> None:
    calls: list[str] = []
    tool = SpyTool("write_file", calls)
    plan = _plan(
        (_step("step_1", "write_file"),),
        requires_confirmation=True,
    )

    result = ExecutionPlanExecutor(_registry(tool)).execute(
        plan,
        _validation(plan),
        confirmation_granted=True,
    )

    assert result.success is True
    assert calls == ["write_file"]


def test_missing_tool_fails_step_without_execution() -> None:
    plan = _plan((_step("step_1", "missing_tool"),))

    result = ExecutionPlanExecutor(_registry()).execute(plan, _validation(plan))

    assert result.success is False
    assert result.failed_step == "step_1"
    assert result.error == "Tool 'missing_tool' is not registered."


def test_failed_result_returned_by_tool_stops_plan() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"success": False, "error": "bad result"})
    second = SpyTool("second_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step("step_2", "second_tool", dependencies=("step_1",)),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        _validation(plan),
    )

    assert result.success is False
    assert result.failed_step == "step_1"
    assert result.skipped_steps == ["step_2"]
    assert result.error == "bad result"
    assert calls == ["first_tool"]


def test_tool_exception_is_captured_as_structured_failure() -> None:
    calls: list[str] = []
    tool = SpyTool("failing_tool", calls, fail=True)
    plan = _plan((_step("step_1", "failing_tool"),))

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert result.success is False
    assert result.failed_step == "step_1"
    assert result.error == "failing_tool exploded"
    assert calls == ["failing_tool"]


def test_fail_fast_stops_on_first_failure() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, fail=True)
    second = SpyTool("second_tool", calls)
    third = SpyTool("third_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step("step_2", "second_tool", dependencies=("step_1",)),
            _step("step_3", "third_tool", dependencies=("step_2",)),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second, third)).execute(
        plan,
        _validation(plan),
    )

    assert result.success is False
    assert result.failed_step == "step_1"
    assert result.skipped_steps == ["step_2", "step_3"]
    assert calls == ["first_tool"]


def test_later_steps_are_not_executed_after_failure() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls)
    second = SpyTool("second_tool", calls, fail=True)
    third = SpyTool("third_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step("step_2", "second_tool", dependencies=("step_1",)),
            _step("step_3", "third_tool", dependencies=("step_2",)),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second, third)).execute(
        plan,
        _validation(plan),
    )

    assert result.success is False
    assert result.failed_step == "step_2"
    assert result.skipped_steps == ["step_3"]
    assert calls == ["first_tool", "second_tool"]


def test_completed_step_is_not_executed_twice() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls)
    second = SpyTool("second_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool", status="completed"),
            _step("step_2", "second_tool", dependencies=("step_1",)),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        _manual_valid_result(),
    )

    assert result.success is True
    assert result.completed_steps == ["step_1", "step_2"]
    assert calls == ["second_tool"]


def test_executor_does_not_modify_plan_structure(tmp_path: Path) -> None:
    calls: list[str] = []
    target = tmp_path / "unchanged.txt"
    tool = SpyTool("safe_tool", calls)
    plan = _plan((_step("step_1", "safe_tool"),))
    before = repr(plan)

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert result.success is True
    assert repr(plan) == before
    assert target.exists() is False


def test_global_result_is_structured() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = _plan((_step("step_1", "safe_tool"),))

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert isinstance(result, PlanExecutionResult)
    assert isinstance(result.step_results[0], StepExecutionResult)
    assert result.plan_status == "completed"


def test_executor_does_not_call_planner_or_replan() -> None:
    source = Path("core/execution_plan_executor.py").read_text(encoding="utf-8")

    assert "Planner" not in source
    assert "create_execution_plan" not in source

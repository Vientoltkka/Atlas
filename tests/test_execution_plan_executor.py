from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from core.execution_plan_executor import (
    ExecutionControl,
    ExecutionErrorCode,
    ExecutionProgress,
    ExecutionPlanExecutor,
    PlanExecutionStatus,
    PlanExecutionResult,
    ResumableExecutionState,
    StepExecutionStatus,
    StepExecutionResult,
)
from core.execution_retry import RetryPolicy
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


class SequenceTool(BaseTool):
    def __init__(
        self,
        name: str,
        calls: list[str],
        outputs: list[Any],
        *,
        requires_confirmation: bool = False,
    ) -> None:
        self._name = name
        self._calls = calls
        self._outputs = list(outputs)
        self._requires_confirmation = requires_confirmation
        self.contexts: list[ToolContext] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Sequence tool {self._name}."

    @property
    def requires_confirmation(self) -> bool:
        return self._requires_confirmation

    def execute(
        self,
        context: ToolContext,
    ) -> Any:
        self._calls.append(self._name)
        self.contexts.append(context)
        output = self._outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _step(
    step_id: str,
    tool: str | None,
    dependencies: tuple[str, ...] = (),
    status: str = "pending",
    arguments: dict[str, Any] | None = None,
) -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        description=f"Execute {step_id}.",
        tool=tool,
        dependencies=dependencies,
        status=status,
        arguments={} if arguments is None else arguments,
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


def _state_from_result(
    *,
    objective: str,
    plan: ExecutionPlan,
    validation: PlanValidationResult,
    result: PlanExecutionResult,
    confirmation_granted: bool = False,
) -> ResumableExecutionState:
    return ResumableExecutionState(
        objective=objective,
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=tuple(result.completed_steps),
        pending_step_ids=tuple(result.pending_steps),
        failed_step_ids=tuple(result.failed_steps),
        interrupted_step_id=result.current_step,
        previous_results={
            step.step_id: step.output
            for step in result.step_results
            if step.success and step.status == StepExecutionStatus.COMPLETED.value
        },
        resumable=result.resumable,
        interruption_reason=result.interruption_reason,
        confirmation_granted=confirmation_granted,
    )


def test_executes_valid_single_step_plan() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls, output="done")
    plan = _plan((_step("step_1", "safe_tool"),))

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert isinstance(result, PlanExecutionResult)
    assert result.plan_status == PlanExecutionStatus.COMPLETED.value
    assert result.status == PlanExecutionStatus.COMPLETED.value
    assert result.success is True
    assert result.completed is True
    assert result.completed_steps == ["step_1"]
    assert result.failed_step is None
    assert result.skipped_steps == []
    assert result.pending_steps == []
    assert result.error is None
    assert result.requires_confirmation is False
    assert result.interrupted is False
    assert result.resumable is False
    assert result.metadata["plan_signature"]
    assert result.step_results[0].step_id == "step_1"
    assert result.step_results[0].status == StepExecutionStatus.COMPLETED.value
    assert result.step_results[0].success is True
    assert result.step_results[0].tool_name == "safe_tool"
    assert result.step_results[0].output == "done"
    assert result.step_results[0].error is None
    assert result.step_results[0].metadata["attempt_number"] == 1
    assert result.step_results[0].metadata["max_attempts"] == 1
    assert calls == ["safe_tool"]
    assert tool.contexts[0].parameters == {}


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


def test_execution_progress_is_typed_safe_and_ordered() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls)
    second = SpyTool("second_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool", arguments={"secret": "hidden"}),
            _step("step_2", "second_tool", dependencies=("step_1",)),
        )
    )
    progress: list[ExecutionProgress] = []

    result = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        _validation(plan),
        on_progress=progress.append,
    )

    assert result.success is True
    assert [event.phase for event in progress] == [
        "preparing",
        "step_started",
        "step_completed",
        "step_started",
        "step_completed",
        "completed",
    ]
    assert progress[1].step_id == "step_1"
    assert progress[1].step_index == 1
    assert progress[1].total_steps == 2
    assert progress[1].tool_name == "first_tool"
    assert all(event.message is None for event in progress)
    assert "hidden" not in repr(progress)


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
    assert result.blocked is True
    assert result.resumable is True
    assert result.error_code == ExecutionErrorCode.CONFIRMATION_REQUIRED.value
    assert result.pending_steps == ["step_1"]
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


def test_step_arguments_are_delivered_to_tool_context() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    arguments = {"path": "README.md", "mode": "safe"}
    plan = _plan((_step("step_1", "safe_tool", arguments=arguments),))
    validation = _validation(plan)

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, validation)

    assert result.success is True
    assert tool.contexts[0].parameters == arguments
    assert tool.contexts[0].arguments == arguments
    assert tool.contexts[0].step_id == "step_1"
    assert tool.contexts[0].plan_signature == validation.plan_signature
    assert tool.contexts[0].metadata == {"executor": "ExecutionPlanExecutor"}


def test_previous_results_are_available_to_later_steps_without_resolution() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"content": "alpha"})
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
    assert second.contexts[0].previous_results == {
        "step_1": {"content": "alpha"},
    }


def test_resolved_arguments_are_delivered_to_tool_context() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"path": "README.md"})
    second = SpyTool("second_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step(
                "step_2",
                "second_tool",
                dependencies=("step_1",),
                arguments={"path": {"$ref": "steps.step_1.output.path"}},
            ),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        _validation(plan),
    )

    assert result.success is True
    assert second.contexts[0].parameters == {"path": "README.md"}
    assert second.contexts[0].arguments == {"path": "README.md"}


def test_step_result_can_feed_next_step() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"content": "alpha"})
    second = SpyTool("second_tool", calls, output={"echo": "alpha"})
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step(
                "step_2",
                "second_tool",
                dependencies=("step_1",),
                arguments={"content": {"$ref": "steps.step_1.output.content"}},
            ),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        _validation(plan),
    )

    assert result.success is True
    assert calls == ["first_tool", "second_tool"]
    assert second.contexts[0].parameters == {"content": "alpha"}


def test_different_steps_can_use_same_previous_result() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"path": "README.md"})
    second = SpyTool("second_tool", calls)
    third = SpyTool("third_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step(
                "step_2",
                "second_tool",
                dependencies=("step_1",),
                arguments={"path": {"$ref": "steps.step_1.output.path"}},
            ),
            _step(
                "step_3",
                "third_tool",
                dependencies=("step_1", "step_2"),
                arguments={"path": {"$ref": "steps.step_1.output.path"}},
            ),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second, third)).execute(
        plan,
        _validation(plan),
    )

    assert result.success is True
    assert second.contexts[0].parameters == {"path": "README.md"}
    assert third.contexts[0].parameters == {"path": "README.md"}


def test_parameter_resolution_failure_prevents_tool_call() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"name": "README.md"})
    second = SpyTool("second_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step(
                "step_2",
                "second_tool",
                dependencies=("step_1",),
                arguments={"path": {"$ref": "steps.step_1.output.path"}},
            ),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        _validation(plan),
    )

    assert result.success is False
    assert result.failed_step == "step_2"
    assert result.error_code == ExecutionErrorCode.PARAMETER_RESOLUTION_FAILED.value
    assert result.step_results[-1].error_code == (
        ExecutionErrorCode.PARAMETER_RESOLUTION_FAILED.value
    )
    assert result.step_results[-1].metadata["parameter_resolution_error_code"] == (
        "REFERENCED_FIELD_NOT_FOUND"
    )
    assert calls == ["first_tool"]


def test_template_resolution_delivers_final_text_to_tool_context() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"path": "README.md"})
    second = SpyTool("second_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step(
                "step_2",
                "second_tool",
                dependencies=("step_1",),
                arguments={
                    "message": {
                        "$template": "Archivo final: {{steps.step_1.output.path}}"
                    }
                },
            ),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        _validation(plan),
    )

    assert result.success is True
    assert second.contexts[0].parameters == {"message": "Archivo final: README.md"}
    assert calls == ["first_tool", "second_tool"]


def test_template_resolution_failure_prevents_tool_call() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"name": "README.md"})
    second = SpyTool("second_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step(
                "step_2",
                "second_tool",
                dependencies=("step_1",),
                arguments={
                    "message": {
                        "$template": "Archivo final: {{steps.step_1.output.path}}"
                    }
                },
            ),
        )
    )

    result = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        _validation(plan),
    )

    assert result.success is False
    assert result.failed_step == "step_2"
    assert result.error_code == ExecutionErrorCode.PARAMETER_RESOLUTION_FAILED.value
    assert result.step_results[-1].metadata["parameter_resolution_error_code"] == (
        "UNRESOLVED_TEMPLATE_REFERENCE"
    )
    assert calls == ["first_tool"]


def test_original_arguments_are_not_mutated_by_tool() -> None:
    class MutatingTool(SpyTool):
        def execute(
            self,
            context: ToolContext,
        ) -> Any:
            context.parameters["path"] = "changed.txt"
            context.parameters["nested"]["value"] = "changed"
            return super().execute(context)

    calls: list[str] = []
    tool = MutatingTool("safe_tool", calls)
    nested = {"value": "original"}
    plan = _plan(
        (
            _step(
                "step_1",
                "safe_tool",
                arguments={"path": "README.md", "nested": nested},
            ),
        )
    )

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert result.success is True
    assert dict(plan.ordered_steps[0].arguments) == {
        "path": "README.md",
        "nested": {"value": "original"},
    }
    assert nested == {"value": "original"}


def test_argument_change_after_validation_is_rejected() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    original = _plan(
        (_step("step_1", "safe_tool", arguments={"path": "README.md"}),)
    )
    validation = _validation(original)
    changed = replace(
        original,
        ordered_steps=(
            _step("step_1", "safe_tool", arguments={"path": "changed.md"}),
        ),
    )

    result = ExecutionPlanExecutor(_registry(tool)).execute(changed, validation)

    assert result.success is False
    assert result.error_code == ExecutionErrorCode.VALIDATION_MISMATCH.value
    assert calls == []


def test_dangerous_step_with_arguments_still_requires_confirmation() -> None:
    calls: list[str] = []
    tool = SpyTool("write_file", calls)
    plan = _plan(
        (
            _step(
                "step_1",
                "write_file",
                arguments={"path": "out.txt", "content": "hello"},
            ),
        ),
        requires_confirmation=True,
    )

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert result.plan_status == PlanExecutionStatus.BLOCKED_CONFIRMATION.value
    assert result.error_code == ExecutionErrorCode.CONFIRMATION_REQUIRED.value
    assert calls == []


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
    assert result.failed_steps == ["step_1"]
    assert result.skipped_steps == ["step_2"]
    assert result.error == "bad result"
    assert result.error_code == ExecutionErrorCode.TOOL_EXECUTION_FAILED.value
    assert result.resumable is False
    assert result.step_results[0].error_code == ExecutionErrorCode.TOOL_EXECUTION_FAILED.value
    assert result.step_results[1].status == StepExecutionStatus.SKIPPED.value
    assert calls == ["first_tool"]


def test_tool_exception_is_captured_as_structured_failure() -> None:
    calls: list[str] = []
    tool = SpyTool("failing_tool", calls, fail=True)
    plan = _plan((_step("step_1", "failing_tool"),))

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert result.success is False
    assert result.failed_step == "step_1"
    assert result.error == "failing_tool exploded"
    assert result.error_code == ExecutionErrorCode.TOOL_EXCEPTION.value
    assert result.step_results[0].metadata["exception_type"] == "RuntimeError"
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
    assert result.plan_status == PlanExecutionStatus.FAILED.value
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
    assert result.plan_status == PlanExecutionStatus.PARTIALLY_COMPLETED.value
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


def test_interruption_before_first_step_is_structured_and_resumable() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = _plan((_step("step_1", "safe_tool"),))
    control = ExecutionControl(
        should_stop=lambda: True,
        interruption_reason="pause requested",
    )

    result = ExecutionPlanExecutor(_registry(tool)).execute(
        plan,
        _validation(plan),
        control=control,
    )

    assert result.plan_status == PlanExecutionStatus.INTERRUPTED.value
    assert result.success is False
    assert result.interrupted is True
    assert result.cancelled is False
    assert result.resumable is True
    assert result.completed_steps == []
    assert result.pending_steps == ["step_1"]
    assert result.current_step == "step_1"
    assert result.interruption_reason == "pause requested"
    assert result.error_code == ExecutionErrorCode.EXECUTION_INTERRUPTED.value
    assert result.step_results[0].status == StepExecutionStatus.INTERRUPTED.value
    assert calls == []


def test_interruption_after_one_step_preserves_completed_and_pending_steps() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls)
    second = SpyTool("second_tool", calls)
    third = SpyTool("third_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step("step_2", "second_tool", dependencies=("step_1",)),
            _step("step_3", "third_tool", dependencies=("step_2",)),
        )
    )
    control = ExecutionControl(
        should_stop=lambda: len(calls) >= 1,
        interruption_reason="time limit",
    )

    result = ExecutionPlanExecutor(_registry(first, second, third)).execute(
        plan,
        _validation(plan),
        control=control,
    )

    assert result.plan_status == PlanExecutionStatus.INTERRUPTED.value
    assert result.completed_steps == ["step_1"]
    assert result.pending_steps == ["step_2", "step_3"]
    assert result.current_step == "step_2"
    assert result.resumable is True
    assert [step.status for step in result.step_results] == [
        StepExecutionStatus.COMPLETED.value,
        StepExecutionStatus.INTERRUPTED.value,
        StepExecutionStatus.NOT_STARTED.value,
    ]
    assert calls == ["first_tool"]


def test_resume_continues_from_first_pending_step_without_repeating_completed() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"content": "alpha"})
    second = SpyTool("second_tool", calls)
    third = SpyTool("third_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step(
                "step_2",
                "second_tool",
                dependencies=("step_1",),
                arguments={"content": {"$ref": "steps.step_1.output.content"}},
            ),
            _step("step_3", "third_tool", dependencies=("step_2",)),
        )
    )
    validation = _validation(plan)
    interrupted = ExecutionPlanExecutor(_registry(first, second, third)).execute(
        plan,
        validation,
        control=ExecutionControl(should_stop=lambda: len(calls) >= 1),
    )
    state = _state_from_result(
        objective="resume",
        plan=plan,
        validation=validation,
        result=interrupted,
    )

    resumed = ExecutionPlanExecutor(_registry(first, second, third)).resume(state)

    assert resumed.success is True
    assert calls == ["first_tool", "second_tool", "third_tool"]
    assert resumed.completed_steps == ["step_1", "step_2", "step_3"]
    assert second.contexts[0].parameters == {"content": "alpha"}


def test_resume_rejects_modified_plan_signature_without_tool_calls() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls)
    second = SpyTool("second_tool", calls)
    plan = _plan((_step("step_1", "first_tool"), _step("step_2", "second_tool")))
    validation = _validation(plan)
    interrupted = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        validation,
        control=ExecutionControl(should_stop=lambda: len(calls) >= 1),
    )
    state = _state_from_result(
        objective="resume",
        plan=replace(plan, goal="changed"),
        validation=validation,
        result=interrupted,
    )

    resumed = ExecutionPlanExecutor(_registry(first, second)).resume(state)

    assert resumed.plan_status == PlanExecutionStatus.REJECTED.value
    assert resumed.error_code == ExecutionErrorCode.VALIDATION_MISMATCH.value
    assert calls == ["first_tool"]


def test_resume_rejects_missing_previous_result() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"content": "alpha"})
    second = SpyTool("second_tool", calls)
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step("step_2", "second_tool", dependencies=("step_1",)),
        )
    )
    validation = _validation(plan)
    interrupted = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        validation,
        control=ExecutionControl(should_stop=lambda: len(calls) >= 1),
    )
    state = replace(
        _state_from_result(
            objective="resume",
            plan=plan,
            validation=validation,
            result=interrupted,
        ),
        previous_results={},
    )

    resumed = ExecutionPlanExecutor(_registry(first, second)).resume(state)

    assert resumed.plan_status == PlanExecutionStatus.REJECTED.value
    assert "missing" in (resumed.error or "")
    assert calls == ["first_tool"]


def test_cancelled_completed_and_failed_states_are_not_resumable() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"success": False, "error": "bad"})
    second = SpyTool("second_tool", calls)
    plan = _plan((_step("step_1", "first_tool"), _step("step_2", "second_tool")))
    validation = _validation(plan)
    executor = ExecutionPlanExecutor(_registry(first, second))

    cancelled = executor.execute(
        plan,
        validation,
        control=ExecutionControl(should_cancel=lambda: True),
    )
    completed = ExecutionPlanExecutor(
        _registry(SpyTool("first_tool", []), SpyTool("second_tool", []))
    ).execute(plan, validation)
    failed = executor.execute(plan, validation)

    for result in (cancelled, completed, failed):
        state = replace(
            _state_from_result(
                objective="resume",
                plan=plan,
                validation=validation,
                result=result,
            ),
            resumable=result.resumable,
        )
        resumed = executor.resume(state)
        assert resumed.plan_status == PlanExecutionStatus.REJECTED.value


def test_resume_confirmation_required_blocks_without_consuming_state() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls)
    second = SpyTool("write_file", calls)
    plan = _plan(
        (_step("step_1", "first_tool"), _step("step_2", "write_file")),
        requires_confirmation=True,
    )
    validation = _validation(plan)
    interrupted = ExecutionPlanExecutor(_registry(first, second)).execute(
        plan,
        validation,
        confirmation_granted=True,
        control=ExecutionControl(should_stop=lambda: len(calls) >= 1),
    )
    state = replace(
        _state_from_result(
            objective="resume",
            plan=plan,
            validation=validation,
            result=interrupted,
        ),
        confirmation_granted=False,
    )

    blocked = ExecutionPlanExecutor(_registry(first, second)).resume(state)
    resumed = ExecutionPlanExecutor(_registry(first, second)).resume(
        state,
        confirmation_granted=True,
    )

    assert blocked.plan_status == PlanExecutionStatus.BLOCKED_CONFIRMATION.value
    assert calls == ["first_tool", "write_file"]
    assert resumed.success is True


def test_resume_preserves_retry_counter_and_does_not_exceed_limit() -> None:
    calls: list[str] = []
    first = SpyTool("first_tool", calls, output={"content": "alpha"})
    second = SequenceTool(
        "second_tool",
        calls,
        [
            {"success": False, "error": "transient", "error_code": "TRANSIENT_ERROR"},
            "must not run",
        ],
    )
    plan = _plan(
        (
            _step("step_1", "first_tool"),
            _step("step_2", "second_tool", dependencies=("step_1",)),
        )
    )
    validation = _validation(plan)
    state = ResumableExecutionState(
        objective="resume",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=("step_1",),
        pending_step_ids=("step_2",),
        failed_step_ids=(),
        interrupted_step_id="step_2",
        previous_results={"step_1": {"content": "alpha"}},
        resumable=True,
        confirmation_granted=False,
        retry_attempts={"step_2": 1},
        retry_history={
            "step_2": (
                {"attempt_number": 1, "error_code": "TRANSIENT_ERROR", "error": "transient"},
            )
        },
    )

    result = ExecutionPlanExecutor(
        _registry(first, second),
        retry_policy=RetryPolicy(max_attempts=2),
    ).resume(state)

    assert result.success is False
    assert calls == ["second_tool"]
    assert result.step_results[0].metadata["attempt_number"] == 2
    assert result.step_results[0].metadata["retry_exhausted"] is True
    assert len(result.step_results[0].metadata["retry_history"]) == 2


def test_controlled_cancellation_is_not_a_technical_failure() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = _plan((_step("step_1", "safe_tool"),))
    control = ExecutionControl(
        should_cancel=lambda: True,
        cancellation_reason="user cancelled",
    )

    result = ExecutionPlanExecutor(_registry(tool)).execute(
        plan,
        _validation(plan),
        control=control,
    )

    assert result.plan_status == PlanExecutionStatus.CANCELLED.value
    assert result.success is False
    assert result.cancelled is True
    assert result.failed is False
    assert result.completed is False
    assert result.resumable is False
    assert result.error is None
    assert result.error_code == ExecutionErrorCode.EXECUTION_CANCELLED.value
    assert result.interruption_reason == "user cancelled"
    assert result.step_results[0].status == StepExecutionStatus.CANCELLED.value
    assert calls == []


def test_rejected_plan_has_stable_error_code_and_is_not_resumable() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = replace(_plan((_step("step_1", "safe_tool"),)), goal="")

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert result.plan_status == PlanExecutionStatus.REJECTED.value
    assert result.error_code == ExecutionErrorCode.INVALID_PLAN.value
    assert result.resumable is False
    assert result.pending_steps == ["step_1"]
    assert calls == []


def test_validation_mismatch_has_specific_error_code() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = _plan((_step("step_1", "safe_tool"),))

    result = ExecutionPlanExecutor(_registry(tool)).execute(
        replace(plan, goal="different"),
        _validation(plan),
    )

    assert result.error_code == ExecutionErrorCode.VALIDATION_MISMATCH.value
    assert calls == []


def test_dependency_failure_uses_dependency_error_code() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = _plan((_step("step_1", "safe_tool", dependencies=("missing",)),))

    result = ExecutionPlanExecutor(_registry(tool)).execute(
        plan,
        _manual_valid_result(),
    )

    assert result.success is False
    assert result.error_code == ExecutionErrorCode.DEPENDENCY_NOT_COMPLETED.value
    assert result.failed_steps == ["step_1"]
    assert result.pending_steps == []
    assert calls == []


def test_missing_tool_uses_tool_not_found_error_code() -> None:
    plan = _plan((_step("step_1", "missing_tool"),))

    result = ExecutionPlanExecutor(_registry()).execute(plan, _validation(plan))

    assert result.error_code == ExecutionErrorCode.TOOL_NOT_FOUND.value
    assert result.step_results[0].error_code == ExecutionErrorCode.TOOL_NOT_FOUND.value


def test_control_callback_error_is_internal_executor_error() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = _plan((_step("step_1", "safe_tool"),))

    def broken_control() -> bool:
        raise RuntimeError("control broke")

    result = ExecutionPlanExecutor(_registry(tool)).execute(
        plan,
        _validation(plan),
        control=ExecutionControl(should_stop=broken_control),
    )

    assert result.success is False
    assert result.error_code == ExecutionErrorCode.INTERNAL_EXECUTOR_ERROR.value
    assert result.failed is True
    assert result.resumable is False
    assert result.error == "Internal executor control error: control broke"
    assert calls == []


def test_outcome_flags_are_not_contradictory() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    plan = _plan((_step("step_1", "safe_tool"),))

    completed = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))
    interrupted = ExecutionPlanExecutor(_registry(tool)).execute(
        plan,
        _validation(plan),
        control=ExecutionControl(should_stop=lambda: True),
    )

    assert completed.completed is True
    assert completed.failed is False
    assert completed.interrupted is False
    assert completed.cancelled is False
    assert completed.success is True
    assert interrupted.completed is False
    assert interrupted.failed is False
    assert interrupted.interrupted is True
    assert interrupted.success is False


def test_traceability_includes_completed_failed_and_skipped_steps() -> None:
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

    assert [step.step_id for step in result.step_results] == [
        "step_1",
        "step_2",
        "step_3",
    ]
    assert [step.tool_name for step in result.step_results] == [
        "first_tool",
        "second_tool",
        "third_tool",
    ]
    assert result.completed_steps == ["step_1"]
    assert result.failed_steps == ["step_2"]
    assert result.skipped_steps == ["step_3"]


def test_executor_does_not_retry_failed_tool() -> None:
    calls: list[str] = []
    tool = SpyTool("failing_tool", calls, fail=True)
    plan = _plan((_step("step_1", "failing_tool"),))

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert result.success is False
    assert calls == ["failing_tool"]


def test_retry_policy_retries_transient_failure_and_then_completes() -> None:
    calls: list[str] = []
    tool = SequenceTool(
        "safe_tool",
        calls,
        [
            {"success": False, "error": "temporary unavailable", "error_code": "TEMPORARY_UNAVAILABLE"},
            "done",
        ],
    )
    plan = _plan((_step("step_1", "safe_tool", arguments={"path": "same.txt"}),))

    result = ExecutionPlanExecutor(
        _registry(tool),
        retry_policy=RetryPolicy(max_attempts=2),
    ).execute(plan, _validation(plan))

    assert result.success is True
    assert calls == ["safe_tool", "safe_tool"]
    assert tool.contexts[0].parameters == tool.contexts[1].parameters == {"path": "same.txt"}
    assert tool.contexts[0].step_id == tool.contexts[1].step_id == "step_1"
    assert result.step_results[0].metadata["attempt_number"] == 2
    assert result.step_results[0].metadata["completed_after_retry"] is True
    assert len(result.step_results[0].metadata["retry_history"]) == 1


def test_retry_policy_stops_at_max_attempts() -> None:
    calls: list[str] = []
    tool = SequenceTool(
        "safe_tool",
        calls,
        [
            {"success": False, "error": "transient", "error_code": "TRANSIENT_ERROR"},
            {"success": False, "error": "transient", "error_code": "TRANSIENT_ERROR"},
            "must not run",
        ],
    )
    plan = _plan((_step("step_1", "safe_tool"),))

    result = ExecutionPlanExecutor(
        _registry(tool),
        retry_policy=RetryPolicy(max_attempts=2),
    ).execute(plan, _validation(plan))

    assert result.success is False
    assert calls == ["safe_tool", "safe_tool"]
    assert result.step_results[0].metadata["retry_exhausted"] is True
    assert result.step_results[0].metadata["attempt_number"] == 2


def test_retry_policy_does_not_retry_non_retryable_error() -> None:
    calls: list[str] = []
    tool = SequenceTool(
        "safe_tool",
        calls,
        [
            {"success": False, "error": "permission denied", "error_code": "PERMISSION_DENIED"},
            "must not run",
        ],
    )
    plan = _plan((_step("step_1", "safe_tool"),))

    result = ExecutionPlanExecutor(
        _registry(tool),
        retry_policy=RetryPolicy(max_attempts=2),
    ).execute(plan, _validation(plan))

    assert result.success is False
    assert calls == ["safe_tool"]
    assert result.step_results[0].metadata["retry_reason"] == "non_retryable_error"


def test_retry_policy_does_not_retry_unknown_exception() -> None:
    calls: list[str] = []
    tool = SequenceTool("safe_tool", calls, [RuntimeError("boom"), "must not run"])
    plan = _plan((_step("step_1", "safe_tool"),))

    result = ExecutionPlanExecutor(
        _registry(tool),
        retry_policy=RetryPolicy(max_attempts=2),
    ).execute(plan, _validation(plan))

    assert result.success is False
    assert calls == ["safe_tool"]
    assert result.step_results[0].error_code == ExecutionErrorCode.TOOL_EXCEPTION.value
    assert result.step_results[0].metadata["retry_reason"] == "non_retryable_error"


def test_confirmed_dangerous_step_can_retry_same_validated_step() -> None:
    calls: list[str] = []
    tool = SequenceTool(
        "write_file",
        calls,
        [
            {"success": False, "error": "temporary unavailable", "error_code": "TEMPORARY_UNAVAILABLE"},
            "written",
        ],
        requires_confirmation=True,
    )
    plan = _plan(
        (_step("step_1", "write_file", arguments={"path": "out.txt", "content": "x"}),),
        requires_confirmation=True,
    )

    result = ExecutionPlanExecutor(
        _registry(tool),
        retry_policy=RetryPolicy(max_attempts=2),
    ).execute(plan, _validation(plan), confirmation_granted=True)

    assert result.success is True
    assert calls == ["write_file", "write_file"]
    assert tool.contexts[0].parameters == tool.contexts[1].parameters


def test_confirmation_pending_blocks_before_retry_policy() -> None:
    calls: list[str] = []
    tool = SequenceTool("write_file", calls, ["must not run"], requires_confirmation=True)
    plan = _plan((_step("step_1", "write_file"),), requires_confirmation=True)

    result = ExecutionPlanExecutor(
        _registry(tool),
        retry_policy=RetryPolicy(max_attempts=2),
    ).execute(plan, _validation(plan))

    assert result.plan_status == PlanExecutionStatus.BLOCKED_CONFIRMATION.value
    assert calls == []


def test_cancellation_during_retry_delay_prevents_next_attempt() -> None:
    calls: list[str] = []
    tool = SequenceTool(
        "safe_tool",
        calls,
        [
            {"success": False, "error": "temporary unavailable", "error_code": "TEMPORARY_UNAVAILABLE"},
            "must not run",
        ],
    )
    control_calls = {"count": 0}

    def should_cancel() -> bool:
        control_calls["count"] += 1
        return control_calls["count"] > 1

    result = ExecutionPlanExecutor(
        _registry(tool),
        retry_policy=RetryPolicy(max_attempts=2, delay_ms=1),
    ).execute(
        _plan((_step("step_1", "safe_tool"),)),
        _validation(_plan((_step("step_1", "safe_tool"),))),
        control=ExecutionControl(should_cancel=should_cancel),
    )

    assert result.plan_status == PlanExecutionStatus.CANCELLED.value
    assert calls == ["safe_tool"]


def test_executor_does_not_call_planner_or_replan() -> None:
    source = Path("core/execution_plan_executor.py").read_text(encoding="utf-8")

    assert "Planner" not in source
    assert "create_execution_plan" not in source


def test_executor_does_not_eval_or_interpret_argument_strings() -> None:
    calls: list[str] = []
    tool = SpyTool("safe_tool", calls)
    payload = "__import__('os').system('echo no')"
    plan = _plan((_step("step_1", "safe_tool", arguments={"payload": payload}),))

    result = ExecutionPlanExecutor(_registry(tool)).execute(plan, _validation(plan))

    assert result.success is True
    assert tool.contexts[0].parameters["payload"] == payload
    source = Path("core/execution_plan_executor.py").read_text(encoding="utf-8")
    assert "eval(" not in source
    assert "exec(" not in source

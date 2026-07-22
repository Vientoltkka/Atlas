from __future__ import annotations

import json
from dataclasses import replace

import pytest

from core.execution_context import (
    ExecutionContext,
    ExecutionStepState,
    ExecutionStepStateTransitionError,
)
from core.execution_condition import ExecutionCondition, ExecutionConditionOperator
from core.execution_dependency_checker import ExecutionDependencyChecker
from core.execution_plan_executor import (
    ExecutionErrorCode,
    ExecutionPlanExecutor,
    PlanExecutionStatus,
    ResumableExecutionState,
    StepExecutionStatus,
)
from core.execution_plan_validator import ExecutionPlanValidator, plan_signature
from core.execution_variable_binding import ExecutionVariableBinding
from core.execution_variable_reference import ExecutionVariableReference
from core.planner import ExecutionPlan, ExecutionStep
from core.resumable_execution_store import JsonResumableExecutionStore
from core.step_output_reference import StepOutputReference
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(self, name: str, calls: list[str], output: object = "ok") -> None:
        self._name = name
        self._calls = calls
        self._output = output

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Spy tool {self._name}."

    def execute(self, context: ToolContext) -> object:
        self._calls.append(self._name)
        return self._output


class CountingConditionEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, condition: ExecutionCondition, context: ExecutionContext):
        self.calls += 1
        from core.execution_condition import ExecutionConditionResult

        return ExecutionConditionResult(
            matched=bool(condition.left),
            operator=condition.operator.value,
        )


def _step(
    step_id: str,
    tool: str | None = "safe_tool",
    *,
    depends_on: tuple[str, ...] = (),
    status: str = "pending",
    arguments: dict[str, object] | None = None,
    condition: object | None = None,
    output_binding: ExecutionVariableBinding | None = None,
) -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        description=f"Execute {step_id}.",
        tool=tool,
        depends_on=depends_on,
        status=status,
        arguments={} if arguments is None else arguments,
        condition=condition,
        output_binding=output_binding,
    )


def _plan(steps: tuple[ExecutionStep, ...]) -> ExecutionPlan:
    tools = tuple(
        dict.fromkeys(
            step.tool
            for step in steps
            if step.tool is not None and step.tool != "direct_response"
        )
    )
    return ExecutionPlan(
        goal="Execute dependencies.",
        ordered_steps=steps,
        estimated_steps=len(steps),
        required_tools=tools,
        detected_risks=(),
        requires_confirmation=False,
    )


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _validate(plan: ExecutionPlan):
    return ExecutionPlanValidator().validate(plan)


def test_execution_step_accepts_depends_on_and_copies_sequence() -> None:
    external = ["read"]

    step = _step("summarize", depends_on=external)  # type: ignore[arg-type]
    external.append("other")

    assert step.depends_on == ("read",)
    assert step.dependencies == ("read",)


def test_validator_rejects_invalid_duplicate_self_cycles_and_limits() -> None:
    too_many = tuple(f"step_{index}" for index in range(65))
    plan = _plan(
        (
            _step("step_1", depends_on=("step_2",)),
            _step("step_2", depends_on=("step_1", "step_1", "", 3)),  # type: ignore[arg-type]
            _step("step_3", depends_on=("step_3",)),
            _step("step_4", depends_on=too_many),
        )
    )

    result = _validate(plan)

    assert result.is_valid is False
    joined = "\n".join(result.errors)
    assert "duplicate dependency 'step_1'" in joined
    assert "dependency at position 2 cannot be empty" in joined
    assert "dependency at position 3 must be a string" in joined
    assert "cannot depend on itself" in joined
    assert "Circular dependency detected" in joined
    assert "TooManyStepDependenciesError" in joined


def test_validator_requires_direct_depends_on_for_step_output_references() -> None:
    invalid = _plan(
        (
            _step("read"),
            _step("write", arguments={"value": StepOutputReference("read")}),
        )
    )
    valid = replace(
        invalid,
        ordered_steps=(
            _step("read"),
            _step(
                "write",
                depends_on=("read",),
                arguments={"value": StepOutputReference("read")},
            ),
        ),
    )

    assert any("ImplicitStepDependencyError" in error for error in _validate(invalid).errors)
    assert _validate(valid).is_valid is True


def test_validator_requires_condition_step_output_reference_depends_on() -> None:
    plan = _plan(
        (
            _step("read"),
            _step(
                "guarded",
                condition=ExecutionCondition(
                    StepOutputReference("read"),
                    ExecutionConditionOperator.EXISTS,
                ),
            ),
        )
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert any("ImplicitStepDependencyError" in error for error in result.errors)


def test_execution_variable_reference_does_not_require_depends_on() -> None:
    plan = _plan(
        (
            _step(
                "use_variable",
                arguments={"path": ExecutionVariableReference("workspace_path")},
            ),
        )
    )

    assert _validate(plan).is_valid is True


@pytest.mark.parametrize(
    ("state", "satisfied"),
    [
        (ExecutionStepState.SUCCESS.value, True),
        (ExecutionStepState.FAILED.value, False),
        (ExecutionStepState.CANCELLED.value, False),
        (ExecutionStepState.SKIPPED.value, False),
        (ExecutionStepState.PENDING.value, False),
        (ExecutionStepState.RUNNING.value, False),
    ],
)
def test_dependency_checker_state_semantics(state: str, satisfied: bool) -> None:
    context = ExecutionContext("exec-dependency-check")
    if state == ExecutionStepState.SUCCESS.value:
        context.mark_step_started("dependency", 1)
        context.mark_step_succeeded("dependency", "ok")
    elif state == ExecutionStepState.FAILED.value:
        context.mark_step_started("dependency", 1)
        context.mark_step_failed("dependency")
    elif state == ExecutionStepState.CANCELLED.value:
        context.mark_step_started("dependency", 1)
        context.mark_step_cancelled("dependency")
    elif state == ExecutionStepState.SKIPPED.value:
        context.mark_step_skipped("dependency")
    elif state == ExecutionStepState.RUNNING.value:
        context.mark_step_started("dependency", 1)

    result = ExecutionDependencyChecker().check(
        _step("dependent", depends_on=("dependency",)),
        context,
    )

    assert result.satisfied is satisfied
    if not satisfied:
        assert result.blocking_dependency_ids == ("dependency",)
        assert result.blocking_states["dependency"] == state


def test_executor_blocks_before_condition_resolution_tool_binding_and_retry() -> None:
    calls: list[str] = []
    evaluator = CountingConditionEvaluator()
    plan = _plan(
        (
            _step("dependency", tool="first"),
            _step(
                "dependent",
                tool="second",
                depends_on=("dependency",),
                arguments={"path": ExecutionVariableReference("missing")},
                condition=ExecutionCondition(False, ExecutionConditionOperator.TRUTHY),
                output_binding=ExecutionVariableBinding("should_not_exist"),
            ),
        )
    )
    context = ExecutionContext("exec-blocked")
    context.mark_step_skipped("dependency")

    result = ExecutionPlanExecutor(
        _registry(SpyTool("first", calls), SpyTool("second", calls)),
        condition_evaluator=evaluator,  # type: ignore[arg-type]
    ).execute(plan, _validate(plan), execution_context=context)

    assert result.plan_status == PlanExecutionStatus.BLOCKED.value
    assert result.blocked is True
    assert result.failed is False
    assert result.skipped_steps == ["dependency"]
    assert result.failed_steps == []
    assert result.blocked_steps == ["dependent"]
    assert result.step_results[0].status == StepExecutionStatus.BLOCKED.value
    assert context.state_for_step("dependency") == ExecutionStepState.SKIPPED.value
    assert context.state_for_step("dependent") == ExecutionStepState.BLOCKED.value
    assert context.has_result("dependent") is False
    assert context.has_variable("should_not_exist") is False
    assert evaluator.calls == 0
    assert calls == []
    assert result.metrics is not None
    assert result.metrics.blocked_steps == 1
    assert result.metrics.failed_steps == 0
    assert result.metrics.skipped_steps == 0


def test_executor_runs_future_physical_dependency_before_consumer() -> None:
    calls: list[str] = []
    read_tool = SpyTool("read_tool", calls, output={"content": "alpha"})
    consume_tool = SpyTool("consume_tool", calls)
    plan = _plan(
        (
            _step(
                "consume",
                tool="consume_tool",
                depends_on=("read",),
                arguments={"content": StepOutputReference("read", ("content",))},
            ),
            _step("read", tool="read_tool"),
        )
    )

    result = ExecutionPlanExecutor(
        _registry(read_tool, consume_tool)
    ).execute(plan, _validate(plan))

    assert result.success is True
    assert calls == ["read_tool", "consume_tool"]
    assert result.completed_steps == ["read", "consume"]
    assert result.partial_state is not None
    assert result.partial_state.completed_step_ids == ("read", "consume")
    assert plan.ordered_steps[0].id == "consume"
    assert read_tool._calls == calls


def test_executor_observes_topological_reorder_without_mutating_plan() -> None:
    calls: list[str] = []
    plan = _plan(
        (
            _step("dependent", depends_on=("dependency",)),
            _step("dependency"),
        )
    )

    result = ExecutionPlanExecutor(
        _registry(SpyTool("safe_tool", calls))
    ).execute(plan, _validate(plan))

    assert result.success is True
    assert plan.ordered_steps[0].id == "dependent"
    assert result.trace is not None
    actions = [event.action for event in result.trace.events]
    assert "execution_topology_started" in actions
    assert "execution_topology_succeeded" in actions
    assert "execution_plan_reordered" in actions
    reorder_events = [
        event
        for event in result.trace.events
        if event.action == "execution_plan_reordered"
    ]
    assert reorder_events[0].details["ordered_step_ids"] == [
        "dependency",
        "dependent",
    ]


def test_executor_reports_running_dependency_state_inconsistency() -> None:
    calls: list[str] = []
    plan = _plan(
        (
            _step("dependency", status="completed"),
            _step("dependent", depends_on=("dependency",)),
        )
    )
    context = ExecutionContext("exec-inconsistent")
    context.mark_step_started("dependency", 1)
    validation = replace(_validate(plan), is_valid=True, errors=[], status="valid")

    result = ExecutionPlanExecutor(
        _registry(SpyTool("safe_tool", calls))
    ).execute(plan, validation, execution_context=context)

    assert result.plan_status == PlanExecutionStatus.REJECTED.value
    assert result.failed is True
    assert result.blocked is False
    assert result.error_code == ExecutionErrorCode.DEPENDENCY_STATE_INCONSISTENCY.value
    assert result.failed_steps == ["dependent"]
    assert result.blocked_steps == []
    assert result.step_results[0].error_code == (
        ExecutionErrorCode.DEPENDENCY_STATE_INCONSISTENCY.value
    )
    assert "dependency:RUNNING" in (result.error or "")
    assert calls == []


def test_failed_dependency_blocks_transitive_dependents() -> None:
    calls: list[str] = []
    plan = _plan(
        (
            _step("root", tool="root_tool"),
            _step("child", tool="safe_tool", depends_on=("root",)),
            _step("grandchild", tool="safe_tool", depends_on=("child",)),
            _step("independent", tool="safe_tool"),
        )
    )

    result = ExecutionPlanExecutor(
        _registry(
            SpyTool("root_tool", calls, output={"success": False, "error": "boom"}),
            SpyTool("safe_tool", calls),
        )
    ).execute(plan, _validate(plan))

    assert result.failed is True
    assert result.failed_steps == ["root"]
    assert result.blocked_steps == ["child", "grandchild"]
    assert result.skipped_steps == ["independent"]
    assert [step.step_id for step in result.step_results] == [
        "root",
        "child",
        "grandchild",
        "independent",
    ]
    assert [step.status for step in result.step_results] == [
        StepExecutionStatus.FAILED.value,
        StepExecutionStatus.BLOCKED.value,
        StepExecutionStatus.BLOCKED.value,
        StepExecutionStatus.SKIPPED.value,
    ]
    assert calls == ["root_tool"]


def test_blocked_state_is_terminal_and_resume_does_not_repeat_it() -> None:
    context = ExecutionContext("exec-resume-blocked")
    context.mark_step_blocked("blocked")

    with pytest.raises(ExecutionStepStateTransitionError):
        context.mark_step_started("blocked", 1)

    calls: list[str] = []
    plan = _plan((_step("blocked"), _step("pending", tool="safe_tool")))
    validation = _validate(plan)
    state = ResumableExecutionState(
        objective="resume blocked",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=(),
        pending_step_ids=("pending",),
        failed_step_ids=(),
        interrupted_step_id="pending",
        previous_results={},
        resumable=True,
        execution_context_snapshot=context.snapshot(),
    )

    result = ExecutionPlanExecutor(_registry(SpyTool("safe_tool", calls))).resume(state)

    assert result.success is True
    assert result.blocked_steps == ["blocked"]
    assert result.completed_steps == ["pending"]
    assert calls == ["safe_tool"]


def test_plan_signature_changes_for_dependency_content_and_order() -> None:
    first = _plan(
        (
            _step("a"),
            _step("b"),
            _step("c", depends_on=("a", "b")),
        )
    )
    reordered = replace(
        first,
        ordered_steps=(
            _step("a"),
            _step("b"),
            _step("c", depends_on=("b", "a")),
        ),
    )
    fewer = replace(
        first,
        ordered_steps=(
            _step("a"),
            _step("b"),
            _step("c", depends_on=("a",)),
        ),
    )

    assert plan_signature(first) != plan_signature(reordered)
    assert plan_signature(first) != plan_signature(fewer)
    assert plan_signature(first) == plan_signature(replace(first))


def test_json_checkpoint_persists_depends_on_and_loads_legacy_dependencies(tmp_path) -> None:
    plan = _plan((_step("read"), _step("write", depends_on=("read",))))
    validation = _validate(plan)
    context = ExecutionContext("exec-store-depends-on")
    context.mark_step_started("read", 1)
    context.mark_step_succeeded("read", "ok")
    state = ResumableExecutionState(
        objective="resume",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=("read",),
        pending_step_ids=("write",),
        failed_step_ids=(),
        interrupted_step_id="write",
        previous_results={"read": "ok"},
        resumable=True,
        execution_context_snapshot=context.snapshot(),
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)
    payload = json.loads(store.path.read_text(encoding="utf-8"))

    assert payload["original_plan"]["ordered_steps"][1]["depends_on"] == ["read"]
    payload["original_plan"]["ordered_steps"][1]["dependencies"] = payload[
        "original_plan"
    ]["ordered_steps"][1].pop("depends_on")
    store.path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = store.load()

    assert loaded is not None
    assert loaded.original_plan.ordered_steps[1].depends_on == ("read",)


def test_json_checkpoint_rejects_blocked_step_with_result(tmp_path) -> None:
    plan = _plan((_step("blocked"), _step("pending")))
    validation = _validate(plan)
    context = ExecutionContext("exec-blocked-result")
    context.mark_step_blocked("blocked")
    context.set_result("blocked", "bad")
    state = ResumableExecutionState(
        objective="resume",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=(),
        pending_step_ids=("pending",),
        failed_step_ids=(),
        interrupted_step_id="pending",
        previous_results={},
        resumable=True,
        execution_context_snapshot=context.snapshot(),
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)

    with pytest.raises(Exception):
        store.load()

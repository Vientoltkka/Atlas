from __future__ import annotations

import json

from core.execution_metrics import ExecutionMetricsCalculator
from core.execution_plan_executor import ExecutionControl, ExecutionPlanExecutor, PlanExecutionResult, ResumableExecutionState
from core.execution_plan_validator import ExecutionPlanValidator, plan_signature
from core.execution_replanner import (
    ExecutionReplanner,
    ReplanningCandidate,
    ReplanningDecision,
    ReplanningPolicy,
    ReplanningStatus,
    ReplanningStrategy,
)
from core.execution_trace import ExecutionTrace, TraceEventStatus
from core.goal_driven_execution import (
    GoalDrivenExecutionController,
    GoalDrivenExecutionCycle,
    GoalDrivenExecutionDecision,
    GoalDrivenExecutionPolicy,
    GoalDrivenExecutionRequest,
    GoalDrivenExecutionStatus,
    goal_driven_cycles_from_dict,
    goal_driven_cycles_to_dict,
    goal_driven_policy_from_dict,
    goal_driven_policy_to_dict,
)
from core.goal_verifier import GoalVerificationReason, GoalVerifier, OutputValidatorKind
from core.planner import ExecutionPlan, ExecutionStep
from core.resumable_execution_store import JsonResumableExecutionStore
from core.step_output_reference import StepOutputReference
from tools.base_tool import BaseTool
from tools.filesystem.list_directory_tool import ListDirectoryTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class ValueTool(BaseTool):
    def __init__(self, value: object = "ok", *, name: str = "demo.value", fail: bool = False) -> None:
        self._name = name
        self._value = value
        self._fail = fail
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Safe value test tool."

    def execute(self, context: ToolContext) -> object:
        del context
        self.calls += 1
        if self._fail:
            raise RuntimeError("tool failed")
        return self._value


class SamePlanReplanner(ExecutionReplanner):
    def decide(self, policy, request):  # type: ignore[override]
        del policy
        return ReplanningDecision(
            should_replan=True,
            status=ReplanningStatus.REPLANNED,
            reason="same plan returned",
            replacement_plan=request.failed_plan,
            replan_attempts=1,
            previous_plan_signature=plan_signature(request.failed_plan),
            replacement_plan_signature=plan_signature(request.failed_plan),
        )


class BrokenGoalVerifier(GoalVerifier):
    def verify(self, plan, execution_result, *, trace=None):  # type: ignore[override]
        del plan, execution_result, trace
        raise RuntimeError("goal verifier failed")


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _controller(
    registry: ToolRegistry,
    *,
    goal_verifier: GoalVerifier | None = None,
    execution_replanner: ExecutionReplanner | None = None,
) -> GoalDrivenExecutionController:
    return GoalDrivenExecutionController(
        ExecutionPlanValidator(registry),
        ExecutionPlanExecutor(registry),
        goal_verifier=goal_verifier,
        execution_replanner=execution_replanner,
    )


def _plan(
    *,
    step_id: str = "step_1",
    tool: str = "demo.value",
    output_name: str = "value",
    required_outputs: tuple[str, ...] = ("value",),
    validators: dict[str, tuple[str, ...]] | None = None,
) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Produce a value.",
        ordered_steps=(
            ExecutionStep(
                step_id,
                "Produce value.",
                tool,
                arguments={},
            ),
        ),
        estimated_steps=1,
        required_tools=(tool,),
        detected_risks=(),
        requires_confirmation=False,
        output={output_name: StepOutputReference(step_id)},
        required_outputs=required_outputs,
        output_validators={} if validators is None else validators,
    )


def _policy(
    *,
    max_cycles: int = 2,
    allow_replanning: bool = False,
) -> GoalDrivenExecutionPolicy:
    return GoalDrivenExecutionPolicy(
        enabled=True,
        max_cycles=max_cycles,
        allow_replanning=allow_replanning,
        replanning_policy=(
            ReplanningPolicy(
                enabled=True,
                max_replans=2,
                strategy=ReplanningStrategy.ALTERNATIVE_PLAN,
                retryable_goal_reasons=(GoalVerificationReason.MISSING_REQUIRED_OUTPUTS.value,),
                retryable_execution_errors=("TOOL_EXCEPTION",),
            )
            if allow_replanning
            else None
        ),
    )


def test_policy_disabled_returns_structured_status() -> None:
    tool = ValueTool()
    result = _controller(_registry(tool)).execute(
        GoalDrivenExecutionRequest(_plan(), policy=GoalDrivenExecutionPolicy())
    )

    assert result.status is GoalDrivenExecutionStatus.POLICY_DISABLED
    assert tool.calls == 0


def test_goal_satisfied_in_first_cycle() -> None:
    tool = ValueTool("done")
    result = _controller(_registry(tool)).execute(
        GoalDrivenExecutionRequest(_plan(), policy=_policy())
    )

    assert result.status is GoalDrivenExecutionStatus.COMPLETED
    assert result.completed is True
    assert len(result.cycles) == 1
    assert result.cycles[0].decision is GoalDrivenExecutionDecision.FINISH_GOAL_SATISFIED


def test_goal_unsatisfied_without_replanning_allowed() -> None:
    tool = ValueTool("done")
    result = _controller(_registry(tool)).execute(
        GoalDrivenExecutionRequest(
            _plan(output_name="other", required_outputs=("value",)),
            policy=_policy(allow_replanning=False),
        )
    )

    assert result.status is GoalDrivenExecutionStatus.GOAL_UNSATISFIED
    assert result.cycles[0].decision is GoalDrivenExecutionDecision.FINISH_GOAL_UNSATISFIED


def test_execution_failure_without_alternative() -> None:
    tool = ValueTool(fail=True)
    result = _controller(_registry(tool)).execute(
        GoalDrivenExecutionRequest(_plan(), policy=_policy(allow_replanning=False))
    )

    assert result.status is GoalDrivenExecutionStatus.EXECUTION_FAILED
    assert result.execution_result is not None
    assert result.execution_result.success is False


def test_replanning_successfully_satisfies_goal_in_next_cycle() -> None:
    tool = ValueTool("done")
    failed = _plan(output_name="other", required_outputs=("value",))
    replacement = _plan(step_id="step_2", output_name="value")

    result = _controller(_registry(tool)).execute(
        GoalDrivenExecutionRequest(
            failed,
            policy=_policy(allow_replanning=True),
            candidates=(ReplanningCandidate(replacement),),
        )
    )

    assert result.status is GoalDrivenExecutionStatus.COMPLETED
    assert len(result.cycles) == 2
    assert result.cycles[0].decision is GoalDrivenExecutionDecision.SELECT_REPLANNED_PLAN
    assert result.cycles[1].decision is GoalDrivenExecutionDecision.FINISH_GOAL_SATISFIED


def test_replanning_same_signature_is_rejected() -> None:
    plan = _plan(output_name="other", required_outputs=("value",))
    result = _controller(_registry(ValueTool()), execution_replanner=SamePlanReplanner()).execute(
        GoalDrivenExecutionRequest(plan, policy=_policy(allow_replanning=True))
    )

    assert result.status is GoalDrivenExecutionStatus.NO_ALTERNATIVE_PLAN
    assert result.cycles[0].decision is GoalDrivenExecutionDecision.REJECT_REPLANNING


def test_repeated_signatures_in_history_are_not_reused() -> None:
    tool = ValueTool("done")
    first = _plan(output_name="other", required_outputs=("value",))
    second = _plan(step_id="step_2", output_name="other", required_outputs=("value",))

    result = _controller(_registry(tool)).execute(
        GoalDrivenExecutionRequest(
            first,
            policy=_policy(max_cycles=3, allow_replanning=True),
            candidates=(ReplanningCandidate(second), ReplanningCandidate(second)),
        )
    )

    assert result.status is GoalDrivenExecutionStatus.NO_ALTERNATIVE_PLAN
    assert result.used_plan_signatures == (plan_signature(first), plan_signature(second))


def test_cycle_limit_is_terminal_failure() -> None:
    result = _controller(_registry(ValueTool())).execute(
        GoalDrivenExecutionRequest(
            _plan(output_name="other", required_outputs=("value",)),
            policy=_policy(max_cycles=1, allow_replanning=True),
            candidates=(ReplanningCandidate(_plan(step_id="step_2")),),
        )
    )

    assert result.status is GoalDrivenExecutionStatus.CYCLE_LIMIT_REACHED


def test_cancellation_returns_cancelled() -> None:
    result = _controller(_registry(ValueTool())).execute(
        GoalDrivenExecutionRequest(
            _plan(),
            policy=_policy(),
            control=ExecutionControl(should_cancel=lambda: True),
        )
    )

    assert result.status is GoalDrivenExecutionStatus.CANCELLED


def test_replanned_plan_validation_failure_is_terminal() -> None:
    invalid = ExecutionPlan(
        goal="Invalid.",
        ordered_steps=(),
        estimated_steps=0,
        required_tools=(),
        detected_risks=(),
        requires_confirmation=False,
    )
    result = _controller(_registry(ValueTool())).execute(
        GoalDrivenExecutionRequest(
            _plan(output_name="other", required_outputs=("value",)),
            policy=_policy(max_cycles=2, allow_replanning=True),
            candidates=(ReplanningCandidate(invalid),),
        )
    )

    assert result.status is GoalDrivenExecutionStatus.VALIDATION_FAILED


def test_goal_verifier_failure_is_internal_error() -> None:
    result = _controller(_registry(ValueTool()), goal_verifier=BrokenGoalVerifier()).execute(
        GoalDrivenExecutionRequest(_plan(), policy=_policy())
    )

    assert result.status is GoalDrivenExecutionStatus.INTERNAL_ERROR
    assert result.error_code == "RuntimeError"


def test_old_single_execution_flow_stays_compatible() -> None:
    tool = ValueTool("done")
    controller = _controller(_registry(tool))

    disabled = controller.execute(GoalDrivenExecutionRequest(_plan(), policy=GoalDrivenExecutionPolicy()))
    enabled = controller.execute(GoalDrivenExecutionRequest(_plan(), policy=_policy()))

    assert disabled.status is GoalDrivenExecutionStatus.POLICY_DISABLED
    assert enabled.status is GoalDrivenExecutionStatus.COMPLETED
    assert tool.calls == 1


def test_policy_cycle_serialization_and_checkpoint_roundtrip(tmp_path) -> None:
    policy = _policy(allow_replanning=True)
    cycle = GoalDrivenExecutionCycle(
        1,
        "abc",
        "exec-1",
        "completed",
        None,
        GoalDrivenExecutionDecision.SELECT_REPLANNED_PLAN,
        replanned_plan_signature="def",
        termination_reason="replanned",
    )

    loaded_policy = goal_driven_policy_from_dict(goal_driven_policy_to_dict(policy))
    loaded_cycles = goal_driven_cycles_from_dict(goal_driven_cycles_to_dict((cycle,)))

    assert loaded_policy == policy
    assert loaded_cycles == (cycle,)

    plan = _plan()
    validation = ExecutionPlanValidator().validate(plan)
    state = ResumableExecutionState(
        objective="goal-driven",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=(),
        pending_step_ids=("step_1",),
        failed_step_ids=(),
        interrupted_step_id="step_1",
        resumable=True,
        goal_driven_policy=policy,
        goal_driven_cycle=1,
        goal_driven_history=(cycle,),
        goal_driven_used_signatures=("abc",),
        goal_driven_last_decision=cycle.decision.value,
        goal_driven_terminal_status=GoalDrivenExecutionStatus.CYCLE_LIMIT_REACHED.value,
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")
    store.save(state)
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    loaded = store.load()

    assert payload["goal_driven_policy"]["enabled"] is True
    assert loaded is not None
    assert loaded.goal_driven_policy == policy
    assert loaded.goal_driven_history == (cycle,)
    assert loaded.goal_driven_terminal_status == "CYCLE_LIMIT_REACHED"


def test_goal_driven_events_and_metrics_are_counted() -> None:
    trace = ExecutionTrace("goal-driven")
    for action, status in (
        ("goal_driven_execution_started", "STARTED"),
        ("goal_driven_cycle_started", "STARTED"),
        ("goal_driven_plan_executed", "FINISHED"),
        ("goal_driven_goal_verified", "FINISHED"),
        ("goal_driven_replanning_requested", "STARTED"),
        ("goal_driven_replanning_selected", "FINISHED"),
        ("goal_driven_cycle_completed", "FINISHED"),
        ("goal_driven_execution_succeeded", "FINISHED"),
        ("goal_driven_cycle_limit_reached", "FAILED"),
    ):
        trace.add_event(component="GoalDrivenExecutionController", action=action, status=status)
    trace.finish("SUCCESS")

    metrics = ExecutionMetricsCalculator().calculate(trace)

    assert metrics.goal_driven_executions_started == 1
    assert metrics.goal_driven_executions_completed == 1
    assert metrics.goal_driven_cycles_completed == 1
    assert metrics.goal_driven_goals_satisfied == 1
    assert metrics.goal_driven_replans_requested == 1
    assert metrics.goal_driven_replans_succeeded == 1
    assert metrics.goal_driven_cycle_limits_reached == 1


def test_observability_events_do_not_include_sensitive_outputs() -> None:
    result = _controller(_registry(ValueTool({"api_token": "hidden"}))).execute(
        GoalDrivenExecutionRequest(_plan(), policy=_policy(), metadata={"request_id": "safe"})
    )

    serialized = " ".join(str(event.details) for event in result.events)

    assert "hidden" not in serialized
    assert "api_token" not in serialized
    assert "request_id" not in serialized


def test_e2e_real_read_only_tool_satisfies_goal(tmp_path) -> None:
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

    result = _controller(_registry(ListDirectoryTool())).execute(
        GoalDrivenExecutionRequest(plan, policy=_policy())
    )

    assert result.status is GoalDrivenExecutionStatus.COMPLETED
    assert result.execution_result is not None
    assert result.execution_result.output == {"entries": ["README.md"]}

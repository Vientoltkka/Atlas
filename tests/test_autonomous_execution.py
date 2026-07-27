from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.autonomous_execution import (
    AutonomousExecutionOptions,
    AutonomousExecutionOrchestrator,
    AutonomousExecutionOutcome,
    ExecutionResumeNotAllowedError,
)
from core.concurrent_step_executor import ConcurrentStepExecutor, ExecutionConcurrencyPolicy
from core.execution_plan_executor import (
    PlanExecutionResult,
    PlanExecutionStatus,
    StepExecutionResult,
)
from core.execution_plan_validator import PlanValidationResult, plan_signature
from core.execution_priority import ExecutionPriorityPolicy
from core.execution_resources import (
    ExecutionBudget,
    ExecutionResourceCatalog,
    ExecutionResourcePolicy,
    ResourceCandidate,
    ResourceHealthStatus,
    ResourceType,
)
from core.execution_supervisor import ExecutionState, ExecutionSupervisor
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult


NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class _Clock:
    def __init__(self) -> None:
        self.current = NOW

    def __call__(self) -> datetime:
        value = self.current
        self.current = self.current + timedelta(seconds=1)
        return value


def _step(step_id: str, **kwargs) -> ExecutionStep:
    defaults = {
        "parallel_safe": True,
        "idempotent": True,
        "recovery_safe": True,
        "side_effect_free": True,
    }
    defaults.update(kwargs)
    return ExecutionStep(step_id, step_id, "read_file", **defaults)


def _plan(steps: tuple[ExecutionStep, ...], *, requires_confirmation: bool = False) -> ExecutionPlan:
    return ExecutionPlan(
        goal="autonomous",
        ordered_steps=steps,
        estimated_steps=len(steps),
        required_tools=("read_file",),
        detected_risks=("risk",) if requires_confirmation else (),
        requires_confirmation=requires_confirmation,
    )


def _candidate(resource_id: str, *, cost: float = 1.0) -> ResourceCandidate:
    return ResourceCandidate(
        resource_id=resource_id,
        resource_type=ResourceType.MODEL,
        provider_id="local",
        capabilities=("text",),
        estimated_cost=cost,
        health_status=ResourceHealthStatus.AVAILABLE,
    )


def test_execute_plan_linear_completes_with_structured_result() -> None:
    plan = _plan((_step("a"),))
    result = _orchestrator(plan).execute_plan(plan)

    assert result.outcome is AutonomousExecutionOutcome.COMPLETED
    assert result.final_state is ExecutionState.COMPLETED
    assert result.completed_step_ids == ("a",)
    assert result.session_id is not None
    assert result.summary
    assert result.trace.entries[-1].event_type == "final_result_built"


def test_execute_objective_uses_existing_planner() -> None:
    plan = _plan((_step("a"),))
    planner = _FixedPlanner(plan)

    result = _orchestrator(plan, planner=planner).execute_objective("build")

    assert result.outcome is AutonomousExecutionOutcome.COMPLETED
    assert planner.calls == 1


def test_dag_concurrent_execution_respects_dependencies_and_priority() -> None:
    calls: list[str] = []
    plan = _plan(
        (
            _step("root", priority=1),
            _step("slow", depends_on=("root",), priority=1),
            _step("fast", depends_on=("root",), priority=9),
        )
    )
    orchestrator = _orchestrator(plan, calls=calls, concurrent=True)

    result = orchestrator.execute_plan(
        plan,
        execution_options=AutonomousExecutionOptions(
            concurrency_policy=ExecutionConcurrencyPolicy(enabled=True, max_concurrency=2),
            priority_policy=ExecutionPriorityPolicy(enabled=True),
        ),
    )

    assert result.outcome is AutonomousExecutionOutcome.COMPLETED
    assert calls[0] == "root"
    assert calls[1:3] == ["fast", "slow"]


def test_dry_run_produces_plan_without_running_tools() -> None:
    calls: list[str] = []
    plan = _plan((_step("a"), _step("b", depends_on=("a",))))
    orchestrator = _orchestrator(plan, calls=calls, concurrent=True)

    result = orchestrator.execute_plan(
        plan,
        execution_options=AutonomousExecutionOptions(
            dry_run=True,
            concurrency_policy=ExecutionConcurrencyPolicy(enabled=True, max_concurrency=2),
            resource_policy=ExecutionResourcePolicy(enabled=True),
            execution_budget=ExecutionBudget(max_total_cost=10),
        ),
    )

    assert result.outcome is AutonomousExecutionOutcome.DRY_RUN
    assert calls == []
    assert result.simulation is not None
    assert result.simulation.planned_order == ("a", "b")
    assert result.simulation.planned_batches == (("a",), ("b",))
    assert result.budget_usage.estimated_cost == 2


def test_confirmation_pauses_and_resume_continues_same_session() -> None:
    calls: list[str] = []
    plan = _plan((_step("a"),), requires_confirmation=True)
    orchestrator = _orchestrator(plan, calls=calls)

    paused = orchestrator.execute_plan(plan)
    resumed = orchestrator.resume_execution(paused.session_id, confirmation=True)

    assert paused.outcome is AutonomousExecutionOutcome.WAITING_CONFIRMATION
    assert resumed.outcome is AutonomousExecutionOutcome.COMPLETED
    assert paused.session_id == resumed.session_id
    assert calls == ["a"]


def test_cancel_execution_is_idempotent_and_terminal_result_is_available() -> None:
    plan = _plan((_step("a"),), requires_confirmation=True)
    orchestrator = _orchestrator(plan)
    paused = orchestrator.execute_plan(plan)

    first = orchestrator.cancel_execution(paused.session_id)
    second = orchestrator.cancel_execution(paused.session_id)
    stored = orchestrator.get_execution_result(paused.session_id)

    assert first.outcome is AutonomousExecutionOutcome.CANCELLED
    assert second.outcome is AutonomousExecutionOutcome.CANCELLED
    assert stored.final_state is ExecutionState.CANCELLED


def test_budget_exhausted_and_incompatible_resource_do_not_run_tools() -> None:
    calls: list[str] = []
    plan = _plan((_step("a"),))
    orchestrator = _orchestrator(
        plan,
        calls=calls,
        catalog=ExecutionResourceCatalog((_candidate("expensive", cost=5),)),
        concurrent=True,
    )

    result = orchestrator.execute_plan(
        plan,
        execution_options=AutonomousExecutionOptions(
            concurrency_policy=ExecutionConcurrencyPolicy(enabled=True, max_concurrency=1),
            resource_policy=ExecutionResourcePolicy(enabled=True),
            execution_budget=ExecutionBudget(max_total_cost=1),
        ),
    )

    assert result.outcome in {
        AutonomousExecutionOutcome.BUDGET_EXHAUSTED,
        AutonomousExecutionOutcome.NO_COMPATIBLE_RESOURCE,
    }
    assert calls == []


def test_get_execution_result_rejects_active_session() -> None:
    plan = _plan((_step("a"),), requires_confirmation=True)
    orchestrator = _orchestrator(plan)
    paused = orchestrator.execute_plan(plan)

    with pytest.raises(ExecutionResumeNotAllowedError):
        orchestrator.get_execution_result(paused.session_id)


def test_terminal_session_is_not_resumed() -> None:
    plan = _plan((_step("a"),))
    orchestrator = _orchestrator(plan)
    result = orchestrator.execute_plan(plan)

    with pytest.raises(ExecutionResumeNotAllowedError):
        orchestrator.resume_execution(result.session_id)


def test_trace_is_immutable_ordered_and_sanitized() -> None:
    plan = _plan((_step("a"),))
    result = _orchestrator(plan).execute_plan(plan)

    assert tuple(entry.sequence for entry in result.trace.entries) == (1, 2, 3)
    with pytest.raises(AttributeError):
        result.trace.entries[0].summary = "changed"  # type: ignore[misc]
    assert "sk-" not in "".join(entry.summary for entry in result.trace.entries)


def _orchestrator(
    plan: ExecutionPlan,
    *,
    planner=None,
    calls: list[str] | None = None,
    catalog: ExecutionResourceCatalog | None = None,
    concurrent: bool = False,
) -> AutonomousExecutionOrchestrator:
    calls = [] if calls is None else calls

    def runner(step: ExecutionStep) -> str:
        calls.append(step.id)
        return step.id

    return AutonomousExecutionOrchestrator(
        planner=planner or _FixedPlanner(plan),
        validator=_Validator(),
        executor=_Executor(calls),
        supervisor=ExecutionSupervisor(clock=_Clock()),
        concurrent_step_executor=ConcurrentStepExecutor(runner) if concurrent else None,
        resource_catalog=catalog or ExecutionResourceCatalog((_candidate("model"),)),
        clock=_Clock(),
    )


class _FixedPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self.calls = 0

    def generate_execution_plan(self, _objective: str, **_kwargs) -> PlanGenerationResult:
        self.calls += 1
        return PlanGenerationResult(success=True, plan=self.plan, generation_attempted=True)


class _Validator:
    def validate(self, plan: ExecutionPlan) -> PlanValidationResult:
        return PlanValidationResult(
            is_valid=True,
            status="valid",
            requires_confirmation=plan.requires_confirmation,
            plan_signature=plan_signature(plan),
        )


class _Executor:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def execute(self, plan: ExecutionPlan, *_args, **_kwargs) -> PlanExecutionResult:
        self.calls.extend(step.id for step in plan.ordered_steps)
        return PlanExecutionResult(
            plan_status=PlanExecutionStatus.COMPLETED.value,
            success=True,
            completed=True,
            completed_steps=[step.id for step in plan.ordered_steps],
            step_results=[
                StepExecutionResult(
                    step_id=step.id,
                    status="completed",
                    success=True,
                    tool_name=step.tool,
                    output=step.id,
                )
                for step in plan.ordered_steps
            ],
        )

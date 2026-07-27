from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.concurrent_step_executor import ConcurrentStepExecutor, ExecutionConcurrencyPolicy
from core.execution_plan_executor import PlanExecutionResult, PlanExecutionStatus
from core.execution_plan_validator import ExecutionPlanValidator, PlanValidationResult, plan_signature
from core.execution_priority import (
    DependencyImpactAnalyzer,
    ExecutionPriorityPolicy,
    ReadyStepPrioritizer,
)
from core.execution_session_persistence import (
    ExecutionRecoveryService,
    ExecutionSessionSnapshot,
    FileExecutionSessionRepository,
    snapshot_from_dict,
    snapshot_to_dict,
)
from core.execution_supervisor import ExecutionState, ExecutionSupervisor
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult
from core.structured_execution import StructuredExecutionCoordinator
from core.structured_plan_replanner import (
    ReplanPolicy,
    ReplanReason,
    ReplanResult,
    ReplanResultStatus,
)


NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _step(step_id: str, **kwargs) -> ExecutionStep:
    defaults = {
        "parallel_safe": True,
        "idempotent": True,
        "recovery_safe": True,
        "side_effect_free": True,
    }
    defaults.update(kwargs)
    return ExecutionStep(step_id, step_id, "read_file", **defaults)


def _plan(steps: tuple[ExecutionStep, ...]) -> ExecutionPlan:
    return ExecutionPlan(
        goal="priority",
        ordered_steps=steps,
        estimated_steps=len(steps),
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
    )


def _prioritize(
    steps: tuple[ExecutionStep, ...],
    *,
    policy: ExecutionPriorityPolicy | None = None,
    ready_since: dict[str, datetime] | None = None,
    completed: tuple[str, ...] = (),
):
    plan = _plan(steps)
    return ReadyStepPrioritizer(policy or ExecutionPriorityPolicy(enabled=True)).prioritize(
        steps,
        plan=plan,
        completed_step_ids=completed,
        ready_since_by_step_id=ready_since or {},
        now=NOW,
    )


def test_disabled_policy_preserves_ready_order() -> None:
    steps = (_step("b", priority=10), _step("a", priority=100))

    decision = _prioritize(steps, policy=ExecutionPriorityPolicy(enabled=False))

    assert decision.ordered_step_ids == ("b", "a")
    assert decision.tie_breaker_used == "plan_order"


def test_declared_priority_urgency_and_criticality_affect_order() -> None:
    assert _prioritize((_step("low"), _step("high", priority=3))).ordered_step_ids[0] == "high"
    assert _prioritize((_step("low"), _step("urgent", urgency=3))).ordered_step_ids[0] == "urgent"
    assert _prioritize((_step("low"), _step("critical", criticality=3))).ordered_step_ids[0] == "critical"


def test_deadline_scores_vencida_proxima_lejana_and_none() -> None:
    decision = _prioritize(
        (
            _step("none"),
            _step("far", deadline=NOW + timedelta(days=5)),
            _step("soon", deadline=NOW + timedelta(minutes=20)),
            _step("late", deadline=NOW - timedelta(minutes=1)),
        )
    )

    assert decision.ordered_step_ids[:2] == ("late", "soon")
    scores = {score.step_id: score.deadline_score for score in decision.scores}
    assert scores["none"] == 0.0
    assert scores["soon"] > scores["far"]


def test_aging_and_starvation_limit() -> None:
    policy = ExecutionPriorityPolicy(enabled=True, age_weight=2, max_age_score=10)

    decision = _prioritize(
        (_step("new", priority=5), _step("old")),
        policy=policy,
        ready_since={"old": NOW - timedelta(hours=10), "new": NOW},
    )

    assert decision.ordered_step_ids[0] == "old"
    assert {score.step_id: score.age_score for score in decision.scores}["old"] == 10.0


def test_cost_duration_and_risk_penalties() -> None:
    policy = ExecutionPriorityPolicy(
        enabled=True,
        prefer_short_tasks=True,
        cost_weight=1,
        duration_weight=1,
        risk_weight=10,
    )
    safe = _step("safe", estimated_cost=1, estimated_duration_seconds=5)
    expensive = _step("expensive", estimated_cost=20, estimated_duration_seconds=50)
    risky = _step("risky", recovery_safe=False)

    decision = _prioritize((expensive, risky, safe), policy=policy)

    assert decision.ordered_step_ids[0] == "safe"
    scores = {score.step_id: score for score in decision.scores}
    assert scores["expensive"].cost_penalty == 20.0
    assert scores["expensive"].duration_penalty == 50.0
    assert scores["risky"].risk_penalty > scores["safe"].risk_penalty


def test_dependency_impact_does_not_overcount_convergence() -> None:
    plan = _plan(
        (
            _step("a"),
            _step("b"),
            ExecutionStep("join", "join", "read_file", ("a", "b")),
        )
    )
    scores = DependencyImpactAnalyzer().impact_scores(
        plan,
        ("a", "b"),
        completed_step_ids=(),
    )

    assert scores["a"] == 1.0
    assert scores["b"] == 1.0


def test_tie_break_uses_plan_order_then_step_id() -> None:
    decision = _prioritize((_step("b"), _step("a")))
    assert decision.ordered_step_ids == ("b", "a")
    assert decision.tie_breaker_used == "deterministic_tie_break"

    no_plan_order = _prioritize(
        (_step("b"), _step("a")),
        policy=ExecutionPriorityPolicy(enabled=True, preserve_plan_order_on_tie=False),
    )
    assert no_plan_order.ordered_step_ids == ("a", "b")


def test_prioritization_is_deterministic() -> None:
    steps = (_step("a", urgency=1), _step("b", priority=1))

    first = _prioritize(steps)
    second = _prioritize(steps)

    assert first.ordered_step_ids == second.ordered_step_ids
    assert first.scores == second.scores


def test_policy_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError):
        ExecutionPriorityPolicy(enabled=True, priority_weight=float("nan"))
    with pytest.raises(ValueError):
        ExecutionPriorityPolicy(enabled=True, cost_weight=-1)


def test_validator_rejects_invalid_priority_metadata() -> None:
    step = _step("a")
    object.__setattr__(step, "estimated_cost", float("inf"))

    result = ExecutionPlanValidator().validate(_plan((step,)))

    assert not result.is_valid
    assert "estimated_cost" in result.errors[0]


def test_concurrent_batch_follows_prioritized_order_and_max_concurrency() -> None:
    plan = _plan(
        (
            _step("slow", priority=1),
            _step("fast", priority=5),
            _step("middle", priority=3),
        )
    )
    calls: list[str] = []

    coordinator = _coordinator(
        plan,
        calls,
        concurrency_policy=ExecutionConcurrencyPolicy(enabled=True, max_concurrency=2),
        priority_policy=ExecutionPriorityPolicy(enabled=True),
    )

    response = coordinator.handle("run")

    assert response.status == "completed"
    assert calls[:2] == ["fast", "middle"]


def test_resource_conflict_keeps_prioritized_batch_compatible() -> None:
    plan = _plan(
        (
            _step("a", priority=5, resource_keys=("file",)),
            _step("b", priority=4, resource_keys=("file",)),
            _step("c", priority=3, resource_keys=("other",)),
        )
    )
    calls: list[str] = []

    coordinator = _coordinator(
        plan,
        calls,
        concurrency_policy=ExecutionConcurrencyPolicy(enabled=True, max_concurrency=2),
        priority_policy=ExecutionPriorityPolicy(enabled=True),
    )
    coordinator.handle("run")

    assert calls[:2] == ["a", "c"]


def test_non_parallel_safe_step_runs_alone_even_when_prioritized() -> None:
    plan = _plan((_step("unsafe", priority=9, parallel_safe=False), _step("safe", priority=1)))
    calls: list[str] = []

    coordinator = _coordinator(
        plan,
        calls,
        concurrency_policy=ExecutionConcurrencyPolicy(enabled=True, max_concurrency=2),
        priority_policy=ExecutionPriorityPolicy(enabled=True),
    )
    coordinator.handle("run")

    assert calls[:2] == ["unsafe", "safe"]


def test_non_ready_step_is_never_prioritized() -> None:
    root = _step("root", priority=1)
    child = ExecutionStep("child", "child", "read_file", ("root",), priority=99)
    plan = _plan((root, child))
    calls: list[str] = []

    _coordinator(
        plan,
        calls,
        concurrency_policy=ExecutionConcurrencyPolicy(enabled=True, max_concurrency=2),
        priority_policy=ExecutionPriorityPolicy(enabled=True),
    ).handle("run")

    assert calls[0] == "root"


def test_priority_decision_is_persisted_and_restored(tmp_path) -> None:
    supervisor = ExecutionSupervisor()
    session = supervisor.start(_plan((_step("a", priority=2), _step("b"))))
    decision = _prioritize(tuple(session.plan.ordered_steps))
    supervisor.record_priority_decision(session.session_id, decision)
    snapshot = ExecutionSessionSnapshot.from_session(supervisor.get_session(session.session_id))

    loaded = snapshot_from_dict(snapshot_to_dict(snapshot))

    assert loaded.last_priority_decision.ordered_step_ids == decision.ordered_step_ids
    assert loaded.priority_history[0].scores[0].step_id == "a"


def test_recovery_recalculates_priority_after_restore(tmp_path) -> None:
    repo = FileExecutionSessionRepository(tmp_path)
    supervisor = ExecutionSupervisor()
    session = supervisor.start(_plan((_step("old", priority=1), _step("new", priority=5))))
    old_decision = _prioritize(tuple(session.plan.ordered_steps))
    supervisor.record_priority_decision(session.session_id, old_decision)
    repo.save(ExecutionSessionSnapshot.from_session(supervisor.get_session(session.session_id)))
    restored_supervisor = ExecutionSupervisor()
    service = ExecutionRecoveryService(repo, restored_supervisor)
    service.recover()
    calls: list[str] = []
    coordinator = _coordinator(
        restored_supervisor.get_session(session.session_id).active_plan,
        calls,
        supervisor=restored_supervisor,
        recovery_service=service,
        concurrency_policy=ExecutionConcurrencyPolicy(enabled=True, max_concurrency=2),
        priority_policy=ExecutionPriorityPolicy(enabled=True),
    )

    coordinator.resume_recovered_session(session.session_id)
    restored = restored_supervisor.get_session(session.session_id)

    assert restored.priority_history[-1].generated_at == NOW
    assert restored.priority_history[-1].ordered_step_ids[0] == "new"


def test_interrupted_step_is_not_selected_automatically_by_recovery(tmp_path) -> None:
    repo = FileExecutionSessionRepository(tmp_path)
    supervisor = ExecutionSupervisor()
    session = supervisor.start(_plan((_step("a"),)))
    supervisor.mark_running(session.session_id)
    supervisor.mark_step_started(session.session_id, "a")
    repo.save(ExecutionSessionSnapshot.from_session(supervisor.get_session(session.session_id)))
    restored = ExecutionSupervisor()
    service = ExecutionRecoveryService(repo, restored)
    report = service.recover()

    assert report.decisions[session.session_id].decision.value == "require_manual_review"


def test_priority_history_is_immutable_and_limited() -> None:
    supervisor = ExecutionSupervisor()
    session = supervisor.start(_plan((_step("a"),)))
    decision = _prioritize(tuple(session.plan.ordered_steps))

    for _ in range(105):
        supervisor.record_priority_decision(session.session_id, decision)

    restored = supervisor.get_session(session.session_id)
    assert isinstance(restored.priority_history, tuple)
    assert len(restored.priority_history) == 100


def test_replan_request_includes_priority_context() -> None:
    plan = _plan((_step("a", priority=9), _step("b")))
    replanner = _FakeReplanner()

    coordinator = _coordinator(
        plan,
        [],
        concurrency_policy=ExecutionConcurrencyPolicy(enabled=True, max_concurrency=2, fail_fast=False),
        priority_policy=ExecutionPriorityPolicy(enabled=True),
        replanner=replanner,
        fail_step="a",
    )
    coordinator.handle("run")

    request = replanner.calls[0]
    assert request.selected_step_ids
    assert request.ordered_ready_step_ids[0] == "a"
    assert request.failed_step_priority is not None
    assert request.priority_rationale_summary


def _coordinator(
    plan: ExecutionPlan,
    calls: list[str],
    *,
    supervisor: ExecutionSupervisor | None = None,
    recovery_service: ExecutionRecoveryService | None = None,
    concurrency_policy: ExecutionConcurrencyPolicy,
    priority_policy: ExecutionPriorityPolicy,
    replanner=None,
    fail_step: str | None = None,
) -> StructuredExecutionCoordinator:
    def runner(step: ExecutionStep) -> str:
        calls.append(step.id)
        if step.id == fail_step:
            raise RuntimeError("boom")
        return step.id

    return StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=_NoopExecutor(),  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        concurrent_step_executor=ConcurrentStepExecutor(runner),
        concurrency_policy=concurrency_policy,
        priority_policy=priority_policy,
        priority_clock=lambda: NOW,
        recovery_service=recovery_service,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1) if replanner is not None else None,
    )


class _FixedPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self._plan = plan

    def generate_execution_plan(self, _objective: str, **_kwargs) -> PlanGenerationResult:
        return PlanGenerationResult(success=True, plan=self._plan, generation_attempted=True)


class _FixedValidator:
    def validate(self, plan: ExecutionPlan) -> PlanValidationResult:
        return PlanValidationResult(
            is_valid=True,
            status="valid",
            requires_confirmation=plan.requires_confirmation,
            plan_signature=plan_signature(plan),
        )


class _NoopExecutor:
    def execute(self, *_args, **_kwargs) -> PlanExecutionResult:
        return PlanExecutionResult(
            plan_status=PlanExecutionStatus.COMPLETED.value,
            success=True,
            completed=True,
        )


class _FakeReplanner:
    def __init__(self) -> None:
        self.calls = []

    def replan(self, request):
        self.calls.append(request)
        return ReplanResult(
            status=ReplanResultStatus.REJECTED,
            reason=ReplanReason.REJECTED,
            error="no replan",
        )

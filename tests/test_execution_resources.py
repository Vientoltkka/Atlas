from __future__ import annotations

from datetime import datetime, timezone
from threading import Thread

import pytest

from core.concurrent_step_executor import ConcurrentStepExecutor, ExecutionConcurrencyPolicy
from core.execution_plan_executor import PlanExecutionResult, PlanExecutionStatus
from core.execution_plan_validator import ExecutionPlanValidator, PlanValidationResult, plan_signature
from core.execution_priority import ExecutionPriorityPolicy
from core.execution_resources import (
    ExecutionBudget,
    ExecutionBudgetExceededError,
    ExecutionBudgetManager,
    ExecutionResourceCatalog,
    ExecutionResourceOptimizer,
    ExecutionResourcePolicy,
    ExecutionResourceRequirements,
    NoCompatibleResourceError,
    OptimizationGoal,
    PrivacyLevel,
    ResourceCandidate,
    ResourceHealthStatus,
    ResourceType,
)
from core.execution_session_persistence import (
    ExecutionSessionSnapshot,
    snapshot_from_dict,
    snapshot_to_dict,
)
from core.execution_supervisor import ExecutionSupervisor
from core.model_manager import ModelManager
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult
from core.structured_execution import StructuredExecutionCoordinator
from core.structured_plan_replanner import (
    ReplanPolicy,
    ReplanReason,
    ReplanResult,
    ReplanResultStatus,
)


NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _candidate(
    resource_id: str,
    *,
    provider_id: str = "local",
    capabilities: tuple[str, ...] = ("text",),
    quality_tier: int = 1,
    estimated_cost: float | None = 1.0,
    estimated_latency: float | None = 1.0,
    context_window: int = 4096,
    local: bool = True,
    health_status: ResourceHealthStatus = ResourceHealthStatus.AVAILABLE,
    privacy_classification: PrivacyLevel = PrivacyLevel.PUBLIC,
    concurrency_limit: int = 1,
) -> ResourceCandidate:
    return ResourceCandidate(
        resource_id=resource_id,
        resource_type=ResourceType.MODEL,
        provider_id=provider_id,
        capabilities=capabilities,
        quality_tier=quality_tier,
        estimated_cost=estimated_cost,
        estimated_latency=estimated_latency,
        context_window=context_window,
        local=local,
        available=health_status is not ResourceHealthStatus.UNAVAILABLE,
        health_status=health_status,
        privacy_classification=privacy_classification,
        concurrency_limit=concurrency_limit,
    )


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
        goal="resources",
        ordered_steps=steps,
        estimated_steps=len(steps),
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
    )


def _select(
    catalog: ExecutionResourceCatalog,
    requirements: ExecutionResourceRequirements | None = None,
    *,
    policy: ExecutionResourcePolicy | None = None,
    budget_usage=None,
):
    return ExecutionResourceOptimizer(policy or ExecutionResourcePolicy(enabled=True)).select(
        step_id="s1",
        requirements=requirements or ExecutionResourceRequirements(),
        catalog=catalog,
        budget_usage=budget_usage,
    )


def test_disabled_policy_preserves_first_available_candidate() -> None:
    catalog = ExecutionResourceCatalog(
        (_candidate("b"), _candidate("a", quality_tier=99))
    )

    decision = _select(catalog, policy=ExecutionResourcePolicy(enabled=False))

    assert decision.selected_resource_id == "a"
    assert decision.reason.value == "policy_disabled"


def test_required_capabilities_model_provider_and_health_filter_candidates() -> None:
    catalog = ExecutionResourceCatalog(
        (
            _candidate("missing", capabilities=("text",)),
            _candidate("forbidden", provider_id="blocked", capabilities=("vision",)),
            _candidate("down", capabilities=("vision",), health_status=ResourceHealthStatus.UNAVAILABLE),
            _candidate("ok", provider_id="local", capabilities=("vision", "text")),
        )
    )

    decision = _select(
        catalog,
        ExecutionResourceRequirements(
            required_capabilities=("vision",),
            forbidden_provider_ids=("blocked",),
        ),
    )

    assert decision.selected_resource_id == "ok"
    assert set(decision.rejected_candidate_ids) == {"missing", "forbidden", "down"}


def test_privacy_context_local_and_remote_constraints_are_enforced() -> None:
    catalog = ExecutionResourceCatalog(
        (
            _candidate("remote", local=False, privacy_classification=PrivacyLevel.RESTRICTED),
            _candidate("short", context_window=512, privacy_classification=PrivacyLevel.SENSITIVE),
            _candidate("public", privacy_classification=PrivacyLevel.PUBLIC),
            _candidate("private", privacy_classification=PrivacyLevel.SENSITIVE, context_window=8192),
        )
    )

    decision = _select(
        catalog,
        ExecutionResourceRequirements(
            local_only=True,
            remote_allowed=False,
            privacy_level=PrivacyLevel.SENSITIVE,
            requires_long_context=True,
            minimum_context_window=4096,
        ),
    )

    assert decision.selected_resource_id == "private"
    assert set(decision.rejected_candidate_ids) == {"remote", "short", "public"}


def test_unknown_cost_and_budget_excess_are_rejected() -> None:
    manager = ExecutionBudgetManager(ExecutionBudget(max_total_cost=2))
    catalog = ExecutionResourceCatalog(
        (_candidate("unknown", estimated_cost=None), _candidate("expensive", estimated_cost=3))
    )

    with pytest.raises(NoCompatibleResourceError):
        _select(
            catalog,
            policy=ExecutionResourcePolicy(enabled=True, require_known_cost=True),
            budget_usage=manager.snapshot(),
        )


def test_optimization_goals_cost_latency_quality_privacy_and_locality() -> None:
    catalog = ExecutionResourceCatalog(
        (
            _candidate("cheap", quality_tier=1, estimated_cost=0.1, estimated_latency=10, local=False),
            _candidate("fast", quality_tier=1, estimated_cost=10, estimated_latency=0.1, local=False),
            _candidate("quality", quality_tier=10, estimated_cost=10, estimated_latency=10, local=False),
            _candidate("private", quality_tier=1, estimated_cost=10, estimated_latency=10, privacy_classification=PrivacyLevel.RESTRICTED, local=False),
            _candidate("local", quality_tier=1, estimated_cost=10, estimated_latency=10, local=True),
        )
    )

    assert _select(
        catalog,
        policy=ExecutionResourcePolicy(enabled=True, optimization_goal=OptimizationGoal.MINIMIZE_COST, quality_weight=0),
    ).selected_resource_id == "cheap"
    assert _select(
        catalog,
        policy=ExecutionResourcePolicy(enabled=True, optimization_goal=OptimizationGoal.MINIMIZE_LATENCY, quality_weight=0),
    ).selected_resource_id == "fast"
    assert _select(
        catalog,
        policy=ExecutionResourcePolicy(enabled=True, optimization_goal=OptimizationGoal.MAXIMIZE_QUALITY),
    ).selected_resource_id == "quality"
    assert _select(
        catalog,
        policy=ExecutionResourcePolicy(enabled=True, optimization_goal=OptimizationGoal.MAXIMIZE_PRIVACY, quality_weight=0),
    ).selected_resource_id == "private"
    assert _select(
        catalog,
        policy=ExecutionResourcePolicy(enabled=True, optimization_goal=OptimizationGoal.LOCAL_FIRST, quality_weight=0),
    ).selected_resource_id == "local"


def test_budget_manager_reserves_confirms_reconciles_releases_and_is_thread_safe() -> None:
    manager = ExecutionBudgetManager(ExecutionBudget(max_total_cost=10, max_tokens=100))
    first = manager.reserve(step_id="a", resource_id="m", estimated_cost=2, estimated_tokens=20)
    usage = manager.confirm_consumption(first.reservation_id, actual_cost=3, actual_tokens=25)

    assert usage.estimated_cost == 3
    assert usage.actual_cost == 3
    assert usage.remaining_cost == 7

    second = manager.reserve(step_id="b", resource_id="m", estimated_cost=1, estimated_tokens=10)
    usage = manager.release(second.reservation_id)
    assert usage.estimated_cost == 3
    assert usage.estimated_tokens == 25

    errors: list[Exception] = []

    def reserve_one(index: int) -> None:
        try:
            manager.reserve(step_id=f"t{index}", resource_id="m", estimated_cost=2)
        except Exception as error:
            errors.append(error)

    threads = [Thread(target=reserve_one, args=(index,)) for index in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert any(isinstance(error, ExecutionBudgetExceededError) for error in errors)
    assert manager.snapshot().estimated_cost <= 10


def test_quality_degradation_marks_lower_cost_choice_when_budget_filters_best() -> None:
    manager = ExecutionBudgetManager(ExecutionBudget(max_total_cost=1))
    catalog = ExecutionResourceCatalog(
        (_candidate("best", quality_tier=10, estimated_cost=5), _candidate("cheap", quality_tier=2, estimated_cost=1))
    )

    decision = _select(
        catalog,
        policy=ExecutionResourcePolicy(
            enabled=True,
            allow_quality_degradation=True,
            optimization_goal=OptimizationGoal.MAXIMIZE_QUALITY,
        ),
        budget_usage=manager.snapshot(),
    )

    assert decision.selected_resource_id == "cheap"
    assert decision.degradation_applied is True


def test_deterministic_tie_break_uses_provider_and_resource_id() -> None:
    catalog = ExecutionResourceCatalog((_candidate("b"), _candidate("a")))

    decision = _select(catalog)

    assert decision.selected_resource_id == "a"
    assert decision.tie_breaker_used == "deterministic_resource_tie"


def test_budget_and_resource_decision_are_persisted_and_restored() -> None:
    supervisor = ExecutionSupervisor()
    session = supervisor.start(_plan((_step("a"),)))
    manager = ExecutionBudgetManager(ExecutionBudget(max_total_cost=5))
    catalog = ExecutionResourceCatalog((_candidate("m", estimated_cost=1),))
    decision = ExecutionResourceOptimizer(ExecutionResourcePolicy(enabled=True)).select(
        step_id="a",
        requirements=ExecutionResourceRequirements(),
        catalog=catalog,
        budget_usage=manager.snapshot(),
    )
    reservation = manager.reserve(step_id="a", resource_id="m", estimated_cost=1)
    usage = manager.confirm_consumption(reservation.reservation_id)
    supervisor.record_resource_decision(session.session_id, decision, budget_usage=usage)
    snapshot = ExecutionSessionSnapshot.from_session(supervisor.get_session(session.session_id))

    loaded = snapshot_from_dict(snapshot_to_dict(snapshot))

    assert loaded.last_resource_decision.selected_resource_id == "m"
    assert loaded.budget_usage.remaining_cost == 4
    assert loaded.selected_resources_by_step["a"] == "m"


def test_model_manager_exposes_local_model_candidates_without_changing_choice() -> None:
    class Client:
        def list_models(self):
            return ["llama3", {"name": "qwen"}]

    manager = ModelManager(Client())

    assert [item.resource_id for item in manager.list_model_candidates()] == ["llama3", "qwen"]
    assert manager.choose_model("anything") == "llama3"


def test_validator_rejects_invalid_resource_metadata() -> None:
    step = _step("a")
    object.__setattr__(
        step,
        "resource_requirements",
        object.__new__(ExecutionResourceRequirements),
    )
    object.__setattr__(step.resource_requirements, "maximum_estimated_cost", -1)
    object.__setattr__(step.resource_requirements, "maximum_latency_seconds", None)
    object.__setattr__(step.resource_requirements, "local_only", False)
    object.__setattr__(step.resource_requirements, "remote_allowed", True)
    object.__setattr__(step.resource_requirements, "preferred_model_ids", ())
    object.__setattr__(step.resource_requirements, "forbidden_model_ids", ())

    result = ExecutionPlanValidator().validate(_plan((step,)))

    assert not result.is_valid
    assert "maximum_estimated_cost" in result.errors[0]


def test_concurrent_execution_selects_resource_after_priority_before_runner() -> None:
    plan = _plan(
        (
            _step("low", priority=1),
            _step("high", priority=9, resource_requirements=ExecutionResourceRequirements(preferred_model_ids=("fast",))),
        )
    )
    calls: list[str] = []
    supervisor = ExecutionSupervisor()
    budget_manager = ExecutionBudgetManager(ExecutionBudget(max_total_cost=10))
    coordinator = _coordinator(
        plan,
        calls,
        supervisor=supervisor,
        resource_policy=ExecutionResourcePolicy(enabled=True),
        resource_catalog=ExecutionResourceCatalog(
            (_candidate("slow", estimated_cost=2), _candidate("fast", estimated_cost=1))
        ),
        budget_manager=budget_manager,
    )

    response = coordinator.handle("run")
    session = supervisor.get_session(next(iter(supervisor.list_sessions())).session_id)

    assert response.status == "completed"
    assert calls[0] == "high"
    assert session.selected_resources_by_step["high"] == "fast"
    assert session.budget_usage.estimated_cost == 3


def test_resource_failure_prevents_runner_and_replan_receives_context() -> None:
    plan = _plan((_step("a", resource_requirements=ExecutionResourceRequirements(required_capabilities=("vision",))),))
    calls: list[str] = []
    replanner = _FakeReplanner()
    supervisor = ExecutionSupervisor()
    coordinator = _coordinator(
        plan,
        calls,
        supervisor=supervisor,
        resource_policy=ExecutionResourcePolicy(enabled=True),
        resource_catalog=ExecutionResourceCatalog((_candidate("text", capabilities=("text",)),)),
        replanner=replanner,
    )

    coordinator.handle("run")

    assert calls == []
    request = replanner.calls[0]
    assert request.resource_selection_failure
    assert request.rejected_candidate_ids == ("text",)
    assert request.optimization_goal == "balanced"


def _coordinator(
    plan: ExecutionPlan,
    calls: list[str],
    *,
    supervisor: ExecutionSupervisor,
    resource_policy: ExecutionResourcePolicy,
    resource_catalog: ExecutionResourceCatalog,
    budget_manager: ExecutionBudgetManager | None = None,
    replanner=None,
) -> StructuredExecutionCoordinator:
    def runner(step: ExecutionStep) -> str:
        calls.append(step.id)
        return step.id

    return StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=_NoopExecutor(),  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        concurrent_step_executor=ConcurrentStepExecutor(runner),
        concurrency_policy=ExecutionConcurrencyPolicy(enabled=True, max_concurrency=2),
        priority_policy=ExecutionPriorityPolicy(enabled=True),
        priority_clock=lambda: NOW,
        resource_policy=resource_policy,
        resource_catalog=resource_catalog,
        budget_manager=budget_manager,
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

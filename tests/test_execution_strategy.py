from __future__ import annotations

from dataclasses import replace

import pytest

from core.execution_history_advisor import (
    HistoricalEvidence,
    HistoricalPlanningContext,
    HistoricalRecommendation,
    HistoricalRecommendationSeverity,
    HistoricalRecommendationType,
)
from core.execution_plan_validator import ExecutionPlanValidator
from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_report import ExecutionReportGenerator
from core.execution_retry import RetryPolicy
from core.execution_session_persistence import (
    ExecutionSessionSnapshot,
    snapshot_from_dict,
    snapshot_to_dict,
)
from core.execution_strategy import (
    ExecutionStrategyGate,
    ExecutionStrategySelectionRequest,
    ExecutionStrategySelector,
    ExecutionStrategyType,
    GlobalExecutionSafetyPolicy,
    StrategyValidationStatus,
    build_strategy_request,
)
from core.historical_plan_adjustment import (
    HistoricalAdjustmentRequest,
    HistoricalPlanAdjuster,
)
from core.execution_supervisor import ExecutionSupervisor
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult
from core.structured_execution import StructuredExecutionCoordinator
from tools.registry import ToolRegistry


def _plan(
    *,
    criticality: int = 0,
    confirmation: bool = False,
    risks: tuple[str, ...] = (),
    status: str = "planned",
) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Preparar informe",
        ordered_steps=(
            ExecutionStep(
                "read",
                "Leer datos",
                "read_file",
                retry_policy=RetryPolicy(max_attempts=2),
                idempotent=True,
                recovery_safe=True,
                side_effect_free=True,
                criticality=criticality,
            ),
            ExecutionStep(
                "render",
                "Generar informe",
                "direct_response",
                dependencies=("read",),
                side_effect_free=True,
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "direct_response"),
        detected_risks=risks,
        requires_confirmation=confirmation,
        status=status,
    )


def _recommendation(
    kind: HistoricalRecommendationType,
    *,
    count: int = 2,
) -> HistoricalRecommendation:
    session_ids = tuple(f"session.{index}" for index in range(count))
    return HistoricalRecommendation(
        type=kind,
        severity=HistoricalRecommendationSeverity.WARNING,
        message="Repeated sanitized evidence.",
        evidence=(
            HistoricalEvidence(
                fact="Repeated sanitized evidence.",
                occurrence_count=count,
                session_ids=session_ids,
            ),
        ),
        supporting_execution_count=count,
        session_ids=session_ids,
        related_step="read",
    )


def _context(
    *recommendations: HistoricalRecommendation,
    recoveries: tuple[str, ...] = (),
) -> HistoricalPlanningContext:
    session_ids = tuple(
        dict.fromkeys(
            session_id
            for recommendation in recommendations
            for session_id in recommendation.session_ids
        )
    )
    return HistoricalPlanningContext(
        objective="Preparar informe",
        reviewed_execution_count=len(session_ids),
        relevant_execution_ids=session_ids,
        recommendations=tuple(recommendations),
        historical_risks=("read failed",) if recommendations else (),
        known_recoveries=recoveries,
        incident_tools_or_steps=("read",) if recommendations else (),
        summary="Bounded historical context.",
    )


def _select(
    plan: ExecutionPlan,
    *,
    context: HistoricalPlanningContext | None = None,
    supervisor: bool = True,
    replanner: bool = False,
    confirmation_available: bool = True,
    adjustment=None,
):
    validation = ExecutionPlanValidator().validate(plan)
    request = build_strategy_request(
        plan,
        validation,
        historical_context=context,
        historical_adjustment=adjustment,
        supervisor_available=supervisor,
        replanner_available=replanner,
        confirmation_available=confirmation_available,
    )
    return ExecutionStrategySelector().select(request)


def test_simple_valid_plan_selects_standard() -> None:
    result = _select(_plan())

    assert result.strategy.type is ExecutionStrategyType.STANDARD
    assert result.executable is True


def test_repeated_historical_risk_selects_conservative() -> None:
    context = _context(
        _recommendation(HistoricalRecommendationType.FREQUENT_FAILURE)
    )

    result = _select(_plan(), context=context)

    assert result.strategy.type is ExecutionStrategyType.CONSERVATIVE
    assert "repeated_historical_risk" in result.trace.activated_rules


def test_critical_step_selects_supervised() -> None:
    result = _select(_plan(criticality=2))

    assert result.strategy.type is ExecutionStrategyType.SUPERVISED
    assert result.strategy.configuration.progress_required is True


def test_confirmation_has_precedence_over_criticality_and_history() -> None:
    context = _context(
        _recommendation(HistoricalRecommendationType.FREQUENT_FAILURE)
    )

    result = _select(
        _plan(criticality=3, confirmation=True),
        context=context,
    )

    assert result.strategy.type is ExecutionStrategyType.CONFIRMATION_REQUIRED
    assert result.strategy.configuration.requires_confirmation is True
    assert result.strategy.configuration.confirmation_step_ids == ("read", "render")


def test_recovery_available_selects_recovery_prepared_without_running_it() -> None:
    context = _context(
        _recommendation(HistoricalRecommendationType.RECOVERY_AVAILABLE),
        recoveries=("Retry read after transient failure.",),
    )

    result = _select(_plan(), context=context, replanner=True)

    assert result.strategy.type is ExecutionStrategyType.RECOVERY_PREPARED
    assert result.strategy.configuration.allow_replanning is True
    assert result.strategy.configuration.max_replans == 1
    assert result.strategy.configuration.recovery_hints


def test_manual_review_from_adjustment_has_highest_precedence() -> None:
    plan = _plan(criticality=3, confirmation=True)
    recommendation = _recommendation(HistoricalRecommendationType.RETRY_RISK)
    unsafe_plan = replace(
        plan,
        ordered_steps=(
            replace(
                plan.ordered_steps[0],
                idempotent=False,
                recovery_safe=False,
                side_effect_free=False,
            ),
            plan.ordered_steps[1],
        ),
    )
    validation = ExecutionPlanValidator()
    adjustment = HistoricalPlanAdjuster(validation).adjust(
        HistoricalAdjustmentRequest(
            plan=unsafe_plan,
            historical_context=_context(recommendation),
        )
    )

    result = _select(unsafe_plan, adjustment=adjustment)

    assert adjustment.requires_manual_review is True
    assert result.strategy.type is ExecutionStrategyType.MANUAL_REVIEW_REQUIRED
    assert result.executable is False


def test_result_is_deterministic_and_collection_order_independent() -> None:
    failure = _recommendation(HistoricalRecommendationType.FREQUENT_FAILURE)
    retry = _recommendation(HistoricalRecommendationType.RETRY_RISK)

    first = _select(_plan(risks=("zeta", "alpha")), context=_context(failure, retry))
    second = _select(_plan(risks=("alpha", "zeta")), context=_context(retry, failure))

    assert first.strategy.type is second.strategy.type
    assert first.strategy.factors == second.strategy.factors
    assert first.summary == second.summary


def test_invalid_or_started_plan_never_gets_executable_strategy() -> None:
    invalid = _plan(status="running")
    validation = ExecutionPlanValidator().validate(invalid)

    result = ExecutionStrategySelector().select(
        ExecutionStrategySelectionRequest(
            plan=invalid,
            plan_validation=validation,
        )
    )

    assert result.strategy.type is ExecutionStrategyType.MANUAL_REVIEW_REQUIRED
    assert result.executable is False
    assert result.validation.status is StrategyValidationStatus.BLOCKED


def test_missing_supervisor_blocks_supervised_strategy_with_safe_fallback() -> None:
    result = _select(_plan(criticality=2), supervisor=False)

    assert result.strategy.type is ExecutionStrategyType.MANUAL_REVIEW_REQUIRED
    assert result.trace.safe_fallback is ExecutionStrategyType.MANUAL_REVIEW_REQUIRED
    assert result.executable is False


def test_missing_replanner_prevents_recovery_strategy() -> None:
    context = _context(
        _recommendation(HistoricalRecommendationType.RECOVERY_AVAILABLE),
        recoveries=("Known recovery.",),
    )

    result = _select(_plan(), context=context, replanner=False)

    assert result.strategy.type is ExecutionStrategyType.SUPERVISED
    assert result.strategy.configuration.allow_replanning is False


def test_missing_confirmation_mechanism_blocks_instead_of_relaxing_control() -> None:
    result = _select(
        _plan(confirmation=True),
        confirmation_available=False,
    )

    assert result.strategy.type is ExecutionStrategyType.MANUAL_REVIEW_REQUIRED
    assert result.executable is False
    assert result.strategy.configuration.requires_confirmation is True


def test_plan_content_and_original_object_are_preserved() -> None:
    plan = _plan(criticality=2, risks=("network",))
    original = (
        plan.goal,
        plan.required_tools,
        tuple(step.id for step in plan.ordered_steps),
        tuple(step.tool for step in plan.ordered_steps),
        tuple(step.dependencies for step in plan.ordered_steps),
        tuple(step.criticality for step in plan.ordered_steps),
    )

    result = _select(plan)

    assert result.original_plan is plan
    assert original == (
        plan.goal,
        plan.required_tools,
        tuple(step.id for step in plan.ordered_steps),
        tuple(step.tool for step in plan.ordered_steps),
        tuple(step.dependencies for step in plan.ordered_steps),
        tuple(step.criticality for step in plan.ordered_steps),
    )


def test_blocking_strategy_gate_never_calls_executor() -> None:
    plan = _plan(status="running")
    validation = ExecutionPlanValidator().validate(plan)
    selection = ExecutionStrategySelector().select(
        ExecutionStrategySelectionRequest(
            plan=plan,
            plan_validation=validation,
        )
    )
    calls = []

    value = ExecutionStrategyGate().execute(
        selection,
        lambda configuration: calls.append(configuration),
    )

    assert value is None
    assert calls == []


def test_executor_consumes_resolved_configuration_but_does_not_select_it() -> None:
    plan = ExecutionPlan(
        goal="Responder",
        ordered_steps=(
            ExecutionStep(
                "respond",
                "Responder",
                "direct_response",
                side_effect_free=True,
            ),
        ),
        estimated_steps=1,
        required_tools=("direct_response",),
        detected_risks=(),
        requires_confirmation=False,
    )
    validation = ExecutionPlanValidator().validate(plan)
    selection = ExecutionStrategySelector().select(
        build_strategy_request(plan, validation)
    )

    result = ExecutionPlanExecutor(ToolRegistry()).execute(
        plan,
        validation,
        operational_config=selection.strategy.configuration,
    )

    assert result.success is True
    assert selection.strategy.type is ExecutionStrategyType.STANDARD

    blocking = _select(_plan(status="running"))
    with pytest.raises(ValueError, match="Blocking operational configuration"):
        ExecutionPlanExecutor(ToolRegistry()).execute(
            plan,
            validation,
            operational_config=blocking.strategy.configuration,
        )


def test_policy_limits_cannot_be_exceeded() -> None:
    plan = _plan()
    policy = GlobalExecutionSafetyPolicy(
        max_retry_attempts=1,
        max_replans=0,
        allow_replanning=False,
    )
    validation = ExecutionPlanValidator().validate(plan)

    result = ExecutionStrategySelector().select(
        build_strategy_request(
            plan,
            validation,
            replanner_available=True,
            safety_policy=policy,
        )
    )

    assert result.strategy.type is ExecutionStrategyType.MANUAL_REVIEW_REQUIRED
    assert result.executable is False
    assert result.strategy.configuration.effective_max_retry_attempts == 1
    assert result.strategy.configuration.max_replans == 0


def test_trace_summary_and_persisted_snapshot_are_bounded_and_safe() -> None:
    result = _select(_plan(criticality=2))
    snapshot = result.persisted_snapshot()

    assert snapshot["strategy"] == "SUPERVISED"
    assert snapshot["validation"]["executable"] is True
    assert "Plan content was not modified." in result.summary
    assert snapshot["trace"]["final_decision"] == "EXECUTE"


def test_strategy_persists_with_session_and_legacy_snapshot_remains_readable() -> None:
    plan = _plan(criticality=2)
    selection = _select(plan)
    supervisor = ExecutionSupervisor()
    session = supervisor.start(
        plan,
        execution_strategy=selection.persisted_snapshot(),
    )
    snapshot = ExecutionSessionSnapshot.from_session(session)
    payload = snapshot_to_dict(snapshot)

    restored = snapshot_from_dict(payload)
    report = ExecutionReportGenerator().generate(
        restored.to_session(),
        supervisor.get_summary(session.session_id),
    )

    assert restored.execution_strategy["strategy"] == "SUPERVISED"
    assert report.execution_strategy == "SUPERVISED"
    assert report.strategy_reinforced_supervision is True

    payload.pop("execution_strategy")
    legacy = snapshot_from_dict(payload)
    legacy_report = ExecutionReportGenerator().generate(
        legacy.to_session(),
        supervisor.get_summary(session.session_id),
    )
    assert legacy.execution_strategy is None
    assert legacy_report.execution_strategy == "STANDARD"
    assert "not recorded" in legacy_report.strategy_reason


def test_coordinator_blocks_manual_strategy_before_executor() -> None:
    plan = _plan(criticality=2)
    unsafe_plan = replace(
        plan,
        ordered_steps=(
            replace(
                plan.ordered_steps[0],
                idempotent=False,
                recovery_safe=False,
                side_effect_free=False,
            ),
            plan.ordered_steps[1],
        ),
    )
    context = _context(
        _recommendation(HistoricalRecommendationType.RETRY_RISK)
    )
    adjustment = HistoricalPlanAdjuster(ExecutionPlanValidator()).adjust(
        HistoricalAdjustmentRequest(
            plan=unsafe_plan,
            historical_context=context,
        )
    )

    class _Planner:
        def generate_execution_plan(self, objective, **kwargs):
            return PlanGenerationResult(success=True, plan=unsafe_plan)

    class _Executor:
        def __init__(self):
            self.calls = 0

        def execute(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("blocking strategy reached executor")

    executor = _Executor()
    coordinator = StructuredExecutionCoordinator(
        planner=_Planner(),
        validator=ExecutionPlanValidator(),
        executor=executor,
        execution_strategy_selector=ExecutionStrategySelector(),
    )

    response = coordinator.handle(
        "Preparar informe",
        historical_adjustment=adjustment,
        historical_context=context,
    )

    assert response.status == "strategy_blocked"
    assert response.strategy_selection is not None
    assert response.strategy_selection.strategy.type is (
        ExecutionStrategyType.MANUAL_REVIEW_REQUIRED
    )
    assert executor.calls == 0

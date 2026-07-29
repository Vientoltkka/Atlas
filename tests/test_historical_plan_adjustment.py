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
from core.execution_plan_validator import (
    ExecutionPlanValidator,
    PlanValidationResult,
)
from core.execution_retry import RetryPolicy
from core.historical_plan_adjustment import (
    HistoricalAdjustmentPolicyLimits,
    HistoricalAdjustmentRequest,
    HistoricalAdjustmentStatus,
    HistoricalAdjustmentTarget,
    HistoricalAdjustmentType,
    HistoricalAdjustmentValidation,
    HistoricalAdjustmentValue,
    HistoricalPlanAdjuster,
    PlanAdjustmentLifecycle,
)
from core.planner import ExecutionPlan, ExecutionStep, Planner


def _plan(
    *,
    attempts: int = 1,
    safe: bool = True,
    criticality: int = 0,
    requires_confirmation: bool = False,
) -> ExecutionPlan:
    retry_policy = RetryPolicy(max_attempts=attempts, delay_ms=25)
    return ExecutionPlan(
        goal="Preparar informe",
        ordered_steps=(
            ExecutionStep(
                "read",
                "Leer datos",
                "read_file",
                retry_policy=retry_policy,
                idempotent=safe,
                recovery_safe=safe,
                side_effect_free=safe,
                criticality=criticality,
            ),
            ExecutionStep(
                "list",
                "Listar resultado",
                "list_directory",
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "list_directory"),
        detected_risks=(),
        requires_confirmation=requires_confirmation,
    )


def _recommendation(
    recommendation_type: HistoricalRecommendationType,
    *,
    count: int = 2,
    severity: HistoricalRecommendationSeverity = (
        HistoricalRecommendationSeverity.CAUTION
    ),
    related_step: str | None = None,
    related_tool: str | None = None,
    message: str = "Historical evidence was observed.",
) -> HistoricalRecommendation:
    session_ids = tuple(f"session.{index}" for index in range(count))
    evidence = (
        HistoricalEvidence(
            fact=message,
            occurrence_count=max(1, count),
            session_ids=session_ids,
        ),
    ) if count else ()
    return HistoricalRecommendation(
        type=recommendation_type,
        severity=severity,
        message=message,
        evidence=evidence,
        supporting_execution_count=count,
        session_ids=session_ids,
        related_step=related_step,
        related_tool=related_tool,
    )


def _context(
    *recommendations: HistoricalRecommendation,
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
        historical_risks=(),
        known_recoveries=(),
        incident_tools_or_steps=(),
        summary="Contexto histórico consultivo.",
    )


def _request(
    plan: ExecutionPlan,
    *recommendations: HistoricalRecommendation,
    lifecycle: PlanAdjustmentLifecycle = PlanAdjustmentLifecycle.GENERATED,
    completed_step_ids: tuple[str, ...] = (),
) -> HistoricalAdjustmentRequest:
    return HistoricalAdjustmentRequest(
        plan=plan,
        historical_context=_context(*recommendations),
        lifecycle=lifecycle,
        completed_step_ids=completed_step_ids,
    )


def _adjuster(
    validator: ExecutionPlanValidator | None = None,
    *,
    limits: HistoricalAdjustmentPolicyLimits | None = None,
) -> HistoricalPlanAdjuster:
    from core.historical_plan_adjustment import HistoricalPlanAdjustmentPolicy

    return HistoricalPlanAdjuster(
        validator or ExecutionPlanValidator(),
        policy=HistoricalPlanAdjustmentPolicy(limits),
    )


@pytest.mark.parametrize(
    "recommendation",
    (
        _recommendation(
            HistoricalRecommendationType.INSUFFICIENT_HISTORY,
            count=0,
            severity=HistoricalRecommendationSeverity.INFORMATION,
        ),
        _recommendation(
            HistoricalRecommendationType.FREQUENT_FAILURE,
            count=1,
            severity=HistoricalRecommendationSeverity.WARNING,
            related_tool="read_file",
        ),
    ),
)
def test_insufficient_and_isolated_failure_do_not_propose(
    recommendation,
) -> None:
    plan = _plan()
    adjuster = _adjuster()

    result = adjuster.adjust(_request(plan, recommendation))

    assert result.proposals == ()
    assert result.selected_plan is plan
    assert result.applied_count == 0


def test_previous_success_produces_informational_proposal_without_change() -> None:
    plan = _plan()
    recommendation = _recommendation(
        HistoricalRecommendationType.PREVIOUS_SUCCESS,
        severity=HistoricalRecommendationSeverity.INFORMATION,
    )

    result = _adjuster().adjust(_request(plan, recommendation))

    assert len(result.proposals) == 1
    assert result.proposals[0].status is HistoricalAdjustmentStatus.INFORMATIONAL
    assert result.proposals[0].applied is False
    assert result.selected_plan is plan


def test_frequent_failure_adds_validated_warning() -> None:
    plan = _plan()
    recommendation = _recommendation(
        HistoricalRecommendationType.FREQUENT_FAILURE,
        severity=HistoricalRecommendationSeverity.WARNING,
        related_tool="read_file",
        message="read_file failed repeatedly.",
    )

    result = _adjuster().adjust(_request(plan, recommendation))

    assert result.applied_count == 1
    proposal = result.proposals[0]
    assert proposal.adjustment_type is HistoricalAdjustmentType.ADD_HISTORICAL_WARNING
    assert proposal.validation is HistoricalAdjustmentValidation.VALID
    assert "read_file failed repeatedly." in result.selected_plan.detected_risks
    assert plan.detected_risks == ()


def test_retry_risk_increases_safe_step_within_limit_and_preserves_policy() -> None:
    plan = _plan(attempts=1, safe=True)
    recommendation = _recommendation(
        HistoricalRecommendationType.RETRY_RISK,
        related_step="read",
    )

    result = _adjuster().adjust(_request(plan, recommendation))

    retry_proposal = next(
        item
        for item in result.proposals
        if item.adjustment_type is HistoricalAdjustmentType.INCREASE_RETRY_LIMIT
    )
    adjusted_step = result.selected_plan.ordered_steps[0]
    assert retry_proposal.applied is True
    assert adjusted_step.retry_policy.max_attempts == 2
    assert adjusted_step.retry_policy.delay_ms == 25
    assert plan.ordered_steps[0].retry_policy.max_attempts == 1


def test_retry_limit_above_absolute_limit_is_rejected() -> None:
    plan = _plan(attempts=2)
    recommendation = _recommendation(
        HistoricalRecommendationType.RETRY_RISK,
        related_step="read",
    )
    request = _request(plan, recommendation)
    adjuster = _adjuster()
    generated = adjuster.propose(request)
    retry = next(
        item
        for item in generated
        if item.adjustment_type is HistoricalAdjustmentType.INCREASE_RETRY_LIMIT
    )
    unsafe = replace(
        retry,
        proposed_value=HistoricalAdjustmentValue.integer_value(4),
    )

    result = adjuster.adjust(request, proposals=(unsafe,))

    assert result.rejected_count == 1
    assert result.selected_plan is plan
    assert result.proposals[0].validation is HistoricalAdjustmentValidation.INVALID


def test_recovery_available_attaches_hint_without_executing_it() -> None:
    plan = _plan()
    recommendation = _recommendation(
        HistoricalRecommendationType.RECOVERY_AVAILABLE,
        severity=HistoricalRecommendationSeverity.INFORMATION,
        message="A controlled retry recovered the previous execution.",
    )

    result = _adjuster().adjust(_request(plan, recommendation))

    assert result.applied_count == 1
    assert result.proposals[0].adjustment_type is HistoricalAdjustmentType.ATTACH_RECOVERY_HINT
    assert result.selected_plan.ordered_steps == plan.ordered_steps
    assert result.selected_plan.detected_risks == (
        "A controlled retry recovered the previous execution.",
    )


def test_user_action_pattern_requires_confirmation_without_reducing_it() -> None:
    plan = _plan()
    recommendation = _recommendation(
        HistoricalRecommendationType.USER_ACTION_PATTERN,
        message="User confirmation was repeatedly required.",
    )

    result = _adjuster().adjust(_request(plan, recommendation))

    proposal = result.proposals[0]
    assert proposal.adjustment_type is HistoricalAdjustmentType.REQUIRE_CONFIRMATION
    assert result.selected_plan.requires_confirmation is True
    assert plan.requires_confirmation is False


def test_attempt_to_reduce_confirmation_is_rejected() -> None:
    plan = _plan(requires_confirmation=False)
    recommendation = _recommendation(
        HistoricalRecommendationType.USER_ACTION_PATTERN,
    )
    request = _request(plan, recommendation)
    adjuster = _adjuster()
    generated = adjuster.propose(request)[0]
    unsafe = replace(
        generated,
        proposed_value=HistoricalAdjustmentValue.boolean_value(False),
    )

    result = adjuster.adjust(request, proposals=(unsafe,))

    assert result.rejected_count == 1
    assert result.selected_plan is plan
    assert result.selected_plan.requires_confirmation is False


@pytest.mark.parametrize(
    "target",
    (
        HistoricalAdjustmentTarget.TOOL,
        HistoricalAdjustmentTarget.STEP_SET,
        HistoricalAdjustmentTarget.OBJECTIVE,
        HistoricalAdjustmentTarget.CRITICALITY,
        HistoricalAdjustmentTarget.OPTIONAL,
    ),
)
def test_forbidden_plan_targets_are_rejected(target) -> None:
    plan = _plan()
    recommendation = _recommendation(
        HistoricalRecommendationType.FREQUENT_FAILURE,
        severity=HistoricalRecommendationSeverity.WARNING,
        related_tool="list_directory",
    )
    request = _request(plan, recommendation)
    adjuster = _adjuster()
    generated = adjuster.propose(request)[0]
    unsafe = replace(generated, target=target)

    result = adjuster.adjust(request, proposals=(unsafe,))

    assert result.proposals[0].status is HistoricalAdjustmentStatus.REJECTED
    assert result.selected_plan is plan
    assert tuple(step.tool for step in result.selected_plan.ordered_steps) == (
        "read_file",
        "list_directory",
    )


@pytest.mark.parametrize(
    "lifecycle",
    (PlanAdjustmentLifecycle.EXECUTING, PlanAdjustmentLifecycle.TERMINAL),
)
def test_executing_and_terminal_plans_reject_adjustments(lifecycle) -> None:
    plan = _plan()
    recommendation = _recommendation(
        HistoricalRecommendationType.FREQUENT_FAILURE,
        severity=HistoricalRecommendationSeverity.WARNING,
        related_tool="read_file",
    )

    result = _adjuster().adjust(
        _request(plan, recommendation, lifecycle=lifecycle)
    )

    assert result.rejected_count == 1
    assert result.selected_plan is plan


def test_completed_step_rejects_step_adjustment() -> None:
    plan = _plan()
    recommendation = _recommendation(
        HistoricalRecommendationType.RETRY_RISK,
        related_step="read",
    )

    result = _adjuster().adjust(
        _request(
            plan,
            recommendation,
            completed_step_ids=("read",),
        )
    )

    assert all(item.rejected for item in result.proposals)
    assert result.selected_plan is plan


def test_unsafe_retry_step_requires_manual_review_and_is_not_increased() -> None:
    plan = _plan(safe=False, criticality=1)
    recommendation = _recommendation(
        HistoricalRecommendationType.RETRY_RISK,
        related_step="read",
    )

    result = _adjuster().adjust(_request(plan, recommendation))

    assert result.requires_manual_review is True
    assert result.selected_plan.ordered_steps[0].retry_policy.max_attempts == 1
    assert not any(
        item.adjustment_type is HistoricalAdjustmentType.INCREASE_RETRY_LIMIT
        for item in result.proposals
    )


class _RejectAdjustedValidator(ExecutionPlanValidator):
    def validate(self, plan, **kwargs):
        if plan.detected_risks:
            return PlanValidationResult(
                is_valid=False,
                errors=["candidate rejected"],
                status="invalid",
            )
        return super().validate(plan, **kwargs)


def test_candidate_validation_failure_preserves_original_plan() -> None:
    plan = _plan()
    recommendation = _recommendation(
        HistoricalRecommendationType.FREQUENT_FAILURE,
        severity=HistoricalRecommendationSeverity.WARNING,
        related_tool="read_file",
    )

    result = _adjuster(_RejectAdjustedValidator()).adjust(
        _request(plan, recommendation)
    )

    assert result.rejected_count == 1
    assert result.selected_plan is plan
    assert result.original_plan is plan
    assert plan.detected_risks == ()


def test_multiple_proposals_never_leave_an_invalid_partial_state() -> None:
    plan = _plan(attempts=2)
    failure = _recommendation(
        HistoricalRecommendationType.FREQUENT_FAILURE,
        severity=HistoricalRecommendationSeverity.WARNING,
        related_tool="list_directory",
    )
    retry = _recommendation(
        HistoricalRecommendationType.RETRY_RISK,
        related_step="read",
    )
    request = _request(plan, failure, retry)
    adjuster = _adjuster()
    generated = adjuster.propose(request)
    retry_change = next(
        item
        for item in generated
        if item.adjustment_type is HistoricalAdjustmentType.INCREASE_RETRY_LIMIT
    )
    unsafe_retry = replace(
        retry_change,
        proposed_value=HistoricalAdjustmentValue.integer_value(4),
    )
    warning = next(
        item
        for item in generated
        if item.adjustment_type is HistoricalAdjustmentType.ADD_HISTORICAL_WARNING
    )

    result = adjuster.adjust(
        request,
        proposals=(warning, unsafe_retry),
    )

    assert result.applied_count == 1
    assert result.rejected_count == 1
    assert result.final_validation.is_valid is True
    assert result.selected_plan.ordered_steps[0].retry_policy.max_attempts == 2
    assert result.selected_plan.detected_risks


def test_proposal_limits_are_enforced_per_plan_and_step() -> None:
    limits = HistoricalAdjustmentPolicyLimits(
        max_proposals_per_plan=2,
        max_proposals_per_step=1,
    )
    plan = _plan()
    recommendations = tuple(
        _recommendation(
            HistoricalRecommendationType.FREQUENT_FAILURE,
            severity=HistoricalRecommendationSeverity.WARNING,
            related_tool="read_file",
            message=f"failure {index}",
        )
        for index in range(5)
    )

    proposals = _adjuster(limits=limits).propose(
        _request(plan, *recommendations)
    )

    assert len(proposals) == 1


def test_trace_summary_and_secret_sanitization_are_deterministic() -> None:
    plan = _plan()
    recommendation = _recommendation(
        HistoricalRecommendationType.FREQUENT_FAILURE,
        severity=HistoricalRecommendationSeverity.WARNING,
        related_tool="read_file",
        message="token sk-secretvalue123456",
    )
    adjuster = _adjuster()
    request = _request(plan, recommendation)

    first = adjuster.adjust(request)
    second = adjuster.adjust(request)

    assert first.summary == second.summary
    assert first.proposals[0].to_dict() == second.proposals[0].to_dict()
    assert "[redacted]" in str(first.proposals[0].to_dict())
    assert "secretvalue" not in str(first.proposals[0].to_dict())
    assert first.traces[0].evidence_session_ids == ("session.0", "session.1")
    assert "herramientas y el orden" in first.summary


def test_original_goal_tools_order_and_step_security_are_preserved() -> None:
    plan = _plan()
    recommendations = (
        _recommendation(
            HistoricalRecommendationType.FREQUENT_FAILURE,
            severity=HistoricalRecommendationSeverity.WARNING,
            related_tool="read_file",
        ),
        _recommendation(
            HistoricalRecommendationType.RETRY_RISK,
            related_step="read",
        ),
    )

    result = _adjuster().adjust(_request(plan, *recommendations))

    assert result.selected_plan.goal == plan.goal
    assert tuple(step.id for step in result.selected_plan.ordered_steps) == (
        "read",
        "list",
    )
    assert tuple(step.tool for step in result.selected_plan.ordered_steps) == (
        "read_file",
        "list_directory",
    )
    assert tuple(step.criticality for step in result.selected_plan.ordered_steps) == (
        0,
        0,
    )
    assert tuple(step.optional for step in result.selected_plan.ordered_steps) == (
        False,
        False,
    )


def test_planner_without_historical_context_remains_backward_compatible() -> None:
    planner = Planner()

    first = planner.generate_execution_plan("Conversar sobre el informe")
    second = planner.generate_execution_plan("Conversar sobre el informe")

    assert first.plan == second.plan
    assert first.success == second.success

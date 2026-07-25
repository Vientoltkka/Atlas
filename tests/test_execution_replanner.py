from __future__ import annotations

from core.execution_replanner import (
    ExecutionReplanner,
    ReplanningCandidate,
    ReplanningPolicy,
    ReplanningRequest,
    ReplanningStatus,
    ReplanningStrategy,
)
from core.goal_verifier import GoalVerificationReason, GoalVerificationResult
from core.planner import ExecutionPlan, ExecutionStep


def _plan(step_id: str = "step_1", *, tool: str = "demo.tool") -> ExecutionPlan:
    return ExecutionPlan(
        goal="Execute workflow.",
        ordered_steps=(ExecutionStep(step_id, "Run.", tool),),
        estimated_steps=1,
        required_tools=(tool,),
        detected_risks=(),
        requires_confirmation=False,
    )


def _policy() -> ReplanningPolicy:
    return ReplanningPolicy(
        enabled=True,
        max_replans=1,
        strategy=ReplanningStrategy.ALTERNATIVE_WORKFLOW,
        retryable_goal_reasons=(GoalVerificationReason.MISSING_REQUIRED_OUTPUTS.value,),
    )


def _failed_goal() -> GoalVerificationResult:
    return GoalVerificationResult(
        satisfied=False,
        reason=GoalVerificationReason.MISSING_REQUIRED_OUTPUTS,
        missing_outputs=("entries",),
    )


def test_disabled_policy_does_not_replan() -> None:
    failed_plan = _plan()

    decision = ExecutionReplanner().decide(
        ReplanningPolicy(),
        ReplanningRequest(
            original_plan=failed_plan,
            failed_plan=failed_plan,
            goal_verification_result=_failed_goal(),
        ),
    )

    assert decision.should_replan is False
    assert decision.status is ReplanningStatus.DISABLED


def test_goal_already_satisfied_does_not_replan() -> None:
    failed_plan = _plan()

    decision = ExecutionReplanner().decide(
        _policy(),
        ReplanningRequest(
            original_plan=failed_plan,
            failed_plan=failed_plan,
            goal_verification_result=GoalVerificationResult(
                satisfied=True,
                reason=GoalVerificationReason.SUCCESS,
            ),
        ),
    )

    assert decision.status is ReplanningStatus.GOAL_ALREADY_SATISFIED
    assert decision.should_replan is False


def test_non_retryable_goal_reason_is_rejected() -> None:
    failed_plan = _plan()

    decision = ExecutionReplanner().decide(
        _policy(),
        ReplanningRequest(
            original_plan=failed_plan,
            failed_plan=failed_plan,
            goal_verification_result=GoalVerificationResult(
                satisfied=False,
                reason=GoalVerificationReason.INVALID_OUTPUT_BINDING,
            ),
        ),
    )

    assert decision.status is ReplanningStatus.FAILURE_NOT_REPLANNABLE
    assert decision.should_replan is False


def test_selects_next_distinct_candidate_and_excludes_failed_signature() -> None:
    failed_plan = _plan("failed")
    alternative = _plan("alternative")

    decision = ExecutionReplanner().decide(
        _policy(),
        ReplanningRequest(
            original_plan=failed_plan,
            failed_plan=failed_plan,
            goal_verification_result=_failed_goal(),
            candidates=(
                ReplanningCandidate(failed_plan),
                ReplanningCandidate(alternative),
            ),
        ),
    )

    assert decision.status is ReplanningStatus.REPLANNED
    assert decision.should_replan is True
    assert decision.replacement_plan is alternative
    assert decision.history_entry is not None
    assert decision.history_entry.attempt == 1


def test_limit_reached_prevents_replanning() -> None:
    failed_plan = _plan("failed")

    decision = ExecutionReplanner().decide(
        _policy(),
        ReplanningRequest(
            original_plan=failed_plan,
            failed_plan=failed_plan,
            goal_verification_result=_failed_goal(),
            candidates=(ReplanningCandidate(_plan("alternative")),),
            replan_attempts=1,
        ),
    )

    assert decision.status is ReplanningStatus.LIMIT_REACHED
    assert decision.should_replan is False


def test_no_alternative_when_only_same_plan_signature_exists() -> None:
    failed_plan = _plan("failed")

    decision = ExecutionReplanner().decide(
        _policy(),
        ReplanningRequest(
            original_plan=failed_plan,
            failed_plan=failed_plan,
            goal_verification_result=_failed_goal(),
            candidates=(ReplanningCandidate(failed_plan),),
        ),
    )

    assert decision.status is ReplanningStatus.NO_ALTERNATIVE_PLAN
    assert decision.should_replan is False

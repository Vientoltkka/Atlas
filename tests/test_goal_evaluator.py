from __future__ import annotations

import pytest

from core.goal_evaluator import GoalEvaluationStatus, GoalEvaluator
from core.goal_verifier import (
    GoalVerificationReason,
    GoalVerificationResult,
    GoalVerificationStatus,
)
from core.test_runner import TestRunResult


class _StubVerifier:
    def __init__(self, result: GoalVerificationResult) -> None:
        self._result = result
        self.calls = 0

    def verify(self, plan: object, execution_result: object) -> GoalVerificationResult:
        self.calls += 1
        return self._result


def _verified() -> GoalVerificationResult:
    return GoalVerificationResult(
        satisfied=True,
        reason=GoalVerificationReason.SUCCESS,
    )


def _not_verified(
    reason: GoalVerificationReason = GoalVerificationReason.OUTPUT_VALIDATION_FAILED,
    status: GoalVerificationStatus | None = None,
) -> GoalVerificationResult:
    return GoalVerificationResult(
        satisfied=False,
        reason=reason,
        verification_status=status,
    )


def _tests(passed: bool) -> TestRunResult:
    return TestRunResult(
        passed=passed,
        exit_code=0 if passed else 1,
        timed_out=False,
        detail="stub",
        output_tail="",
        command=("pytest",),
        basetemp=None,
    )


_PLAN = object()
_EXECUTION = object()


def test_verified_without_tests_maps_to_success() -> None:
    verifier = _StubVerifier(_verified())

    evaluation = GoalEvaluator(verifier).evaluate(_PLAN, _EXECUTION)

    assert evaluation.status is GoalEvaluationStatus.SUCCESS
    assert evaluation.reason == "verified"
    assert evaluation.test_result is None
    assert verifier.calls == 1


def test_verified_with_passing_tests_maps_to_success() -> None:
    evaluation = GoalEvaluator(_StubVerifier(_verified())).evaluate(
        _PLAN,
        _EXECUTION,
        test_result=_tests(True),
    )

    assert evaluation.status is GoalEvaluationStatus.SUCCESS


def test_failed_tests_map_to_retry_even_when_verified() -> None:
    evaluation = GoalEvaluator(_StubVerifier(_verified())).evaluate(
        _PLAN,
        _EXECUTION,
        test_result=_tests(False),
    )

    assert evaluation.status is GoalEvaluationStatus.RETRY
    assert evaluation.reason == "tests_failed"


def test_failed_verification_maps_to_retry() -> None:
    verifier = _StubVerifier(_not_verified())

    evaluation = GoalEvaluator(verifier).evaluate(
        _PLAN,
        _EXECUTION,
        test_result=_tests(True),
    )

    assert evaluation.status is GoalEvaluationStatus.RETRY
    assert evaluation.reason == "verification_failed:OUTPUT_VALIDATION_FAILED"


def test_user_action_required_maps_to_blocked() -> None:
    result = _not_verified(
        GoalVerificationReason.USER_ACTION_REQUIRED,
        GoalVerificationStatus.USER_ACTION_REQUIRED,
    )

    evaluation = GoalEvaluator(_StubVerifier(result)).evaluate(_PLAN, _EXECUTION)

    assert evaluation.status is GoalEvaluationStatus.BLOCKED
    assert evaluation.reason == "blocked:USER_ACTION_REQUIRED"


def test_plan_blocked_and_cancelled_map_to_blocked() -> None:
    blocked = GoalEvaluator(_StubVerifier(_not_verified(GoalVerificationReason.PLAN_BLOCKED))).evaluate(_PLAN, _EXECUTION)
    cancelled = GoalEvaluator(_StubVerifier(_not_verified(GoalVerificationReason.PLAN_CANCELLED))).evaluate(_PLAN, _EXECUTION)

    assert blocked.status is GoalEvaluationStatus.BLOCKED
    assert cancelled.status is GoalEvaluationStatus.BLOCKED


def test_inconclusive_verification_maps_to_blocked() -> None:
    result = _not_verified(
        GoalVerificationReason.INSUFFICIENT_EVIDENCE,
        GoalVerificationStatus.INCONCLUSIVE,
    )

    evaluation = GoalEvaluator(_StubVerifier(result)).evaluate(_PLAN, _EXECUTION)

    assert evaluation.status is GoalEvaluationStatus.BLOCKED


def test_optional_reviewer_is_injected_and_consulted_only_for_success() -> None:
    seen: list[tuple[GoalVerificationResult, TestRunResult | None]] = []

    def reviewer(verification: GoalVerificationResult, test_result: TestRunResult | None) -> bool:
        seen.append((verification, test_result))
        return False

    evaluator = GoalEvaluator(_StubVerifier(_verified()), reviewer=reviewer)
    rejected = evaluator.evaluate(_PLAN, _EXECUTION, test_result=_tests(True))

    assert rejected.status is GoalEvaluationStatus.RETRY
    assert rejected.reason == "reviewer_rejected"
    assert seen and seen[0][1].passed is True

    def accepting(verification: GoalVerificationResult, test_result: TestRunResult | None) -> bool:
        return True

    accepted = GoalEvaluator(_StubVerifier(_verified()), reviewer=accepting).evaluate(
        _PLAN,
        _EXECUTION,
        test_result=_tests(True),
    )

    assert accepted.status is GoalEvaluationStatus.SUCCESS


def test_reviewer_is_not_consulted_when_blocked() -> None:
    def failing_reviewer(verification: object, test_result: object) -> bool:
        raise AssertionError("reviewer must not run for blocked evaluations.")

    result = _not_verified(
        GoalVerificationReason.USER_ACTION_REQUIRED,
        GoalVerificationStatus.USER_ACTION_REQUIRED,
    )

    evaluation = GoalEvaluator(_StubVerifier(result), reviewer=failing_reviewer).evaluate(_PLAN, _EXECUTION)

    assert evaluation.status is GoalEvaluationStatus.BLOCKED

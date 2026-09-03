"""Thin decision wrapper over the deterministic GoalVerifier.

It adds no verification logic of its own: it combines the verifier outcome
with an optional test-run result and an optional injected reviewer callable,
and maps everything onto the minimal SUCCESS / RETRY / BLOCKED states.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

from core.goal_verifier import (
    GoalVerifier,
    GoalVerificationReason,
    GoalVerificationResult,
    GoalVerificationStatus,
)

if TYPE_CHECKING:
    from core.test_runner import TestRunResult


class GoalEvaluationStatus(str, Enum):
    """Minimal autonomous decision states."""

    SUCCESS = "SUCCESS"
    RETRY = "RETRY"
    BLOCKED = "BLOCKED"


_TERMINAL_BLOCKED_REASONS = {
    GoalVerificationReason.PLAN_CANCELLED,
    GoalVerificationReason.PLAN_BLOCKED,
    GoalVerificationReason.USER_ACTION_REQUIRED,
}

Reviewer = Callable[["GoalVerificationResult", "TestRunResult | None"], bool]


@dataclass(frozen=True, slots=True)
class GoalEvaluation:
    """Serializable decision with the evidence that produced it."""

    status: GoalEvaluationStatus
    reason: str
    verification: GoalVerificationResult
    test_result: "TestRunResult | None" = None


class GoalEvaluator:
    """Decide SUCCESS/RETRY/BLOCKED from verifier evidence and test results."""

    def __init__(self, verifier: GoalVerifier, *, reviewer: Reviewer | None = None) -> None:
        self._verifier = verifier
        self._reviewer = reviewer

    def evaluate(
        self,
        plan: object,
        execution_result: object,
        *,
        test_result: "TestRunResult | None" = None,
    ) -> GoalEvaluation:
        verification = self._verifier.verify(plan, execution_result)
        if (
            verification.verification_status
            in {GoalVerificationStatus.USER_ACTION_REQUIRED, GoalVerificationStatus.INCONCLUSIVE}
            or verification.reason in _TERMINAL_BLOCKED_REASONS
        ):
            return GoalEvaluation(
                GoalEvaluationStatus.BLOCKED,
                "blocked:" + verification.reason.value,
                verification,
                test_result,
            )
        if test_result is not None and not test_result.passed:
            return GoalEvaluation(
                GoalEvaluationStatus.RETRY,
                "tests_failed",
                verification,
                test_result,
            )
        if not verification.satisfied:
            return GoalEvaluation(
                GoalEvaluationStatus.RETRY,
                "verification_failed:" + verification.reason.value,
                verification,
                test_result,
            )
        if self._reviewer is not None and not self._reviewer(verification, test_result):
            return GoalEvaluation(
                GoalEvaluationStatus.RETRY,
                "reviewer_rejected",
                verification,
                test_result,
            )
        return GoalEvaluation(
            GoalEvaluationStatus.SUCCESS,
            "verified",
            verification,
            test_result,
        )

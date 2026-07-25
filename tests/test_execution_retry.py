from __future__ import annotations

from core.execution_retry import (
    RetryEngine,
    RetryPolicy,
    RetryReason,
    RetryStrategy,
)


def test_retry_engine_retries_transient_failure_immediately() -> None:
    decision = RetryEngine().decide(
        RetryPolicy(max_attempts=3, strategy=RetryStrategy.IMMEDIATE),
        attempt_number=1,
        error_code="TRANSIENT_ERROR",
        metadata={},
    )

    assert decision.should_retry is True
    assert decision.reason is RetryReason.TRANSIENT_FAILURE
    assert decision.next_attempt == 2


def test_retry_engine_stops_at_max_attempts() -> None:
    decision = RetryEngine().decide(
        RetryPolicy(max_attempts=2),
        attempt_number=2,
        error_code="TRANSIENT_ERROR",
        metadata={},
    )

    assert decision.should_retry is False
    assert decision.reason is RetryReason.MAX_RETRIES_REACHED
    assert decision.next_attempt == 2


def test_retry_engine_classifies_validation_and_user_cancelled_without_retry() -> None:
    validation = RetryEngine().decide(
        RetryPolicy(max_attempts=2),
        attempt_number=1,
        error_code="TOOL_SCHEMA_VALIDATION_FAILED",
        metadata={},
    )
    cancelled = RetryEngine().decide(
        RetryPolicy(max_attempts=2),
        attempt_number=1,
        error_code="EXECUTION_CANCELLED",
        metadata={},
    )

    assert validation.should_retry is False
    assert validation.reason is RetryReason.VALIDATION_FAILURE
    assert cancelled.should_retry is False
    assert cancelled.reason is RetryReason.USER_CANCELLED


def test_retry_policy_rejects_invalid_contract() -> None:
    try:
        RetryPolicy(max_attempts=0)
    except ValueError as error:
        assert "greater than 0" in str(error)
    else:
        raise AssertionError("RetryPolicy must reject max_attempts=0")

    try:
        RetryPolicy(max_attempts=2, strategy=RetryStrategy.NO_RETRY)
    except ValueError as error:
        assert "NO_RETRY" in str(error)
    else:
        raise AssertionError("NO_RETRY must require one attempt")

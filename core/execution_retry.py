"""Declarative retry decisions for structured execution steps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


MAX_RETRY_ATTEMPTS = 10


class RetryStrategy(str, Enum):
    """Supported retry strategies for deterministic execution."""

    IMMEDIATE = "IMMEDIATE"
    NO_RETRY = "NO_RETRY"


class RetryReason(str, Enum):
    """Stable reasons for retry decisions."""

    SUCCESS = "SUCCESS"
    MAX_RETRIES_REACHED = "MAX_RETRIES_REACHED"
    TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
    PERMANENT_FAILURE = "PERMANENT_FAILURE"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    TOOL_FAILURE = "TOOL_FAILURE"
    USER_CANCELLED = "USER_CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Decision returned by a retry policy for one failed attempt."""

    should_retry: bool
    reason: RetryReason
    next_attempt: int
    max_attempts: int = 1
    delay_ms: int = 0

    @property
    def attempt_number(self) -> int:
        """Return the next attempt number for backward-compatible callers."""
        return self.next_attempt


@dataclass(frozen=True, slots=True)
class RetryableErrorClassifier:
    """Classify structured step failures without guessing unknown exceptions."""

    retryable_error_codes: frozenset[str] = frozenset(
        {
            "TIMEOUT",
            "TEMPORARY_UNAVAILABLE",
            "TRANSIENT_ERROR",
            "EXECUTION_PLAN_OUTPUT_RESOLUTION_FAILED",
            "EXECUTION_PLAN_REFERENCE_NOT_FOUND",
        }
    )

    def classify(
        self,
        *,
        error_code: str | None,
        metadata: dict[str, object],
    ) -> RetryReason | None:
        """Return a retry reason for explicitly retryable failures."""
        normalized_code = (error_code or "").upper()
        if normalized_code in self.retryable_error_codes:
            return RetryReason.TRANSIENT_FAILURE
        child_error_code = str(metadata.get("child_error_code") or "").upper()
        if child_error_code in self.retryable_error_codes:
            return RetryReason.TRANSIENT_FAILURE

        exception_type = str(metadata.get("exception_type") or "")
        if exception_type == "TimeoutError":
            return RetryReason.TRANSIENT_FAILURE

        return None


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Declarative retry policy for individual execution steps."""

    max_attempts: int = 1
    strategy: RetryStrategy = RetryStrategy.IMMEDIATE
    classifier: RetryableErrorClassifier = RetryableErrorClassifier()
    delay_ms: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.strategy, str):
            object.__setattr__(self, "strategy", RetryStrategy(self.strategy))
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ValueError("max_attempts must be an integer.")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than 0.")
        if self.max_attempts > MAX_RETRY_ATTEMPTS:
            raise ValueError(f"max_attempts cannot exceed {MAX_RETRY_ATTEMPTS}.")
        if self.strategy is RetryStrategy.NO_RETRY and self.max_attempts != 1:
            raise ValueError("NO_RETRY strategy requires max_attempts=1.")

    def decide(
        self,
        *,
        attempt_number: int,
        error_code: str | None,
        error: str | None,
        metadata: dict[str, object],
    ) -> RetryDecision:
        """Decide whether another attempt is allowed for this failure."""
        del error
        return RetryEngine().decide(
            self,
            attempt_number=attempt_number,
            error_code=error_code,
            metadata=metadata,
        )


class RetryEngine:
    """Decide whether a failed step should be retried."""

    def decide(
        self,
        policy: RetryPolicy | None,
        *,
        attempt_number: int,
        error_code: str | None,
        metadata: dict[str, object] | None = None,
    ) -> RetryDecision:
        """Return a deterministic retry decision without executing anything."""
        if policy is None:
            return RetryDecision(
                should_retry=False,
                reason=RetryReason.PERMANENT_FAILURE,
                next_attempt=attempt_number,
                max_attempts=attempt_number,
            )

        if policy.strategy is RetryStrategy.NO_RETRY or policy.max_attempts <= 1:
            return RetryDecision(
                should_retry=False,
                reason=self._reason_for_failure(error_code, metadata or {}, policy),
                next_attempt=attempt_number,
                max_attempts=policy.max_attempts,
            )

        if attempt_number >= policy.max_attempts:
            return RetryDecision(
                should_retry=False,
                reason=RetryReason.MAX_RETRIES_REACHED,
                next_attempt=attempt_number,
                max_attempts=policy.max_attempts,
            )

        reason = self._reason_for_failure(error_code, metadata or {}, policy)
        if reason not in {RetryReason.TRANSIENT_FAILURE, RetryReason.TOOL_FAILURE}:
            return RetryDecision(
                should_retry=False,
                reason=reason,
                next_attempt=attempt_number,
                max_attempts=policy.max_attempts,
            )
        return RetryDecision(
            should_retry=True,
            reason=reason,
            next_attempt=attempt_number + 1,
            max_attempts=policy.max_attempts,
        )

    def _reason_for_failure(
        self,
        error_code: str | None,
        metadata: dict[str, object],
        policy: RetryPolicy,
    ) -> RetryReason:
        normalized = (error_code or "").upper()
        if normalized in {"EXECUTION_CANCELLED", "USER_CANCELLED"}:
            return RetryReason.USER_CANCELLED
        if normalized in {
            "PARAMETER_RESOLUTION_FAILED",
            "TOOL_SCHEMA_VALIDATION_FAILED",
            "EXECUTION_CONDITION_FAILED",
            "INVALID_PLAN",
            "VALIDATION_MISMATCH",
        }:
            return RetryReason.VALIDATION_FAILURE
        retryable = policy.classifier.classify(
            error_code=error_code,
            metadata=metadata,
        )
        if retryable is not None:
            return retryable
        if normalized == "TOOL_NOT_FOUND":
            return RetryReason.PERMANENT_FAILURE
        if normalized in {
            "TOOL_EXCEPTION",
            "TOOL_EXECUTION_FAILED",
            "SUBPLAN_FAILED",
            "EXECUTION_VARIABLE_BINDING_FAILED",
        }:
            return RetryReason.TOOL_FAILURE
        if normalized in {"PERMISSION_DENIED", "PERMANENT_FAILURE"}:
            return RetryReason.PERMANENT_FAILURE
        return RetryReason.UNKNOWN


def copy_retry_policy(
    policy: RetryPolicy | None,
) -> RetryPolicy | None:
    """Return an immutable retry policy copy."""
    if policy is None:
        return None
    if not isinstance(policy, RetryPolicy):
        raise TypeError("retry_policy must be RetryPolicy or None.")
    return RetryPolicy(
        max_attempts=policy.max_attempts,
        strategy=policy.strategy,
        classifier=policy.classifier,
        delay_ms=policy.delay_ms,
    )

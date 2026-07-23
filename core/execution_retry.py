"""Controlled retry policy for structured execution steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RetryReason(str, Enum):
    """Stable reasons for retry decisions."""

    TIMEOUT = "timeout"
    TEMPORARY_UNAVAILABLE = "temporary_unavailable"
    TRANSIENT_ERROR = "transient_error"
    ALLOWLISTED_ERROR_CODE = "allowlisted_error_code"
    MAX_ATTEMPTS_REACHED = "max_attempts_reached"
    NON_RETRYABLE_ERROR = "non_retryable_error"
    RETRIES_DISABLED = "retries_disabled"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Decision returned by a retry policy for one failed attempt."""

    should_retry: bool
    reason: str
    attempt_number: int
    max_attempts: int
    delay_ms: int = 0


@dataclass(frozen=True, slots=True)
class RetryableErrorClassifier:
    """Classify structured step failures without guessing unknown exceptions."""

    retryable_error_codes: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "TIMEOUT",
                "TEMPORARY_UNAVAILABLE",
                "TRANSIENT_ERROR",
                "EXECUTION_PLAN_OUTPUT_RESOLUTION_FAILED",
            }
        )
    )

    def classify(
        self,
        *,
        error_code: str | None,
        error: str | None,
        metadata: dict[str, object],
    ) -> str | None:
        """Return a retry reason for explicitly retryable failures."""
        normalized_code = (error_code or "").upper()
        if normalized_code in self.retryable_error_codes:
            return RetryReason.ALLOWLISTED_ERROR_CODE.value
        child_error_code = str(metadata.get("child_error_code") or "").upper()
        if child_error_code in self.retryable_error_codes:
            return RetryReason.ALLOWLISTED_ERROR_CODE.value

        exception_type = str(metadata.get("exception_type") or "")
        if exception_type == "TimeoutError":
            return RetryReason.TIMEOUT.value

        lowered_error = (error or "").lower()
        if "temporary unavailable" in lowered_error:
            return RetryReason.TEMPORARY_UNAVAILABLE.value

        if "transient" in lowered_error:
            return RetryReason.TRANSIENT_ERROR.value

        return None


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Conservative retry policy for individual execution steps."""

    max_attempts: int = 1
    delay_ms: int = 0
    classifier: RetryableErrorClassifier = field(
        default_factory=RetryableErrorClassifier
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            object.__setattr__(self, "max_attempts", 1)
        if self.delay_ms < 0:
            object.__setattr__(self, "delay_ms", 0)

    def decide(
        self,
        *,
        attempt_number: int,
        error_code: str | None,
        error: str | None,
        metadata: dict[str, object],
    ) -> RetryDecision:
        """Decide whether another attempt is allowed for this failure."""
        if self.max_attempts <= 1:
            return RetryDecision(
                should_retry=False,
                reason=RetryReason.RETRIES_DISABLED.value,
                attempt_number=attempt_number,
                max_attempts=self.max_attempts,
            )

        if attempt_number >= self.max_attempts:
            return RetryDecision(
                should_retry=False,
                reason=RetryReason.MAX_ATTEMPTS_REACHED.value,
                attempt_number=attempt_number,
                max_attempts=self.max_attempts,
            )

        retry_reason = self.classifier.classify(
            error_code=error_code,
            error=error,
            metadata=metadata,
        )
        if retry_reason is None:
            return RetryDecision(
                should_retry=False,
                reason=RetryReason.NON_RETRYABLE_ERROR.value,
                attempt_number=attempt_number,
                max_attempts=self.max_attempts,
            )

        return RetryDecision(
            should_retry=True,
            reason=retry_reason,
            attempt_number=attempt_number + 1,
            max_attempts=self.max_attempts,
            delay_ms=self.delay_ms,
        )

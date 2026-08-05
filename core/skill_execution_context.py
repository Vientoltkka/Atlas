"""Shared execution context for cooperative skill cancellation."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True, slots=True)
class SkillExecutionContext:
    """Cooperative execution context with deadline and cancellation signal.

    This context is passed to skill targets that support cooperative cancellation.
    Targets should periodically check `is_cancelled` or `remaining_seconds` to
    voluntarily stop work when the deadline expires.

    Attributes:
        deadline: Monotonic deadline timestamp (seconds since epoch). None if no deadline.
        cancelled: Whether cancellation has been requested.
    """

    deadline: float | None = None
    cancelled: bool = False
    def cancel(self) -> None:
        """Request cooperative cancellation for every target sharing this context."""

        object.__setattr__(self, "cancelled", True)


    @property
    def is_cancelled(self) -> bool:
        """Return True if the execution has been cancelled or deadline expired."""
        if self.cancelled:
            return True
        if self.deadline is not None:
            return time.monotonic() >= self.deadline
        return False

    @property
    def remaining_seconds(self) -> float | None:
        """Return remaining seconds until deadline, or None if no deadline."""
        if self.deadline is None:
            return None
        remaining = self.deadline - time.monotonic()
        return max(0.0, remaining)
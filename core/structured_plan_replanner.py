"""Controlled automatic replanning for structured execution plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType

from core.planner import ExecutionPlan, PlanGenerationResult, Planner


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReplanReason(str, Enum):
    """Closed reasons used to classify one structured replanning decision."""

    RECOVERABLE_FAILURE = "recoverable_failure"
    NON_RECOVERABLE_FAILURE = "non_recoverable_failure"
    LIMIT_REACHED = "limit_reached"
    CANCELLED = "cancelled"
    CONFIRMATION_REQUIRED = "confirmation_required"
    VALIDATION_ERROR = "validation_error"
    PLANNER_ERROR = "planner_error"
    REJECTED = "rejected"


class ReplanResultStatus(str, Enum):
    """Terminal statuses for one controlled replanning attempt."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NOT_RECOVERABLE = "not_recoverable"
    PLANNER_ERROR = "planner_error"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class ReplanRequest:
    """Input provided to a replanner after a supervised execution failure."""

    session_id: str
    original_plan: ExecutionPlan
    failed_step: str | None
    error: str
    partial_results: Mapping[str, object]
    attempt_number: int
    max_attempts: int
    active_plan: ExecutionPlan | None = None
    error_code: str | None = None
    completed_step_ids: tuple[str, ...] = ()
    failed_step_ids: tuple[str, ...] = ()
    cancelled_step_ids: tuple[str, ...] = ()
    pending_step_ids: tuple[str, ...] = ()
    blocked_step_ids: tuple[str, ...] = ()
    dependency_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    batch_id: str | None = None
    errors_by_step: Mapping[str, str] = field(default_factory=dict)
    priority_decision_id: str | None = None
    ordered_ready_step_ids: tuple[str, ...] = ()
    selected_step_ids: tuple[str, ...] = ()
    priority_scores: Mapping[str, float] = field(default_factory=dict)
    failed_step_priority: float | None = None
    priority_rationale_summary: str | None = None
    resource_selection_failure: str | None = None
    rejected_candidate_ids: tuple[str, ...] = ()
    budget_snapshot: object | None = None
    selected_resource_id: str | None = None
    previous_resource_id: str | None = None
    optimization_goal: str | None = None
    degradation_applied: bool = False

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if not isinstance(self.original_plan, ExecutionPlan):
            raise TypeError("original_plan must be an ExecutionPlan.")
        if self.active_plan is not None and not isinstance(self.active_plan, ExecutionPlan):
            raise TypeError("active_plan must be an ExecutionPlan or None.")
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be greater than zero.")
        if self.max_attempts < 0:
            raise ValueError("max_attempts cannot be negative.")
        object.__setattr__(
            self,
            "partial_results",
            MappingProxyType(dict(self.partial_results)),
        )
        object.__setattr__(
            self,
            "completed_step_ids",
            tuple(self.completed_step_ids),
        )
        object.__setattr__(
            self,
            "failed_step_ids",
            tuple(self.failed_step_ids),
        )
        object.__setattr__(
            self,
            "cancelled_step_ids",
            tuple(self.cancelled_step_ids),
        )
        object.__setattr__(
            self,
            "pending_step_ids",
            tuple(self.pending_step_ids),
        )
        object.__setattr__(
            self,
            "blocked_step_ids",
            tuple(self.blocked_step_ids),
        )
        object.__setattr__(
            self,
            "dependency_graph",
            MappingProxyType(
                {
                    str(step_id): tuple(dependencies)
                    for step_id, dependencies in self.dependency_graph.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "errors_by_step",
            MappingProxyType(
                {
                    str(step_id): str(error)
                    for step_id, error in self.errors_by_step.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "ordered_ready_step_ids",
            tuple(self.ordered_ready_step_ids),
        )
        object.__setattr__(
            self,
            "selected_step_ids",
            tuple(self.selected_step_ids),
        )
        object.__setattr__(
            self,
            "priority_scores",
            MappingProxyType(
                {
                    str(step_id): float(score)
                    for step_id, score in self.priority_scores.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "rejected_candidate_ids",
            tuple(str(item) for item in self.rejected_candidate_ids),
        )
        if type(self.degradation_applied) is not bool:
            raise TypeError("degradation_applied must be a bool.")


@dataclass(frozen=True, slots=True)
class ReplanResult:
    """Result produced by a controlled structured replanner."""

    status: ReplanResultStatus
    revised_plan: ExecutionPlan | None = None
    reason: ReplanReason = ReplanReason.REJECTED
    error: str | None = None
    planner_result: PlanGenerationResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ReplanResultStatus):
            raise TypeError("status must be a ReplanResultStatus.")
        if not isinstance(self.reason, ReplanReason):
            raise TypeError("reason must be a ReplanReason.")
        if self.status is ReplanResultStatus.ACCEPTED:
            if not isinstance(self.revised_plan, ExecutionPlan):
                raise ValueError("accepted replanning requires a revised_plan.")
        elif self.revised_plan is not None:
            raise ValueError("only accepted replanning can include a revised_plan.")

    @property
    def accepted(self) -> bool:
        """Return whether this result contains a revised execution plan."""
        return self.status is ReplanResultStatus.ACCEPTED


@dataclass(frozen=True, slots=True)
class ReplanRecord:
    """Immutable trace record for one accepted replanning attempt."""

    attempt_number: int
    previous_plan: ExecutionPlan
    revised_plan: ExecutionPlan
    reason: ReplanReason
    failed_step: str | None
    error: str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be greater than zero.")
        if not isinstance(self.previous_plan, ExecutionPlan):
            raise TypeError("previous_plan must be an ExecutionPlan.")
        if not isinstance(self.revised_plan, ExecutionPlan):
            raise TypeError("revised_plan must be an ExecutionPlan.")
        if not isinstance(self.reason, ReplanReason):
            raise TypeError("reason must be a ReplanReason.")


@dataclass(frozen=True, slots=True)
class ReplanPolicy:
    """Conservative policy for one-session automatic replanning."""

    max_replans_per_session: int = 1
    recoverable_error_codes: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "TOOL_EXECUTION_FAILED",
                "TOOL_EXCEPTION",
                "TOOL_NOT_FOUND",
                "PARAMETER_RESOLUTION_FAILED",
                "NO_COMPATIBLE_RESOURCE",
                "EXECUTION_BUDGET_EXCEEDED",
                "EXECUTION_TOKEN_BUDGET_EXCEEDED",
            }
        )
    )
    non_recoverable_error_codes: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "INVALID_PLAN",
                "VALIDATION_MISMATCH",
                "CONFIRMATION_REQUIRED",
                "EXECUTION_CANCELLED",
                "TOOL_SCHEMA_VALIDATION_FAILED",
                "SUBPLAN_VALIDATION_FAILED",
            }
        )
    )

    def __post_init__(self) -> None:
        if self.max_replans_per_session < 0:
            raise ValueError("max_replans_per_session cannot be negative.")
        object.__setattr__(
            self,
            "recoverable_error_codes",
            frozenset(self.recoverable_error_codes),
        )
        object.__setattr__(
            self,
            "non_recoverable_error_codes",
            frozenset(self.non_recoverable_error_codes),
        )

    def evaluate(
        self,
        request: ReplanRequest,
        *,
        current_replan_count: int,
    ) -> ReplanResult:
        """Return a policy-only decision without invoking a planner."""
        if current_replan_count >= self.max_replans_per_session:
            return ReplanResult(
                status=ReplanResultStatus.LIMIT_REACHED,
                reason=ReplanReason.LIMIT_REACHED,
                error="replan limit reached",
            )
        if request.error_code in self.non_recoverable_error_codes:
            return ReplanResult(
                status=ReplanResultStatus.NOT_RECOVERABLE,
                reason=ReplanReason.NON_RECOVERABLE_FAILURE,
                error="failure is not recoverable",
            )
        if request.error_code not in self.recoverable_error_codes:
            return ReplanResult(
                status=ReplanResultStatus.NOT_RECOVERABLE,
                reason=ReplanReason.NON_RECOVERABLE_FAILURE,
                error="failure is not classified as recoverable",
            )
        return ReplanResult(
            status=ReplanResultStatus.ACCEPTED,
            revised_plan=request.active_plan or request.original_plan,
            reason=ReplanReason.RECOVERABLE_FAILURE,
        )


class ExecutionReplanner:
    """Generate revised plans through the existing structured planner."""

    def __init__(self, planner: Planner) -> None:
        self._planner = planner

    def replan(self, request: ReplanRequest) -> ReplanResult:
        """Request one revised plan without executing it."""
        try:
            generation = self._planner.generate_execution_plan(
                self._replan_objective(request)
            )
        except Exception as error:
            return ReplanResult(
                status=ReplanResultStatus.PLANNER_ERROR,
                reason=ReplanReason.PLANNER_ERROR,
                error=str(error) or type(error).__name__,
            )

        if not generation.success or generation.plan is None:
            return ReplanResult(
                status=ReplanResultStatus.REJECTED,
                reason=ReplanReason.REJECTED,
                error="; ".join(generation.errors) if generation.errors else generation.error_code,
                planner_result=generation,
            )

        return ReplanResult(
            status=ReplanResultStatus.ACCEPTED,
            revised_plan=generation.plan,
            reason=ReplanReason.RECOVERABLE_FAILURE,
            planner_result=generation,
        )

    def _replan_objective(self, request: ReplanRequest) -> str:
        completed = ", ".join(sorted(request.partial_results)) or "none"
        return "\n".join(
            [
                "Revise the structured execution plan after a recoverable failure.",
                f"Original objective: {request.original_plan.goal}",
                f"Failed step: {request.failed_step or 'unknown'}",
                f"Error: {request.error}",
                f"Completed step results available: {completed}",
                f"Pending steps: {', '.join(request.pending_step_ids) or 'none'}",
                f"Blocked steps: {', '.join(request.blocked_step_ids) or 'none'}",
                f"Replan attempt: {request.attempt_number} of {request.max_attempts}",
            ]
        )


def replan_record(
    request: ReplanRequest,
    result: ReplanResult,
    *,
    previous_plan: ExecutionPlan,
    created_at: datetime | None = None,
) -> ReplanRecord:
    """Build a trace record for an accepted replanning result."""
    if not result.accepted or result.revised_plan is None:
        raise ValueError("accepted ReplanResult with revised_plan is required.")
    return ReplanRecord(
        attempt_number=request.attempt_number,
        previous_plan=previous_plan,
        revised_plan=result.revised_plan,
        reason=result.reason,
        failed_step=request.failed_step,
        error=request.error,
        created_at=created_at or _utc_now(),
    )

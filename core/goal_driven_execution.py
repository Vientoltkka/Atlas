"""Bounded goal-driven execution controller."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable

from core.execution_context import ExecutionContext
from core.execution_plan_executor import ExecutionControl, ExecutionPlanExecutor, PlanExecutionResult
from core.execution_plan_validator import ExecutionPlanValidator, PlanValidationResult, plan_signature
from core.execution_replanner import (
    ExecutionReplanner,
    ReplanningCandidate,
    ReplanningDecision,
    ReplanningPolicy,
    ReplanningRequest,
    ReplanningStatus,
)
from core.goal_verifier import (
    GoalVerificationResult,
    GoalVerifier,
    goal_verification_result_from_dict,
    goal_verification_result_to_dict,
)
from core.planner import ExecutionPlan


MAX_GOAL_DRIVEN_CYCLES = 10
MAX_GOAL_DRIVEN_METADATA_ITEMS = 32


class GoalDrivenExecutionError(RuntimeError):
    """Base error for bounded goal-driven execution contracts."""


class InvalidGoalDrivenExecutionRequestError(GoalDrivenExecutionError):
    """Raised when a goal-driven request is malformed."""


class GoalDrivenExecutionStatus(str, Enum):
    """Terminal and structured statuses for goal-driven execution."""

    COMPLETED = "COMPLETED"
    GOAL_SATISFIED = "GOAL_SATISFIED"
    GOAL_UNSATISFIED = "GOAL_UNSATISFIED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    REPLANNING_FAILED = "REPLANNING_FAILED"
    NO_ALTERNATIVE_PLAN = "NO_ALTERNATIVE_PLAN"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    CYCLE_LIMIT_REACHED = "CYCLE_LIMIT_REACHED"
    POLICY_DISABLED = "POLICY_DISABLED"
    INVALID_REQUEST = "INVALID_REQUEST"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class GoalDrivenExecutionDecision(str, Enum):
    """Closed decisions made after one cycle."""

    FINISH_GOAL_SATISFIED = "FINISH_GOAL_SATISFIED"
    FINISH_POLICY_DISABLED = "FINISH_POLICY_DISABLED"
    FINISH_GOAL_UNSATISFIED = "FINISH_GOAL_UNSATISFIED"
    FINISH_EXECUTION_FAILED = "FINISH_EXECUTION_FAILED"
    FINISH_VALIDATION_FAILED = "FINISH_VALIDATION_FAILED"
    FINISH_CANCELLED = "FINISH_CANCELLED"
    FINISH_BLOCKED = "FINISH_BLOCKED"
    FINISH_CYCLE_LIMIT_REACHED = "FINISH_CYCLE_LIMIT_REACHED"
    REQUEST_REPLANNING = "REQUEST_REPLANNING"
    SELECT_REPLANNED_PLAN = "SELECT_REPLANNED_PLAN"
    REJECT_REPLANNING = "REJECT_REPLANNING"


@dataclass(frozen=True, slots=True)
class GoalDrivenExecutionPolicy:
    """Immutable limits for bounded goal-driven execution."""

    enabled: bool = False
    max_cycles: int = 1
    allow_step_retries: bool = True
    allow_replanning: bool = False
    stop_on_goal_success: bool = True
    replanning_policy: ReplanningPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise InvalidGoalDrivenExecutionRequestError("enabled must be a bool.")
        if isinstance(self.max_cycles, bool) or not isinstance(self.max_cycles, int):
            raise InvalidGoalDrivenExecutionRequestError("max_cycles must be an integer.")
        if self.max_cycles <= 0 or self.max_cycles > MAX_GOAL_DRIVEN_CYCLES:
            raise InvalidGoalDrivenExecutionRequestError(
                f"max_cycles must be between 1 and {MAX_GOAL_DRIVEN_CYCLES}."
            )
        if not isinstance(self.allow_step_retries, bool):
            raise InvalidGoalDrivenExecutionRequestError("allow_step_retries must be a bool.")
        if not isinstance(self.allow_replanning, bool):
            raise InvalidGoalDrivenExecutionRequestError("allow_replanning must be a bool.")
        if not isinstance(self.stop_on_goal_success, bool):
            raise InvalidGoalDrivenExecutionRequestError("stop_on_goal_success must be a bool.")
        if self.replanning_policy is not None and not isinstance(self.replanning_policy, ReplanningPolicy):
            raise InvalidGoalDrivenExecutionRequestError("replanning_policy must be ReplanningPolicy or None.")


@dataclass(frozen=True, slots=True)
class GoalDrivenExecutionEvent:
    """Safe event with no arguments, outputs, prompts, or secrets."""

    name: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _safe_text(self.name, "event name"))
        object.__setattr__(self, "status", _safe_text(self.status, "event status"))
        object.__setattr__(self, "details", MappingProxyType(_safe_metadata(self.details)))


@dataclass(frozen=True, slots=True)
class GoalDrivenExecutionCycle:
    """Safe summary for one bounded execution cycle."""

    cycle_number: int
    plan_signature: str
    execution_id: str | None
    execution_status: str | None
    goal_verification_result: GoalVerificationResult | None
    decision: GoalDrivenExecutionDecision
    replanned_plan_signature: str | None = None
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.cycle_number, bool) or not isinstance(self.cycle_number, int) or self.cycle_number <= 0:
            raise InvalidGoalDrivenExecutionRequestError("cycle_number must be a positive int.")
        object.__setattr__(self, "plan_signature", _safe_text(self.plan_signature, "plan_signature"))
        object.__setattr__(self, "decision", _decision(self.decision))
        for field_name in ("execution_id", "execution_status", "replanned_plan_signature", "termination_reason"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _safe_text(value, field_name))


@dataclass(frozen=True, slots=True)
class GoalDrivenExecutionRequest:
    """Structured request for one bounded goal-driven execution."""

    plan: ExecutionPlan
    policy: GoalDrivenExecutionPolicy = field(default_factory=GoalDrivenExecutionPolicy)
    candidates: tuple[ReplanningCandidate, ...] = ()
    inputs: Mapping[str, object] = field(default_factory=dict)
    confirmation_granted: bool = False
    control: ExecutionControl | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExecutionPlan):
            raise InvalidGoalDrivenExecutionRequestError("plan must be ExecutionPlan.")
        if not isinstance(self.policy, GoalDrivenExecutionPolicy):
            raise InvalidGoalDrivenExecutionRequestError("policy must be GoalDrivenExecutionPolicy.")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not all(isinstance(candidate, ReplanningCandidate) for candidate in self.candidates):
            raise InvalidGoalDrivenExecutionRequestError("candidates must contain ReplanningCandidate values.")
        if not isinstance(self.confirmation_granted, bool):
            raise InvalidGoalDrivenExecutionRequestError("confirmation_granted must be a bool.")
        if self.control is not None and not isinstance(self.control, ExecutionControl):
            raise InvalidGoalDrivenExecutionRequestError("control must be ExecutionControl or None.")
        object.__setattr__(self, "inputs", MappingProxyType(_safe_metadata(self.inputs)))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class GoalDrivenExecutionResult:
    """Final structured result of a bounded goal-driven execution."""

    status: GoalDrivenExecutionStatus
    cycles: tuple[GoalDrivenExecutionCycle, ...] = ()
    current_plan: ExecutionPlan | None = None
    validation_result: PlanValidationResult | None = None
    execution_result: PlanExecutionResult | None = None
    goal_verification_result: GoalVerificationResult | None = None
    replanning_decision: ReplanningDecision | None = None
    used_plan_signatures: tuple[str, ...] = ()
    events: tuple[GoalDrivenExecutionEvent, ...] = ()
    terminal_reason: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(self, "cycles", tuple(self.cycles))
        object.__setattr__(self, "used_plan_signatures", tuple(self.used_plan_signatures))
        object.__setattr__(self, "events", tuple(self.events))

    @property
    def completed(self) -> bool:
        """Return whether the goal-driven execution completed successfully."""

        return self.status in {GoalDrivenExecutionStatus.COMPLETED, GoalDrivenExecutionStatus.GOAL_SATISFIED}


GoalDrivenObserver = Callable[[GoalDrivenExecutionEvent], None]


class GoalDrivenExecutionController:
    """Coordinate bounded execute -> verify -> decide -> replan cycles."""

    def __init__(
        self,
        execution_plan_validator: ExecutionPlanValidator,
        execution_plan_executor: ExecutionPlanExecutor,
        *,
        goal_verifier: GoalVerifier | None = None,
        execution_replanner: ExecutionReplanner | None = None,
        observer: GoalDrivenObserver | None = None,
    ) -> None:
        if not isinstance(execution_plan_validator, ExecutionPlanValidator):
            raise GoalDrivenExecutionError("GoalDrivenExecutionController requires ExecutionPlanValidator.")
        if not isinstance(execution_plan_executor, ExecutionPlanExecutor):
            raise GoalDrivenExecutionError("GoalDrivenExecutionController requires ExecutionPlanExecutor.")
        if goal_verifier is not None and not isinstance(goal_verifier, GoalVerifier):
            raise GoalDrivenExecutionError("goal_verifier must be GoalVerifier or None.")
        if execution_replanner is not None and not isinstance(execution_replanner, ExecutionReplanner):
            raise GoalDrivenExecutionError("execution_replanner must be ExecutionReplanner or None.")
        if observer is not None and not callable(observer):
            raise GoalDrivenExecutionError("observer must be callable or None.")
        self._validator = execution_plan_validator
        self._executor = execution_plan_executor
        self._goal_verifier = goal_verifier or GoalVerifier()
        self._replanner = execution_replanner or ExecutionReplanner()
        self._observer = observer

    def execute(
        self,
        request: GoalDrivenExecutionRequest,
    ) -> GoalDrivenExecutionResult:
        """Run a bounded goal-driven execution cycle."""
        events: list[GoalDrivenExecutionEvent] = []
        if not isinstance(request, GoalDrivenExecutionRequest):
            _record(events, self._observer, "goal_driven_execution_started", "failed")
            return _result(
                GoalDrivenExecutionStatus.INVALID_REQUEST,
                events,
                error_code="INVALID_REQUEST",
                error_message="request must be GoalDrivenExecutionRequest.",
            )
        _record(events, self._observer, "goal_driven_execution_started", "started")
        if not request.policy.enabled:
            _record(events, self._observer, "goal_driven_execution_failed", "finished", {"reason": "policy_disabled"})
            return _result(
                GoalDrivenExecutionStatus.POLICY_DISABLED,
                events,
                current_plan=request.plan,
                terminal_reason="policy disabled",
            )
        if not request.policy.allow_step_retries and _plan_has_retry_policy(request.plan):
            _record(events, self._observer, "goal_driven_execution_failed", "failed", {"reason": "step_retries_disabled"})
            return _result(
                GoalDrivenExecutionStatus.INVALID_REQUEST,
                events,
                current_plan=request.plan,
                error_code="STEP_RETRIES_DISABLED",
                error_message="Plan contains retry policies but goal-driven step retries are disabled.",
            )

        current_plan = request.plan
        original_plan = request.plan
        used_signatures: list[str] = []
        cycles: list[GoalDrivenExecutionCycle] = []
        last_execution: PlanExecutionResult | None = None
        last_goal: GoalVerificationResult | None = None
        last_validation: PlanValidationResult | None = None
        last_replanning: ReplanningDecision | None = None

        for cycle_number in range(1, request.policy.max_cycles + 1):
            signature = plan_signature(current_plan)
            if signature in used_signatures:
                _record(events, self._observer, "goal_driven_execution_failed", "failed", {"reason": "repeated_signature"})
                return _result(
                    GoalDrivenExecutionStatus.NO_ALTERNATIVE_PLAN,
                    events,
                    cycles=tuple(cycles),
                    current_plan=current_plan,
                    validation_result=last_validation,
                    execution_result=last_execution,
                    goal_verification_result=last_goal,
                    replanning_decision=last_replanning,
                    used_plan_signatures=tuple(used_signatures),
                    terminal_reason="plan signature already used",
                )
            used_signatures.append(signature)
            _record(
                events,
                self._observer,
                "goal_driven_cycle_started",
                "started",
                {"cycle": cycle_number, "plan_signature": signature},
            )
            validation = self._validator.validate(current_plan)
            last_validation = validation
            if not validation.is_valid:
                cycle = GoalDrivenExecutionCycle(
                    cycle_number=cycle_number,
                    plan_signature=signature,
                    execution_id=None,
                    execution_status=None,
                    goal_verification_result=None,
                    decision=GoalDrivenExecutionDecision.FINISH_VALIDATION_FAILED,
                    termination_reason="validation failed",
                )
                cycles.append(cycle)
                _record(events, self._observer, "goal_driven_cycle_completed", "failed", {"cycle": cycle_number})
                _record(events, self._observer, "goal_driven_execution_failed", "failed", {"reason": "validation_failed"})
                return _result(
                    GoalDrivenExecutionStatus.VALIDATION_FAILED,
                    events,
                    cycles=tuple(cycles),
                    current_plan=current_plan,
                    validation_result=validation,
                    used_plan_signatures=tuple(used_signatures),
                    terminal_reason="validation failed",
                    error_code="VALIDATION_FAILED",
                    error_message="Current execution plan did not pass validation.",
                )

            try:
                execution = self._executor.execute(
                    current_plan,
                    validation,
                    confirmation_granted=request.confirmation_granted,
                    control=request.control,
                    execution_context=ExecutionContext(initial_variables=request.inputs),
                )
                goal = self._goal_verifier.verify(current_plan, execution)
            except (TypeError, ValueError, RuntimeError) as error:
                _record(events, self._observer, "goal_driven_execution_failed", "failed", {"reason": type(error).__name__})
                return _result(
                    GoalDrivenExecutionStatus.INTERNAL_ERROR,
                    events,
                    cycles=tuple(cycles),
                    current_plan=current_plan,
                    validation_result=validation,
                    execution_result=last_execution,
                    goal_verification_result=last_goal,
                    used_plan_signatures=tuple(used_signatures),
                    terminal_reason="internal error",
                    error_code=type(error).__name__,
                    error_message=str(error),
                )
            last_execution = execution
            last_goal = goal
            execution_id = execution.trace.execution_id if execution.trace is not None else None
            _record(
                events,
                self._observer,
                "goal_driven_plan_executed",
                "finished" if execution.success else "failed",
                {"cycle": cycle_number, "execution_status": execution.status},
            )
            _record(
                events,
                self._observer,
                "goal_driven_goal_verified",
                "finished" if goal.satisfied else "failed",
                {"cycle": cycle_number, "goal_reason": goal.reason.value},
            )
            if execution.cancelled:
                cycles.append(
                    GoalDrivenExecutionCycle(
                        cycle_number,
                        signature,
                        execution_id,
                        execution.status,
                        goal,
                        GoalDrivenExecutionDecision.FINISH_CANCELLED,
                        termination_reason="execution cancelled",
                    )
                )
                _record(events, self._observer, "goal_driven_execution_cancelled", "finished")
                return _result(
                    GoalDrivenExecutionStatus.CANCELLED,
                    events,
                    cycles=tuple(cycles),
                    current_plan=current_plan,
                    validation_result=validation,
                    execution_result=execution,
                    goal_verification_result=goal,
                    used_plan_signatures=tuple(used_signatures),
                    terminal_reason="execution cancelled",
                    error_code=execution.error_code,
                    error_message=execution.error,
                )
            if execution.blocked:
                cycles.append(
                    GoalDrivenExecutionCycle(
                        cycle_number,
                        signature,
                        execution_id,
                        execution.status,
                        goal,
                        GoalDrivenExecutionDecision.FINISH_BLOCKED,
                        termination_reason="execution blocked",
                    )
                )
                _record(events, self._observer, "goal_driven_execution_failed", "failed", {"reason": "blocked"})
                return _result(
                    GoalDrivenExecutionStatus.BLOCKED,
                    events,
                    cycles=tuple(cycles),
                    current_plan=current_plan,
                    validation_result=validation,
                    execution_result=execution,
                    goal_verification_result=goal,
                    used_plan_signatures=tuple(used_signatures),
                    terminal_reason="execution blocked",
                )
            if goal.satisfied and request.policy.stop_on_goal_success:
                cycles.append(
                    GoalDrivenExecutionCycle(
                        cycle_number,
                        signature,
                        execution_id,
                        execution.status,
                        goal,
                        GoalDrivenExecutionDecision.FINISH_GOAL_SATISFIED,
                        termination_reason="goal satisfied",
                    )
                )
                _record(events, self._observer, "goal_driven_cycle_completed", "finished", {"cycle": cycle_number})
                _record(events, self._observer, "goal_driven_execution_succeeded", "finished")
                return _result(
                    GoalDrivenExecutionStatus.COMPLETED,
                    events,
                    cycles=tuple(cycles),
                    current_plan=current_plan,
                    validation_result=validation,
                    execution_result=execution,
                    goal_verification_result=goal,
                    used_plan_signatures=tuple(used_signatures),
                    terminal_reason="goal satisfied",
                )

            if cycle_number >= request.policy.max_cycles:
                cycles.append(
                    GoalDrivenExecutionCycle(
                        cycle_number,
                        signature,
                        execution_id,
                        execution.status,
                        goal,
                        GoalDrivenExecutionDecision.FINISH_CYCLE_LIMIT_REACHED,
                        termination_reason="cycle limit reached",
                    )
                )
                _record(events, self._observer, "goal_driven_cycle_limit_reached", "failed", {"cycle": cycle_number})
                return _result(
                    GoalDrivenExecutionStatus.CYCLE_LIMIT_REACHED,
                    events,
                    cycles=tuple(cycles),
                    current_plan=current_plan,
                    validation_result=validation,
                    execution_result=execution,
                    goal_verification_result=goal,
                    used_plan_signatures=tuple(used_signatures),
                    terminal_reason="cycle limit reached",
                )

            if request.policy.allow_replanning:
                replanning_policy = _effective_replanning_policy(request.policy, current_plan)
                _record(events, self._observer, "goal_driven_replanning_requested", "started", {"cycle": cycle_number})
                decision = self._replanner.decide(
                    replanning_policy,
                    ReplanningRequest(
                        original_plan=original_plan,
                        failed_plan=current_plan,
                        execution_result=execution,
                        goal_verification_result=goal,
                        candidates=request.candidates,
                        replan_attempts=len([item for item in cycles if item.replanned_plan_signature is not None]),
                        excluded_plan_signatures=tuple(used_signatures),
                    ),
                )
                last_replanning = decision
                if decision.should_replan and decision.replacement_plan is not None:
                    replacement_signature = decision.replacement_plan_signature or plan_signature(decision.replacement_plan)
                    if replacement_signature in used_signatures:
                        cycles.append(
                            GoalDrivenExecutionCycle(
                                cycle_number,
                                signature,
                                execution_id,
                                execution.status,
                                goal,
                                GoalDrivenExecutionDecision.REJECT_REPLANNING,
                                replanned_plan_signature=replacement_signature,
                                termination_reason="replanned signature already used",
                            )
                        )
                        _record(events, self._observer, "goal_driven_execution_failed", "failed", {"reason": "repeated_signature"})
                        return _result(
                            GoalDrivenExecutionStatus.NO_ALTERNATIVE_PLAN,
                            events,
                            cycles=tuple(cycles),
                            current_plan=current_plan,
                            validation_result=validation,
                            execution_result=execution,
                            goal_verification_result=goal,
                            replanning_decision=decision,
                            used_plan_signatures=tuple(used_signatures),
                            terminal_reason="replanned signature already used",
                        )
                    _record(
                        events,
                        self._observer,
                        "goal_driven_replanning_selected",
                        "finished",
                        {"cycle": cycle_number, "replacement_plan_signature": replacement_signature},
                    )
                    cycles.append(
                        GoalDrivenExecutionCycle(
                            cycle_number,
                            signature,
                            execution_id,
                            execution.status,
                            goal,
                            GoalDrivenExecutionDecision.SELECT_REPLANNED_PLAN,
                            replanned_plan_signature=replacement_signature,
                            termination_reason="replanned",
                        )
                    )
                    _record(events, self._observer, "goal_driven_cycle_completed", "finished", {"cycle": cycle_number})
                    current_plan = decision.replacement_plan
                    continue

                cycles.append(
                    GoalDrivenExecutionCycle(
                        cycle_number,
                        signature,
                        execution_id,
                        execution.status,
                        goal,
                        GoalDrivenExecutionDecision.REJECT_REPLANNING,
                        termination_reason=decision.reason,
                    )
                )
                _record(events, self._observer, "goal_driven_execution_failed", "failed", {"reason": decision.status.value})
                return _result(
                    _status_for_replanning_failure(decision.status, execution),
                    events,
                    cycles=tuple(cycles),
                    current_plan=current_plan,
                    validation_result=validation,
                    execution_result=execution,
                    goal_verification_result=goal,
                    replanning_decision=decision,
                    used_plan_signatures=tuple(used_signatures),
                    terminal_reason=decision.reason,
                    error_code=decision.status.value,
                    error_message=decision.reason,
                )

            cycles.append(
                GoalDrivenExecutionCycle(
                    cycle_number,
                    signature,
                    execution_id,
                    execution.status,
                    goal,
                    (
                        GoalDrivenExecutionDecision.FINISH_EXECUTION_FAILED
                        if not execution.success
                        else GoalDrivenExecutionDecision.FINISH_GOAL_UNSATISFIED
                    ),
                    termination_reason="no valid action",
                )
            )
            _record(events, self._observer, "goal_driven_execution_failed", "failed", {"reason": "no_valid_action"})
            return _result(
                GoalDrivenExecutionStatus.EXECUTION_FAILED
                if not execution.success
                else GoalDrivenExecutionStatus.GOAL_UNSATISFIED,
                events,
                cycles=tuple(cycles),
                current_plan=current_plan,
                validation_result=validation,
                execution_result=execution,
                goal_verification_result=goal,
                used_plan_signatures=tuple(used_signatures),
                terminal_reason="no valid action",
                error_code=execution.error_code,
                error_message=execution.error,
            )

        _record(events, self._observer, "goal_driven_cycle_limit_reached", "failed")
        return _result(
            GoalDrivenExecutionStatus.CYCLE_LIMIT_REACHED,
            events,
            cycles=tuple(cycles),
            current_plan=current_plan,
            validation_result=last_validation,
            execution_result=last_execution,
            goal_verification_result=last_goal,
            replanning_decision=last_replanning,
            used_plan_signatures=tuple(used_signatures),
            terminal_reason="cycle limit reached",
        )


def goal_driven_policy_to_dict(
    policy: GoalDrivenExecutionPolicy | None,
) -> dict[str, object] | None:
    """Serialize a goal-driven policy to safe JSON-compatible data."""
    if policy is None:
        return None
    return {
        "$type": "goal_driven_execution_policy",
        "enabled": policy.enabled,
        "max_cycles": policy.max_cycles,
        "allow_step_retries": policy.allow_step_retries,
        "allow_replanning": policy.allow_replanning,
        "stop_on_goal_success": policy.stop_on_goal_success,
        "replanning_policy": _replanning_policy_to_dict(policy.replanning_policy),
    }


def goal_driven_policy_from_dict(
    payload: Any,
) -> GoalDrivenExecutionPolicy | None:
    """Load a goal-driven policy from checkpoint data."""
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "$type",
        "enabled",
        "max_cycles",
        "allow_step_retries",
        "allow_replanning",
        "stop_on_goal_success",
        "replanning_policy",
    }:
        raise ValueError("goal-driven policy must be an explicit object or null.")
    if payload.get("$type") != "goal_driven_execution_policy":
        raise ValueError("goal-driven policy type is invalid.")
    return GoalDrivenExecutionPolicy(
        enabled=_bool(payload, "enabled"),
        max_cycles=_int(payload, "max_cycles"),
        allow_step_retries=_bool(payload, "allow_step_retries"),
        allow_replanning=_bool(payload, "allow_replanning"),
        stop_on_goal_success=_bool(payload, "stop_on_goal_success"),
        replanning_policy=_replanning_policy_from_dict(payload.get("replanning_policy")),
    )


def goal_driven_cycles_to_dict(
    cycles: tuple[GoalDrivenExecutionCycle, ...],
) -> list[dict[str, object]]:
    """Serialize safe goal-driven cycle summaries."""
    return [
        {
            "cycle_number": cycle.cycle_number,
            "plan_signature": cycle.plan_signature,
            "execution_id": cycle.execution_id,
            "execution_status": cycle.execution_status,
            "goal_verification_result": goal_verification_result_to_dict(cycle.goal_verification_result),
            "decision": cycle.decision.value,
            "replanned_plan_signature": cycle.replanned_plan_signature,
            "termination_reason": cycle.termination_reason,
        }
        for cycle in cycles
    ]


def goal_driven_cycles_from_dict(
    payload: Any,
) -> tuple[GoalDrivenExecutionCycle, ...]:
    """Load safe goal-driven cycle summaries."""
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError("goal-driven cycles must be a list.")
    return tuple(
        GoalDrivenExecutionCycle(
            cycle_number=_int(item, "cycle_number"),
            plan_signature=_str(item, "plan_signature"),
            execution_id=_optional_str(item, "execution_id"),
            execution_status=_optional_str(item, "execution_status"),
            goal_verification_result=goal_verification_result_from_dict(item.get("goal_verification_result")),
            decision=GoalDrivenExecutionDecision(_str(item, "decision")),
            replanned_plan_signature=_optional_str(item, "replanned_plan_signature"),
            termination_reason=_optional_str(item, "termination_reason"),
        )
        for item in payload
        if isinstance(item, dict)
    )


def _result(
    status: GoalDrivenExecutionStatus,
    events: list[GoalDrivenExecutionEvent],
    *,
    cycles: tuple[GoalDrivenExecutionCycle, ...] = (),
    current_plan: ExecutionPlan | None = None,
    validation_result: PlanValidationResult | None = None,
    execution_result: PlanExecutionResult | None = None,
    goal_verification_result: GoalVerificationResult | None = None,
    replanning_decision: ReplanningDecision | None = None,
    used_plan_signatures: tuple[str, ...] = (),
    terminal_reason: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> GoalDrivenExecutionResult:
    return GoalDrivenExecutionResult(
        status=status,
        cycles=cycles,
        current_plan=current_plan,
        validation_result=validation_result,
        execution_result=execution_result,
        goal_verification_result=goal_verification_result,
        replanning_decision=replanning_decision,
        used_plan_signatures=used_plan_signatures,
        events=tuple(events),
        terminal_reason=terminal_reason,
        error_code=error_code,
        error_message=error_message,
    )


def _record(
    events: list[GoalDrivenExecutionEvent],
    observer: GoalDrivenObserver | None,
    name: str,
    status: str,
    details: Mapping[str, object] | None = None,
) -> None:
    event = GoalDrivenExecutionEvent(name, status, {} if details is None else details)
    events.append(event)
    if observer is not None:
        observer(event)


def _effective_replanning_policy(
    policy: GoalDrivenExecutionPolicy,
    plan: ExecutionPlan,
) -> ReplanningPolicy | None:
    if policy.replanning_policy is not None:
        return policy.replanning_policy
    if isinstance(plan.replanning_policy, ReplanningPolicy):
        return plan.replanning_policy
    return None


def _status_for_replanning_failure(
    status: ReplanningStatus,
    execution: PlanExecutionResult,
) -> GoalDrivenExecutionStatus:
    if status is ReplanningStatus.NO_ALTERNATIVE_PLAN:
        return GoalDrivenExecutionStatus.NO_ALTERNATIVE_PLAN
    if status is ReplanningStatus.LIMIT_REACHED:
        return GoalDrivenExecutionStatus.CYCLE_LIMIT_REACHED
    if execution.cancelled:
        return GoalDrivenExecutionStatus.CANCELLED
    if not execution.success:
        return GoalDrivenExecutionStatus.EXECUTION_FAILED
    return GoalDrivenExecutionStatus.REPLANNING_FAILED


def _plan_has_retry_policy(
    plan: ExecutionPlan,
) -> bool:
    for step in plan.ordered_steps:
        policy = getattr(step, "retry_policy", None)
        if policy is not None and getattr(policy, "max_attempts", 1) > 1:
            return True
        if step.subplan is not None and _plan_has_retry_policy(step.subplan):
            return True
    return False


def _replanning_policy_to_dict(policy: ReplanningPolicy | None) -> dict[str, object] | None:
    from core.execution_replanner import replanning_policy_to_dict

    return replanning_policy_to_dict(policy)


def _replanning_policy_from_dict(payload: Any) -> ReplanningPolicy | None:
    from core.execution_replanner import replanning_policy_from_dict

    return replanning_policy_from_dict(payload)


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidGoalDrivenExecutionRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_GOAL_DRIVEN_METADATA_ITEMS:
        raise InvalidGoalDrivenExecutionRequestError("metadata has too many items.")
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        safe[_safe_text(key, "metadata key")] = _safe_value(value)
    return safe


def _safe_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise InvalidGoalDrivenExecutionRequestError("metadata floats must be finite.")
        return value
    raise InvalidGoalDrivenExecutionRequestError("metadata values must be primitive safe values.")


def _safe_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidGoalDrivenExecutionRequestError(f"{field_name} must be a string.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise InvalidGoalDrivenExecutionRequestError(f"{field_name} cannot be empty.")
    return normalized


def _status(value: GoalDrivenExecutionStatus | str) -> GoalDrivenExecutionStatus:
    if isinstance(value, GoalDrivenExecutionStatus):
        return value
    if isinstance(value, str):
        return GoalDrivenExecutionStatus(value)
    raise InvalidGoalDrivenExecutionRequestError("status must be GoalDrivenExecutionStatus.")


def _decision(value: GoalDrivenExecutionDecision | str) -> GoalDrivenExecutionDecision:
    if isinstance(value, GoalDrivenExecutionDecision):
        return value
    if isinstance(value, str):
        return GoalDrivenExecutionDecision(value)
    raise InvalidGoalDrivenExecutionRequestError("decision must be GoalDrivenExecutionDecision.")


def _str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null.")
    return value


def _int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return value


def _bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a bool.")
    return value

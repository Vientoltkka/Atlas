"""Controlled orchestration for capability-selected execution plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable

from core.capability_planner import (
    CapabilityPlanner,
    CapabilityPlanningDecision,
    CapabilityPlanningError,
    CapabilityPlanningRequest,
    CapabilityPlanningStatus,
)
from core.execution_plan_executor import (
    ExecutionControl,
    ExecutionPlanExecutor,
    PlanExecutionResult,
)
from core.execution_context import ExecutionContext
from core.execution_plan_validator import ExecutionPlanValidator, PlanValidationResult
from core.planner import ExecutionPlan


MAX_CAPABILITY_ORCHESTRATION_METADATA_ITEMS = 32


class CapabilityOrchestrationError(RuntimeError):
    """Base error for capability orchestration contract violations."""


class InvalidCapabilityOrchestrationRequestError(CapabilityOrchestrationError):
    """Raised when a capability orchestration request is malformed."""


class CapabilityPlanningFailedError(CapabilityOrchestrationError):
    """Raised when capability planning fails unexpectedly."""


class CapabilityPlanValidationFailedError(CapabilityOrchestrationError):
    """Raised when plan validation fails unexpectedly."""


class CapabilityExecutionFailedError(CapabilityOrchestrationError):
    """Raised when plan execution fails unexpectedly."""


class CapabilityOrchestrationStatus(str, Enum):
    """Stable states for CapabilityPlanner -> Validator -> Executor orchestration."""

    COMPLETED = "completed"
    NO_CAPABILITY_CANDIDATES = "no_capability_candidates"
    CAPABILITY_AMBIGUOUS = "capability_ambiguous"
    NO_WORKFLOW_CANDIDATES = "no_workflow_candidates"
    WORKFLOW_BELOW_MINIMUM_SCORE = "workflow_below_minimum_score"
    WORKFLOW_AMBIGUOUS = "workflow_ambiguous"
    INVALID_REQUEST = "invalid_request"
    PLANNING_FAILED = "planning_failed"
    PLAN_VALIDATION_FAILED = "plan_validation_failed"
    EXECUTION_FAILED = "execution_failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class CapabilityOrchestrationEvent:
    """Safe orchestration event with no arguments, outputs, prompts, or secrets."""

    name: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidCapabilityOrchestrationRequestError("event name must be a non-empty string.")
        if not isinstance(self.status, str) or not self.status.strip():
            raise InvalidCapabilityOrchestrationRequestError("event status must be a non-empty string.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "status", self.status.strip())
        object.__setattr__(self, "details", MappingProxyType(_safe_metadata(self.details)))


@dataclass(frozen=True, slots=True)
class CapabilityOrchestrationPolicy:
    """Explicit execution policy for a selected and validated capability plan."""

    confirmation_granted: bool = False
    control: ExecutionControl | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.confirmation_granted, bool):
            raise InvalidCapabilityOrchestrationRequestError("confirmation_granted must be a bool.")
        if self.control is not None and not isinstance(self.control, ExecutionControl):
            raise InvalidCapabilityOrchestrationRequestError("control must be ExecutionControl or None.")


@dataclass(frozen=True, slots=True)
class CapabilityOrchestrationRequest:
    """Structured request for controlled capability orchestration."""

    planning_request: CapabilityPlanningRequest
    policy: CapabilityOrchestrationPolicy = field(default_factory=CapabilityOrchestrationPolicy)
    inputs: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.planning_request, CapabilityPlanningRequest):
            raise InvalidCapabilityOrchestrationRequestError(
                "planning_request must be CapabilityPlanningRequest."
            )
        if not isinstance(self.policy, CapabilityOrchestrationPolicy):
            raise InvalidCapabilityOrchestrationRequestError("policy must be CapabilityOrchestrationPolicy.")
        object.__setattr__(self, "inputs", MappingProxyType(_safe_metadata(self.inputs)))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class CapabilityOrchestrationResult:
    """Immutable outcome for controlled capability orchestration."""

    status: CapabilityOrchestrationStatus
    planning_decision: CapabilityPlanningDecision | None = None
    selected_plan: ExecutionPlan | None = None
    validation_result: PlanValidationResult | None = None
    execution_result: PlanExecutionResult | None = None
    error_code: str | None = None
    error_message: str | None = None
    events: tuple[CapabilityOrchestrationEvent, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _validate_status(self.status))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))

    @property
    def completed(self) -> bool:
        """Return whether the complete Planner -> Validator -> Executor chain succeeded."""

        return self.status is CapabilityOrchestrationStatus.COMPLETED


Observer = Callable[[CapabilityOrchestrationEvent], None]


class CapabilityOrchestrator:
    """Coordinate capability planning, plan validation, and plan execution."""

    _PLANNING_STATUS_MAP = {
        CapabilityPlanningStatus.NO_CAPABILITY_CANDIDATES: CapabilityOrchestrationStatus.NO_CAPABILITY_CANDIDATES,
        CapabilityPlanningStatus.CAPABILITY_AMBIGUOUS: CapabilityOrchestrationStatus.CAPABILITY_AMBIGUOUS,
        CapabilityPlanningStatus.NO_WORKFLOW_CANDIDATES: CapabilityOrchestrationStatus.NO_WORKFLOW_CANDIDATES,
        CapabilityPlanningStatus.WORKFLOW_BELOW_MINIMUM_SCORE: (
            CapabilityOrchestrationStatus.WORKFLOW_BELOW_MINIMUM_SCORE
        ),
        CapabilityPlanningStatus.WORKFLOW_AMBIGUOUS: CapabilityOrchestrationStatus.WORKFLOW_AMBIGUOUS,
        CapabilityPlanningStatus.INVALID_REQUEST: CapabilityOrchestrationStatus.INVALID_REQUEST,
    }

    def __init__(
        self,
        capability_planner: CapabilityPlanner,
        execution_plan_validator: ExecutionPlanValidator,
        execution_plan_executor: ExecutionPlanExecutor,
        *,
        observer: Observer | None = None,
    ) -> None:
        if not isinstance(capability_planner, CapabilityPlanner):
            raise CapabilityOrchestrationError("CapabilityOrchestrator requires CapabilityPlanner.")
        if not isinstance(execution_plan_validator, ExecutionPlanValidator):
            raise CapabilityOrchestrationError("CapabilityOrchestrator requires ExecutionPlanValidator.")
        if not isinstance(execution_plan_executor, ExecutionPlanExecutor):
            raise CapabilityOrchestrationError("CapabilityOrchestrator requires ExecutionPlanExecutor.")
        if observer is not None and not callable(observer):
            raise CapabilityOrchestrationError("observer must be callable or None.")
        self._capability_planner = capability_planner
        self._execution_plan_validator = execution_plan_validator
        self._execution_plan_executor = execution_plan_executor
        self._observer = observer

    def orchestrate(self, request: CapabilityOrchestrationRequest) -> CapabilityOrchestrationResult:
        """Run the controlled orchestration flow without touching external entrypoints."""

        events: list[CapabilityOrchestrationEvent] = []
        if not isinstance(request, CapabilityOrchestrationRequest):
            _record(events, self._observer, "capability_orchestration_started", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.INVALID_REQUEST,
                events,
                error_code="INVALID_REQUEST",
                error_message="request must be CapabilityOrchestrationRequest.",
            )

        _record(events, self._observer, "capability_orchestration_started", "started")
        _record(events, self._observer, "capability_planning_started", "started")
        try:
            decision = self._capability_planner.plan(request.planning_request)
        except CapabilityPlanningError as error:
            _record(events, self._observer, "capability_planning_failed", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.PLANNING_FAILED,
                events,
                error_code=type(error).__name__,
                error_message=str(error),
            )

        if not isinstance(decision, CapabilityPlanningDecision):
            _record(events, self._observer, "capability_planning_failed", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.PLANNING_FAILED,
                events,
                error_code="INVALID_PLANNING_DECISION",
                error_message="CapabilityPlanner returned an invalid decision.",
            )

        _record(
            events,
            self._observer,
            "capability_planning_succeeded",
            "finished",
            {"planning_status": decision.status.value, "has_plan": decision.plan is not None},
        )
        if decision.status is not CapabilityPlanningStatus.SELECTED or decision.plan is None:
            return self._complete(
                self._status_for_unselected_decision(decision.status),
                events,
                planning_decision=decision,
                selected_plan=None,
            )

        plan = decision.plan
        _record(
            events,
            self._observer,
            "capability_plan_validation_started",
            "started",
            {"step_count": len(plan.ordered_steps), "required_tool_count": len(plan.required_tools)},
        )
        try:
            validation = self._execution_plan_validator.validate(plan)
        except (TypeError, ValueError, RuntimeError) as error:
            _record(events, self._observer, "capability_plan_validation_failed", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.PLAN_VALIDATION_FAILED,
                events,
                planning_decision=decision,
                selected_plan=plan,
                error_code=type(error).__name__,
                error_message=str(error),
            )

        if not validation.is_valid:
            _record(
                events,
                self._observer,
                "capability_plan_validation_failed",
                "failed",
                {"error_count": len(validation.errors), "warning_count": len(validation.warnings)},
            )
            return self._complete(
                CapabilityOrchestrationStatus.PLAN_VALIDATION_FAILED,
                events,
                planning_decision=decision,
                selected_plan=plan,
                validation_result=validation,
                error_code="PLAN_VALIDATION_FAILED",
                error_message="Selected execution plan did not pass validation.",
            )

        _record(
            events,
            self._observer,
            "capability_plan_validation_succeeded",
            "finished",
            {"warning_count": len(validation.warnings), "requires_confirmation": validation.requires_confirmation},
        )
        _record(
            events,
            self._observer,
            "capability_execution_started",
            "started",
            {"step_count": len(plan.ordered_steps)},
        )
        try:
            execution = self._execution_plan_executor.execute(
                plan,
                validation,
                confirmation_granted=request.policy.confirmation_granted,
                control=request.policy.control,
                execution_context=ExecutionContext(initial_variables=request.inputs),
            )
        except (TypeError, ValueError, RuntimeError) as error:
            _record(events, self._observer, "capability_execution_failed", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.EXECUTION_FAILED,
                events,
                planning_decision=decision,
                selected_plan=plan,
                validation_result=validation,
                error_code=type(error).__name__,
                error_message=str(error),
            )

        if execution.cancelled:
            _record(events, self._observer, "capability_execution_failed", "failed", {"cancelled": True})
            return self._complete(
                CapabilityOrchestrationStatus.CANCELLED,
                events,
                planning_decision=decision,
                selected_plan=plan,
                validation_result=validation,
                execution_result=execution,
                error_code=execution.error_code,
                error_message=execution.error,
            )
        if not execution.success:
            _record(
                events,
                self._observer,
                "capability_execution_failed",
                "failed",
                {"execution_status": execution.status, "failed_step_count": len(execution.failed_steps)},
            )
            return self._complete(
                CapabilityOrchestrationStatus.EXECUTION_FAILED,
                events,
                planning_decision=decision,
                selected_plan=plan,
                validation_result=validation,
                execution_result=execution,
                error_code=execution.error_code,
                error_message=execution.error,
            )

        _record(
            events,
            self._observer,
            "capability_execution_succeeded",
            "finished",
            {"completed_step_count": len(execution.completed_steps), "skipped_step_count": len(execution.skipped_steps)},
        )
        return self._complete(
            CapabilityOrchestrationStatus.COMPLETED,
            events,
            planning_decision=decision,
            selected_plan=plan,
            validation_result=validation,
            execution_result=execution,
        )

    def _complete(
        self,
        status: CapabilityOrchestrationStatus,
        events: list[CapabilityOrchestrationEvent],
        *,
        planning_decision: CapabilityPlanningDecision | None = None,
        selected_plan: ExecutionPlan | None = None,
        validation_result: PlanValidationResult | None = None,
        execution_result: PlanExecutionResult | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> CapabilityOrchestrationResult:
        _record(
            events,
            self._observer,
            "capability_orchestration_completed",
            "finished",
            {"orchestration_status": status.value},
        )
        return _result(
            status,
            events,
            planning_decision=planning_decision,
            selected_plan=selected_plan,
            validation_result=validation_result,
            execution_result=execution_result,
            error_code=error_code,
            error_message=error_message,
        )

    def _status_for_unselected_decision(
        self,
        status: CapabilityPlanningStatus,
    ) -> CapabilityOrchestrationStatus:
        return self._PLANNING_STATUS_MAP.get(status, CapabilityOrchestrationStatus.PLANNING_FAILED)


def _record(
    events: list[CapabilityOrchestrationEvent],
    observer: Observer | None,
    name: str,
    status: str,
    details: Mapping[str, object] | None = None,
) -> None:
    event = CapabilityOrchestrationEvent(name=name, status=status, details={} if details is None else details)
    events.append(event)
    if observer is not None:
        observer(event)


def _result(
    status: CapabilityOrchestrationStatus,
    events: list[CapabilityOrchestrationEvent],
    *,
    planning_decision: CapabilityPlanningDecision | None = None,
    selected_plan: ExecutionPlan | None = None,
    validation_result: PlanValidationResult | None = None,
    execution_result: PlanExecutionResult | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> CapabilityOrchestrationResult:
    return CapabilityOrchestrationResult(
        status=status,
        planning_decision=planning_decision,
        selected_plan=selected_plan,
        validation_result=validation_result,
        execution_result=execution_result,
        error_code=error_code,
        error_message=error_message,
        events=tuple(events),
    )


def _validate_status(status: CapabilityOrchestrationStatus | str) -> CapabilityOrchestrationStatus:
    if isinstance(status, CapabilityOrchestrationStatus):
        return status
    if isinstance(status, str):
        try:
            return CapabilityOrchestrationStatus(status)
        except ValueError as error:
            raise InvalidCapabilityOrchestrationRequestError("invalid orchestration status.") from error
    raise InvalidCapabilityOrchestrationRequestError("status must be CapabilityOrchestrationStatus.")


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidCapabilityOrchestrationRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_CAPABILITY_ORCHESTRATION_METADATA_ITEMS:
        raise InvalidCapabilityOrchestrationRequestError("metadata has too many items.")
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise InvalidCapabilityOrchestrationRequestError("metadata keys must be non-empty strings.")
        safe[key] = _safe_value(value)
    return safe


def _safe_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise InvalidCapabilityOrchestrationRequestError("metadata floats must be finite.")
        return value
    raise InvalidCapabilityOrchestrationRequestError("metadata values must be primitive safe values.")

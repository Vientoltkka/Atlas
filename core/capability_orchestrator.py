"""Controlled orchestration for capability-selected execution plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable

from core.agent_executor import (
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentExecutionStatus,
    AgentExecutor,
)
from core.agent_registry import validate_agent_id
from core.agent_resolver import AgentResolutionRequest
from core.skill_execution_context import SkillExecutionContext
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
from core.goal_verifier import (
    GoalVerificationResult,
    GoalVerificationStatus,
)
from core.execution_plan_validator import ExecutionPlanValidator, PlanValidationResult
from core.execution_plan_validator import plan_signature
from core.execution_replanner import (
    ExecutionReplanner,
    ReplanningCandidate,
    ReplanningDecision,
    ReplanningHistoryEntry,
    ReplanningPolicy,
    ReplanningRequest,
    ReplanningStatus,
    ReplanningStrategy,
)
from core.goal_driven_execution import (
    GoalDrivenExecutionController,
    GoalDrivenExecutionPolicy,
    GoalDrivenExecutionRequest,
    GoalDrivenExecutionResult,
    GoalDrivenExecutionStatus,
)
from core.planner import ExecutionPlan


MAX_CAPABILITY_ORCHESTRATION_METADATA_ITEMS = 32
_PERMISSION_IDS = frozenset(
    {
        "can_read_project",
        "can_write_files",
        "can_execute_tools",
        "can_modify_memory",
        "can_use_network",
        "requires_confirmation",
    }
)


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
class AgentExecutionPolicy:
    """Explicit opt-in policy for running a specialized agent pipeline."""

    enabled: bool = False
    allow_agent_execution: bool = False
    preferred_agent_id: str | None = None
    require_explicit_agent: bool = True
    fail_if_agent_missing: bool = True
    required_capability_ids: tuple[str, ...] = ()
    required_permission_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "enabled",
            "allow_agent_execution",
            "require_explicit_agent",
            "fail_if_agent_missing",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise InvalidCapabilityOrchestrationRequestError(f"{field_name} must be a bool.")
        if self.preferred_agent_id is not None:
            object.__setattr__(self, "preferred_agent_id", validate_agent_id(self.preferred_agent_id))
        object.__setattr__(
            self,
            "required_capability_ids",
            _identifier_tuple(self.required_capability_ids, "required_capability_ids"),
        )
        object.__setattr__(
            self,
            "required_permission_ids",
            _permission_tuple(self.required_permission_ids, "required_permission_ids"),
        )


@dataclass(frozen=True, slots=True)
class CapabilityOrchestrationPolicy:
    """Explicit execution policy for a selected and validated capability plan."""

    confirmation_granted: bool = False
    control: ExecutionControl | None = None
    replanning_policy: ReplanningPolicy | None = None
    goal_driven_policy: GoalDrivenExecutionPolicy | None = None
    agent_execution_policy: AgentExecutionPolicy = field(default_factory=AgentExecutionPolicy)
    execution_context: SkillExecutionContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.confirmation_granted, bool):
            raise InvalidCapabilityOrchestrationRequestError("confirmation_granted must be a bool.")
        if self.control is not None and not isinstance(self.control, ExecutionControl):
            raise InvalidCapabilityOrchestrationRequestError("control must be ExecutionControl or None.")
        if self.replanning_policy is not None and not isinstance(self.replanning_policy, ReplanningPolicy):
            raise InvalidCapabilityOrchestrationRequestError("replanning_policy must be ReplanningPolicy or None.")
        if self.goal_driven_policy is not None and not isinstance(
            self.goal_driven_policy,
            GoalDrivenExecutionPolicy,
        ):
            raise InvalidCapabilityOrchestrationRequestError(
                "goal_driven_policy must be GoalDrivenExecutionPolicy or None."
            )
        if not isinstance(self.agent_execution_policy, AgentExecutionPolicy):
            raise InvalidCapabilityOrchestrationRequestError(
                "agent_execution_policy must be AgentExecutionPolicy."
            )
        if self.execution_context is not None and not isinstance(
            self.execution_context,
            SkillExecutionContext,
        ):
            raise InvalidCapabilityOrchestrationRequestError(
                "execution_context must be SkillExecutionContext or None."
            )


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
    goal_verification_result: GoalVerificationResult | None = None
    error_code: str | None = None
    error_message: str | None = None
    replanning_attempted: bool = False
    replan_attempts: int = 0
    replanning_status: str | None = None
    replanning_reason: str | None = None
    original_plan_signature: str | None = None
    final_plan_signature: str | None = None
    replanning_history: tuple[Mapping[str, object], ...] = ()
    goal_driven_result: GoalDrivenExecutionResult | None = None
    agent_execution_result: AgentExecutionResult | None = None
    checkpoint: Mapping[str, object] = field(default_factory=dict)
    metrics: Mapping[str, object] = field(default_factory=dict)
    events: tuple[CapabilityOrchestrationEvent, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _validate_status(self.status))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "replanning_history", tuple(self.replanning_history))
        if self.agent_execution_result is not None and not isinstance(
            self.agent_execution_result,
            AgentExecutionResult,
        ):
            raise InvalidCapabilityOrchestrationRequestError(
                "agent_execution_result must be AgentExecutionResult or None."
            )
        object.__setattr__(self, "checkpoint", MappingProxyType(_safe_metadata(self.checkpoint)))
        object.__setattr__(self, "metrics", MappingProxyType(_safe_metadata(self.metrics)))
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
        execution_replanner: ExecutionReplanner | None = None,
        goal_driven_controller: GoalDrivenExecutionController | None = None,
        agent_executor: AgentExecutor | None = None,
        observer: Observer | None = None,
    ) -> None:
        if not isinstance(capability_planner, CapabilityPlanner):
            raise CapabilityOrchestrationError("CapabilityOrchestrator requires CapabilityPlanner.")
        if not isinstance(execution_plan_validator, ExecutionPlanValidator):
            raise CapabilityOrchestrationError("CapabilityOrchestrator requires ExecutionPlanValidator.")
        if not isinstance(execution_plan_executor, ExecutionPlanExecutor):
            raise CapabilityOrchestrationError("CapabilityOrchestrator requires ExecutionPlanExecutor.")
        if agent_executor is not None and not isinstance(agent_executor, AgentExecutor):
            raise CapabilityOrchestrationError("agent_executor must be AgentExecutor or None.")
        if observer is not None and not callable(observer):
            raise CapabilityOrchestrationError("observer must be callable or None.")
        self._capability_planner = capability_planner
        self._execution_plan_validator = execution_plan_validator
        self._execution_plan_executor = execution_plan_executor
        self._execution_replanner = execution_replanner or ExecutionReplanner()
        self._goal_driven_controller = goal_driven_controller
        self._agent_executor = agent_executor
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
        if request.policy.agent_execution_policy.enabled:
            return self._execute_agent_pipeline(request, events)

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

        return self._execute_selected_plan(
            decision.plan,
            request.policy,
            request.inputs,
            events,
            planning_decision=decision,
        )

    def orchestrate_plan(
        self,
        plan: ExecutionPlan,
        *,
        policy: CapabilityOrchestrationPolicy | None = None,
        inputs: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> CapabilityOrchestrationResult:
        """Validate and execute an already-planned composed capability plan."""

        events: list[CapabilityOrchestrationEvent] = []
        _record(events, self._observer, "capability_orchestration_started", "started")
        if not isinstance(plan, ExecutionPlan):
            _record(events, self._observer, "capability_planning_failed", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.INVALID_REQUEST,
                events,
                error_code="INVALID_PLAN",
                error_message="plan must be ExecutionPlan.",
            )
        if policy is None:
            active_policy = CapabilityOrchestrationPolicy()
        elif isinstance(policy, CapabilityOrchestrationPolicy):
            active_policy = policy
        else:
            _record(events, self._observer, "capability_planning_failed", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.INVALID_REQUEST,
                events,
                error_code="INVALID_POLICY",
                error_message="policy must be CapabilityOrchestrationPolicy or None.",
            )
        try:
            safe_inputs = MappingProxyType(_safe_metadata({} if inputs is None else inputs))
            safe_metadata = MappingProxyType(_safe_metadata({} if metadata is None else metadata))
        except InvalidCapabilityOrchestrationRequestError as error:
            _record(events, self._observer, "capability_planning_failed", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.INVALID_REQUEST,
                events,
                error_code=type(error).__name__,
                error_message=str(error),
            )
        _record(
            events,
            self._observer,
            "capability_planning_succeeded",
            "finished",
            {
                "planning_status": "multi_capability_planned",
                "has_plan": True,
                "metadata_items": len(safe_metadata),
            },
        )
        return self._execute_selected_plan(plan, active_policy, safe_inputs, events)

    def _execute_agent_pipeline(
        self,
        request: CapabilityOrchestrationRequest,
        events: list[CapabilityOrchestrationEvent],
    ) -> CapabilityOrchestrationResult:
        policy = request.policy.agent_execution_policy
        _record(
            events,
            self._observer,
            "agent_pipeline_started",
            "started",
            {"preferred_agent": policy.preferred_agent_id is not None},
        )
        if not policy.allow_agent_execution:
            _record(events, self._observer, "agent_pipeline_failed", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.INVALID_REQUEST,
                events,
                error_code="AGENT_EXECUTION_NOT_ALLOWED",
                error_message="Agent execution policy is enabled but allow_agent_execution is false.",
                checkpoint=_agent_checkpoint(policy, None, CapabilityOrchestrationStatus.INVALID_REQUEST),
                metrics=_agent_metrics(started=1, completed=0, failed=1),
            )
        if policy.require_explicit_agent and policy.preferred_agent_id is None:
            _record(events, self._observer, "agent_pipeline_failed", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.INVALID_REQUEST,
                events,
                error_code="EXPLICIT_AGENT_REQUIRED",
                error_message="Agent execution requires preferred_agent_id.",
                checkpoint=_agent_checkpoint(policy, None, CapabilityOrchestrationStatus.INVALID_REQUEST),
                metrics=_agent_metrics(started=1, completed=0, failed=1),
            )
        if self._agent_executor is None:
            _record(events, self._observer, "agent_pipeline_failed", "failed")
            status = (
                CapabilityOrchestrationStatus.EXECUTION_FAILED
                if policy.fail_if_agent_missing
                else CapabilityOrchestrationStatus.NO_CAPABILITY_CANDIDATES
            )
            return self._complete(
                status,
                events,
                error_code="AGENT_EXECUTOR_UNAVAILABLE",
                error_message="Agent executor is not configured.",
                checkpoint=_agent_checkpoint(policy, None, status),
                metrics=_agent_metrics(started=1, completed=0, failed=1),
            )

        agent_request = AgentExecutionRequest(
            resolution_request=AgentResolutionRequest(
                required_agent_ids=(
                    (policy.preferred_agent_id,)
                    if policy.preferred_agent_id is not None
                    else ()
                ),
                preferred_agent_ids=(
                    (policy.preferred_agent_id,)
                    if policy.preferred_agent_id is not None
                    else ()
                ),
                enabled_only=False,
                require_unique_top_score=True,
            ),
            structured_input=request.inputs,
            metadata={
                "source": "capability_orchestrator",
                "request_metadata_items": len(request.metadata),
            },
            required_capability_ids=policy.required_capability_ids,
            required_permission_ids=policy.required_permission_ids,
            execution_context=request.policy.execution_context,
        )
        try:
            agent_result = self._agent_executor.execute(agent_request)
        except (TypeError, ValueError, RuntimeError) as error:
            _record(events, self._observer, "agent_pipeline_failed", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.EXECUTION_FAILED,
                events,
                error_code=type(error).__name__,
                error_message=str(error),
                checkpoint=_agent_checkpoint(policy, None, CapabilityOrchestrationStatus.EXECUTION_FAILED),
                metrics=_agent_metrics(started=1, completed=0, failed=1),
            )

        status = _orchestration_status_for_agent(agent_result.status)
        if agent_result.completed:
            _record(
                events,
                self._observer,
                "agent_pipeline_completed",
                "finished",
                {"agent_id": agent_result.agent_id or ""},
            )
            metrics = _agent_metrics(started=1, completed=1, failed=0)
        else:
            _record(
                events,
                self._observer,
                "agent_pipeline_failed",
                "failed",
                {"agent_status": agent_result.status.value, "agent_id": agent_result.agent_id or ""},
            )
            metrics = _agent_metrics(started=1, completed=0, failed=1)

        return self._complete(
            status,
            events,
            error_code=agent_result.error_code,
            error_message=agent_result.safe_message,
            agent_execution_result=agent_result,
            checkpoint=_agent_checkpoint(policy, agent_result, status),
            metrics=metrics,
        )

    def _execute_selected_plan(
        self,
        plan: ExecutionPlan,
        policy: CapabilityOrchestrationPolicy,
        inputs: Mapping[str, object],
        events: list[CapabilityOrchestrationEvent],
        *,
        planning_decision: CapabilityPlanningDecision | None = None,
    ) -> CapabilityOrchestrationResult:
        if policy.goal_driven_policy is not None and policy.goal_driven_policy.enabled:
            return self._execute_goal_driven_plan(
                plan,
                policy,
                inputs,
                events,
                planning_decision=planning_decision,
            )
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
                planning_decision=planning_decision,
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
                planning_decision=planning_decision,
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
                confirmation_granted=policy.confirmation_granted,
                control=policy.control,
                execution_context=ExecutionContext(initial_variables=inputs),
            )
        except (TypeError, ValueError, RuntimeError) as error:
            _record(events, self._observer, "capability_execution_failed", "failed")
            return self._complete(
                CapabilityOrchestrationStatus.EXECUTION_FAILED,
                events,
                planning_decision=planning_decision,
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
                planning_decision=planning_decision,
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
            replanned = self._try_replanning(
                original_plan=plan,
                failed_plan=plan,
                policy=policy,
                inputs=inputs,
                events=events,
                planning_decision=planning_decision,
                failed_validation=validation,
                failed_execution=execution,
                failed_goal_verification=execution.goal_verification_result,
            )
            if replanned is not None:
                return replanned
            return self._complete(
                CapabilityOrchestrationStatus.EXECUTION_FAILED,
                events,
                planning_decision=planning_decision,
                selected_plan=plan,
                validation_result=validation,
                execution_result=execution,
                error_code=execution.error_code,
                error_message=execution.error,
            )

        goal_verification = execution.goal_verification_result
        legacy_inconclusive = (
            goal_verification is not None
            and goal_verification.verification_status
            is GoalVerificationStatus.INCONCLUSIVE
            and not plan.acceptance_criteria
            and not plan.required_outputs
            and not plan.output_validators
        )
        if (
            goal_verification is None
            or (
                not goal_verification.satisfied
                and not legacy_inconclusive
            )
        ):
            _record(
                events,
                self._observer,
                "capability_execution_failed",
                "failed",
                {
                    "execution_status": execution.status,
                    "goal_satisfied": False,
                },
            )
            replanned = self._try_replanning(
                original_plan=plan,
                failed_plan=plan,
                policy=policy,
                inputs=inputs,
                events=events,
                planning_decision=planning_decision,
                failed_validation=validation,
                failed_execution=execution,
                failed_goal_verification=goal_verification,
            )
            if replanned is not None:
                return replanned
            return self._complete(
                CapabilityOrchestrationStatus.EXECUTION_FAILED,
                events,
                planning_decision=planning_decision,
                selected_plan=plan,
                validation_result=validation,
                execution_result=execution,
                goal_verification_result=goal_verification,
                error_code=(
                    goal_verification.reason.value
                    if goal_verification is not None
                    else "UNKNOWN"
                ),
                error_message="Capability goal verification failed.",
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
            planning_decision=planning_decision,
            selected_plan=plan,
            validation_result=validation,
            execution_result=execution,
            goal_verification_result=goal_verification,
        )

    def _execute_goal_driven_plan(
        self,
        plan: ExecutionPlan,
        policy: CapabilityOrchestrationPolicy,
        inputs: Mapping[str, object],
        events: list[CapabilityOrchestrationEvent],
        *,
        planning_decision: CapabilityPlanningDecision | None = None,
    ) -> CapabilityOrchestrationResult:
        controller = self._goal_driven_controller or GoalDrivenExecutionController(
            self._execution_plan_validator,
            self._execution_plan_executor,
            execution_replanner=self._execution_replanner,
        )
        candidates = self._replanning_candidates(planning_decision)
        result = controller.execute(
            GoalDrivenExecutionRequest(
                plan=plan,
                policy=policy.goal_driven_policy or GoalDrivenExecutionPolicy(),
                candidates=candidates,
                inputs=inputs,
                confirmation_granted=policy.confirmation_granted,
                control=policy.control,
            )
        )
        for event in result.events:
            _record(events, self._observer, event.name, event.status, event.details)
        status = _orchestration_status_for_goal_driven(result.status)
        return self._complete(
            status,
            events,
            planning_decision=planning_decision,
            selected_plan=result.current_plan or plan,
            validation_result=result.validation_result,
            execution_result=result.execution_result,
            goal_verification_result=result.goal_verification_result,
            error_code=result.error_code or result.status.value,
            error_message=result.error_message or result.terminal_reason,
            replanning_decision=result.replanning_decision,
            original_plan_signature=plan_signature(plan),
            final_plan_signature=(
                plan_signature(result.current_plan)
                if result.current_plan is not None
                else plan_signature(plan)
            ),
            goal_driven_result=result,
        )

    def _try_replanning(
        self,
        *,
        original_plan: ExecutionPlan,
        failed_plan: ExecutionPlan,
        policy: CapabilityOrchestrationPolicy,
        inputs: Mapping[str, object],
        events: list[CapabilityOrchestrationEvent],
        planning_decision: CapabilityPlanningDecision | None,
        failed_validation: PlanValidationResult | None,
        failed_execution: PlanExecutionResult | None,
        failed_goal_verification: GoalVerificationResult | None,
    ) -> CapabilityOrchestrationResult | None:
        replanning_policy = _effective_replanning_policy(policy, original_plan)
        if not replanning_policy.enabled:
            return None
        candidates = self._replanning_candidates(planning_decision)
        history: list[ReplanningHistoryEntry] = []
        current_plan = failed_plan
        current_execution = failed_execution
        current_goal = failed_goal_verification
        latest_validation: PlanValidationResult | None = failed_validation
        latest_status = CapabilityOrchestrationStatus.EXECUTION_FAILED
        latest_error_code = (
            current_goal.reason.value
            if current_goal is not None
            else (current_execution.error_code if current_execution is not None else "UNKNOWN")
        )
        latest_error_message = "Capability goal verification failed."

        for attempt_index in range(replanning_policy.max_replans):
            _record(
                events,
                self._observer,
                "replanning_requested",
                "started",
                {"attempt": attempt_index + 1, "strategy": replanning_policy.strategy.value},
            )
            decision = self._execution_replanner.decide(
                replanning_policy,
                ReplanningRequest(
                    original_plan=original_plan,
                    failed_plan=current_plan,
                    execution_result=current_execution,
                    goal_verification_result=current_goal,
                    candidates=candidates,
                    replan_attempts=attempt_index,
                    history=tuple(history),
                ),
            )
            if not decision.should_replan:
                _record(
                    events,
                    self._observer,
                    _event_name_for_replanning_status(decision.status),
                    "finished",
                    {"status": decision.status.value, "reason": decision.reason},
                )
                return self._complete(
                    latest_status,
                    events,
                    planning_decision=planning_decision,
                    selected_plan=current_plan,
                    validation_result=latest_validation,
                    execution_result=current_execution,
                    goal_verification_result=current_goal,
                    error_code=latest_error_code,
                    error_message=latest_error_message,
                    replanning_decision=decision,
                    replanning_history=tuple(history),
                    original_plan_signature=plan_signature(original_plan),
                    final_plan_signature=plan_signature(current_plan),
                )

            if decision.history_entry is not None:
                history.append(decision.history_entry)
            _record(
                events,
                self._observer,
                "replanning_plan_selected",
                "finished",
                {
                    "attempt": decision.replan_attempts,
                    "previous_plan_signature": decision.previous_plan_signature,
                    "replacement_plan_signature": decision.replacement_plan_signature,
                },
            )
            replacement = decision.replacement_plan
            if replacement is None:
                return None

            _record(
                events,
                self._observer,
                "replanned_plan_validation_started",
                "started",
                {"attempt": decision.replan_attempts, "step_count": len(replacement.ordered_steps)},
            )
            validation = self._execution_plan_validator.validate(replacement)
            latest_validation = validation
            if not validation.is_valid:
                _record(
                    events,
                    self._observer,
                    "replanning_failed",
                    "failed",
                    {"attempt": decision.replan_attempts, "error_count": len(validation.errors)},
                )
                current_plan = replacement
                latest_error_code = "PLAN_VALIDATION_FAILED"
                latest_error_message = "Replanned execution plan did not pass validation."
                continue

            _record(
                events,
                self._observer,
                "replanned_plan_execution_started",
                "started",
                {"attempt": decision.replan_attempts, "step_count": len(replacement.ordered_steps)},
            )
            execution = self._execution_plan_executor.execute(
                replacement,
                validation,
                confirmation_granted=policy.confirmation_granted,
                control=policy.control,
                execution_context=ExecutionContext(initial_variables=inputs),
            )
            current_plan = replacement
            current_execution = execution
            current_goal = execution.goal_verification_result
            latest_error_code = execution.error_code
            latest_error_message = execution.error or "Replanned capability execution failed."
            if execution.cancelled:
                latest_status = CapabilityOrchestrationStatus.CANCELLED
                _record(events, self._observer, "replanning_failed", "failed", {"cancelled": True})
                break
            if execution.success and current_goal is not None and current_goal.satisfied:
                _record(
                    events,
                    self._observer,
                    "replanning_succeeded",
                    "finished",
                    {"attempt": decision.replan_attempts},
                )
                return self._complete(
                    CapabilityOrchestrationStatus.COMPLETED,
                    events,
                    planning_decision=planning_decision,
                    selected_plan=replacement,
                    validation_result=validation,
                    execution_result=execution,
                    goal_verification_result=current_goal,
                    replanning_decision=decision,
                    replanning_history=tuple(history),
                    original_plan_signature=plan_signature(original_plan),
                    final_plan_signature=plan_signature(replacement),
                )

            _record(
                events,
                self._observer,
                "replanning_failed",
                "failed",
                {
                    "attempt": decision.replan_attempts,
                    "execution_status": execution.status,
                    "goal_satisfied": current_goal.satisfied if current_goal is not None else False,
                },
            )

        return self._complete(
            latest_status,
            events,
            planning_decision=planning_decision,
            selected_plan=current_plan,
            validation_result=latest_validation,
            execution_result=current_execution,
            goal_verification_result=current_goal,
            error_code=latest_error_code,
            error_message=latest_error_message,
            replanning_decision=ReplanningDecision(
                should_replan=False,
                status=ReplanningStatus.LIMIT_REACHED,
                reason="replanning limit reached",
                replan_attempts=len(history),
                previous_plan_signature=plan_signature(current_plan),
            ),
            replanning_history=tuple(history),
            original_plan_signature=plan_signature(original_plan),
            final_plan_signature=plan_signature(current_plan),
        )

    def _replanning_candidates(
        self,
        planning_decision: CapabilityPlanningDecision | None,
    ) -> tuple[ReplanningCandidate, ...]:
        if planning_decision is None or planning_decision.workflow_selection_result is None:
            return ()
        resolver = getattr(self._capability_planner, "_workflow_for", None)
        if not callable(resolver):
            return ()
        candidates: list[ReplanningCandidate] = []
        for scored in planning_decision.workflow_selection_result.considered_candidates:
            capability = scored.candidate.capability
            try:
                workflow = resolver(capability)
            except (TypeError, ValueError, RuntimeError):
                continue
            candidates.append(
                ReplanningCandidate(
                    plan=workflow.plan,
                    plan_signature=plan_signature(workflow.plan),
                    workflow_reference=workflow.reference,
                    library_id=(
                        capability.source_reference.library.library_id
                        if hasattr(capability.source_reference, "library")
                        else None
                    ),
                    score=scored.final_score,
                )
            )
        return tuple(candidates)

    def _complete(
        self,
        status: CapabilityOrchestrationStatus,
        events: list[CapabilityOrchestrationEvent],
        *,
        planning_decision: CapabilityPlanningDecision | None = None,
        selected_plan: ExecutionPlan | None = None,
        validation_result: PlanValidationResult | None = None,
        execution_result: PlanExecutionResult | None = None,
        goal_verification_result: GoalVerificationResult | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        replanning_decision: ReplanningDecision | None = None,
        replanning_history: tuple[ReplanningHistoryEntry, ...] = (),
        original_plan_signature: str | None = None,
        final_plan_signature: str | None = None,
        goal_driven_result: GoalDrivenExecutionResult | None = None,
        agent_execution_result: AgentExecutionResult | None = None,
        checkpoint: Mapping[str, object] | None = None,
        metrics: Mapping[str, object] | None = None,
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
            goal_verification_result=goal_verification_result,
            error_code=error_code,
            error_message=error_message,
            replanning_decision=replanning_decision,
            replanning_history=replanning_history,
            original_plan_signature=original_plan_signature,
            final_plan_signature=final_plan_signature,
            goal_driven_result=goal_driven_result,
            agent_execution_result=agent_execution_result,
            checkpoint=checkpoint or {},
            metrics=metrics or {},
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
    goal_verification_result: GoalVerificationResult | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    replanning_decision: ReplanningDecision | None = None,
    replanning_history: tuple[ReplanningHistoryEntry, ...] = (),
    original_plan_signature: str | None = None,
    final_plan_signature: str | None = None,
    goal_driven_result: GoalDrivenExecutionResult | None = None,
    agent_execution_result: AgentExecutionResult | None = None,
    checkpoint: Mapping[str, object] | None = None,
    metrics: Mapping[str, object] | None = None,
) -> CapabilityOrchestrationResult:
    return CapabilityOrchestrationResult(
        status=status,
        planning_decision=planning_decision,
        selected_plan=selected_plan,
        validation_result=validation_result,
        execution_result=execution_result,
        goal_verification_result=goal_verification_result,
        error_code=error_code,
        error_message=error_message,
        replanning_attempted=replanning_decision is not None,
        replan_attempts=(
            replanning_decision.replan_attempts
            if replanning_decision is not None
            else 0
        ),
        replanning_status=(
            replanning_decision.status.value
            if replanning_decision is not None
            else None
        ),
        replanning_reason=(
            replanning_decision.reason
            if replanning_decision is not None
            else None
        ),
        original_plan_signature=original_plan_signature,
        final_plan_signature=final_plan_signature,
        replanning_history=tuple(_history_entry_to_metadata(entry) for entry in replanning_history),
        goal_driven_result=goal_driven_result,
        agent_execution_result=agent_execution_result,
        checkpoint=checkpoint or {},
        metrics=metrics or {},
        events=tuple(events),
    )


def _orchestration_status_for_goal_driven(
    status: GoalDrivenExecutionStatus,
) -> CapabilityOrchestrationStatus:
    if status in {GoalDrivenExecutionStatus.COMPLETED, GoalDrivenExecutionStatus.GOAL_SATISFIED}:
        return CapabilityOrchestrationStatus.COMPLETED
    if status is GoalDrivenExecutionStatus.POLICY_DISABLED:
        return CapabilityOrchestrationStatus.INVALID_REQUEST
    if status is GoalDrivenExecutionStatus.VALIDATION_FAILED:
        return CapabilityOrchestrationStatus.PLAN_VALIDATION_FAILED
    if status is GoalDrivenExecutionStatus.CANCELLED:
        return CapabilityOrchestrationStatus.CANCELLED
    if status is GoalDrivenExecutionStatus.INVALID_REQUEST:
        return CapabilityOrchestrationStatus.INVALID_REQUEST
    return CapabilityOrchestrationStatus.EXECUTION_FAILED


def _orchestration_status_for_agent(
    status: AgentExecutionStatus,
) -> CapabilityOrchestrationStatus:
    if status is AgentExecutionStatus.COMPLETED:
        return CapabilityOrchestrationStatus.COMPLETED
    if status is AgentExecutionStatus.INVALID_REQUEST:
        return CapabilityOrchestrationStatus.INVALID_REQUEST
    if status is AgentExecutionStatus.NO_AGENT_CANDIDATES:
        return CapabilityOrchestrationStatus.NO_CAPABILITY_CANDIDATES
    if status is AgentExecutionStatus.AGENT_AMBIGUOUS:
        return CapabilityOrchestrationStatus.CAPABILITY_AMBIGUOUS
    if status is AgentExecutionStatus.CANCELLED:
        return CapabilityOrchestrationStatus.CANCELLED
    return CapabilityOrchestrationStatus.EXECUTION_FAILED


def _agent_metrics(
    *,
    started: int,
    completed: int,
    failed: int,
) -> dict[str, object]:
    return {
        "agent_pipeline_started": started,
        "agent_pipeline_completed": completed,
        "agent_pipeline_failed": failed,
    }


def _agent_checkpoint(
    policy: AgentExecutionPolicy,
    result: AgentExecutionResult | None,
    status: CapabilityOrchestrationStatus,
) -> dict[str, object]:
    output = result.output if result is not None else None
    output_keys = tuple(output.keys()) if isinstance(output, Mapping) else ()
    return {
        "agent_policy_enabled": policy.enabled,
        "agent_policy_allow_agent_execution": policy.allow_agent_execution,
        "agent_policy_preferred_agent_id": policy.preferred_agent_id,
        "agent_policy_require_explicit_agent": policy.require_explicit_agent,
        "agent_policy_fail_if_agent_missing": policy.fail_if_agent_missing,
        "agent_id": result.agent_id if result is not None else None,
        "agent_status": result.status.value if result is not None else status.value,
        "result_has_output": bool(output),
        "result_output_key_count": len(output_keys),
        "result_output_keys": tuple(str(key) for key in output_keys[:16]),
        "result_sanitized_output_fields": (
            int(result.metadata.get("sanitized_output_fields", 0))
            if result is not None
            else 0
        ),
    }


def _effective_replanning_policy(
    policy: CapabilityOrchestrationPolicy,
    plan: ExecutionPlan,
) -> ReplanningPolicy:
    if policy.replanning_policy is not None:
        return policy.replanning_policy
    if isinstance(plan.replanning_policy, ReplanningPolicy):
        return plan.replanning_policy
    return ReplanningPolicy()


def _event_name_for_replanning_status(
    status: ReplanningStatus,
) -> str:
    if status is ReplanningStatus.LIMIT_REACHED:
        return "replanning_limit_reached"
    if status in {ReplanningStatus.NO_ALTERNATIVE_PLAN, ReplanningStatus.FAILURE_NOT_REPLANNABLE}:
        return "replanning_skipped"
    return "replanning_failed"


def _history_entry_to_metadata(
    entry: ReplanningHistoryEntry,
) -> dict[str, object]:
    return {
        "attempt": entry.attempt,
        "status": entry.status.value,
        "reason": entry.reason,
        "previous_plan_signature": entry.previous_plan_signature,
        "replacement_plan_signature": entry.replacement_plan_signature,
        "workflow_plan_id": entry.workflow_plan_id,
        "workflow_version": entry.workflow_version,
        "library_id": entry.library_id,
    }


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
    if isinstance(value, tuple):
        return tuple(_safe_value(item) for item in value)
    raise InvalidCapabilityOrchestrationRequestError("metadata values must be primitive safe values.")


def _identifier_tuple(
    values: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise InvalidCapabilityOrchestrationRequestError(f"{field_name} must be a tuple of strings.")
    if len(values) > MAX_CAPABILITY_ORCHESTRATION_METADATA_ITEMS:
        raise InvalidCapabilityOrchestrationRequestError(f"{field_name} has too many items.")
    normalized: list[str] = []
    for value in values:
        item = validate_agent_id(value)
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _permission_tuple(
    values: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    normalized = _identifier_tuple(values, field_name)
    unknown = tuple(value for value in normalized if value not in _PERMISSION_IDS)
    if unknown:
        raise InvalidCapabilityOrchestrationRequestError(f"{field_name} contains an unknown permission id.")
    return normalized

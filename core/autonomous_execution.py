"""High-level autonomous execution facade for Atlas structured plans."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from core.concurrent_step_executor import (
    ConcurrentStepExecutor,
    ExecutionBatch,
    ExecutionConcurrencyPolicy,
    build_execution_batch,
)
from core.execution_dependency_resolver import ExecutionDependencyResolver
from core.execution_plan_executor import (
    ExecutionControl,
    ExecutionPlanExecutor,
    PlanExecutionResult,
    PlanExecutionStatus,
)
from core.execution_plan_validator import (
    ExecutionPlanValidator,
    PlanValidationResult,
    plan_signature,
)
from core.execution_priority import ExecutionPriorityPolicy, ReadyStepPrioritizer
from core.execution_resources import (
    ExecutionBudget,
    ExecutionBudgetManager,
    ExecutionBudgetUsage,
    ExecutionResourceCatalog,
    ExecutionResourceOptimizer,
    ExecutionResourcePolicy,
    NoCompatibleResourceError,
    ResourceSelectionError,
)
from core.execution_session_persistence import (
    ExecutionRecoveryPolicy,
    ExecutionRecoveryService,
    RecoveryDecision,
    RecoveryDecisionType,
    RecoveryReport,
)
from core.execution_supervisor import (
    ExecutionSession,
    ExecutionState,
    ExecutionSupervisor,
    StepExecutionState,
)
from core.planner import ExecutionPlan, PlanGenerationResult, Planner
from core.structured_plan_replanner import ReplanPolicy
from core.structured_execution import (
    StructuredExecutionCoordinator,
    StructuredExecutionResponse,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AutonomousExecutionOutcome(str, Enum):
    """Closed autonomous execution outcomes."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_CONFIRMATION = "waiting_confirmation"
    INTERRUPTED = "interrupted"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_COMPATIBLE_RESOURCE = "no_compatible_resource"
    INVALID_PLAN = "invalid_plan"
    DRY_RUN = "dry_run"


class AutonomousExecutionError(RuntimeError):
    """Base typed error raised by the autonomous facade."""

    def __init__(
        self,
        code: str,
        summary: str,
        *,
        session_id: str | None = None,
        step_id: str | None = None,
        recoverable: bool = False,
        cause: str | None = None,
    ) -> None:
        super().__init__(summary)
        self.code = code
        self.summary = summary
        self.session_id = session_id
        self.step_id = step_id
        self.recoverable = recoverable
        self.cause = cause


class ExecutionNoProgressError(AutonomousExecutionError):
    pass


class ExecutionLimitReachedError(AutonomousExecutionError):
    pass


class ExecutionTimeoutError(ExecutionLimitReachedError):
    pass


class ExecutionResumeNotAllowedError(AutonomousExecutionError):
    pass


class ExecutionDryRunError(AutonomousExecutionError):
    pass


@dataclass(frozen=True, slots=True)
class AutonomousExecutionOptions:
    """Explicit, immutable options for one autonomous execution request."""

    concurrency_policy: ExecutionConcurrencyPolicy = field(default_factory=ExecutionConcurrencyPolicy)
    priority_policy: ExecutionPriorityPolicy = field(default_factory=ExecutionPriorityPolicy)
    resource_policy: ExecutionResourcePolicy = field(default_factory=ExecutionResourcePolicy)
    execution_budget: ExecutionBudget | None = None
    replan_policy: ReplanPolicy = field(default_factory=lambda: ReplanPolicy(max_replans_per_session=0))
    recovery_policy: ExecutionRecoveryPolicy = field(default_factory=ExecutionRecoveryPolicy)
    persistence_enabled: bool = False
    stop_on_first_failure: bool = True
    allow_automatic_recovery: bool = False
    require_confirmation_for_ambiguous_recovery: bool = True
    max_execution_steps: int = 100
    max_wall_time_seconds: float | None = 300.0
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    max_remote_calls: int | None = None
    max_total_cost: float | None = None
    max_tokens: int | None = None
    dry_run: bool = False
    trace_enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_execution_steps < 1:
            raise ValueError("max_execution_steps must be greater than zero.")
        if self.max_wall_time_seconds is not None and self.max_wall_time_seconds < 0:
            raise ValueError("max_wall_time_seconds cannot be negative.")
        for name in (
            "max_model_calls",
            "max_tool_calls",
            "max_remote_calls",
            "max_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative.")
        if self.max_total_cost is not None and self.max_total_cost < 0:
            raise ValueError("max_total_cost cannot be negative.")
        for name in (
            "persistence_enabled",
            "stop_on_first_failure",
            "allow_automatic_recovery",
            "require_confirmation_for_ambiguous_recovery",
            "dry_run",
            "trace_enabled",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")


@dataclass(frozen=True, slots=True)
class ExecutionTraceEntry:
    """Immutable autonomous trace entry with sanitized details."""

    sequence: int
    timestamp: datetime
    session_id: str | None
    event_type: str
    step_id: str | None = None
    batch_id: str | None = None
    state_before: str | None = None
    state_after: str | None = None
    decision_reference: str | None = None
    error_code: str | None = None
    summary: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    """Bounded immutable trace returned by autonomous execution."""

    entries: tuple[ExecutionTraceEntry, ...] = ()
    max_entries: int = 200

    def append(
        self,
        *,
        event_type: str,
        session_id: str | None = None,
        step_id: str | None = None,
        batch_id: str | None = None,
        state_before: str | None = None,
        state_after: str | None = None,
        decision_reference: str | None = None,
        error_code: str | None = None,
        summary: str = "",
        timestamp: datetime | None = None,
    ) -> ExecutionTrace:
        entry = ExecutionTraceEntry(
            sequence=(self.entries[-1].sequence + 1 if self.entries else 1),
            timestamp=timestamp or _utc_now(),
            session_id=session_id,
            event_type=event_type,
            step_id=step_id,
            batch_id=batch_id,
            state_before=state_before,
            state_after=state_after,
            decision_reference=decision_reference,
            error_code=error_code,
            summary=_sanitize_summary(summary),
        )
        return replace(self, entries=(self.entries + (entry,))[-self.max_entries :])


@dataclass(frozen=True, slots=True)
class ExecutionSimulationResult:
    """Dry-run result that never executes tools or consumes budget."""

    planned_order: tuple[str, ...]
    planned_batches: tuple[tuple[str, ...], ...]
    selected_resources: Mapping[str, str]
    estimated_budget: ExecutionBudgetUsage | None
    confirmation_points: tuple[str, ...]
    risks: tuple[str, ...]
    validation_errors: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_resources",
            MappingProxyType(dict(self.selected_resources)),
        )


@dataclass(frozen=True, slots=True)
class AutonomousExecutionResult:
    """Final structured result returned by the autonomous facade."""

    session_id: str | None
    outcome: AutonomousExecutionOutcome
    final_state: ExecutionState | None
    objective: str
    original_plan: ExecutionPlan | None
    active_plan: ExecutionPlan | None
    completed_step_ids: tuple[str, ...] = ()
    failed_step_ids: tuple[str, ...] = ()
    blocked_step_ids: tuple[str, ...] = ()
    cancelled_step_ids: tuple[str, ...] = ()
    results: Mapping[str, object] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    replan_count: int = 0
    selected_resources: Mapping[str, str] = field(default_factory=dict)
    budget_usage: ExecutionBudgetUsage | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration: float | None = None
    requires_confirmation: bool = False
    requires_manual_review: bool = False
    recovery_decision: RecoveryDecision | None = None
    summary: str = ""
    trace: ExecutionTrace = field(default_factory=ExecutionTrace)
    simulation: ExecutionSimulationResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
        object.__setattr__(
            self,
            "selected_resources",
            MappingProxyType(dict(self.selected_resources)),
        )
        object.__setattr__(self, "completed_step_ids", tuple(self.completed_step_ids))
        object.__setattr__(self, "failed_step_ids", tuple(self.failed_step_ids))
        object.__setattr__(self, "blocked_step_ids", tuple(self.blocked_step_ids))
        object.__setattr__(self, "cancelled_step_ids", tuple(self.cancelled_step_ids))
        object.__setattr__(self, "errors", tuple(self.errors))


class AutonomousExecutionOrchestrator:
    """High-level facade that composes the existing structured execution stack."""

    def __init__(
        self,
        *,
        planner: Planner,
        validator: ExecutionPlanValidator,
        executor: ExecutionPlanExecutor,
        supervisor: ExecutionSupervisor | None = None,
        dependency_resolver: ExecutionDependencyResolver | None = None,
        concurrent_step_executor: ConcurrentStepExecutor | None = None,
        resource_catalog: ExecutionResourceCatalog | None = None,
        recovery_service: ExecutionRecoveryService | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._planner = planner
        self._validator = validator
        self._executor = executor
        self._supervisor = supervisor or ExecutionSupervisor()
        self._dependency_resolver = dependency_resolver or ExecutionDependencyResolver()
        self._concurrent_step_executor = concurrent_step_executor
        self._resource_catalog = resource_catalog or ExecutionResourceCatalog()
        self._recovery_service = recovery_service
        self._clock = clock or _utc_now
        self._last_results: dict[str, AutonomousExecutionResult] = {}
        self._coordinators: dict[str, StructuredExecutionCoordinator] = {}

    def execute_objective(
        self,
        objective: str,
        *,
        planning_context: object | None = None,
        execution_options: AutonomousExecutionOptions | None = None,
    ) -> AutonomousExecutionResult:
        del planning_context
        options = execution_options or AutonomousExecutionOptions()
        if options.dry_run:
            generation = self._planner.generate_execution_plan(objective)
            if generation.plan is None:
                return self._invalid_planning_result(objective, generation)
            return self.execute_plan(
                generation.plan,
                execution_options=options,
                objective=objective,
            )
        coordinator = self._coordinator(self._planner, options)
        trace = ExecutionTrace().append(
            event_type="autonomous_execution_started",
            summary="objective execution started",
        )
        response = coordinator.handle(
            objective,
            control=self._control_for_options(options, self._clock()),
        )
        return self._result_from_response(
            objective,
            response,
            trace=trace,
            coordinator=coordinator,
        )

    def execute_plan(
        self,
        plan: ExecutionPlan,
        *,
        execution_options: AutonomousExecutionOptions | None = None,
        objective: str | None = None,
    ) -> AutonomousExecutionResult:
        options = execution_options or AutonomousExecutionOptions()
        objective_value = objective or plan.goal
        limit_result = self._limit_result_if_needed(plan, objective_value, options)
        if limit_result is not None:
            return limit_result
        if options.dry_run:
            simulation = self._simulate_plan(plan, options)
            outcome = (
                AutonomousExecutionOutcome.INVALID_PLAN
                if simulation.validation_errors
                else AutonomousExecutionOutcome.DRY_RUN
            )
            trace = ExecutionTrace().append(
                event_type="dry_run_started",
                summary="dry-run started",
            ).append(
                event_type="dry_run_completed",
                summary="dry-run completed",
            )
            return AutonomousExecutionResult(
                session_id=None,
                outcome=outcome,
                final_state=None,
                objective=objective_value,
                original_plan=plan,
                active_plan=plan,
                blocked_step_ids=(),
                errors=simulation.validation_errors,
                requires_confirmation=bool(simulation.confirmation_points),
                summary=_summary_for_simulation(simulation),
                trace=trace,
                simulation=simulation,
                budget_usage=simulation.estimated_budget,
                selected_resources=simulation.selected_resources,
            )
        validation = self._validator.validate(plan)
        if not validation.is_valid:
            return self._invalid_plan_result(objective_value, plan, validation)
        coordinator = self._coordinator(_FixedPlanPlanner(plan), options)
        trace = ExecutionTrace().append(
            event_type="autonomous_execution_started",
            summary="plan execution started",
        )
        response = coordinator.handle(
            objective_value,
            control=self._control_for_options(options, self._clock()),
        )
        return self._result_from_response(
            objective_value,
            response,
            trace=trace,
            coordinator=coordinator,
        )

    def resume_execution(
        self,
        session_id: str,
        *,
        confirmation: bool | None = None,
        recovery_authorization: bool | None = None,
        execution_options: AutonomousExecutionOptions | None = None,
    ) -> AutonomousExecutionResult:
        options = execution_options or AutonomousExecutionOptions()
        session = self._supervisor.get_session(session_id)
        if session.state in {
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        }:
            raise ExecutionResumeNotAllowedError(
                "EXECUTION_RESUME_NOT_ALLOWED",
                "Terminal sessions cannot be resumed.",
                session_id=session_id,
            )
        coordinator = self._coordinators.get(session_id) or self._coordinator(
            _FixedPlanPlanner(session.active_plan),
            options,
        )
        trace = ExecutionTrace().append(
            event_type="autonomous_execution_resumed",
            session_id=session_id,
            summary="resume requested",
        )
        decision = (
            self._recovery_service.decision_for(session_id)
            if self._recovery_service is not None
            else None
        )
        if decision is not None and decision.decision is not RecoveryDecisionType.RESUME_AUTOMATICALLY:
            if not recovery_authorization:
                return self._manual_review_result(session, decision, trace)
        if session.state is ExecutionState.WAITING_CONFIRMATION:
            if confirmation is not True:
                return self._waiting_confirmation_result(session, trace)
            response = coordinator.confirm_pending()
        else:
            response = coordinator.resume_recovered_session(session_id)
            if response.status == "recovery_not_configured":
                response = coordinator.handle(
                    session.active_plan.goal,
                    confirmation_granted=confirmation is True,
                    control=self._control_for_options(options, self._clock()),
                )
        return self._result_from_response(
            session.active_plan.goal,
            response,
            trace=trace,
            recovery_decision=decision,
            coordinator=coordinator,
        )

    def recover_executions(
        self,
        *,
        execution_options: AutonomousExecutionOptions | None = None,
    ) -> RecoveryReport | None:
        options = execution_options or AutonomousExecutionOptions()
        if self._recovery_service is None:
            return None
        report = self._recovery_service.recover()
        if options.allow_automatic_recovery:
            for session_id, decision in report.decisions.items():
                if decision.decision is RecoveryDecisionType.RESUME_AUTOMATICALLY:
                    self.resume_execution(
                        session_id,
                        recovery_authorization=True,
                        execution_options=options,
                    )
        return report

    def cancel_execution(self, session_id: str) -> AutonomousExecutionResult:
        session = self._supervisor.get_session(session_id)
        trace = ExecutionTrace().append(
            event_type="autonomous_execution_cancelled",
            session_id=session_id,
            summary="cancel requested",
        )
        if session.state is ExecutionState.COMPLETED:
            return self._result_from_session(
                session,
                outcome=AutonomousExecutionOutcome.COMPLETED,
                trace=trace,
            )
        if session.state is not ExecutionState.CANCELLED:
            session = self._supervisor.mark_cancelled(
                session_id,
                error="autonomous cancellation requested",
                current_step=session.current_step,
            )
        return self._result_from_session(
            session,
            outcome=AutonomousExecutionOutcome.CANCELLED,
            trace=trace,
        )

    def get_execution_status(self, session_id: str) -> ExecutionSession:
        return self._supervisor.get_session(session_id)

    def get_execution_result(self, session_id: str) -> AutonomousExecutionResult:
        session = self._supervisor.get_session(session_id)
        if session.state not in {
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        }:
            raise ExecutionResumeNotAllowedError(
                "EXECUTION_RESULT_NOT_READY",
                "Execution result is only available for terminal sessions.",
                session_id=session_id,
                recoverable=True,
            )
        if session_id in self._last_results:
            stored = self._last_results[session_id]
            if stored.final_state is session.state:
                return stored
        return self._result_from_session(
            session,
            outcome=_outcome_from_session(session, None),
            trace=ExecutionTrace(),
        )

    def _coordinator(
        self,
        planner: Planner,
        options: AutonomousExecutionOptions,
    ) -> StructuredExecutionCoordinator:
        budget = self._budget_for_options(options)
        return StructuredExecutionCoordinator(
            planner=planner,
            validator=self._validator,
            executor=self._executor,
            execution_supervisor=self._supervisor,
            execution_replanner=None,
            replan_policy=options.replan_policy,
            dependency_resolver=self._dependency_resolver,
            concurrency_policy=options.concurrency_policy,
            concurrent_step_executor=self._concurrent_step_executor,
            recovery_service=self._recovery_service,
            priority_policy=options.priority_policy,
            priority_clock=self._clock,
            resource_policy=options.resource_policy,
            resource_catalog=self._resource_catalog,
            budget_manager=ExecutionBudgetManager(budget) if budget is not None else None,
        )

    def _budget_for_options(
        self,
        options: AutonomousExecutionOptions,
    ) -> ExecutionBudget | None:
        explicit = options.execution_budget
        if explicit is None and not any(
            value is not None
            for value in (
                options.max_model_calls,
                options.max_tool_calls,
                options.max_remote_calls,
                options.max_total_cost,
                options.max_tokens,
            )
        ):
            return None
        return ExecutionBudget(
            max_total_cost=options.max_total_cost
            if options.max_total_cost is not None
            else getattr(explicit, "max_total_cost", None),
            max_tokens=options.max_tokens
            if options.max_tokens is not None
            else getattr(explicit, "max_tokens", None),
            max_remote_calls=options.max_remote_calls
            if options.max_remote_calls is not None
            else getattr(explicit, "max_remote_calls", None),
            max_model_calls=options.max_model_calls
            if options.max_model_calls is not None
            else getattr(explicit, "max_model_calls", None),
            max_tool_calls=options.max_tool_calls
            if options.max_tool_calls is not None
            else getattr(explicit, "max_tool_calls", None),
            max_replans=getattr(explicit, "max_replans", None),
            reserved_cost=getattr(explicit, "reserved_cost", 0.0),
            reserved_tokens=getattr(explicit, "reserved_tokens", 0),
            currency_code=getattr(explicit, "currency_code", "UNIT"),
            hard_limit=getattr(explicit, "hard_limit", True),
        )

    def _limit_result_if_needed(
        self,
        plan: ExecutionPlan,
        objective: str,
        options: AutonomousExecutionOptions,
    ) -> AutonomousExecutionResult | None:
        if len(plan.ordered_steps) <= options.max_execution_steps:
            return None
        trace = ExecutionTrace().append(
            event_type="autonomous_limit_reached",
            error_code="MAX_EXECUTION_STEPS_REACHED",
            summary="max_execution_steps reached before starting work",
        )
        return AutonomousExecutionResult(
            session_id=None,
            outcome=AutonomousExecutionOutcome.FAILED,
            final_state=None,
            objective=objective,
            original_plan=plan,
            active_plan=plan,
            errors=("MAX_EXECUTION_STEPS_REACHED",),
            summary="Limite max_execution_steps alcanzado; no se inicio trabajo.",
            trace=trace,
        )

    def _simulate_plan(
        self,
        plan: ExecutionPlan,
        options: AutonomousExecutionOptions,
    ) -> ExecutionSimulationResult:
        validation = self._validator.validate(plan)
        if not validation.is_valid:
            return ExecutionSimulationResult(
                planned_order=(),
                planned_batches=(),
                selected_resources={},
                estimated_budget=None,
                confirmation_points=tuple(step.id for step in plan.ordered_steps)
                if validation.requires_confirmation
                else (),
                risks=tuple(plan.detected_risks),
                validation_errors=tuple(validation.errors),
            )
        completed: tuple[str, ...] = ()
        planned_order: list[str] = []
        planned_batches: list[tuple[str, ...]] = []
        selected_resources: dict[str, str] = {}
        budget = ExecutionBudgetManager(self._budget_for_options(options))
        optimizer = ExecutionResourceOptimizer(options.resource_policy)
        iterations = 0
        while len(completed) < len(plan.ordered_steps):
            iterations += 1
            if iterations > options.max_execution_steps:
                break
            resolution = self._dependency_resolver.resolve(
                plan,
                completed_step_ids=completed,
            )
            if not resolution.ready_steps:
                break
            ready = ReadyStepPrioritizer(options.priority_policy).prioritize(
                resolution.ready_steps,
                plan=plan,
                completed_step_ids=completed,
                ready_since_by_step_id={},
                now=self._clock(),
            )
            ready_steps = tuple(
                step
                for step_id in ready.ordered_step_ids
                for step in resolution.ready_steps
                if step.id == step_id
            )
            batch = build_execution_batch(
                ready_steps,
                options.concurrency_policy,
                batch_id=f"dry-run.batch.{iterations:06d}",
            )
            planned_batches.append(tuple(batch.step_ids))
            for step in ready_steps:
                if step.id not in batch.step_ids:
                    continue
                if options.resource_policy.enabled:
                    decision = optimizer.select(
                        step_id=step.id,
                        requirements=step.resource_requirements,
                        catalog=self._resource_catalog,
                        budget_usage=budget.snapshot(),
                    )
                    if decision.selected_resource_id is not None:
                        selected_resources[step.id] = decision.selected_resource_id
                        budget.reserve(
                            step_id=step.id,
                            resource_id=decision.selected_resource_id,
                            estimated_cost=decision.estimated_cost,
                            estimated_tokens=decision.estimated_tokens,
                        )
                planned_order.append(step.id)
            for resource_id in selected_resources.values():
                optimizer.release(resource_id)
            completed = tuple(dict.fromkeys(completed + tuple(batch.step_ids)))
        return ExecutionSimulationResult(
            planned_order=tuple(planned_order),
            planned_batches=tuple(planned_batches),
            selected_resources=selected_resources,
            estimated_budget=budget.snapshot(),
            confirmation_points=tuple(step.id for step in plan.ordered_steps)
            if validation.requires_confirmation or plan.requires_confirmation
            else (),
            risks=tuple(plan.detected_risks),
            validation_errors=(),
        )

    def _control_for_options(
        self,
        options: AutonomousExecutionOptions,
        started_at: datetime,
    ) -> ExecutionControl:
        def should_stop() -> bool:
            if options.max_wall_time_seconds is None:
                return False
            elapsed = (self._clock() - started_at).total_seconds()
            return elapsed > options.max_wall_time_seconds

        return ExecutionControl(
            should_stop=should_stop,
            interruption_reason="Autonomous execution limit reached.",
        )

    def _result_from_response(
        self,
        objective: str,
        response: StructuredExecutionResponse,
        *,
        trace: ExecutionTrace,
        recovery_decision: RecoveryDecision | None = None,
        coordinator: StructuredExecutionCoordinator | None = None,
    ) -> AutonomousExecutionResult:
        session = _latest_session(self._supervisor)
        execution = response.execution_result
        trace = trace.append(
            event_type=_event_for_response(response),
            session_id=session.session_id if session is not None else None,
            state_after=session.state.value if session is not None else response.status,
            error_code=response.error_code,
            summary=response.message,
        ).append(
            event_type="final_result_built",
            session_id=session.session_id if session is not None else None,
            summary="final autonomous result built",
        )
        result = AutonomousExecutionResult(
            session_id=session.session_id if session is not None else None,
            outcome=_outcome_from_response(response, session),
            final_state=session.state if session is not None else None,
            objective=objective,
            original_plan=session.original_plan if session is not None else response.plan,
            active_plan=session.active_plan if session is not None else response.plan,
            completed_step_ids=tuple(execution.completed_steps)
            if execution is not None
            else _step_ids_in_state(session, StepExecutionState.COMPLETED),
            failed_step_ids=tuple(execution.failed_steps)
            if execution is not None
            else _step_ids_in_state(session, StepExecutionState.FAILED),
            blocked_step_ids=tuple(execution.blocked_steps)
            if execution is not None
            else _step_ids_in_state(session, StepExecutionState.BLOCKED),
            cancelled_step_ids=tuple(
                execution.metadata.get("cancelled_step_ids", ())
            )
            if execution is not None
            else _step_ids_in_state(session, StepExecutionState.CANCELLED),
            results=session.results if session is not None else {},
            errors=tuple(
                item
                for item in (response.error, response.error_code)
                if item is not None
            ),
            replan_count=session.replan_count if session is not None else 0,
            selected_resources=session.selected_resources_by_step
            if session is not None
            else {},
            budget_usage=session.budget_usage if session is not None else None,
            started_at=session.started_at if session is not None else None,
            finished_at=session.finished_at if session is not None else None,
            duration=_duration_seconds(session),
            requires_confirmation=response.requires_confirmation,
            requires_manual_review=(
                recovery_decision is not None
                and recovery_decision.decision is RecoveryDecisionType.REQUIRE_MANUAL_REVIEW
            ),
            recovery_decision=recovery_decision,
            summary=_summary_for_response(response, session),
            trace=trace,
        )
        if result.session_id is not None:
            self._last_results[result.session_id] = result
            if coordinator is not None and result.requires_confirmation:
                self._coordinators[result.session_id] = coordinator
            elif result.session_id in self._coordinators and not result.requires_confirmation:
                self._coordinators.pop(result.session_id, None)
        return result

    def _invalid_plan_result(
        self,
        objective: str,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
    ) -> AutonomousExecutionResult:
        return AutonomousExecutionResult(
            session_id=None,
            outcome=AutonomousExecutionOutcome.INVALID_PLAN,
            final_state=None,
            objective=objective,
            original_plan=plan,
            active_plan=plan,
            errors=tuple(validation.errors),
            summary="Plan invalido; no se ejecuto ningun paso.",
            trace=ExecutionTrace().append(
                event_type="autonomous_execution_failed",
                error_code="INVALID_PLAN",
                summary="invalid plan",
            ),
        )

    def _invalid_planning_result(
        self,
        objective: str,
        generation: PlanGenerationResult,
    ) -> AutonomousExecutionResult:
        return AutonomousExecutionResult(
            session_id=None,
            outcome=AutonomousExecutionOutcome.INVALID_PLAN,
            final_state=None,
            objective=objective,
            original_plan=None,
            active_plan=None,
            errors=tuple(generation.errors),
            summary="No se pudo construir un plan valido.",
        )

    def _waiting_confirmation_result(
        self,
        session: ExecutionSession,
        trace: ExecutionTrace,
    ) -> AutonomousExecutionResult:
        return self._result_from_session(
            session,
            outcome=AutonomousExecutionOutcome.WAITING_CONFIRMATION,
            trace=trace,
            requires_confirmation=True,
        )

    def _manual_review_result(
        self,
        session: ExecutionSession,
        decision: RecoveryDecision,
        trace: ExecutionTrace,
    ) -> AutonomousExecutionResult:
        return self._result_from_session(
            session,
            outcome=AutonomousExecutionOutcome.MANUAL_REVIEW_REQUIRED,
            trace=trace,
            recovery_decision=decision,
            requires_manual_review=True,
        )

    def _result_from_session(
        self,
        session: ExecutionSession,
        *,
        outcome: AutonomousExecutionOutcome,
        trace: ExecutionTrace,
        recovery_decision: RecoveryDecision | None = None,
        requires_confirmation: bool = False,
        requires_manual_review: bool = False,
    ) -> AutonomousExecutionResult:
        return AutonomousExecutionResult(
            session_id=session.session_id,
            outcome=outcome,
            final_state=session.state,
            objective=getattr(session.active_plan, "goal", ""),
            original_plan=session.original_plan,
            active_plan=session.active_plan,
            completed_step_ids=_step_ids_in_state(session, StepExecutionState.COMPLETED),
            failed_step_ids=_step_ids_in_state(session, StepExecutionState.FAILED),
            blocked_step_ids=_step_ids_in_state(session, StepExecutionState.BLOCKED),
            cancelled_step_ids=_step_ids_in_state(session, StepExecutionState.CANCELLED),
            results=session.results,
            errors=tuple([session.last_error] if session.last_error else ()),
            replan_count=session.replan_count,
            selected_resources=session.selected_resources_by_step,
            budget_usage=session.budget_usage,
            started_at=session.started_at,
            finished_at=session.finished_at,
            duration=_duration_seconds(session),
            requires_confirmation=requires_confirmation,
            requires_manual_review=requires_manual_review,
            recovery_decision=recovery_decision,
            summary=_summary_for_session(session, outcome),
            trace=trace,
        )


class _FixedPlanPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self._plan = plan

    def generate_execution_plan(self, _objective: str, **_kwargs: object) -> PlanGenerationResult:
        return PlanGenerationResult(
            success=True,
            plan=self._plan,
            generation_attempted=True,
        )


def _latest_session(supervisor: ExecutionSupervisor) -> ExecutionSession | None:
    sessions = supervisor.list_sessions(limit=1)
    return sessions[0] if sessions else None


def _duration_seconds(session: ExecutionSession | None) -> float | None:
    if session is None or session.finished_at is None:
        return None
    return max(0.0, (session.finished_at - session.started_at).total_seconds())


def _step_ids_in_state(
    session: ExecutionSession | None,
    state: StepExecutionState,
) -> tuple[str, ...]:
    if session is None:
        return ()
    return tuple(
        step_id
        for step_id, snapshot in session.step_states.items()
        if snapshot.state is state
    )


def _outcome_from_response(
    response: StructuredExecutionResponse,
    session: ExecutionSession | None,
) -> AutonomousExecutionOutcome:
    if response.error_code == "INVALID_PLAN":
        return AutonomousExecutionOutcome.INVALID_PLAN
    if response.error_code in {
        "EXECUTION_BUDGET_EXCEEDED",
        "EXECUTION_TOKEN_BUDGET_EXCEEDED",
    }:
        return AutonomousExecutionOutcome.BUDGET_EXHAUSTED
    if response.error_code == "NO_COMPATIBLE_RESOURCE":
        return AutonomousExecutionOutcome.NO_COMPATIBLE_RESOURCE
    if response.requires_confirmation:
        return AutonomousExecutionOutcome.WAITING_CONFIRMATION
    if session is not None:
        return _outcome_from_session(session, response.execution_result)
    if response.status == "completed":
        return AutonomousExecutionOutcome.COMPLETED
    return AutonomousExecutionOutcome.FAILED


def _outcome_from_session(
    session: ExecutionSession,
    execution: PlanExecutionResult | None,
) -> AutonomousExecutionOutcome:
    if session.state is ExecutionState.COMPLETED:
        return AutonomousExecutionOutcome.COMPLETED
    if session.state is ExecutionState.CANCELLED:
        return AutonomousExecutionOutcome.CANCELLED
    if session.state is ExecutionState.WAITING_CONFIRMATION:
        return AutonomousExecutionOutcome.WAITING_CONFIRMATION
    if session.state is ExecutionState.INTERRUPTED:
        return AutonomousExecutionOutcome.INTERRUPTED
    if execution is not None and execution.error_code in {
        "EXECUTION_BUDGET_EXCEEDED",
        "EXECUTION_TOKEN_BUDGET_EXCEEDED",
    }:
        return AutonomousExecutionOutcome.BUDGET_EXHAUSTED
    if execution is not None and execution.error_code == "NO_COMPATIBLE_RESOURCE":
        return AutonomousExecutionOutcome.NO_COMPATIBLE_RESOURCE
    return AutonomousExecutionOutcome.FAILED


def _event_for_response(response: StructuredExecutionResponse) -> str:
    if response.requires_confirmation:
        return "autonomous_execution_paused"
    if response.status == "completed":
        return "autonomous_execution_completed"
    if response.status == "cancelled":
        return "autonomous_execution_cancelled"
    return "autonomous_execution_failed"


def _summary_for_response(
    response: StructuredExecutionResponse,
    session: ExecutionSession | None,
) -> str:
    if response.status == "completed":
        completed = len(_step_ids_in_state(session, StepExecutionState.COMPLETED))
        return f"Ejecucion completada. Pasos completados: {completed}."
    if response.requires_confirmation:
        return "Ejecucion pausada por confirmacion requerida."
    if response.error:
        return f"Ejecucion terminada con error conocido: {response.error}."
    if response.error_code:
        return f"Ejecucion terminada con codigo: {response.error_code}."
    return response.message or "Ejecucion terminada sin detalle adicional."


def _summary_for_session(
    session: ExecutionSession,
    outcome: AutonomousExecutionOutcome,
) -> str:
    completed = len(_step_ids_in_state(session, StepExecutionState.COMPLETED))
    failed = len(_step_ids_in_state(session, StepExecutionState.FAILED))
    return (
        f"Estado final {outcome.value}. Completados: {completed}. "
        f"Fallidos: {failed}. Replans: {session.replan_count}."
    )


def _summary_for_simulation(simulation: ExecutionSimulationResult) -> str:
    if simulation.validation_errors:
        return "Dry-run invalido; el plan no paso validacion."
    return (
        "Dry-run completado sin efectos externos. "
        f"Pasos previstos: {len(simulation.planned_order)}."
    )


def _sanitize_summary(value: str) -> str:
    redacted = value.replace("sk-", "[redacted]")
    if len(redacted) > 300:
        return redacted[:297] + "..."
    return redacted

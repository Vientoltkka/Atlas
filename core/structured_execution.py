"""Structured execution pipeline coordination for Atlas."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable

from core.concurrent_step_executor import (
    ConcurrentStepExecutor,
    ExecutionBatchResult,
    ExecutionConcurrencyPolicy,
    build_execution_batch,
)
from core.execution_context import ExecutionContext
from core.execution_dependency_resolver import ExecutionDependencyResolver
from core.execution_plan_executor import (
    ExecutionControl,
    ExecutionErrorCode,
    ExecutionProgress,
    ExecutionPlanExecutor,
    PartialExecutionState,
    PlanExecutionResult,
    PlanExecutionStatus,
    ResumableExecutionState,
    StepExecutionResult,
    StepExecutionStatus,
)
from core.execution_supervisor import (
    ExecutionOverview,
    ExecutionSession,
    ExecutionState,
    ExecutionSupervisor,
    StepExecutionState,
)
from core.execution_plan_validator import (
    ExecutionPlanValidator,
    PlanValidationResult,
    plan_signature,
)
from core.execution_priority import ExecutionPriorityPolicy, ReadyStepPrioritizer
from core.execution_session_persistence import (
    ExecutionRecoveryService,
    RecoveryDecisionType,
    RecoveryReport,
)
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult, Planner
from core.resumable_execution_store import (
    ResumableExecutionStore,
    ResumableExecutionStoreError,
)
from core.structured_plan_replanner import (
    ExecutionReplanner as StructuredExecutionReplanner,
    ReplanPolicy,
    ReplanRequest,
    ReplanResult,
    ReplanResultStatus,
    replan_record,
)
from core.structured_reference_path import navigate_structured_path


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class StructuredExecutionResponse:
    """Structured response returned to the orchestrator."""

    handled: bool
    status: str
    message: str
    plan: ExecutionPlan | None = None
    validation_result: PlanValidationResult | None = None
    execution_result: PlanExecutionResult | None = None
    requires_confirmation: bool = False
    confirmation_token: str | None = None
    error_code: str | None = None
    error: str | None = None
    resumable_state: ResumableExecutionState | None = None
    partial_state: PartialExecutionState | None = None


@dataclass(frozen=True, slots=True)
class PendingStructuredExecution:
    """Public read model for one pending structured execution."""

    objective: str
    plan: ExecutionPlan
    validation_result: PlanValidationResult
    confirmation_token: str
    status: str
    summary: str
    risks: list[str]
    required_tools: list[str]


@dataclass(frozen=True, slots=True)
class _PendingPlan:
    execution: PendingStructuredExecution
    session_id: str | None = None


class StructuredExecutionCoordinator:
    """Coordinate Planner, Validator and Executor without duplicating them."""

    def __init__(
        self,
        planner: Planner,
        validator: ExecutionPlanValidator,
        executor: ExecutionPlanExecutor,
        resumable_store: ResumableExecutionStore | None = None,
        execution_supervisor: ExecutionSupervisor | None = None,
        execution_replanner: StructuredExecutionReplanner | None = None,
        replan_policy: ReplanPolicy | None = None,
        dependency_resolver: ExecutionDependencyResolver | None = None,
        concurrency_policy: ExecutionConcurrencyPolicy | None = None,
        concurrent_step_executor: ConcurrentStepExecutor | None = None,
        recovery_service: ExecutionRecoveryService | None = None,
        priority_policy: ExecutionPriorityPolicy | None = None,
        ready_step_prioritizer: ReadyStepPrioritizer | None = None,
        priority_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._planner = planner
        self._validator = validator
        self._executor = executor
        self._execution_supervisor = execution_supervisor or ExecutionSupervisor()
        self._execution_replanner = execution_replanner
        self._replan_policy = replan_policy or ReplanPolicy(max_replans_per_session=0)
        self._dependency_resolver = dependency_resolver or ExecutionDependencyResolver()
        self._concurrency_policy = concurrency_policy or ExecutionConcurrencyPolicy()
        self._concurrent_step_executor = concurrent_step_executor
        self._recovery_service = recovery_service
        self._priority_policy = priority_policy or ExecutionPriorityPolicy()
        self._ready_step_prioritizer = (
            ready_step_prioritizer
            or ReadyStepPrioritizer(self._priority_policy)
        )
        self._priority_clock = priority_clock or (lambda: datetime.now(timezone.utc))
        self._pending_plans: dict[str, _PendingPlan] = {}
        self._active_confirmation_token: str | None = None
        self._resumable_execution: ResumableExecutionState | None = None
        self._resumable_store = resumable_store

    def handle(
        self,
        objective: str,
        *,
        confirmation_granted: bool = False,
        control: ExecutionControl | None = None,
        on_planning_progress: Callable[[Any], None] | None = None,
        planning_control: Any | None = None,
        on_execution_progress: Callable[[ExecutionProgress], None] | None = None,
        execute_after_planning: bool = True,
    ) -> StructuredExecutionResponse:
        """Generate, validate and optionally execute one structured plan."""
        generation = self._planner.generate_execution_plan(
            objective,
            on_planning_progress=on_planning_progress,
            planning_control=planning_control,
        )

        if not self._should_handle_generation(generation):
            return StructuredExecutionResponse(
                handled=False,
                status="fallback",
                message="Structured execution is not applicable.",
                plan=generation.plan,
                error_code=generation.error_code,
                error="; ".join(generation.errors) if generation.errors else None,
            )

        if generation.plan is None:
            if generation.error_code == "STRUCTURED_PLAN_PROVIDER_CANCELLED":
                return StructuredExecutionResponse(
                    handled=True,
                    status="planning_cancelled",
                    message="Planificación cancelada.",
                    error_code=generation.error_code,
                    error="; ".join(generation.errors) if generation.errors else None,
                )
            return StructuredExecutionResponse(
                handled=True,
                status="planning_failed",
                message=self._planning_failed_message(generation),
                error_code=generation.error_code,
                error="; ".join(generation.errors) if generation.errors else None,
            )

        validation = self._validator.validate(generation.plan)
        if not validation.is_valid:
            return StructuredExecutionResponse(
                handled=True,
                status="validation_failed",
                message=self._validation_failed_message(validation),
                plan=generation.plan,
                validation_result=validation,
                error_code="INVALID_PLAN",
                error="; ".join(validation.errors),
            )

        if validation.requires_confirmation and not confirmation_granted:
            session = self._start_waiting_confirmation_session(generation.plan)
            token = self._store_pending_plan(
                objective,
                generation.plan,
                validation,
                session.session_id,
            )
            return StructuredExecutionResponse(
                handled=True,
                status="confirmation_required",
                message=self._confirmation_message(generation.plan, validation, token),
                plan=generation.plan,
                validation_result=validation,
                requires_confirmation=True,
                confirmation_token=token,
                error_code="CONFIRMATION_REQUIRED",
            )

        if not execute_after_planning:
            return StructuredExecutionResponse(
                handled=True,
                status="planned",
                message=self._planned_message(generation.plan, validation),
                plan=generation.plan,
                validation_result=validation,
                requires_confirmation=False,
            )

        execution = self._execute_plan_under_supervision(
            generation.plan,
            validation,
            confirmation_granted=confirmation_granted,
            control=control,
            on_progress=on_execution_progress,
        )
        response = self._execution_response(
            generation.plan,
            validation,
            execution,
        )
        persistence_error = self._sync_resumable_state(
            response,
            objective=objective,
            confirmation_granted=confirmation_granted,
        )
        if persistence_error is not None:
            return replace(response, error_code=persistence_error)
        return response

    def confirm(
        self,
        confirmation_token: str,
        *,
        objective: str | None = None,
        control: ExecutionControl | None = None,
        on_execution_progress: Callable[[ExecutionProgress], None] | None = None,
    ) -> StructuredExecutionResponse:
        """Execute exactly the validated plan associated with a confirmation token."""
        pending_record = self._pending_plans.get(confirmation_token)
        pending = pending_record.execution if pending_record is not None else None
        if pending is None:
            return StructuredExecutionResponse(
                handled=True,
                status="confirmation_not_found",
                message="No hay ninguna ejecucion pendiente que confirmar.",
                requires_confirmation=False,
                confirmation_token=confirmation_token,
                error_code="NO_PENDING_EXECUTION",
                error="confirmation token is not pending",
            )

        if objective is not None and objective != pending.objective:
            self._cancel_pending_session(
                pending_record,
                "objective changed after validation",
            )
            self._discard_pending(confirmation_token)
            return StructuredExecutionResponse(
                handled=True,
                status="validation_mismatch",
                message="La confirmacion no coincide con el objetivo validado.",
                plan=pending.plan,
                validation_result=pending.validation_result,
                confirmation_token=confirmation_token,
                error_code="VALIDATION_MISMATCH",
                error="objective changed after validation",
            )

        current_signature = plan_signature(pending.plan)
        if current_signature != pending.validation_result.plan_signature:
            self._cancel_pending_session(
                pending_record,
                "plan signature changed after validation",
            )
            self._discard_pending(confirmation_token)
            return StructuredExecutionResponse(
                handled=True,
                status="validation_mismatch",
                message="El plan pendiente cambio despues de ser validado.",
                plan=pending.plan,
                validation_result=pending.validation_result,
                confirmation_token=confirmation_token,
                error_code="VALIDATION_MISMATCH",
                error="plan signature changed after validation",
            )

        self._discard_pending(confirmation_token)
        execution = self._execute_plan_under_supervision(
            pending.plan,
            pending.validation_result,
            confirmation_granted=True,
            control=control,
            on_progress=on_execution_progress,
            session_id=pending_record.session_id,
        )

        response = self._execution_response(
            pending.plan,
            pending.validation_result,
            execution,
            confirmation_token=confirmation_token,
        )
        persistence_error = self._sync_resumable_state(
            response,
            objective=pending.objective,
            confirmation_granted=True,
        )
        if persistence_error is not None:
            return replace(response, error_code=persistence_error)
        return response

    def pending_plan(
        self,
        confirmation_token: str,
    ) -> ExecutionPlan | None:
        """Return a pending plan by token without exposing internal mutation."""
        pending = self._pending_plans.get(confirmation_token)
        return pending.execution.plan if pending is not None else None

    def get_execution_overview(self) -> ExecutionOverview:
        """Return the global supervised execution overview."""
        return self._execution_supervisor.get_overview()

    def list_execution_sessions(
        self,
        *,
        state: ExecutionState | None = None,
        limit: int | None = None,
        newest_first: bool = True,
    ) -> tuple[ExecutionSession, ...]:
        """Return supervised execution session snapshots."""
        return self._execution_supervisor.list_sessions(
            state=state,
            limit=limit,
            newest_first=newest_first,
        )

    def has_pending_execution(self) -> bool:
        """Return whether this coordinator has one active pending plan."""
        return self._active_confirmation_token in self._pending_plans

    def pending_execution(self) -> PendingStructuredExecution | None:
        """Return the active pending execution summary, if any."""
        if self._active_confirmation_token is None:
            return None

        pending = self._pending_plans.get(self._active_confirmation_token)
        return pending.execution if pending is not None else None

    def has_resumable_execution(self) -> bool:
        """Return whether there is one valid interrupted execution to resume."""
        return self.resumable_execution() is not None

    def resumable_execution(self) -> ResumableExecutionState | None:
        """Return the current in-memory resumable execution state."""
        if self._resumable_execution is None:
            self.load_persisted_resumable_execution()
        return self._resumable_execution

    def load_persisted_resumable_execution(self) -> StructuredExecutionResponse:
        """Load a persisted resumable execution state without executing it."""
        if self._resumable_store is None:
            return StructuredExecutionResponse(
                handled=True,
                status="no_resumable_execution",
                message="No hay ninguna ejecución pendiente que pueda reanudarse.",
                error_code="EXECUTION_STATE_NOT_FOUND",
            )

        try:
            state = self._resumable_store.load()
        except ResumableExecutionStoreError as error:
            return StructuredExecutionResponse(
                handled=True,
                status="resumable_execution_invalid",
                message="No se puede reanudar la ejecución.",
                error_code=error.error_code,
                error=error.message,
            )

        if state is None:
            return StructuredExecutionResponse(
                handled=True,
                status="no_resumable_execution",
                message="No hay ninguna ejecución pendiente que pueda reanudarse.",
                error_code="EXECUTION_STATE_NOT_FOUND",
            )

        self._resumable_execution = state
        return StructuredExecutionResponse(
            handled=True,
            status="resumable_execution_loaded",
            message="Atlas encontró una ejecución interrumpida que puede reanudarse.",
            plan=state.original_plan,
            validation_result=state.validation_result,
            requires_confirmation=state.validation_result.requires_confirmation,
            resumable_state=state,
        )

    def discard_resumable_execution(self) -> StructuredExecutionResponse:
        """Discard any in-memory and persisted resumable execution state."""
        self._resumable_execution = None
        delete_error = self._delete_persisted_resumable_state()
        return StructuredExecutionResponse(
            handled=True,
            status="resumable_execution_discarded",
            message="La ejecución pendiente fue descartada.",
            error_code=delete_error or "EXECUTION_STATE_DISCARDED",
        )

    def recover_execution_sessions(self) -> RecoveryReport | None:
        """Restore persisted supervised sessions without executing them."""
        if self._recovery_service is None:
            return None
        return self._recovery_service.recover()

    def resume_recovered_session(
        self,
        session_id: str,
        *,
        control: ExecutionControl | None = None,
        on_execution_progress: Callable[[ExecutionProgress], None] | None = None,
    ) -> StructuredExecutionResponse:
        """Resume one restored session only when recovery policy allows it."""
        if self._recovery_service is None:
            return StructuredExecutionResponse(
                handled=True,
                status="recovery_not_configured",
                message="La recuperacion de sesiones no esta configurada.",
                error_code="RECOVERY_NOT_CONFIGURED",
            )
        decision = self._recovery_service.decision_for(session_id)
        if decision is None:
            return StructuredExecutionResponse(
                handled=True,
                status="recovered_session_not_found",
                message="La sesion recuperada no existe o no fue restaurada.",
                error_code="RECOVERED_SESSION_NOT_FOUND",
            )
        if decision.decision is not RecoveryDecisionType.RESUME_AUTOMATICALLY:
            return StructuredExecutionResponse(
                handled=True,
                status=decision.decision.value,
                message=decision.reason,
                error_code="RECOVERY_NOT_SAFE",
                error=decision.reason,
            )

        session = self._execution_supervisor.get_session(session_id)
        if session.state in {
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
            ExecutionState.WAITING_CONFIRMATION,
        }:
            return StructuredExecutionResponse(
                handled=True,
                status="recovery_not_resumable",
                message="La sesion recuperada no puede reanudarse en su estado actual.",
                plan=session.active_plan,
                error_code="RECOVERY_NOT_RESUMABLE",
            )
        execution_plan = self._remaining_recovery_plan(session)
        validation = self._validator.validate(execution_plan)
        if not validation.is_valid:
            return StructuredExecutionResponse(
                handled=True,
                status="validation_failed",
                message=self._validation_failed_message(validation),
                plan=execution_plan,
                validation_result=validation,
                error_code="INVALID_PLAN",
                error="; ".join(validation.errors),
            )
        execution = self._execute_plan_under_supervision(
            execution_plan,
            validation,
            confirmation_granted=False,
            control=control,
            on_progress=on_execution_progress,
            session_id=session_id,
        )
        return self._execution_response(execution_plan, validation, execution)

    def _remaining_recovery_plan(self, session: ExecutionSession) -> ExecutionPlan:
        completed = {
            step_id
            for step_id, snapshot in session.step_states.items()
            if snapshot.state is StepExecutionState.COMPLETED
        }
        if not completed:
            return session.active_plan
        remaining_steps = tuple(
            self._copy_step_without_completed_dependencies(step, completed)
            for step in session.active_plan.ordered_steps
            if step.id not in completed
        )
        return ExecutionPlan(
            goal=session.active_plan.goal,
            ordered_steps=remaining_steps,
            estimated_steps=len(remaining_steps),
            required_tools=tuple(
                dict.fromkeys(
                    step.tool
                    for step in remaining_steps
                    if isinstance(step.tool, str)
                )
            ),
            detected_risks=session.active_plan.detected_risks,
            requires_confirmation=session.active_plan.requires_confirmation,
            status=session.active_plan.status,
            output=session.active_plan.output,
            required_outputs=session.active_plan.required_outputs,
            output_validators=session.active_plan.output_validators,
            replanning_policy=session.active_plan.replanning_policy,
        )

    def _copy_step_without_completed_dependencies(
        self,
        step: ExecutionStep,
        completed_step_ids: set[str],
    ) -> ExecutionStep:
        return ExecutionStep(
            step.id,
            step.description,
            step.tool,
            tuple(
                dependency
                for dependency in step.depends_on
                if dependency not in completed_step_ids
            ),
            subplan=step.subplan,
            subplan_ref=step.subplan_ref,
            branch=step.branch,
            loop=step.loop,
            status=step.status,
            arguments=step.arguments,
            output_binding=step.output_binding,
            condition=step.condition,
            retry_policy=step.retry_policy,
            parallel_safe=step.parallel_safe,
            resource_keys=step.resource_keys,
            idempotent=step.idempotent,
            recovery_safe=step.recovery_safe,
            side_effect_free=step.side_effect_free,
            priority=step.priority,
            urgency=step.urgency,
            estimated_cost=step.estimated_cost,
            estimated_duration_seconds=step.estimated_duration_seconds,
            criticality=step.criticality,
            deadline=step.deadline,
        )

    def resume_pending_execution(
        self,
        *,
        confirmation_granted: bool = False,
        control: ExecutionControl | None = None,
        on_execution_progress: Callable[[ExecutionProgress], None] | None = None,
    ) -> StructuredExecutionResponse:
        """Resume the current interrupted execution without regenerating a plan."""
        state = self._resumable_execution
        if state is None:
            load_response = self.load_persisted_resumable_execution()
            if load_response.status == "resumable_execution_loaded":
                state = self._resumable_execution
                assert state is not None
            else:
                return load_response
        if state is None:
            return StructuredExecutionResponse(
                handled=True,
                status="no_resumable_execution",
                message="No hay ninguna ejecución pendiente que pueda reanudarse.",
                requires_confirmation=False,
                error_code="NO_RESUMABLE_EXECUTION",
                error="no resumable structured execution",
            )

        claim_error = self._delete_persisted_resumable_state()
        if claim_error is not None:
            return StructuredExecutionResponse(
                handled=True,
                status="resumable_execution_claim_failed",
                message="No se puede reanudar la ejecución.",
                plan=state.original_plan,
                validation_result=state.validation_result,
                requires_confirmation=state.validation_result.requires_confirmation,
                error_code=claim_error,
                error="could not claim persisted resumable execution state",
            )

        execution = self._resume_under_supervision(
            state,
            confirmation_granted=confirmation_granted,
            control=control,
            on_progress=on_execution_progress,
        )
        response = self._execution_response(
            state.original_plan,
            state.validation_result,
            execution,
        )
        persistence_error = self._sync_resumable_state(
            response,
            objective=state.objective,
            confirmation_granted=confirmation_granted or state.confirmation_granted,
        )
        if persistence_error is not None:
            return replace(response, error_code=persistence_error)
        return response

    def confirm_pending(
        self,
        *,
        control: ExecutionControl | None = None,
        on_execution_progress: Callable[[ExecutionProgress], None] | None = None,
    ) -> StructuredExecutionResponse:
        """Confirm the single active pending plan without exposing its token."""
        pending = self.pending_execution()
        if pending is None:
            return StructuredExecutionResponse(
                handled=True,
                status="no_pending_execution",
                message="No hay ninguna ejecucion pendiente que confirmar.",
                requires_confirmation=False,
                error_code="NO_PENDING_EXECUTION",
                error="no pending structured execution",
            )

        return self.confirm(
            pending.confirmation_token,
            objective=pending.objective,
            control=control,
            on_execution_progress=on_execution_progress,
        )

    def cancel_pending(
        self,
    ) -> StructuredExecutionResponse:
        """Cancel the single active pending plan without executing tools."""
        pending = self.pending_execution()
        if pending is None:
            return StructuredExecutionResponse(
                handled=True,
                status="no_pending_execution",
                message="No hay ninguna ejecucion pendiente que cancelar.",
                requires_confirmation=False,
                error_code="NO_PENDING_EXECUTION",
                error="no pending structured execution",
            )

        pending_record = self._pending_plans.get(pending.confirmation_token)
        self._discard_pending(pending.confirmation_token)
        self._resumable_execution = None
        delete_error = self._delete_persisted_resumable_state()
        self._cancel_pending_session(pending_record, "pending execution cancelled")
        return StructuredExecutionResponse(
            handled=True,
            status="pending_execution_cancelled",
            message="El plan pendiente fue cancelado. No se ejecuto ninguna herramienta.",
            plan=pending.plan,
            validation_result=pending.validation_result,
            requires_confirmation=False,
            confirmation_token=pending.confirmation_token,
            error_code=delete_error or "PENDING_EXECUTION_CANCELLED",
        )

    def show_pending(
        self,
    ) -> StructuredExecutionResponse:
        """Show the active pending plan without consuming its confirmation."""
        pending = self.pending_execution()
        if pending is None:
            return StructuredExecutionResponse(
                handled=True,
                status="no_pending_execution",
                message="No hay ninguna ejecucion pendiente que revisar.",
                requires_confirmation=False,
                error_code="NO_PENDING_EXECUTION",
                error="no pending structured execution",
            )

        return StructuredExecutionResponse(
            handled=True,
            status="pending_plan",
            message=pending.summary,
            plan=pending.plan,
            validation_result=pending.validation_result,
            requires_confirmation=True,
            confirmation_token=pending.confirmation_token,
            error_code="CONFIRMATION_REQUIRED",
        )

    def _should_handle_generation(
        self,
        generation: PlanGenerationResult,
    ) -> bool:
        if generation.error_code in {
            "PLAN_PARSE_ERROR",
            "INVALID_PLAN_RESPONSE",
            "UNKNOWN_TOOL",
            "INVALID_MODEL_RESPONSE",
            "MODEL_PLAN_PARSE_ERROR",
            "MODEL_PROPOSED_UNKNOWN_TOOL",
            "MODEL_PLAN_VALIDATION_FAILED",
            "MODEL_INSUFFICIENT_INFORMATION",
            "UNSUPPORTED_OBJECTIVE",
            "STRUCTURED_PLAN_PROVIDER_TIMEOUT",
            "STRUCTURED_PLAN_EMPTY_RESPONSE",
            "STRUCTURED_PLAN_PROVIDER_ERROR",
            "STRUCTURED_PLAN_PROVIDER_CANCELLED",
            "STRUCTURED_PLAN_RESPONSE_TOO_LARGE",
        }:
            return True

        if not generation.success:
            return False

        if generation.plan is None:
            return False

        if generation.plan.required_tools:
            return True

        return False

    def _store_pending_plan(
        self,
        objective: str,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
        session_id: str | None = None,
    ) -> str:
        token = validation.plan_signature or plan_signature(plan)
        if self._active_confirmation_token is not None:
            previous = self._pending_plans.get(self._active_confirmation_token)
            self._cancel_pending_session(previous, "pending execution replaced")
            self._discard_pending(self._active_confirmation_token)
        summary = self._pending_summary(plan)
        self._pending_plans[token] = _PendingPlan(
            execution=PendingStructuredExecution(
                objective=objective,
                plan=plan,
                validation_result=validation,
                confirmation_token=token,
                status="pending_confirmation",
                summary=summary,
                risks=list(plan.detected_risks),
                required_tools=list(plan.required_tools),
            ),
            session_id=session_id,
        )
        self._active_confirmation_token = token
        return token

    def _discard_pending(
        self,
        confirmation_token: str,
    ) -> None:
        self._pending_plans.pop(confirmation_token, None)
        if self._active_confirmation_token == confirmation_token:
            self._active_confirmation_token = None

    def _start_waiting_confirmation_session(
        self,
        plan: ExecutionPlan,
    ) -> ExecutionSession:
        session = self._execution_supervisor.start(plan)
        session = self._execution_supervisor.mark_running(session.session_id)
        return self._execution_supervisor.mark_waiting_confirmation(session.session_id)

    def _execute_plan_under_supervision(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
        *,
        confirmation_granted: bool,
        control: ExecutionControl | None = None,
        on_progress: Callable[[ExecutionProgress], None] | None = None,
        session_id: str | None = None,
    ) -> PlanExecutionResult:
        session = (
            self._execution_supervisor.get_session(session_id)
            if session_id is not None
            else self._execution_supervisor.start(plan)
        )
        self._execution_supervisor.mark_running(session.session_id)
        active_plan = plan
        active_validation = validation
        try:
            while True:
                self._record_dependency_graph_state(
                    session.session_id,
                    active_plan,
                )
                if self._can_use_concurrent_execution(active_plan, active_validation):
                    execution = self._execute_plan_concurrently_under_supervision(
                        session.session_id,
                        active_plan,
                        control=control,
                        on_progress=on_progress,
                    )
                else:
                    execution = self._executor.execute(
                        active_plan,
                        active_validation,
                        confirmation_granted=confirmation_granted,
                        control=control,
                        on_progress=on_progress,
                    )
                self._record_step_states_from_execution(
                    session.session_id,
                    active_plan,
                    execution,
                )
                if self._execution_can_finish_without_replan(execution):
                    self._finalize_supervised_session(session.session_id, execution)
                    return execution

                replanned = self._attempt_replan(
                    session.session_id,
                    active_plan,
                    execution,
                )
                if replanned is None:
                    return execution
                active_plan, active_validation = replanned
                self._execution_supervisor.mark_running(session.session_id)
        except Exception as error:
            self._execution_supervisor.mark_failed(session.session_id, error)
            raise

    def _can_use_concurrent_execution(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
    ) -> bool:
        return (
            self._concurrent_step_executor is not None
            and self._concurrency_policy.enabled
            and self._concurrency_policy.max_concurrency > 1
            and not plan.requires_confirmation
            and not getattr(validation, "requires_confirmation", False)
        )

    def _execute_plan_concurrently_under_supervision(
        self,
        session_id: str,
        plan: ExecutionPlan,
        *,
        control: ExecutionControl | None = None,
        on_progress: Callable[[ExecutionProgress], None] | None = None,
    ) -> PlanExecutionResult:
        assert self._concurrent_step_executor is not None
        completed: list[str] = []
        failed: list[str] = []
        cancelled: list[str] = []
        blocked: list[str] = []
        step_results: list[StepExecutionResult] = []
        started_at = _utc_iso()
        batch_number = 0
        total_steps = len(plan.ordered_steps)

        while len(completed) < total_steps:
            if control is not None and control.should_cancel and control.should_cancel():
                pending = [
                    step.id
                    for step in plan.ordered_steps
                    if step.id not in completed and step.id not in failed
                ]
                return PlanExecutionResult(
                    plan_status=PlanExecutionStatus.CANCELLED.value,
                    success=False,
                    completed_steps=completed,
                    failed_steps=failed,
                    pending_steps=pending,
                    step_results=step_results,
                    error=control.cancellation_reason,
                    interrupted=True,
                    cancelled=True,
                    error_code=ExecutionErrorCode.EXECUTION_CANCELLED.value,
                    started_at=started_at,
                    finished_at=_utc_iso(),
                )

            resolution = self._dependency_resolver.resolve(
                plan,
                completed_step_ids=tuple(completed),
                failed_step_ids=tuple(failed),
            )
            blocked = list(resolution.blocked_step_ids)
            if failed or blocked:
                pending = list(resolution.pending_step_ids)
                return PlanExecutionResult(
                    plan_status=PlanExecutionStatus.FAILED.value
                    if failed
                    else PlanExecutionStatus.BLOCKED.value,
                    success=False,
                    completed_steps=completed,
                    failed_step=failed[0] if failed else None,
                    failed_steps=failed,
                    blocked_steps=blocked,
                    pending_steps=pending,
                    step_results=step_results,
                    error="one or more concurrent steps failed"
                    if failed
                    else "execution blocked by dependencies",
                    failed=bool(failed),
                    blocked=bool(blocked),
                    current_step=failed[0] if failed else None,
                    error_code=ExecutionErrorCode.TOOL_EXECUTION_FAILED.value
                    if failed
                    else ExecutionErrorCode.DEPENDENCY_NOT_COMPLETED.value,
                    started_at=started_at,
                    finished_at=_utc_iso(),
                    metadata=self._concurrent_metadata(
                        step_results,
                        last_batch=None,
                    ),
                )

            ready_steps = list(resolution.ready_steps)
            if not ready_steps:
                pending = list(resolution.pending_step_ids)
                return PlanExecutionResult(
                    plan_status=PlanExecutionStatus.BLOCKED.value,
                    success=False,
                    completed_steps=completed,
                    failed_steps=failed,
                    blocked_steps=blocked,
                    pending_steps=pending,
                    step_results=step_results,
                    error="no ready steps available",
                    blocked=True,
                    error_code=ExecutionErrorCode.DEPENDENCY_NOT_COMPLETED.value,
                    started_at=started_at,
                    finished_at=_utc_iso(),
                )

            ready_steps = list(
                self._prioritize_ready_steps(
                    session_id,
                    plan,
                    tuple(ready_steps),
                    completed_step_ids=tuple(completed),
                )
            )
            batch_number += 1
            batch = build_execution_batch(
                ready_steps,
                self._concurrency_policy,
                batch_id=f"{session_id}.batch.{batch_number:06d}",
            )
            self._execution_supervisor.record_execution_batch_created(
                session_id,
                batch,
            )
            if len(batch.step_ids) == self._concurrency_policy.max_concurrency:
                self._execution_supervisor.record_concurrency_limit_applied(
                    session_id,
                    max_concurrency=self._concurrency_policy.max_concurrency,
                    selected_step_count=len(batch.step_ids),
                )
            selected_steps = [
                step for step in ready_steps if step.id in set(batch.step_ids)
            ]
            for step in selected_steps:
                self._execution_supervisor.mark_step_started(
                    session_id,
                    step.id,
                    dependency_ids=tuple(step.depends_on),
                )
            self._execution_supervisor.mark_execution_batch_started(
                session_id,
                batch,
            )
            if on_progress is not None:
                on_progress(
                    ExecutionProgress(
                        phase="concurrent_batch_running",
                        total_steps=total_steps,
                        elapsed_ms=0,
                        message=batch.batch_id,
                    )
                )
            batch_result = self._concurrent_step_executor.run_batch(
                batch,
                selected_steps,
                self._concurrency_policy,
            )
            self._execution_supervisor.record_execution_batch_result(
                session_id,
                batch_result,
            )
            if batch_result.fail_fast_triggered:
                self._execution_supervisor.record_fail_fast_triggered(
                    session_id,
                    batch_id=batch_result.batch_id,
                )

            converted = self._batch_step_results(batch_result, selected_steps)
            step_results.extend(converted)
            completed.extend(batch_result.completed_step_ids)
            failed.extend(batch_result.failed_step_ids)
            cancelled.extend(batch_result.cancelled_step_ids)
            if failed or cancelled:
                resolution = self._dependency_resolver.resolve(
                    plan,
                    completed_step_ids=tuple(completed),
                    failed_step_ids=tuple(failed + cancelled),
                )
                return PlanExecutionResult(
                    plan_status=PlanExecutionStatus.CANCELLED.value
                    if cancelled and not failed
                    else PlanExecutionStatus.FAILED.value,
                    success=False,
                    completed_steps=completed,
                    failed_step=(failed + cancelled)[0],
                    failed_steps=failed,
                    blocked_steps=list(resolution.blocked_step_ids),
                    pending_steps=list(resolution.pending_step_ids),
                    step_results=step_results,
                    error="one or more concurrent steps failed"
                    if failed
                    else "one or more concurrent steps were cancelled",
                    failed=bool(failed),
                    cancelled=bool(cancelled and not failed),
                    current_step=(failed + cancelled)[0],
                    error_code=ExecutionErrorCode.TOOL_EXECUTION_FAILED.value
                    if failed
                    else ExecutionErrorCode.EXECUTION_CANCELLED.value,
                    started_at=started_at,
                    finished_at=_utc_iso(),
                    metadata=self._concurrent_metadata(
                        step_results,
                        last_batch=batch_result,
                    ),
                )

        return PlanExecutionResult(
            plan_status=PlanExecutionStatus.COMPLETED.value,
            success=True,
            completed_steps=completed,
            step_results=step_results,
            completed=True,
            started_at=started_at,
            finished_at=_utc_iso(),
            metadata={"concurrent_execution": True},
        )

    def _batch_step_results(
        self,
        batch_result: ExecutionBatchResult,
        selected_steps: list[Any],
    ) -> list[StepExecutionResult]:
        step_by_id = {step.id: step for step in selected_steps}
        converted: list[StepExecutionResult] = []
        for result in batch_result.step_results:
            step = step_by_id.get(result.step_id)
            if result.status == "completed":
                status = StepExecutionStatus.COMPLETED.value
                error_code = None
            elif result.status == "cancelled":
                status = StepExecutionStatus.CANCELLED.value
                error_code = ExecutionErrorCode.EXECUTION_CANCELLED.value
            else:
                status = StepExecutionStatus.FAILED.value
                error_code = ExecutionErrorCode.TOOL_EXECUTION_FAILED.value
            converted.append(
                StepExecutionResult(
                    step_id=result.step_id,
                    status=status,
                    success=result.status == "completed",
                    tool_name=getattr(step, "tool", None),
                    output=result.result,
                    error=result.error,
                    error_code=error_code,
                    started_at=result.started_at,
                    finished_at=result.finished_at,
                    metadata={"batch_id": batch_result.batch_id},
                )
            )
        return converted

    def _concurrent_metadata(
        self,
        step_results: list[StepExecutionResult],
        *,
        last_batch: ExecutionBatchResult | None,
    ) -> dict[str, object]:
        errors_by_step = {
            result.step_id: result.error or result.error_code or "step failed"
            for result in step_results
            if not result.success
        }
        metadata: dict[str, object] = {
            "concurrent_execution": True,
            "errors_by_step": errors_by_step,
        }
        if last_batch is not None:
            metadata["batch_id"] = last_batch.batch_id
            metadata["cancelled_step_ids"] = tuple(last_batch.cancelled_step_ids)
        return metadata

    def _resume_under_supervision(
        self,
        state: ResumableExecutionState,
        *,
        confirmation_granted: bool,
        control: ExecutionControl | None = None,
        on_progress: Callable[[ExecutionProgress], None] | None = None,
    ) -> PlanExecutionResult:
        session = self._execution_supervisor.start(state.original_plan)
        session = self._execution_supervisor.mark_running(session.session_id)
        try:
            execution = self._executor.resume(
                state,
                confirmation_granted=confirmation_granted,
                control=control,
                on_progress=on_progress,
            )
        except Exception as error:
            self._execution_supervisor.mark_failed(session.session_id, error)
            raise

        self._finalize_supervised_session(session.session_id, execution)
        return execution

    def _prioritize_ready_steps(
        self,
        session_id: str,
        plan: ExecutionPlan,
        ready_steps: tuple[ExecutionStep, ...],
        *,
        completed_step_ids: tuple[str, ...],
    ) -> tuple[ExecutionStep, ...]:
        session = self._execution_supervisor.get_session(session_id)
        ready_since = {
            step_id: snapshot.ready_since
            for step_id, snapshot in session.step_states.items()
            if snapshot.ready_since is not None
        }
        for step in ready_steps:
            self._execution_supervisor.mark_step_ready(
                session_id,
                step.id,
                dependency_ids=tuple(step.depends_on),
            )
        session = self._execution_supervisor.get_session(session_id)
        ready_since = {
            step_id: snapshot.ready_since
            for step_id, snapshot in session.step_states.items()
            if snapshot.ready_since is not None
        }
        decision = self._ready_step_prioritizer.prioritize(
            ready_steps,
            plan=plan,
            completed_step_ids=completed_step_ids,
            ready_since_by_step_id=ready_since,
            now=self._priority_clock(),
        )
        self._execution_supervisor.record_priority_decision(session_id, decision)
        return tuple(
            step
            for step_id in decision.ordered_step_ids
            for step in ready_steps
            if step.id == step_id
        )

    def _record_dependency_graph_state(
        self,
        session_id: str,
        plan: ExecutionPlan,
    ) -> None:
        step_count = len(plan.ordered_steps)
        dependency_count = sum(len(step.depends_on) for step in plan.ordered_steps)
        try:
            ready_steps = self._dependency_resolver.get_ready_steps(
                plan,
                completed_step_ids=(),
            )
        except Exception as error:
            self._execution_supervisor.record_dependency_graph_rejected(
                session_id,
                error=error,
            )
            return

        self._execution_supervisor.record_dependency_graph_validated(
            session_id,
            step_count=step_count,
            dependency_count=dependency_count,
        )
        for step in ready_steps:
            self._execution_supervisor.mark_step_ready(
                session_id,
                step.id,
                dependency_ids=tuple(step.depends_on),
            )

    def _record_step_states_from_execution(
        self,
        session_id: str,
        plan: ExecutionPlan,
        execution: PlanExecutionResult,
    ) -> None:
        step_by_id = {step.id: step for step in plan.ordered_steps}
        for step_result in execution.step_results:
            step = step_by_id.get(step_result.step_id)
            dependency_ids = tuple(step.depends_on) if step is not None else ()
            if step_result.status in {"completed", "failed", "blocked"}:
                self._execution_supervisor.mark_step_started(
                    session_id,
                    step_result.step_id,
                    dependency_ids=dependency_ids,
                )
            if step_result.status == "completed":
                self._execution_supervisor.mark_step_completed(
                    session_id,
                    step_result.step_id,
                    dependency_ids=dependency_ids,
                )
            elif step_result.status == "failed":
                self._execution_supervisor.mark_step_failed(
                    session_id,
                    step_result.step_id,
                    step_result.error or step_result.error_code or "step failed",
                    dependency_ids=dependency_ids,
                )
            elif step_result.status == "blocked":
                self._execution_supervisor.mark_step_blocked(
                    session_id,
                    step_result.step_id,
                    dependency_ids=dependency_ids,
                    error=step_result.error,
                )
            elif step_result.status == "cancelled":
                self._execution_supervisor.mark_step_cancelled(
                    session_id,
                    step_result.step_id,
                    dependency_ids=dependency_ids,
                    error=step_result.error,
                )

        for step_id in execution.completed_steps:
            if step_id not in step_by_id:
                continue
            step = step_by_id[step_id]
            current = self._execution_supervisor.get_session(session_id).step_states.get(step_id)
            if current is None or current.state.value != "completed":
                self._execution_supervisor.mark_step_started(
                    session_id,
                    step_id,
                    dependency_ids=tuple(step.depends_on),
                )
                self._execution_supervisor.mark_step_completed(
                    session_id,
                    step_id,
                    dependency_ids=tuple(step.depends_on),
                )

        if execution.plan_status == PlanExecutionStatus.BLOCKED.value:
            self._execution_supervisor.record_execution_blocked(
                session_id,
                error=execution.error or execution.failure_reason or "execution blocked",
            )

    def _execution_can_finish_without_replan(
        self,
        execution: PlanExecutionResult,
    ) -> bool:
        if execution.success:
            return True
        if execution.plan_status in {
            PlanExecutionStatus.CANCELLED.value,
            PlanExecutionStatus.BLOCKED_CONFIRMATION.value,
        }:
            return True
        return self._execution_replanner is None

    def _attempt_replan(
        self,
        session_id: str,
        failed_plan: ExecutionPlan,
        execution: PlanExecutionResult,
    ) -> tuple[ExecutionPlan, PlanValidationResult] | None:
        session = self._execution_supervisor.get_session(session_id)
        current_step = execution.current_step or execution.failed_step
        results = self._supervisor_results(execution)
        error = (
            execution.error
            or execution.failure_reason
            or execution.error_code
            or execution.plan_status
        )
        failed = self._execution_supervisor.mark_failed(
            session_id,
            error,
            current_step=current_step,
            results=results,
        )
        attempt_number = failed.replan_count + 1
        request = self._replan_request(
            failed,
            failed_plan,
            execution,
            attempt_number=attempt_number,
        )
        self._execution_supervisor.record_replan_event(
            session_id,
            "replan_requested",
            attempt_number=attempt_number,
            failed_step=request.failed_step,
            reason="execution_failed",
        )

        policy_decision = self._replan_policy.evaluate(
            request,
            current_replan_count=failed.replan_count,
        )
        if policy_decision.status is ReplanResultStatus.LIMIT_REACHED:
            self._execution_supervisor.record_replan_event(
                session_id,
                "replan_limit_reached",
                attempt_number=attempt_number,
                failed_step=request.failed_step,
                reason=policy_decision.reason.value,
                error=policy_decision.error,
            )
            return None
        if not policy_decision.accepted:
            self._execution_supervisor.record_replan_event(
                session_id,
                "replan_rejected",
                attempt_number=attempt_number,
                failed_step=request.failed_step,
                reason=policy_decision.reason.value,
                error=policy_decision.error,
            )
            return None

        self._execution_supervisor.mark_replanning(
            session_id,
            attempt_number=attempt_number,
            current_step=request.failed_step,
            reason=policy_decision.reason.value,
        )
        assert self._execution_replanner is not None
        replan_result = self._execution_replanner.replan(request)
        if not replan_result.accepted or replan_result.revised_plan is None:
            self._record_replan_failure(session_id, request, replan_result)
            return None

        revised_validation = self._validator.validate(replan_result.revised_plan)
        if not revised_validation.is_valid:
            self._execution_supervisor.record_replan_event(
                session_id,
                "replan_failed",
                attempt_number=attempt_number,
                failed_step=request.failed_step,
                reason="validation_error",
                error="; ".join(revised_validation.errors),
            )
            self._execution_supervisor.mark_failed(
                session_id,
                "replanned plan validation failed",
                current_step=request.failed_step,
                results=results,
            )
            return None

        record = replan_record(
            request,
            replan_result,
            previous_plan=failed_plan,
        )
        self._execution_supervisor.record_replan(session_id, record)
        self._execution_supervisor.record_replan_event(
            session_id,
            "execution_resumed_with_revised_plan",
            attempt_number=attempt_number,
            failed_step=request.failed_step,
            reason=replan_result.reason.value,
        )
        return replan_result.revised_plan, revised_validation

    def _record_replan_failure(
        self,
        session_id: str,
        request: ReplanRequest,
        replan_result: ReplanResult,
    ) -> None:
        event_type = (
            "replan_failed"
            if replan_result.status is ReplanResultStatus.PLANNER_ERROR
            else "replan_rejected"
        )
        self._execution_supervisor.record_replan_event(
            session_id,
            event_type,
            attempt_number=request.attempt_number,
            failed_step=request.failed_step,
            reason=replan_result.reason.value,
            error=replan_result.error,
        )
        self._execution_supervisor.mark_failed(
            session_id,
            replan_result.error or replan_result.status.value,
            current_step=request.failed_step,
        )

    def _replan_request(
        self,
        session: ExecutionSession,
        failed_plan: ExecutionPlan,
        execution: PlanExecutionResult,
        *,
        attempt_number: int,
    ) -> ReplanRequest:
        return ReplanRequest(
            session_id=session.session_id,
            original_plan=session.original_plan,
            active_plan=failed_plan,
            failed_step=execution.current_step or execution.failed_step,
            error=(
                execution.error
                or execution.failure_reason
                or execution.error_code
                or execution.plan_status
            ),
            error_code=execution.error_code,
            partial_results=self._partial_results(execution),
            completed_step_ids=tuple(execution.completed_steps),
            failed_step_ids=tuple(execution.failed_steps),
            cancelled_step_ids=tuple(
                item
                for item in execution.metadata.get("cancelled_step_ids", ())
                if isinstance(item, str)
            ),
            pending_step_ids=tuple(execution.pending_steps),
            blocked_step_ids=tuple(execution.blocked_steps),
            dependency_graph={
                step.id: tuple(step.depends_on)
                for step in failed_plan.ordered_steps
            },
            batch_id=(
                execution.metadata.get("batch_id")
                if isinstance(execution.metadata.get("batch_id"), str)
                else None
            ),
            errors_by_step={
                str(step_id): str(error)
                for step_id, error in execution.metadata.get(
                    "errors_by_step",
                    {},
                ).items()
            }
            if isinstance(execution.metadata.get("errors_by_step"), dict)
            else {},
            priority_decision_id=(
                f"{session.session_id}.priority.{len(session.priority_history):06d}"
                if session.last_priority_decision is not None
                else None
            ),
            ordered_ready_step_ids=tuple(
                getattr(session.last_priority_decision, "ordered_step_ids", ())
            ),
            selected_step_ids=tuple(
                getattr(session.last_priority_decision, "selected_step_ids", ())
            ),
            priority_scores={
                score.step_id: float(score.final_score)
                for score in getattr(session.last_priority_decision, "scores", ())
            },
            failed_step_priority=(
                next(
                    (
                        float(score.final_score)
                        for score in getattr(
                            session.last_priority_decision,
                            "scores",
                            (),
                        )
                        if score.step_id == (execution.current_step or execution.failed_step)
                    ),
                    None,
                )
            ),
            priority_rationale_summary=getattr(
                session.last_priority_decision,
                "rationale_summary",
                None,
            ),
            attempt_number=attempt_number,
            max_attempts=self._replan_policy.max_replans_per_session,
        )

    def _partial_results(
        self,
        execution: PlanExecutionResult,
    ) -> dict[str, object]:
        return {
            result.step_id: result.output
            for result in execution.step_results
            if result.success and result.status == "completed"
        }

    def _finalize_supervised_session(
        self,
        session_id: str,
        execution: PlanExecutionResult,
    ) -> None:
        current_step = execution.current_step or execution.failed_step
        results = self._supervisor_results(execution)
        if execution.success:
            self._execution_supervisor.mark_completed(
                session_id,
                current_step=current_step,
                results=results,
            )
            return

        if execution.plan_status == PlanExecutionStatus.CANCELLED.value:
            self._execution_supervisor.mark_cancelled(
                session_id,
                error=execution.interruption_reason or execution.error,
                current_step=current_step,
                results=results,
            )
            return

        if execution.plan_status == PlanExecutionStatus.BLOCKED_CONFIRMATION.value:
            self._execution_supervisor.mark_waiting_confirmation(
                session_id,
                current_step=current_step,
            )
            return

        self._execution_supervisor.mark_failed(
            session_id,
            execution.error
            or execution.failure_reason
            or execution.error_code
            or execution.plan_status,
            current_step=current_step,
            results=results,
        )

    def _cancel_pending_session(
        self,
        pending_record: _PendingPlan | None,
        reason: str,
    ) -> None:
        if pending_record is None or pending_record.session_id is None:
            return
        self._execution_supervisor.mark_cancelled(
            pending_record.session_id,
            error=reason,
        )

    def _supervisor_results(
        self,
        execution: PlanExecutionResult,
    ) -> dict[str, object]:
        return {
            "plan_status": execution.plan_status,
            "completed_steps": tuple(execution.completed_steps),
            "failed_steps": tuple(execution.failed_steps),
            "blocked_steps": tuple(execution.blocked_steps),
            "skipped_steps": tuple(execution.skipped_steps),
            "error_code": execution.error_code,
        }

    def _execution_response(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
        execution: PlanExecutionResult,
        *,
        confirmation_token: str | None = None,
    ) -> StructuredExecutionResponse:
        if execution.success:
            return StructuredExecutionResponse(
                handled=True,
                status="completed",
                message=self._completed_message(execution),
                plan=plan,
                validation_result=validation,
                execution_result=execution,
                requires_confirmation=False,
                confirmation_token=confirmation_token,
                resumable_state=None,
                partial_state=execution.partial_state,
            )

        return StructuredExecutionResponse(
            handled=True,
            status=execution.plan_status,
            message=self._failed_message(execution),
            plan=plan,
            validation_result=validation,
            execution_result=execution,
            requires_confirmation=execution.requires_confirmation,
            confirmation_token=confirmation_token,
            error_code=execution.error_code,
            error=execution.error,
            resumable_state=None,
            partial_state=execution.partial_state,
        )

    def _sync_resumable_state(
        self,
        response: StructuredExecutionResponse,
        *,
        objective: str,
        confirmation_granted: bool,
    ) -> str | None:
        execution = response.execution_result
        if (
            execution is not None
            and execution.plan_status == PlanExecutionStatus.BLOCKED_CONFIRMATION.value
        ):
            if self._resumable_execution is not None:
                return self._save_resumable_state(self._resumable_execution)
            return None

        if (
            response.plan is None
            or response.validation_result is None
            or execution is None
            or execution.plan_status != PlanExecutionStatus.INTERRUPTED.value
            or not execution.resumable
        ):
            if (
                self._resumable_execution is not None
                and response.validation_result is not None
                and response.validation_result.plan_signature
                != self._resumable_execution.validated_plan_signature
            ):
                return None
            self._resumable_execution = None
            self._delete_persisted_resumable_state()
            return None

        state = self._build_resumable_state(
            objective=objective,
            plan=response.plan,
            validation=response.validation_result,
            execution=execution,
            confirmation_granted=confirmation_granted,
        )
        self._resumable_execution = state
        return self._save_resumable_state(state)

    def _build_resumable_state(
        self,
        *,
        objective: str,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
        execution: PlanExecutionResult,
        confirmation_granted: bool,
    ) -> ResumableExecutionState:
        previous_results = {
            result.step_id: result.output
            for result in execution.step_results
            if result.success and result.status == "completed"
        }
        retry_attempts = {
            result.step_id: int(result.metadata["attempt_number"])
            for result in execution.step_results
            if isinstance(result.metadata.get("attempt_number"), int)
        }
        retry_history = {
            result.step_id: tuple(
                entry
                for entry in result.metadata.get("retry_history", ())
                if isinstance(entry, dict)
            )
            for result in execution.step_results
            if result.metadata.get("retry_history")
        }
        retry_decisions = {
            result.step_id: {
                "retry_reason": result.metadata.get("retry_reason"),
                "retry_scheduled": result.metadata.get("retry_scheduled"),
                "retry_exhausted": result.metadata.get("retry_exhausted"),
                "attempt_number": result.metadata.get("attempt_number"),
                "max_attempts": result.metadata.get("max_attempts"),
            }
            for result in execution.step_results
            if "retry_reason" in result.metadata
        }
        execution_id = (
            execution.trace.execution_id
            if execution.trace is not None
            else None
        )
        execution_context = ExecutionContext(execution_id)
        step_by_id = {step.id: step for step in plan.ordered_steps}
        for result in execution.step_results:
            if result.success and result.status == "completed":
                execution_context.mark_step_started(
                    result.step_id,
                    int(result.metadata.get("attempt_number", 1)),
                )
                execution_context.mark_step_succeeded(result.step_id, result.output)
                step = step_by_id.get(result.step_id)
                if step is not None and step.output_binding is not None:
                    binding = step.output_binding
                    execution_context.set_variable(
                        binding.variable_name,
                        navigate_structured_path(
                            result.output,
                            binding.path,
                            owner_label=f"output binding for step '{step.id}'",
                        ),
                    )
                continue
            if result.status == "skipped":
                execution_context.mark_step_skipped(result.step_id)
        return ResumableExecutionState(
            objective=objective,
            original_plan=plan,
            validation_result=validation,
            validated_plan_signature=validation.plan_signature,
            completed_step_ids=tuple(execution.completed_steps),
            pending_step_ids=tuple(execution.pending_steps),
            failed_step_ids=tuple(execution.failed_steps),
            interrupted_step_id=execution.current_step,
            previous_results=previous_results,
            resumable=execution.resumable,
            interruption_reason=execution.interruption_reason,
            confirmation_granted=confirmation_granted,
            retry_attempts=retry_attempts,
            retry_history=retry_history,
            retry_decisions=retry_decisions,
            execution_context_snapshot=execution_context.snapshot(),
            goal_verification_result=execution.goal_verification_result,
        )

    def _save_resumable_state(
        self,
        state: ResumableExecutionState,
    ) -> str | None:
        if self._resumable_store is None:
            return None

        try:
            self._resumable_store.save(state)
        except ResumableExecutionStoreError as error:
            return error.error_code
        return None

    def _delete_persisted_resumable_state(self) -> str | None:
        if self._resumable_store is None:
            return None

        try:
            self._resumable_store.delete()
        except ResumableExecutionStoreError as error:
            return error.error_code
        return None

    def _planning_failed_message(
        self,
        generation: PlanGenerationResult,
    ) -> str:
        detail = "; ".join(generation.errors) if generation.errors else "unknown error"
        return f"No se pudo generar un plan estructurado: {detail}"

    def _validation_failed_message(
        self,
        validation: PlanValidationResult,
    ) -> str:
        detail = "; ".join(validation.errors[:3])
        return f"El plan estructurado no puede ejecutarse: {detail}"

    def _confirmation_message(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
        token: str,
    ) -> str:
        lines = [
            "Plan estructurado pendiente de confirmacion.",
            f"Objetivo: {plan.goal}",
            "Pasos:",
        ]
        for step in plan.ordered_steps:
            lines.append(f"- {step.id}: {step.description} [{step.tool}]")

        if plan.required_tools:
            lines.append("Herramientas: " + ", ".join(plan.required_tools))
        if plan.detected_risks:
            lines.append("Riesgos: " + "; ".join(plan.detected_risks))
        lines.append(
            "La ejecucion no se realizo porque requiere confirmacion explicita."
        )
        lines.append(
            "Responde 'confirmo' para ejecutarlo, 'cancela' para descartarlo "
            "o 'muestrame el plan' para revisarlo."
        )
        return "\n".join(lines)

    def _pending_summary(
        self,
        plan: ExecutionPlan,
    ) -> str:
        lines = [
            "Plan estructurado pendiente de confirmacion.",
            f"Objetivo: {plan.goal}",
            "Pasos:",
        ]
        for step in plan.ordered_steps:
            lines.append(f"- {step.id}: {step.description} [{step.tool}]")
        if plan.required_tools:
            lines.append("Herramientas: " + ", ".join(plan.required_tools))
        if plan.detected_risks:
            lines.append("Riesgos: " + "; ".join(plan.detected_risks))
        lines.append("Confirmacion requerida: si.")
        lines.append(
            "Responde 'confirmo' para ejecutarlo, 'cancela' para descartarlo "
            "o 'muestrame el plan' para revisarlo."
        )
        return "\n".join(lines)

    def _planned_message(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
    ) -> str:
        lines = [
            "Plan estructurado generado.",
            f"Objetivo: {plan.goal}",
            "Pasos:",
        ]
        for step in plan.ordered_steps:
            lines.append(f"- {step.id}: {step.description} [{step.tool}]")
        if plan.required_tools:
            lines.append("Herramientas: " + ", ".join(plan.required_tools))
        if validation.warnings:
            lines.append("Advertencias: " + "; ".join(validation.warnings))
        lines.append("La ejecucion no se realizo en esta fase.")
        return "\n".join(lines)

    def _completed_message(
        self,
        execution: PlanExecutionResult,
    ) -> str:
        return (
            "Ejecucion estructurada completada. "
            f"Pasos completados: {', '.join(execution.completed_steps)}."
        )

    def _failed_message(
        self,
        execution: PlanExecutionResult,
    ) -> str:
        if execution.plan_status == PlanExecutionStatus.PARTIALLY_COMPLETED.value:
            return self._partial_execution_message(execution)
        if execution.plan_status == PlanExecutionStatus.INTERRUPTED.value:
            return execution.interruption_reason or "Ejecucion estructurada interrumpida."
        if execution.plan_status == PlanExecutionStatus.CANCELLED.value:
            return execution.interruption_reason or "Ejecucion estructurada cancelada."

        failed = execution.failed_step or execution.current_step or "desconocido"
        reason = execution.error or execution.failure_reason or "motivo no especificado"
        skipped = (
            f" Pasos no ejecutados: {', '.join(execution.skipped_steps)}."
            if execution.skipped_steps
            else ""
        )
        return f"Ejecucion estructurada fallida en {failed}: {reason}.{skipped}"

    def _partial_execution_message(
        self,
        execution: PlanExecutionResult,
    ) -> str:
        partial = execution.partial_state
        completed = len(execution.completed_steps)
        total = (
            len(partial.step_results)
            if partial is not None
            else completed + len(execution.failed_steps) + len(execution.skipped_steps)
        )
        failed_step = execution.failed_step or execution.current_step or "desconocido"
        pending_count = (
            len(partial.pending_step_ids)
            if partial is not None
            else len(execution.pending_steps)
        )
        resumable = (
            "Atlas: La ejecucion puede reanudarse."
            if execution.resumable
            else "Atlas: La ejecucion no puede reanudarse."
        )
        return "\n".join(
            [
                "Atlas: Ejecucion completada parcialmente.",
                f"Atlas: Se completaron {completed} de {total} pasos.",
                f"Atlas: El paso {failed_step} ha fallado.",
                f"Atlas: Quedan {pending_count} pasos pendientes.",
                resumable,
            ]
        )

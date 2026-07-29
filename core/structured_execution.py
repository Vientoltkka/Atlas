"""Structured execution pipeline coordination for Atlas."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from uuid import uuid4

from core.concurrent_step_executor import (
    ConcurrentStepExecutor,
    ExecutionBatchResult,
    ExecutionConcurrencyPolicy,
    build_execution_batch,
)
from core.execution_context import ExecutionContext, ExecutionContextSnapshot
from core.execution_report import (
    ExecutionReportGenerator,
    OperationalExecutionReport,
)
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
from core.execution_resources import (
    BudgetReservation,
    ExecutionBudgetExceededError,
    ExecutionBudgetManager,
    ExecutionResourceCatalog,
    ExecutionResourceOptimizer,
    ExecutionResourcePolicy,
    NoCompatibleResourceError,
    ResourceSelectionError,
    ResourceSelectionDecision,
    ResourceSelectionReason,
)
from core.execution_session_persistence import (
    ExecutionRecoveryService,
    RecoveryDecisionType,
    RecoveryReport,
)
from core.execution_strategy import (
    ExecutionStrategySelectionResult,
    ExecutionStrategySelector,
    GlobalExecutionSafetyPolicy,
    build_strategy_request,
)
from core.execution_history_advisor import (
    ExecutionHistoryAdvisor,
    HistoricalPlanningContext,
)
from core.goal_verifier import (
    GoalVerificationResult,
    GoalVerificationStatus,
    GoalVerifier,
    goal_verification_result_from_dict,
    goal_verification_result_to_dict,
)
from core.historical_plan_adjustment import (
    HistoricalAdjustmentRequest,
    HistoricalPlanAdjuster,
    HistoricalPlanAdjustmentResult,
)
from core.execution_authorization import (
    DispatchStatus,
    ExecutionAuthorizationDecision,
    ExecutionAuthorizationGate,
    ExecutionAuthorizationRequest,
    ExecutionAuthorizationResult,
    ExecutionConfirmationReference,
    ExecutionDispatchResult,
    ExecutionDispatcher,
    authorization_with_dispatch,
    build_confirmation_references,
)
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult, Planner
from core.objective_correction import (
    CorrectionClassification,
    CorrectionLifecycleStatus,
    ObjectiveCorrectionDecision,
    ObjectiveCorrectionPolicy,
    correction_fragment_fingerprint,
    merge_corrective_execution,
)
from core.resumable_execution_store import (
    ResumableExecutionStore,
    ResumableExecutionStoreError,
)
from core.structured_plan_replanner import (
    ExecutionReplanner as StructuredExecutionReplanner,
    ReplanFailureClassification,
    ReplanPolicy,
    ReplanReason,
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
    operational_report: OperationalExecutionReport | None = None
    strategy_selection: ExecutionStrategySelectionResult | None = None
    authorization_result: ExecutionAuthorizationResult | None = None
    dispatch_result: ExecutionDispatchResult | None = None
    original_request: str | None = None
    route_decision: object | None = None
    historical_context: HistoricalPlanningContext | None = None
    historical_adjustment: HistoricalPlanAdjustmentResult | None = None
    objective_correction: ObjectiveCorrectionDecision | None = None
    corrected_verification: GoalVerificationResult | None = None


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
    strategy_selection: ExecutionStrategySelectionResult | None = None
    authorization_result: ExecutionAuthorizationResult | None = None
    confirmation_references: tuple[ExecutionConfirmationReference, ...] = ()
    authorization_scope: str | None = None


@dataclass(frozen=True, slots=True)
class _PendingPlan:
    execution: PendingStructuredExecution
    session_id: str | None = None
    correction_decision: ObjectiveCorrectionDecision | None = None
    original_plan: ExecutionPlan | None = None
    original_execution: PlanExecutionResult | None = None
    original_session_id: str | None = None


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
        resource_policy: ExecutionResourcePolicy | None = None,
        resource_catalog: ExecutionResourceCatalog | None = None,
        resource_optimizer: ExecutionResourceOptimizer | None = None,
        budget_manager: ExecutionBudgetManager | None = None,
        execution_report_generator: ExecutionReportGenerator | None = None,
        execution_strategy_selector: ExecutionStrategySelector | None = None,
        execution_safety_policy: GlobalExecutionSafetyPolicy | None = None,
        execution_authorization_gate: ExecutionAuthorizationGate | None = None,
        execution_dispatcher: ExecutionDispatcher | None = None,
        execution_history_advisor: ExecutionHistoryAdvisor | None = None,
        historical_plan_adjuster: HistoricalPlanAdjuster | None = None,
        objective_correction_policy: ObjectiveCorrectionPolicy | None = None,
    ) -> None:
        self._planner = planner
        self._validator = validator
        self._executor = executor
        self._execution_supervisor = execution_supervisor or ExecutionSupervisor()
        self._execution_replanner = execution_replanner
        self._replan_policy = replan_policy or ReplanPolicy()
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
        self._resource_policy = resource_policy or ExecutionResourcePolicy()
        self._resource_catalog = resource_catalog or ExecutionResourceCatalog()
        self._resource_optimizer = (
            resource_optimizer
            or ExecutionResourceOptimizer(self._resource_policy)
        )
        self._budget_manager = budget_manager
        self._execution_report_generator = (
            execution_report_generator or ExecutionReportGenerator()
        )
        self._execution_strategy_selector = execution_strategy_selector
        self._execution_safety_policy = (
            execution_safety_policy or GlobalExecutionSafetyPolicy()
        )
        self._execution_dispatcher = execution_dispatcher or ExecutionDispatcher()
        self._execution_history_advisor = execution_history_advisor
        self._historical_plan_adjuster = historical_plan_adjuster
        self._objective_correction_policy = (
            objective_correction_policy or ObjectiveCorrectionPolicy()
        )
        self._execution_authorization_gate = (
            execution_authorization_gate
            or ExecutionAuthorizationGate(
                already_dispatched=self._execution_dispatcher.has_dispatched,
                confirmation_consumed=(
                    self._execution_dispatcher.confirmation_consumed
                ),
            )
        )
        self._pending_plans: dict[str, _PendingPlan] = {}
        self._active_confirmation_token: str | None = None
        self._resumable_execution: ResumableExecutionState | None = None
        self._resumable_store = resumable_store
        self._correction_fingerprints: set[str] = set()

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
        historical_adjustment: HistoricalPlanAdjustmentResult | None = None,
        historical_context: HistoricalPlanningContext | None = None,
        request_id: str | None = None,
    ) -> StructuredExecutionResponse:
        """Generate, validate and optionally execute one structured plan."""
        authorization_scope = request_id or f"execution-request.{uuid4().hex}"
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

        active_plan = generation.plan
        validation = self._validator.validate(active_plan)
        if not validation.is_valid:
            session = self._record_pre_execution_failure(
                active_plan,
                "Plan validation failed: " + "; ".join(validation.errors),
                results={
                    "plan_status": "validation_failed",
                    "validation_errors": tuple(validation.errors),
                },
            )
            return StructuredExecutionResponse(
                handled=True,
                status="validation_failed",
                message=self._validation_failed_message(validation),
                plan=active_plan,
                validation_result=validation,
                error_code="INVALID_PLAN",
                error="; ".join(validation.errors),
                operational_report=self.get_execution_report(session.session_id),
            )

        if historical_context is None and self._execution_history_advisor is not None:
            historical_context = self._execution_history_advisor.build_planning_context(
                objective
            )
        if (
            historical_adjustment is None
            and historical_context is not None
            and self._historical_plan_adjuster is not None
        ):
            historical_adjustment = self._historical_plan_adjuster.adjust(
                HistoricalAdjustmentRequest(
                    plan=active_plan,
                    historical_context=historical_context,
                )
            )
            active_plan = historical_adjustment.selected_plan
            validation = historical_adjustment.final_validation

        strategy_selection = self._select_execution_strategy(
            active_plan,
            validation,
            historical_adjustment=historical_adjustment,
            historical_context=historical_context,
        )
        if strategy_selection is None:
            authorization = None
            confirmation_references: tuple[
                ExecutionConfirmationReference,
                ...,
            ] = ()
        else:
            confirmation_references = self._confirmation_references(
                active_plan,
                validation,
                strategy_selection,
            )
            granted_references = (
                tuple(
                    reference.granted_copy()
                    for reference in confirmation_references
                )
                if confirmation_granted
                else ()
            )
            authorization = self._authorize_execution(
                active_plan,
                validation,
                strategy_selection,
                required_confirmations=confirmation_references,
                granted_confirmations=granted_references,
                source=(
                    "StructuredExecutionCoordinator.handle:"
                    + authorization_scope
                ),
            )

        if (
            authorization is not None
            and authorization.decision
            is ExecutionAuthorizationDecision.MANUAL_REVIEW_PENDING
        ):
            session = self._start_waiting_confirmation_session(
                active_plan,
                strategy_selection,
                authorization,
            )
            return StructuredExecutionResponse(
                handled=True,
                status="strategy_blocked",
                message=authorization.reason,
                plan=active_plan,
                validation_result=validation,
                error_code="EXECUTION_STRATEGY_BLOCKED",
                operational_report=self.get_execution_report(session.session_id),
                strategy_selection=strategy_selection,
                authorization_result=authorization,
                historical_context=historical_context,
                historical_adjustment=historical_adjustment,
            )

        if (
            authorization is not None
            and authorization.decision
            in {
                ExecutionAuthorizationDecision.BLOCKED,
                ExecutionAuthorizationDecision.REJECTED,
                ExecutionAuthorizationDecision.ALREADY_DISPATCHED,
            }
        ):
            report = None
            if (
                authorization.decision
                is not ExecutionAuthorizationDecision.ALREADY_DISPATCHED
            ):
                session = self._record_pre_execution_failure(
                    active_plan,
                    authorization.reason,
                    strategy_selection=strategy_selection,
                    authorization_result=authorization,
                    results={
                        "plan_status": authorization.decision.value.lower(),
                        "authorization_error": authorization.reason,
                    },
                )
                report = self.get_execution_report(session.session_id)
            return StructuredExecutionResponse(
                handled=True,
                status=authorization.decision.value.lower(),
                message=authorization.reason,
                plan=active_plan,
                validation_result=validation,
                error_code=f"EXECUTION_AUTHORIZATION_{authorization.decision.value}",
                error=authorization.reason,
                operational_report=report,
                strategy_selection=strategy_selection,
                authorization_result=authorization,
                historical_context=historical_context,
                historical_adjustment=historical_adjustment,
            )

        if (
            (
                authorization is not None
                and authorization.decision
                is ExecutionAuthorizationDecision.CONFIRMATION_PENDING
            )
            or (
                authorization is None
                and validation.requires_confirmation
                and not confirmation_granted
            )
        ):
            session = self._start_waiting_confirmation_session(
                active_plan,
                strategy_selection,
                authorization,
            )
            token = self._store_pending_plan(
                objective,
                active_plan,
                validation,
                session.session_id,
                strategy_selection,
                authorization,
                confirmation_references,
                authorization_scope,
            )
            return StructuredExecutionResponse(
                handled=True,
                status="confirmation_required",
                message=self._confirmation_message(active_plan, validation, token),
                plan=active_plan,
                validation_result=validation,
                requires_confirmation=True,
                confirmation_token=token,
                error_code="CONFIRMATION_REQUIRED",
                operational_report=self.get_execution_report(session.session_id),
                strategy_selection=strategy_selection,
                authorization_result=authorization,
                historical_context=historical_context,
                historical_adjustment=historical_adjustment,
            )

        if not execute_after_planning:
            return StructuredExecutionResponse(
                handled=True,
                status="planned",
                message=self._planned_message(active_plan, validation),
                plan=active_plan,
                validation_result=validation,
                requires_confirmation=False,
                strategy_selection=strategy_selection,
                authorization_result=authorization,
                historical_context=historical_context,
                historical_adjustment=historical_adjustment,
            )

        if authorization is None:
            execution = self._execute_plan_under_supervision(
                active_plan,
                validation,
                confirmation_granted=confirmation_granted,
                control=control,
                on_progress=on_execution_progress,
                strategy_selection=strategy_selection,
            )
            dispatch = None
        else:
            dispatch = self._execution_dispatcher.dispatch(
                authorization,
                lambda plan, validated, selected: (
                    self._execute_plan_under_supervision(
                        plan,
                        validated,
                        confirmation_granted=confirmation_granted,
                        control=control,
                        on_progress=on_execution_progress,
                        strategy_selection=selected,
                        authorization_result=authorization,
                    )
                ),
            )
            if dispatch.execution_result is None:
                return StructuredExecutionResponse(
                    handled=True,
                    status="dispatch_failed",
                    message=dispatch.reason,
                    plan=active_plan,
                    validation_result=validation,
                    error_code=dispatch.status.value,
                    error=dispatch.error or dispatch.reason,
                    strategy_selection=strategy_selection,
                    authorization_result=authorization,
                    dispatch_result=dispatch,
                    historical_context=historical_context,
                    historical_adjustment=historical_adjustment,
                )
            execution = dispatch.execution_result
            self._record_dispatch_authorization(authorization, dispatch)
        response = self._execution_response(
            active_plan,
            validation,
            execution,
        )
        persistence_error = self._sync_resumable_state(
            response,
            objective=objective,
            confirmation_granted=confirmation_granted,
        )
        if persistence_error is not None:
            return replace(
                response,
                error_code=persistence_error,
                strategy_selection=strategy_selection,
                authorization_result=authorization,
                dispatch_result=dispatch,
                historical_context=historical_context,
                historical_adjustment=historical_adjustment,
            )
        return replace(
            response,
            strategy_selection=strategy_selection,
            authorization_result=authorization,
            dispatch_result=dispatch,
            historical_context=historical_context,
            historical_adjustment=historical_adjustment,
        )

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

        if pending_record.correction_decision is not None:
            return self._confirm_objective_correction(
                confirmation_token,
                pending_record,
                control=control,
                on_execution_progress=on_execution_progress,
            )

        if pending.strategy_selection is None:
            self._discard_pending(confirmation_token)
            execution = self._execute_plan_under_supervision(
                pending.plan,
                pending.validation_result,
                confirmation_granted=True,
                control=control,
                on_progress=on_execution_progress,
                session_id=pending_record.session_id,
                strategy_selection=None,
            )
            authorization = None
            dispatch = None
        else:
            authorization = self._authorize_execution(
                pending.plan,
                pending.validation_result,
                pending.strategy_selection,
                required_confirmations=pending.confirmation_references,
                granted_confirmations=tuple(
                    reference.granted_copy()
                    for reference in pending.confirmation_references
                ),
                session_state="waiting_confirmation",
                source=(
                    "StructuredExecutionCoordinator.handle:"
                    + (
                        pending.authorization_scope
                        or pending.confirmation_token
                    )
                ),
            )
            if (
                authorization.decision
                is not ExecutionAuthorizationDecision.AUTHORIZED
            ):
                return StructuredExecutionResponse(
                    handled=True,
                    status=authorization.decision.value.lower(),
                    message=authorization.reason,
                    plan=pending.plan,
                    validation_result=pending.validation_result,
                    requires_confirmation=True,
                    confirmation_token=confirmation_token,
                    error_code=(
                        f"EXECUTION_AUTHORIZATION_{authorization.decision.value}"
                    ),
                    error=authorization.reason,
                    strategy_selection=pending.strategy_selection,
                    authorization_result=authorization,
                )
            dispatch = self._execution_dispatcher.dispatch(
                authorization,
                lambda plan, validated, selected: (
                    self._execute_plan_under_supervision(
                        plan,
                        validated,
                        confirmation_granted=True,
                        control=control,
                        on_progress=on_execution_progress,
                        session_id=pending_record.session_id,
                        strategy_selection=selected,
                        authorization_result=authorization,
                    )
                ),
            )
            if dispatch.execution_result is None:
                return StructuredExecutionResponse(
                    handled=True,
                    status="dispatch_failed",
                    message=dispatch.reason,
                    plan=pending.plan,
                    validation_result=pending.validation_result,
                    requires_confirmation=True,
                    confirmation_token=confirmation_token,
                    error_code=dispatch.status.value,
                    error=dispatch.error or dispatch.reason,
                    strategy_selection=pending.strategy_selection,
                    authorization_result=authorization,
                    dispatch_result=dispatch,
                )
            self._discard_pending(confirmation_token)
            execution = dispatch.execution_result
            self._record_dispatch_authorization(authorization, dispatch)

        response = self._execution_response(
            pending.plan,
            pending.validation_result,
            execution,
            confirmation_token=confirmation_token,
        )
        if response.objective_correction is not None:
            return response
        persistence_error = self._sync_resumable_state(
            response,
            objective=pending.objective,
            confirmation_granted=True,
        )
        if persistence_error is not None:
            return replace(
                response,
                error_code=persistence_error,
                strategy_selection=pending.strategy_selection,
                authorization_result=authorization,
                dispatch_result=dispatch,
            )
        return replace(
            response,
            strategy_selection=pending.strategy_selection,
            authorization_result=authorization,
            dispatch_result=dispatch,
        )

    def _confirm_objective_correction(
        self,
        confirmation_token: str,
        pending_record: _PendingPlan,
        *,
        control: ExecutionControl | None,
        on_execution_progress: Callable[[ExecutionProgress], None] | None,
    ) -> StructuredExecutionResponse:
        """Authorize, dispatch and reverify one already prepared correction."""

        pending = pending_record.execution
        decision = pending_record.correction_decision
        original_plan = pending_record.original_plan
        original_execution = pending_record.original_execution
        original_session_id = pending_record.original_session_id
        if (
            decision is None
            or original_plan is None
            or original_execution is None
            or original_session_id is None
            or pending.strategy_selection is None
        ):
            self._discard_pending(confirmation_token)
            return StructuredExecutionResponse(
                handled=True,
                status="correction_rejected",
                message="La corrección pendiente perdió su trazabilidad.",
                error_code="CORRECTION_CONTEXT_MISSING",
            )

        authorization = self._authorize_execution(
            pending.plan,
            pending.validation_result,
            pending.strategy_selection,
            required_confirmations=pending.confirmation_references,
            granted_confirmations=tuple(
                reference.granted_copy()
                for reference in pending.confirmation_references
            ),
            session_state="waiting_confirmation",
            source=(
                "StructuredExecutionCoordinator.objective_correction:"
                + decision.request.correction_request_id
            ),
        )
        if authorization.decision is not ExecutionAuthorizationDecision.AUTHORIZED:
            self._execution_supervisor.record_objective_correction(
                original_session_id,
                {
                    **self._correction_snapshot(
                        decision,
                        status=CorrectionLifecycleStatus.REJECTED,
                    ),
                    "authorization": authorization.persisted_snapshot(),
                    "rejection_reason": authorization.reason,
                },
                event_type="objective_correction_authorization_rejected",
            )
            return StructuredExecutionResponse(
                handled=True,
                status=authorization.decision.value.lower(),
                message=authorization.reason,
                plan=original_plan,
                execution_result=original_execution,
                requires_confirmation=True,
                confirmation_token=confirmation_token,
                error_code=f"CORRECTION_AUTHORIZATION_{authorization.decision.value}",
                error=authorization.reason,
                strategy_selection=pending.strategy_selection,
                authorization_result=authorization,
                objective_correction=decision,
            )

        dispatch = self._execution_dispatcher.dispatch(
            authorization,
            lambda plan, validated, selected: self._execute_plan_under_supervision(
                plan,
                validated,
                confirmation_granted=True,
                control=control,
                on_progress=on_execution_progress,
                session_id=pending_record.session_id,
                strategy_selection=selected,
                authorization_result=authorization,
                execution_context=ExecutionContext(
                    initial_variables=decision.expected_context
                ),
            ),
        )
        if dispatch.execution_result is None:
            self._execution_supervisor.record_objective_correction(
                original_session_id,
                {
                    **self._correction_snapshot(
                        decision,
                        status=CorrectionLifecycleStatus.FAILED,
                    ),
                    "authorization": authorization.persisted_snapshot(),
                    "dispatch": dispatch.persisted_snapshot(),
                    "failure_reason": dispatch.error or dispatch.reason,
                },
                event_type="objective_correction_dispatch_failed",
            )
            return StructuredExecutionResponse(
                handled=True,
                status="correction_dispatch_failed",
                message=dispatch.reason,
                plan=original_plan,
                execution_result=original_execution,
                requires_confirmation=False,
                confirmation_token=confirmation_token,
                error_code=dispatch.status.value,
                error=dispatch.error or dispatch.reason,
                strategy_selection=pending.strategy_selection,
                authorization_result=authorization,
                dispatch_result=dispatch,
                objective_correction=decision,
            )

        correction_execution = dispatch.execution_result
        self._record_dispatch_authorization(authorization, dispatch)
        if correction_execution.interrupted and correction_execution.resumable:
            state = self._build_resumable_state(
                objective=original_plan.goal,
                plan=pending.plan,
                validation=pending.validation_result,
                execution=correction_execution,
                confirmation_granted=True,
            )
            state = replace(
                state,
                metadata={
                    **state.metadata,
                    "objective_correction_resume": {
                        "original_session_id": original_session_id,
                        "correction_request_id": (
                            decision.request.correction_request_id
                        ),
                        "fragment_signature": decision.fragment_signature,
                        "affected_criteria": list(
                            decision.affected_criterion_ids
                        ),
                    },
                },
            )
            self._resumable_execution = state
            persistence_error = self._save_resumable_state(state)
            self._discard_pending(confirmation_token)
            self._execution_supervisor.record_objective_correction(
                original_session_id,
                {
                    **self._correction_snapshot(
                        decision,
                        status=CorrectionLifecycleStatus.RUNNING,
                    ),
                    "authorization": authorization.persisted_snapshot(),
                    "dispatch": dispatch.persisted_snapshot(),
                    "correction_result": "interrupted",
                    "completed_corrective_steps": list(
                        correction_execution.completed_steps
                    ),
                },
                event_type="objective_correction_interrupted",
            )
            return StructuredExecutionResponse(
                handled=True,
                status="correction_interrupted",
                message=(
                    "La corrección se interrumpió de forma reanudable; "
                    "los pasos correctivos completados no se repetirán."
                ),
                plan=original_plan,
                execution_result=correction_execution,
                resumable_state=state,
                error_code=persistence_error
                or correction_execution.error_code,
                error=correction_execution.error,
                operational_report=self.get_execution_report(original_session_id),
                strategy_selection=pending.strategy_selection,
                authorization_result=authorization,
                dispatch_result=dispatch,
                objective_correction=decision,
            )

        self._discard_pending(confirmation_token)
        if not correction_execution.success:
            self._execution_supervisor.record_objective_correction(
                original_session_id,
                {
                    **self._correction_snapshot(
                        decision,
                        status=CorrectionLifecycleStatus.FAILED,
                    ),
                    "authorization": authorization.persisted_snapshot(),
                    "dispatch": dispatch.persisted_snapshot(),
                    "correction_result": correction_execution.plan_status,
                    "failure_reason": (
                        correction_execution.error
                        or correction_execution.failure_reason
                        or correction_execution.error_code
                    ),
                },
                event_type="objective_correction_failed",
            )
            return StructuredExecutionResponse(
                handled=True,
                status="correction_failed",
                message=(
                    "La corrección falló y no se inició otro ciclo. "
                    + self._failed_message(correction_execution)
                ),
                plan=original_plan,
                execution_result=correction_execution,
                error_code=correction_execution.error_code,
                error=correction_execution.error,
                operational_report=self.get_execution_report(original_session_id),
                strategy_selection=pending.strategy_selection,
                authorization_result=authorization,
                dispatch_result=dispatch,
                objective_correction=decision,
            )

        merged = merge_corrective_execution(
            original_plan,
            original_execution,
            correction_execution,
            decision,
        )
        verification = GoalVerifier().verify(original_plan, merged)
        verification = replace(
            verification,
            session_id=original_session_id,
        )
        merged = replace(merged, goal_verification_result=verification)
        lifecycle_status = (
            CorrectionLifecycleStatus.VERIFIED_AFTER_CORRECTION
            if verification.verification_status is GoalVerificationStatus.VERIFIED
            else CorrectionLifecycleStatus.LIMIT_REACHED
        )
        self._execution_supervisor.record_objective_correction(
            original_session_id,
            {
                **self._correction_snapshot(
                    decision,
                    status=lifecycle_status,
                ),
                "authorization": authorization.persisted_snapshot(),
                "dispatch": dispatch.persisted_snapshot(),
                "correction_result": correction_execution.plan_status,
                "final_verification": goal_verification_result_to_dict(verification),
                "final_verification_status": verification.verification_status.value,
                "confirmation": "granted",
            },
            event_type="objective_correction_reverified",
        )
        message = (
            "El objetivo no se verificó en el primer intento. Atlas corrigió "
            "el recurso, lo volvió a comprobar y ahora el objetivo está verificado."
            if verification.verification_status is GoalVerificationStatus.VERIFIED
            else (
                "La corrección terminó, pero el objetivo sigue sin verificarse. "
                "Se alcanzó el límite de un ciclo correctivo."
            )
        )
        return StructuredExecutionResponse(
            handled=True,
            status=(
                "verified_after_correction"
                if verification.verification_status is GoalVerificationStatus.VERIFIED
                else "correction_limit_reached"
            ),
            message=message,
            plan=original_plan,
            execution_result=merged,
            requires_confirmation=False,
            confirmation_token=confirmation_token,
            operational_report=self.get_execution_report(original_session_id),
            strategy_selection=pending.strategy_selection,
            authorization_result=authorization,
            dispatch_result=dispatch,
            objective_correction=decision,
            corrected_verification=verification,
        )

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

    def get_execution_report(
        self,
        session_id: str,
    ) -> OperationalExecutionReport:
        """Build a user-facing report from current or restored session data."""
        session = self._execution_supervisor.get_session(session_id)
        summary = self._execution_supervisor.get_summary(session_id)
        return self._execution_report_generator.generate(session, summary)

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
            optional=step.optional,
            priority=step.priority,
            urgency=step.urgency,
            estimated_cost=step.estimated_cost,
            estimated_duration_seconds=step.estimated_duration_seconds,
            criticality=step.criticality,
            deadline=step.deadline,
            resource_requirements=step.resource_requirements,
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
        if isinstance(
            state.metadata.get("objective_correction_resume"),
            Mapping,
        ):
            response = self._finalize_resumed_objective_correction(
                state,
                execution,
            )
        else:
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

    def _finalize_resumed_objective_correction(
        self,
        state: ResumableExecutionState,
        correction_execution: PlanExecutionResult,
    ) -> StructuredExecutionResponse:
        """Finish a persisted correction without repeating completed steps."""

        marker = state.metadata.get("objective_correction_resume")
        if not isinstance(marker, Mapping):
            raise ValueError("objective correction resume marker is required.")
        original_session_id = marker.get("original_session_id")
        if not isinstance(original_session_id, str):
            return StructuredExecutionResponse(
                handled=True,
                status="correction_resume_failed",
                message="La corrección reanudada perdió la sesión original.",
                error_code="CORRECTION_ORIGINAL_SESSION_MISSING",
                execution_result=correction_execution,
            )
        try:
            original_session = self._execution_supervisor.get_session(
                original_session_id
            )
        except Exception as error:
            return StructuredExecutionResponse(
                handled=True,
                status="correction_resume_failed",
                message="No se pudo restaurar la sesión original de la corrección.",
                error_code="CORRECTION_ORIGINAL_SESSION_NOT_FOUND",
                error=str(error),
                execution_result=correction_execution,
            )
        original_execution = self._execution_from_supervisor_session(
            original_session
        )
        verification = original_execution.goal_verification_result
        corrective_builder = getattr(
            self._execution_replanner,
            "corrective_fragment",
            None,
        )
        if verification is None or not callable(corrective_builder):
            return StructuredExecutionResponse(
                handled=True,
                status="correction_resume_failed",
                message="No existe evidencia original suficiente para reverificar.",
                error_code="CORRECTION_EVIDENCE_MISSING",
                execution_result=correction_execution,
            )
        decision = corrective_builder(
            original_session.original_plan,
            original_execution,
            verification,
            session_id=original_session_id,
            previous_attempts=0,
            policy=self._objective_correction_policy,
        )
        request_id = marker.get("correction_request_id")
        if isinstance(request_id, str):
            decision = replace(
                decision,
                request=replace(
                    decision.request,
                    correction_request_id=request_id,
                ),
            )

        if correction_execution.interrupted and correction_execution.resumable:
            self._execution_supervisor.record_objective_correction(
                original_session_id,
                {
                    **self._correction_snapshot(
                        decision,
                        status=CorrectionLifecycleStatus.RUNNING,
                    ),
                    "correction_result": "interrupted",
                    "completed_corrective_steps": list(
                        correction_execution.completed_steps
                    ),
                },
                event_type="objective_correction_reinterrupted",
            )
            return StructuredExecutionResponse(
                handled=True,
                status="correction_interrupted",
                message="La corrección continúa siendo reanudable.",
                plan=state.original_plan,
                validation_result=state.validation_result,
                execution_result=correction_execution,
                resumable_state=state,
                objective_correction=decision,
            )
        if not correction_execution.success:
            self._execution_supervisor.record_objective_correction(
                original_session_id,
                {
                    **self._correction_snapshot(
                        decision,
                        status=CorrectionLifecycleStatus.FAILED,
                    ),
                    "correction_result": correction_execution.plan_status,
                    "failure_reason": (
                        correction_execution.error
                        or correction_execution.failure_reason
                        or correction_execution.error_code
                    ),
                },
                event_type="objective_correction_resume_failed",
            )
            return StructuredExecutionResponse(
                handled=True,
                status="correction_failed",
                message="La corrección reanudada falló; no se inició otro ciclo.",
                plan=original_session.original_plan,
                execution_result=correction_execution,
                error_code=correction_execution.error_code,
                error=correction_execution.error,
                operational_report=self.get_execution_report(original_session_id),
                objective_correction=decision,
            )

        merged = merge_corrective_execution(
            original_session.original_plan,
            original_execution,
            correction_execution,
            decision,
        )
        final_verification = GoalVerifier().verify(
            original_session.original_plan,
            merged,
        )
        final_verification = replace(
            final_verification,
            session_id=original_session_id,
        )
        merged = replace(
            merged,
            goal_verification_result=final_verification,
        )
        lifecycle = (
            CorrectionLifecycleStatus.VERIFIED_AFTER_CORRECTION
            if final_verification.verification_status
            is GoalVerificationStatus.VERIFIED
            else CorrectionLifecycleStatus.LIMIT_REACHED
        )
        self._execution_supervisor.record_objective_correction(
            original_session_id,
            {
                **self._correction_snapshot(decision, status=lifecycle),
                "correction_result": correction_execution.plan_status,
                "final_verification": goal_verification_result_to_dict(
                    final_verification
                ),
                "final_verification_status": (
                    final_verification.verification_status.value
                ),
                "confirmation": "granted",
                "resumed": True,
            },
            event_type="objective_correction_resumed_and_reverified",
        )
        return StructuredExecutionResponse(
            handled=True,
            status=(
                "verified_after_correction"
                if final_verification.verification_status
                is GoalVerificationStatus.VERIFIED
                else "correction_limit_reached"
            ),
            message=(
                "La corrección se reanudó sin repetir pasos completados y "
                "el objetivo quedó verificado."
                if final_verification.verification_status
                is GoalVerificationStatus.VERIFIED
                else (
                    "La corrección se reanudó, pero el objetivo sigue sin "
                    "verificarse y se alcanzó el límite."
                )
            ),
            plan=original_session.original_plan,
            execution_result=merged,
            operational_report=self.get_execution_report(original_session_id),
            objective_correction=decision,
            corrected_verification=final_verification,
        )

    @staticmethod
    def _execution_from_supervisor_session(
        session: ExecutionSession,
    ) -> PlanExecutionResult:
        outputs = session.results.get("step_outputs")
        tools = session.results.get("step_tools")
        completed = tuple(session.results.get("completed_steps", ()))
        step_results = [
            StepExecutionResult(
                step_id=step_id,
                status=StepExecutionStatus.COMPLETED.value,
                success=True,
                tool_name=(
                    tools.get(step_id)
                    if isinstance(tools, Mapping)
                    and isinstance(tools.get(step_id), str)
                    else None
                ),
                output=output,
            )
            for step_id, output in (
                outputs.items() if isinstance(outputs, Mapping) else ()
            )
        ]
        raw_verification = session.results.get("goal_verification")
        verification = (
            goal_verification_result_from_dict(raw_verification)
            if isinstance(raw_verification, Mapping)
            else None
        )
        return PlanExecutionResult(
            plan_status=PlanExecutionStatus.COMPLETED.value,
            success=True,
            completed=True,
            completed_steps=list(completed),
            step_results=step_results,
            metadata={
                "execution_session_id": session.session_id,
                "confirmation_granted": True,
            },
            goal_verification_result=verification,
        )

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
        strategy_selection: ExecutionStrategySelectionResult | None = None,
        authorization_result: ExecutionAuthorizationResult | None = None,
        confirmation_references: tuple[
            ExecutionConfirmationReference,
            ...,
        ] = (),
        authorization_scope: str | None = None,
        correction_decision: ObjectiveCorrectionDecision | None = None,
        original_plan: ExecutionPlan | None = None,
        original_execution: PlanExecutionResult | None = None,
        original_session_id: str | None = None,
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
                strategy_selection=strategy_selection,
                authorization_result=authorization_result,
                confirmation_references=confirmation_references,
                authorization_scope=authorization_scope,
            ),
            session_id=session_id,
            correction_decision=correction_decision,
            original_plan=original_plan,
            original_execution=original_execution,
            original_session_id=original_session_id,
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

    def _select_execution_strategy(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
        *,
        historical_adjustment: HistoricalPlanAdjustmentResult | None = None,
        historical_context: HistoricalPlanningContext | None = None,
    ) -> ExecutionStrategySelectionResult | None:
        """Resolve operational controls outside Planner, Executor and Supervisor."""
        if self._execution_strategy_selector is None:
            return None
        request = build_strategy_request(
            plan,
            validation,
            historical_adjustment=historical_adjustment,
            historical_context=historical_context,
            supervisor_available=self._execution_supervisor is not None,
            replanner_available=self._execution_replanner is not None,
            confirmation_available=True,
            safety_policy=self._execution_safety_policy,
        )
        return self._execution_strategy_selector.select(request)

    def _confirmation_references(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
        strategy_selection: ExecutionStrategySelectionResult,
    ) -> tuple[ExecutionConfirmationReference, ...]:
        if not strategy_selection.strategy.configuration.requires_confirmation:
            return ()
        token = validation.plan_signature or plan_signature(plan)
        return build_confirmation_references(
            plan,
            strategy_selection,
            confirmation_id=token,
        )

    def _authorize_execution(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
        strategy_selection: ExecutionStrategySelectionResult,
        *,
        required_confirmations: tuple[
            ExecutionConfirmationReference,
            ...,
        ] = (),
        granted_confirmations: tuple[
            ExecutionConfirmationReference,
            ...,
        ] = (),
        session_state: str | None = None,
        source: str,
    ) -> ExecutionAuthorizationResult:
        return self._execution_authorization_gate.authorize(
            ExecutionAuthorizationRequest(
                plan=plan,
                plan_validation=validation,
                strategy_selection=strategy_selection,
                plan_state=plan.status,
                session_state=session_state,
                required_confirmations=required_confirmations,
                granted_confirmations=granted_confirmations,
                protected_step_ids=(
                    strategy_selection.strategy.configuration.confirmation_step_ids
                ),
                required_capabilities=plan.required_tools,
                executor_available=self._executor is not None,
                supervisor_available=self._execution_supervisor is not None,
                replanner_available=self._execution_replanner is not None,
                source=source,
                safety_policy=self._execution_safety_policy,
            )
        )

    def _start_waiting_confirmation_session(
        self,
        plan: ExecutionPlan,
        strategy_selection: ExecutionStrategySelectionResult | None = None,
        authorization_result: ExecutionAuthorizationResult | None = None,
    ) -> ExecutionSession:
        session = self._execution_supervisor.start(
            plan,
            execution_strategy=(
                None
                if strategy_selection is None
                else strategy_selection.persisted_snapshot()
            ),
            execution_authorization=(
                None
                if authorization_result is None
                else authorization_result.persisted_snapshot()
            ),
        )
        session = self._execution_supervisor.mark_running(session.session_id)
        return self._execution_supervisor.mark_waiting_confirmation(session.session_id)

    def _record_pre_execution_failure(
        self,
        plan: ExecutionPlan,
        error: object,
        *,
        strategy_selection: ExecutionStrategySelectionResult | None = None,
        authorization_result: ExecutionAuthorizationResult | None = None,
        results: Mapping[str, object] | None = None,
    ) -> ExecutionSession:
        """Persist a controlled failure that occurs before Executor delivery."""
        session = self._execution_supervisor.start(
            plan,
            execution_strategy=(
                None
                if strategy_selection is None
                else strategy_selection.persisted_snapshot()
            ),
            execution_authorization=(
                None
                if authorization_result is None
                else authorization_result.persisted_snapshot()
            ),
        )
        self._execution_supervisor.mark_running(session.session_id)
        return self._execution_supervisor.mark_failed(
            session.session_id,
            error,
            results=results,
        )

    def _execute_plan_under_supervision(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
        *,
        confirmation_granted: bool,
        control: ExecutionControl | None = None,
        on_progress: Callable[[ExecutionProgress], None] | None = None,
        session_id: str | None = None,
        strategy_selection: ExecutionStrategySelectionResult | None = None,
        authorization_result: ExecutionAuthorizationResult | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> PlanExecutionResult:
        session = (
            self._execution_supervisor.get_session(session_id)
            if session_id is not None
            else self._execution_supervisor.start(
                plan,
                execution_strategy=(
                    None
                    if strategy_selection is None
                    else strategy_selection.persisted_snapshot()
                ),
                execution_authorization=(
                    None
                    if authorization_result is None
                    else authorization_result.persisted_snapshot()
                ),
            )
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
                        operational_config=(
                            None
                            if strategy_selection is None
                            else strategy_selection.strategy.configuration
                        ),
                        execution_context=execution_context,
                    )
                if execution.goal_verification_result is None:
                    execution = replace(
                        execution,
                        metadata={
                            **execution.metadata,
                            "plan_signature": active_validation.plan_signature,
                            "confirmation_granted": confirmation_granted,
                        },
                    )
                    execution = replace(
                        execution,
                        goal_verification_result=GoalVerifier().verify(
                            active_plan,
                            execution,
                        ),
                    )
                execution = self._with_execution_session_id(
                    execution,
                    session.session_id,
                )
                self._record_step_states_from_execution(
                    session.session_id,
                    active_plan,
                    execution,
                )
                if (
                    self._execution_supervisor.get_session(session.session_id).state
                    is ExecutionState.CANCELLED
                ):
                    return self._with_execution_session_id(
                        execution,
                        session.session_id,
                    )
                strategy_allows_replanning = (
                    strategy_selection is None
                    or strategy_selection.strategy.configuration.allow_replanning
                )
                if (
                    self._execution_can_finish_without_replan(execution)
                    or not strategy_allows_replanning
                ):
                    self._finalize_supervised_session(session.session_id, execution)
                    return self._with_execution_session_id(
                        execution,
                        session.session_id,
                    )

                replanned = self._attempt_replan(
                    session.session_id,
                    active_plan,
                    execution,
                )
                if replanned is None:
                    if (
                        self._execution_supervisor.get_session(session.session_id).state
                        is ExecutionState.COMPLETED
                    ):
                        omitted_step = execution.current_step or execution.failed_step
                        return self._with_execution_session_id(
                            replace(
                                execution,
                                plan_status=PlanExecutionStatus.COMPLETED.value,
                                success=True,
                                completed=True,
                                failed=False,
                                failed_step=None,
                                failed_steps=[],
                                skipped_steps=list(
                                    dict.fromkeys(
                                        execution.skipped_steps
                                        + ([omitted_step] if omitted_step else [])
                                    )
                                ),
                                error=None,
                                error_code=None,
                            ),
                            session.session_id,
                        )
                    return self._with_execution_session_id(
                        execution,
                        session.session_id,
                    )
                active_plan, active_validation = replanned
                if (
                    self._execution_supervisor.get_session(session.session_id).state
                    is not ExecutionState.RUNNING
                ):
                    self._execution_supervisor.mark_running(session.session_id)
        except Exception as error:
            self._execution_supervisor.mark_failed(session.session_id, error)
            raise

    def _record_dispatch_authorization(
        self,
        authorization: ExecutionAuthorizationResult,
        dispatch: ExecutionDispatchResult,
    ) -> None:
        if dispatch.session_id is None:
            return
        self._execution_supervisor.record_execution_authorization(
            dispatch.session_id,
            authorization_with_dispatch(authorization, dispatch),
        )

    def _can_use_concurrent_execution(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
    ) -> bool:
        return (
            self._concurrent_step_executor is not None
            and (
                (
                    self._concurrency_policy.enabled
                    and self._concurrency_policy.max_concurrency > 1
                )
                or self._resource_policy.enabled
            )
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
            try:
                resource_reservations = self._select_resources_for_steps(
                    session_id,
                    selected_steps,
                )
            except ResourceSelectionError as error:
                pending = list(resolution.pending_step_ids)
                return PlanExecutionResult(
                    plan_status=PlanExecutionStatus.FAILED.value,
                    success=False,
                    completed_steps=completed,
                    failed_step=error.step_id,
                    failed_steps=tuple([error.step_id] if error.step_id else ()),
                    blocked_steps=blocked,
                    pending_steps=pending,
                    step_results=step_results,
                    error=error.message,
                    failed=True,
                    current_step=error.step_id,
                    error_code=error.error_code,
                    started_at=started_at,
                    finished_at=_utc_iso(),
                    metadata=self._concurrent_metadata(
                        step_results,
                        last_batch=None,
                        resource_selection_failure=error.message,
                        rejected_candidate_ids=error.candidates_considered,
                    ),
                )
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
            self._confirm_resource_reservations(session_id, resource_reservations)
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

    def _select_resources_for_steps(
        self,
        session_id: str,
        selected_steps: list[ExecutionStep],
    ) -> list[BudgetReservation]:
        if not self._resource_policy.enabled:
            return []
        reservations: list[BudgetReservation] = []
        selected_resource_ids: list[str | None] = []
        try:
            for step in selected_steps:
                budget_snapshot = (
                    self._budget_manager.snapshot()
                    if self._budget_manager is not None
                    else None
                )
                decision = self._resource_optimizer.select(
                    step_id=step.id,
                    requirements=step.resource_requirements,
                    catalog=self._resource_catalog,
                    budget_usage=budget_snapshot,
                )
                selected_resource_ids.append(decision.selected_resource_id)
                if (
                    self._budget_manager is not None
                    and decision.selected_resource_id is not None
                ):
                    reservation = self._budget_manager.reserve(
                        step_id=step.id,
                        resource_id=decision.selected_resource_id,
                        estimated_cost=decision.estimated_cost,
                        estimated_tokens=decision.estimated_tokens,
                    )
                    reservations.append(reservation)
                    self._execution_supervisor.record_budget_usage(
                        session_id,
                        self._budget_manager.snapshot(),
                        event_type="execution_budget_reserved",
                    )
                self._execution_supervisor.record_resource_decision(
                    session_id,
                    decision,
                    budget_usage=(
                        self._budget_manager.snapshot()
                        if self._budget_manager is not None
                        else decision.budget_snapshot
                    ),
                )
        except ResourceSelectionError as error:
            self._execution_supervisor.record_resource_decision(
                session_id,
                ResourceSelectionDecision(
                    step_id=error.step_id or "",
                    selected_resource_id=None,
                    provider_id=None,
                    scores=(),
                    rejected_candidate_ids=error.candidates_considered,
                    reason=(
                        ResourceSelectionReason.BUDGET_EXCEEDED
                        if isinstance(error, ExecutionBudgetExceededError)
                        else ResourceSelectionReason.NO_COMPATIBLE_RESOURCE
                    ),
                    optimization_goal=self._resource_policy.optimization_goal,
                    budget_snapshot=(
                        self._budget_manager.snapshot()
                        if self._budget_manager is not None
                        else None
                    ),
                ),
                budget_usage=(
                    self._budget_manager.snapshot()
                    if self._budget_manager is not None
                    else None
                ),
            )
            self._release_resource_reservations(reservations)
            for resource_id in selected_resource_ids:
                self._resource_optimizer.release(resource_id)
            raise
        return reservations

    def _confirm_resource_reservations(
        self,
        session_id: str,
        reservations: list[BudgetReservation],
    ) -> None:
        if self._budget_manager is None:
            for reservation in reservations:
                self._resource_optimizer.release(reservation.resource_id)
            return
        for reservation in reservations:
            usage = self._budget_manager.confirm_consumption(
                reservation.reservation_id,
            )
            self._resource_optimizer.release(reservation.resource_id)
            self._execution_supervisor.record_budget_usage(
                session_id,
                usage,
            )

    def _release_resource_reservations(
        self,
        reservations: list[BudgetReservation],
    ) -> None:
        if self._budget_manager is None:
            return
        for reservation in reservations:
            self._budget_manager.release(reservation.reservation_id)

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
        resource_selection_failure: str | None = None,
        rejected_candidate_ids: tuple[str, ...] = (),
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
        if resource_selection_failure is not None:
            metadata["resource_selection_failure"] = resource_selection_failure
            metadata["rejected_candidate_ids"] = tuple(rejected_candidate_ids)
            metadata["optimization_goal"] = self._resource_policy.optimization_goal.value
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
            attempt_number = step_result.metadata.get("attempt_number")
            max_attempts = step_result.metadata.get("max_attempts")
            retry_history = step_result.metadata.get("retry_history")
            if (
                isinstance(attempt_number, int)
                and not isinstance(attempt_number, bool)
                and attempt_number > 1
                and isinstance(max_attempts, int)
                and not isinstance(max_attempts, bool)
                and max_attempts >= attempt_number
            ):
                retry_error = None
                if isinstance(retry_history, (list, tuple)) and retry_history:
                    last_retry = retry_history[-1]
                    if isinstance(last_retry, dict):
                        retry_error = last_retry.get("error") or last_retry.get("error_code")
                self._execution_supervisor.mark_step_retrying(
                    session_id,
                    step_result.step_id,
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                    dependency_ids=dependency_ids,
                    error=retry_error,
                )
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
            elif step_result.status == "skipped":
                self._execution_supervisor.mark_step_skipped(
                    session_id,
                    step_result.step_id,
                    dependency_ids=dependency_ids,
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
        failed_step = next(
            (step for step in failed_plan.ordered_steps if step.id == current_step),
            None,
        )
        classification = self._replan_policy.classify(
            error_code=execution.error_code,
            critical=bool(failed_step is not None and failed_step.criticality > 0),
            optional=bool(failed_step is not None and failed_step.optional),
        )
        if classification is ReplanFailureClassification.OPTIONAL:
            return self._continue_after_optional_failure(
                session_id,
                failed_plan,
                execution,
                failed_step,
            )
        results = self._supervisor_results(execution)
        error = (
            execution.error
            or execution.failure_reason
            or execution.error_code
            or execution.plan_status
        )
        if (
            classification is ReplanFailureClassification.CRITICAL
            and failed_step is not None
            and failed_step.criticality > 0
        ):
            self._execution_supervisor.record_replan_event(
                session_id,
                "replan_rejected",
                attempt_number=session.replan_count + 1,
                failed_step=current_step,
                reason=ReplanReason.NON_RECOVERABLE_FAILURE.value,
                error="critical failure cannot be replanned",
                details={"classification": classification.value},
            )
            self._execution_supervisor.mark_cancelled(
                session_id,
                error=error,
                current_step=current_step,
                results=results,
            )
            return None
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
            classification=classification,
        )
        self._execution_supervisor.record_replan_event(
            session_id,
            "replan_requested",
            attempt_number=attempt_number,
            failed_step=request.failed_step,
            reason="execution_failed",
            details={
                "plan_id": request.plan_id or "",
                "attempts_performed": request.attempts_performed,
                "classification": request.classification.value,
            },
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
            event_type = (
                "replan_no_safe_alternative"
                if policy_decision.status is ReplanResultStatus.NO_SAFE_ALTERNATIVE
                else "replan_rejected"
            )
            self._execution_supervisor.record_replan_event(
                session_id,
                event_type,
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

        fragment = replan_result.revised_plan
        fragment_error = self._replacement_fragment_error(fragment, request)
        if fragment_error is not None:
            self._reject_replan_validation(
                session_id,
                request,
                fragment_error,
                results,
            )
            return None
        fragment_validation = self._validator.validate(fragment)
        if not fragment_validation.is_valid:
            self._reject_replan_validation(
                session_id,
                request,
                "; ".join(fragment_validation.errors),
                results,
            )
            return None

        revised_plan = self._compose_recovery_plan(
            failed_plan,
            fragment,
            request,
        )
        revised_validation = self._validator.validate(revised_plan)
        if not revised_validation.is_valid:
            self._reject_replan_validation(
                session_id,
                request,
                "; ".join(revised_validation.errors),
                results,
            )
            return None
        attempted_signatures = {
            plan_signature(failed_plan),
            *(
                plan_signature(record.revised_plan)
                for record in failed.replan_history
            ),
        }
        if plan_signature(revised_plan) in attempted_signatures:
            self._execution_supervisor.record_replan_event(
                session_id,
                "replan_no_safe_alternative",
                attempt_number=attempt_number,
                failed_step=request.failed_step,
                reason=ReplanReason.NO_SAFE_ALTERNATIVE.value,
                error="replacement would repeat an already attempted plan",
            )
            self._execution_supervisor.mark_failed(
                session_id,
                "no safe alternative plan",
                current_step=request.failed_step,
                results=results,
            )
            return None

        replacement_step_ids = tuple(step.id for step in fragment.ordered_steps)
        accepted_result = replace(replan_result, revised_plan=revised_plan)
        record = replan_record(
            request,
            accepted_result,
            previous_plan=failed_plan,
            replacement_step_ids=replacement_step_ids,
            validation_status=revised_validation.status,
        )
        self._execution_supervisor.record_replan(session_id, record)
        self._execution_supervisor.record_replan_event(
            session_id,
            "execution_resumed_with_revised_plan",
            attempt_number=attempt_number,
            failed_step=request.failed_step,
            reason=replan_result.reason.value,
            details={
                "replacement_step_ids": list(replacement_step_ids),
                "validation_status": revised_validation.status,
            },
        )
        return revised_plan, revised_validation

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

    def _continue_after_optional_failure(
        self,
        session_id: str,
        failed_plan: ExecutionPlan,
        execution: PlanExecutionResult,
        failed_step: ExecutionStep | None,
    ) -> tuple[ExecutionPlan, PlanValidationResult] | None:
        if failed_step is None:
            return None
        completed = set(execution.completed_steps)
        omitted = completed | {failed_step.id}
        remaining_steps = tuple(
            self._copy_step_without_completed_dependencies(step, omitted)
            for step in failed_plan.ordered_steps
            if step.id not in omitted
        )
        self._execution_supervisor.mark_step_skipped(
            session_id,
            failed_step.id,
            dependency_ids=tuple(failed_step.depends_on),
        )
        self._execution_supervisor.record_replan_event(
            session_id,
            "optional_step_omitted",
            attempt_number=1,
            failed_step=failed_step.id,
            reason=ReplanReason.OPTIONAL_STEP_OMITTED.value,
        )
        if not remaining_steps:
            self._execution_supervisor.mark_completed(
                session_id,
                current_step=failed_step.id,
                results=self._supervisor_results(execution),
            )
            return None
        continuation = self._execution_plan_from_steps(failed_plan, remaining_steps)
        validation = self._validator.validate(continuation)
        if not validation.is_valid:
            self._execution_supervisor.mark_failed(
                session_id,
                "optional-step continuation validation failed",
                current_step=failed_step.id,
                results=self._supervisor_results(execution),
            )
            return None
        return continuation, validation

    def _replacement_fragment_error(
        self,
        fragment: ExecutionPlan,
        request: ReplanRequest,
    ) -> str | None:
        if not fragment.ordered_steps:
            return "replacement fragment cannot be empty"
        if request.failed_step is None:
            return "replacement fragment requires a known failed step"
        if (
            request.pending_step_ids
            and fragment.ordered_steps[-1].id != request.failed_step
        ):
            return (
                "replacement fragment final step must reuse failed step ID "
                f"'{request.failed_step}'"
            )
        completed = set(request.completed_step_ids)
        duplicate_completed = completed.intersection(
            step.id for step in fragment.ordered_steps
        )
        if duplicate_completed:
            return "replacement fragment repeats completed steps"
        pending = set(request.pending_step_ids)
        duplicate_pending = pending.intersection(
            step.id for step in fragment.ordered_steps
        )
        if duplicate_pending:
            return "replacement fragment repeats pending steps"
        return None

    def _compose_recovery_plan(
        self,
        failed_plan: ExecutionPlan,
        fragment: ExecutionPlan,
        request: ReplanRequest,
    ) -> ExecutionPlan:
        if not request.pending_step_ids:
            return fragment
        completed = set(request.completed_step_ids)
        excluded = completed | {request.failed_step}
        pending_steps = tuple(
            self._copy_step_without_completed_dependencies(step, completed)
            for step in failed_plan.ordered_steps
            if step.id not in excluded
        )
        return self._execution_plan_from_steps(
            failed_plan,
            fragment.ordered_steps + pending_steps,
        )

    @staticmethod
    def _execution_plan_from_steps(
        base_plan: ExecutionPlan,
        steps: tuple[ExecutionStep, ...],
    ) -> ExecutionPlan:
        return ExecutionPlan(
            goal=base_plan.goal,
            ordered_steps=steps,
            estimated_steps=len(steps),
            required_tools=tuple(
                dict.fromkeys(
                    step.tool for step in steps if isinstance(step.tool, str)
                )
            ),
            detected_risks=base_plan.detected_risks,
            requires_confirmation=base_plan.requires_confirmation,
            status=base_plan.status,
            output=base_plan.output,
            required_outputs=base_plan.required_outputs,
            output_validators=base_plan.output_validators,
            replanning_policy=base_plan.replanning_policy,
        )

    def _reject_replan_validation(
        self,
        session_id: str,
        request: ReplanRequest,
        error: str,
        results: Mapping[str, object],
    ) -> None:
        self._execution_supervisor.record_replan_event(
            session_id,
            "replan_validation_rejected",
            attempt_number=request.attempt_number,
            failed_step=request.failed_step,
            reason=ReplanReason.VALIDATION_ERROR.value,
            error=error,
            details={"validation_status": "invalid"},
        )
        self._execution_supervisor.mark_failed(
            session_id,
            "replanned fragment validation failed",
            current_step=request.failed_step,
            results=results,
        )

    def _replan_request(
        self,
        session: ExecutionSession,
        failed_plan: ExecutionPlan,
        execution: PlanExecutionResult,
        *,
        attempt_number: int,
        classification: ReplanFailureClassification,
    ) -> ReplanRequest:
        failed_step_id = execution.current_step or execution.failed_step
        step_result = next(
            (
                result
                for result in execution.step_results
                if result.step_id == failed_step_id
            ),
            None,
        )
        attempts_performed = (
            step_result.metadata.get("attempt_number", 1)
            if step_result is not None
            else 1
        )
        if isinstance(attempts_performed, bool) or not isinstance(
            attempts_performed,
            int,
        ):
            attempts_performed = 1
        max_execution_attempts = (
            step_result.metadata.get("max_attempts", attempts_performed)
            if step_result is not None
            else attempts_performed
        )
        retries_exhausted = not bool(
            step_result is not None
            and step_result.metadata.get("retry_scheduled") is True
        ) and (
            step_result is None
            or step_result.metadata.get("retry_exhausted") is True
            or (
                isinstance(max_execution_attempts, int)
                and not isinstance(max_execution_attempts, bool)
                and attempts_performed >= max_execution_attempts
            )
        )
        completed_ids = tuple(execution.completed_steps)
        failed_ids = tuple(execution.failed_steps)
        unavailable_ids = set(completed_ids) | set(failed_ids) | set(
            execution.blocked_steps
        )
        pending_ids = tuple(
            dict.fromkeys(
                tuple(execution.pending_steps)
                + tuple(
                    step.id
                    for step in failed_plan.ordered_steps
                    if step.id not in unavailable_ids
                )
            )
        )
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
            completed_step_ids=completed_ids,
            failed_step_ids=failed_ids,
            cancelled_step_ids=tuple(
                item
                for item in execution.metadata.get("cancelled_step_ids", ())
                if isinstance(item, str)
            ),
            pending_step_ids=pending_ids,
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
            resource_selection_failure=(
                str(execution.metadata.get("resource_selection_failure"))
                if execution.metadata.get("resource_selection_failure") is not None
                else None
            ),
            rejected_candidate_ids=tuple(
                str(item)
                for item in execution.metadata.get("rejected_candidate_ids", ())
            ),
            budget_snapshot=getattr(session, "budget_usage", None),
            selected_resource_id=getattr(
                session.last_resource_decision,
                "selected_resource_id",
                None,
            ),
            previous_resource_id=(
                session.selected_resources_by_step.get(
                    execution.current_step or execution.failed_step or "",
                )
                if session.selected_resources_by_step
                else None
            ),
            optimization_goal=(
                getattr(
                    getattr(session.last_resource_decision, "optimization_goal", None),
                    "value",
                    None,
                )
                or (
                    str(execution.metadata.get("optimization_goal"))
                    if execution.metadata.get("optimization_goal") is not None
                    else None
                )
            ),
            degradation_applied=bool(
                getattr(session.last_resource_decision, "degradation_applied", False)
            ),
            attempt_number=attempt_number,
            max_attempts=self._replan_policy.max_replans_per_session,
            plan_id=plan_signature(failed_plan),
            attempts_performed=attempts_performed,
            retries_exhausted=retries_exhausted,
            classification=classification,
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
            "goal_verification": goal_verification_result_to_dict(
                execution.goal_verification_result
            ),
            "step_outputs": {
                result.step_id: result.output
                for result in execution.step_results
                if result.success and result.status == "completed"
            },
            "step_tools": {
                result.step_id: result.tool_name
                for result in execution.step_results
                if result.tool_name is not None
            },
            "step_resolution": {
                result.step_id: {
                    "status": result.metadata.get(
                        "parameter_resolution_status",
                        (
                            "failed"
                            if result.metadata.get(
                                "parameter_resolution_error_code"
                            )
                            else "not_required"
                        ),
                    ),
                    "argument_keys": tuple(
                        str(value)
                        for value in result.metadata.get(
                            "resolved_argument_keys",
                            (),
                        )
                    ),
                    "references": tuple(
                        str(value)
                        for value in result.metadata.get(
                            "used_references",
                            (),
                        )
                    ),
                }
                for result in execution.step_results
            },
        }

    def _prepare_objective_correction(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
        execution: PlanExecutionResult,
    ) -> StructuredExecutionResponse | None:
        """Classify and stage one bounded correction after technical completion."""

        verification = execution.goal_verification_result
        session_id = execution.metadata.get("execution_session_id")
        corrective_builder = getattr(
            self._execution_replanner,
            "corrective_fragment",
            None,
        )
        if (
            plan.status == "corrective"
            or not execution.success
            or verification is None
            or verification.verification_status
            not in {
                GoalVerificationStatus.NOT_VERIFIED,
                GoalVerificationStatus.PARTIALLY_VERIFIED,
                GoalVerificationStatus.INCONCLUSIVE,
                GoalVerificationStatus.USER_ACTION_REQUIRED,
            }
            or not isinstance(session_id, str)
            or not callable(corrective_builder)
        ):
            return None

        decision = corrective_builder(
            plan,
            execution,
            verification,
            session_id=session_id,
            previous_attempts=0,
            policy=self._objective_correction_policy,
        )
        base_snapshot = self._correction_snapshot(
            decision,
            status=CorrectionLifecycleStatus.NOT_STARTED,
        )
        if decision.classification is not CorrectionClassification.CORRECTABLE:
            status = (
                CorrectionLifecycleStatus.LIMIT_REACHED
                if decision.classification is CorrectionClassification.LIMIT_REACHED
                else CorrectionLifecycleStatus.REJECTED
            )
            self._execution_supervisor.record_objective_correction(
                session_id,
                {**base_snapshot, "status": status.value},
                event_type="objective_correction_not_started",
            )
            return StructuredExecutionResponse(
                handled=True,
                status="completed_objective_not_verified",
                message=(
                    self._completed_message(execution)
                    + " No se inició una corrección: "
                    + decision.reason
                ),
                plan=plan,
                validation_result=validation,
                execution_result=execution,
                operational_report=self.get_execution_report(session_id),
                objective_correction=decision,
            )

        fragment = decision.fragment
        if fragment is None:
            return None
        fragment_validation = self._validator.validate(fragment)
        safety_error = self._corrective_fragment_error(plan, execution, decision)
        fingerprint = (
            session_id
            + ":"
            + correction_fragment_fingerprint(
                decision.request.correction_request_id,
                fragment,
            )
        )
        if fingerprint in self._correction_fingerprints:
            safety_error = "The same corrective fragment was already prepared."
        if not fragment_validation.is_valid or safety_error is not None:
            reason = safety_error or "; ".join(fragment_validation.errors)
            self._execution_supervisor.record_objective_correction(
                session_id,
                {
                    **base_snapshot,
                    "status": CorrectionLifecycleStatus.REJECTED.value,
                    "validation_status": "invalid",
                    "rejection_reason": reason,
                },
                event_type="objective_correction_rejected",
            )
            return StructuredExecutionResponse(
                handled=True,
                status="correction_rejected",
                message="La corrección propuesta fue rechazada: " + reason,
                plan=plan,
                validation_result=validation,
                execution_result=execution,
                error_code="CORRECTION_VALIDATION_FAILED",
                error=reason,
                operational_report=self.get_execution_report(session_id),
                objective_correction=decision,
            )

        strategy = self._select_execution_strategy(fragment, fragment_validation)
        if strategy is None:
            reason = "The correction cannot bypass strategy selection."
            self._execution_supervisor.record_objective_correction(
                session_id,
                {
                    **base_snapshot,
                    "status": CorrectionLifecycleStatus.REJECTED.value,
                    "validation_status": "valid",
                    "rejection_reason": reason,
                },
                event_type="objective_correction_rejected",
            )
            return StructuredExecutionResponse(
                handled=True,
                status="correction_rejected",
                message="La corrección no puede ejecutarse sin estrategia validada.",
                plan=plan,
                validation_result=validation,
                execution_result=execution,
                error_code="CORRECTION_STRATEGY_UNAVAILABLE",
                error=reason,
                operational_report=self.get_execution_report(session_id),
                objective_correction=decision,
            )
        confirmations = self._confirmation_references(
            fragment,
            fragment_validation,
            strategy,
        )
        authorization = self._authorize_execution(
            fragment,
            fragment_validation,
            strategy,
            required_confirmations=confirmations,
            source=(
                "StructuredExecutionCoordinator.objective_correction:"
                + decision.request.correction_request_id
            ),
        )
        if (
            authorization.decision
            is not ExecutionAuthorizationDecision.CONFIRMATION_PENDING
        ):
            self._execution_supervisor.record_objective_correction(
                session_id,
                {
                    **base_snapshot,
                    "status": CorrectionLifecycleStatus.REJECTED.value,
                    "validation_status": "valid",
                    "authorization": authorization.persisted_snapshot(),
                    "rejection_reason": authorization.reason,
                },
                event_type="objective_correction_rejected",
            )
            return StructuredExecutionResponse(
                handled=True,
                status=authorization.decision.value.lower(),
                message=authorization.reason,
                plan=plan,
                validation_result=validation,
                execution_result=execution,
                error_code=f"CORRECTION_AUTHORIZATION_{authorization.decision.value}",
                error=authorization.reason,
                operational_report=self.get_execution_report(session_id),
                strategy_selection=strategy,
                authorization_result=authorization,
                objective_correction=decision,
            )

        correction_session = self._start_waiting_confirmation_session(
            fragment,
            strategy,
            authorization,
        )
        token = self._store_pending_plan(
            plan.goal,
            fragment,
            fragment_validation,
            correction_session.session_id,
            strategy,
            authorization,
            confirmations,
            decision.request.correction_request_id,
            correction_decision=decision,
            original_plan=plan,
            original_execution=execution,
            original_session_id=session_id,
        )
        self._correction_fingerprints.add(fingerprint)
        self._execution_supervisor.record_objective_correction(
            session_id,
            {
                **base_snapshot,
                "status": CorrectionLifecycleStatus.PENDING_CONFIRMATION.value,
                "validation_status": "valid",
                "strategy": strategy.persisted_snapshot(),
                "authorization": authorization.persisted_snapshot(),
                "correction_session_id": correction_session.session_id,
                "confirmation_token": token,
            },
            event_type="objective_correction_pending_confirmation",
        )
        return StructuredExecutionResponse(
            handled=True,
            status="correction_confirmation_required",
            message=(
                "El objetivo no se verificó en el primer intento. "
                "Atlas preparó una corrección limitada del recurso declarado; "
                "requiere confirmación explícita antes de escribir."
            ),
            plan=plan,
            validation_result=validation,
            execution_result=execution,
            requires_confirmation=True,
            confirmation_token=token,
            error_code="CORRECTION_CONFIRMATION_REQUIRED",
            operational_report=self.get_execution_report(session_id),
            strategy_selection=strategy,
            authorization_result=authorization,
            objective_correction=decision,
        )

    def _corrective_fragment_error(
        self,
        original_plan: ExecutionPlan,
        original_execution: PlanExecutionResult,
        decision: ObjectiveCorrectionDecision,
    ) -> str | None:
        fragment = decision.fragment
        if fragment is None:
            return "Corrective fragment is missing."
        policy = self._objective_correction_policy
        if fragment.goal != original_plan.goal:
            return "Corrective fragment changes the original objective."
        if len(fragment.ordered_steps) > policy.max_steps:
            return "Corrective fragment exceeds the step limit."
        writes = tuple(
            step for step in fragment.ordered_steps if step.tool == "write_file"
        )
        resources = {
            step.arguments.get("path")
            for step in writes
            if isinstance(step.arguments.get("path"), str)
        }
        if len(resources) > policy.max_resources:
            return "Corrective fragment exceeds the resource limit."
        if not fragment.requires_confirmation or len(writes) != 1:
            return "Resource correction must retain one explicit confirmation."
        if set(original_execution.completed_steps).intersection(
            step.id for step in fragment.ordered_steps
        ):
            return "Corrective fragment repeats completed step identifiers."
        if any(
            step.retry_policy is not None
            and getattr(step.retry_policy, "max_attempts", 1) > 1
            for step in fragment.ordered_steps
        ):
            return "Corrective fragment cannot increase retry attempts."
        return None

    @staticmethod
    def _correction_snapshot(
        decision: ObjectiveCorrectionDecision,
        *,
        status: CorrectionLifecycleStatus,
    ) -> dict[str, object]:
        cycle_started = status in {
            CorrectionLifecycleStatus.AUTHORIZED,
            CorrectionLifecycleStatus.DISPATCHED,
            CorrectionLifecycleStatus.RUNNING,
            CorrectionLifecycleStatus.COMPLETED,
            CorrectionLifecycleStatus.FAILED,
            CorrectionLifecycleStatus.VERIFIED_AFTER_CORRECTION,
        } or (
            status is CorrectionLifecycleStatus.LIMIT_REACHED
            and decision.fragment is not None
        )
        return {
            "correction_request_id": decision.request.correction_request_id,
            "original_session_id": decision.request.session_id,
            "initial_verification_status": decision.request.verification_status,
            "classification": decision.classification.value,
            "correction_type": decision.correction_type.value,
            "status": status.value,
            "cycle": (
                decision.request.previous_attempts + 1
                if cycle_started
                else decision.request.previous_attempts
            ),
            "cycle_limit": (
                decision.request.previous_attempts
                + decision.request.remaining_cycles
            ),
            "affected_criteria": list(decision.affected_criterion_ids),
            "failed_criteria": [
                item.to_dict() for item in decision.request.failed_criteria
            ],
            "fragment_signature": decision.fragment_signature,
            "fragment_step_ids": (
                []
                if decision.fragment is None
                else [step.id for step in decision.fragment.ordered_steps]
            ),
            "reason": decision.reason,
        }

    def _execution_response(
        self,
        plan: ExecutionPlan,
        validation: PlanValidationResult,
        execution: PlanExecutionResult,
        *,
        confirmation_token: str | None = None,
    ) -> StructuredExecutionResponse:
        correction_response = self._prepare_objective_correction(
            plan,
            validation,
            execution,
        )
        if correction_response is not None:
            return correction_response
        report = self._report_for_execution(execution)
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
                operational_report=report,
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
            operational_report=report,
        )

    def _report_for_execution(
        self,
        execution: PlanExecutionResult,
    ) -> OperationalExecutionReport | None:
        session_id = execution.metadata.get("execution_session_id")
        if not isinstance(session_id, str):
            return None
        return self.get_execution_report(session_id)

    @staticmethod
    def _with_execution_session_id(
        execution: PlanExecutionResult,
        session_id: str,
    ) -> PlanExecutionResult:
        verification = execution.goal_verification_result
        return replace(
            execution,
            metadata={
                **execution.metadata,
                "execution_session_id": session_id,
            },
            goal_verification_result=(
                None
                if verification is None
                else replace(verification, session_id=session_id)
            ),
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
        saved_context = execution.metadata.get("execution_context_snapshot")
        execution_context = (
            ExecutionContext.restore(saved_context)
            if isinstance(saved_context, ExecutionContextSnapshot)
            else ExecutionContext(execution_id)
        )
        step_by_id = {step.id: step for step in plan.ordered_steps}
        for result in execution.step_results:
            if (
                not isinstance(saved_context, ExecutionContextSnapshot)
                and result.success
                and result.status == "completed"
            ):
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
            if (
                not isinstance(saved_context, ExecutionContextSnapshot)
                and result.status == "skipped"
            ):
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
        verification = execution.goal_verification_result
        technical = (
            "Ejecucion estructurada completada. "
            f"Pasos completados: {', '.join(execution.completed_steps)}."
        )
        if verification is None:
            return technical + " El cumplimiento del objetivo no fue evaluado."
        if verification.verification_status is GoalVerificationStatus.VERIFIED:
            return technical + " Objetivo verificado."
        return technical + " " + (
            verification.message
            or "El objetivo no quedó verificado."
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

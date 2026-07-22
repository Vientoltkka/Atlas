"""Structured execution pipeline coordination for Atlas."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from core.execution_context import ExecutionContext
from core.execution_plan_executor import (
    ExecutionControl,
    ExecutionProgress,
    ExecutionPlanExecutor,
    PartialExecutionState,
    PlanExecutionResult,
    PlanExecutionStatus,
    ResumableExecutionState,
)
from core.execution_plan_validator import (
    ExecutionPlanValidator,
    PlanValidationResult,
    plan_signature,
)
from core.planner import ExecutionPlan, PlanGenerationResult, Planner
from core.resumable_execution_store import (
    ResumableExecutionStore,
    ResumableExecutionStoreError,
)


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


class StructuredExecutionCoordinator:
    """Coordinate Planner, Validator and Executor without duplicating them."""

    def __init__(
        self,
        planner: Planner,
        validator: ExecutionPlanValidator,
        executor: ExecutionPlanExecutor,
        resumable_store: ResumableExecutionStore | None = None,
    ) -> None:
        self._planner = planner
        self._validator = validator
        self._executor = executor
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
            token = self._store_pending_plan(objective, generation.plan, validation)
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

        execution = self._executor.execute(
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
        execution = self._executor.execute(
            pending.plan,
            pending.validation_result,
            confirmation_granted=True,
            control=control,
            on_progress=on_execution_progress,
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

        execution = self._executor.resume(
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

        self._discard_pending(pending.confirmation_token)
        self._resumable_execution = None
        delete_error = self._delete_persisted_resumable_state()
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
    ) -> str:
        token = validation.plan_signature or plan_signature(plan)
        if self._active_confirmation_token is not None:
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
            )
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
        execution_id = (
            execution.trace.execution_id
            if execution.trace is not None
            else None
        )
        execution_context = ExecutionContext(execution_id)
        for result in execution.step_results:
            if result.success and result.status == "completed":
                execution_context.mark_step_started(
                    result.step_id,
                    int(result.metadata.get("attempt_number", 1)),
                )
                execution_context.mark_step_succeeded(result.step_id, result.output)
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
            execution_context_snapshot=execution_context.snapshot(),
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

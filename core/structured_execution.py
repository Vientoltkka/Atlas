"""Structured execution pipeline coordination for Atlas."""

from __future__ import annotations

from dataclasses import dataclass

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
from core.planner import ExecutionPlan, PlanGenerationResult, Planner


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


@dataclass(frozen=True, slots=True)
class _PendingPlan:
    objective: str
    plan: ExecutionPlan
    validation_result: PlanValidationResult


class StructuredExecutionCoordinator:
    """Coordinate Planner, Validator and Executor without duplicating them."""

    def __init__(
        self,
        planner: Planner,
        validator: ExecutionPlanValidator,
        executor: ExecutionPlanExecutor,
    ) -> None:
        self._planner = planner
        self._validator = validator
        self._executor = executor
        self._pending_plans: dict[str, _PendingPlan] = {}

    def handle(
        self,
        objective: str,
        *,
        confirmation_granted: bool = False,
        control: ExecutionControl | None = None,
    ) -> StructuredExecutionResponse:
        """Generate, validate and optionally execute one structured plan."""
        generation = self._planner.generate_execution_plan(objective)

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

        execution = self._executor.execute(
            generation.plan,
            validation,
            confirmation_granted=confirmation_granted,
            control=control,
        )
        return self._execution_response(generation.plan, validation, execution)

    def confirm(
        self,
        confirmation_token: str,
        *,
        objective: str | None = None,
        control: ExecutionControl | None = None,
    ) -> StructuredExecutionResponse:
        """Execute exactly the validated plan associated with a confirmation token."""
        pending = self._pending_plans.get(confirmation_token)
        if pending is None:
            return StructuredExecutionResponse(
                handled=True,
                status="confirmation_not_found",
                message="No hay ningun plan pendiente para esa confirmacion.",
                requires_confirmation=False,
                confirmation_token=confirmation_token,
                error_code="CONFIRMATION_NOT_FOUND",
                error="confirmation token is not pending",
            )

        if objective is not None and objective != pending.objective:
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

        execution = self._executor.execute(
            pending.plan,
            pending.validation_result,
            confirmation_granted=True,
            control=control,
        )
        if execution.success:
            self._pending_plans.pop(confirmation_token, None)

        return self._execution_response(
            pending.plan,
            pending.validation_result,
            execution,
            confirmation_token=confirmation_token,
        )

    def pending_plan(
        self,
        confirmation_token: str,
    ) -> ExecutionPlan | None:
        """Return a pending plan by token without exposing internal mutation."""
        pending = self._pending_plans.get(confirmation_token)
        return pending.plan if pending is not None else None

    def _should_handle_generation(
        self,
        generation: PlanGenerationResult,
    ) -> bool:
        if generation.error_code in {"PLAN_PARSE_ERROR", "INVALID_PLAN_RESPONSE", "UNKNOWN_TOOL"}:
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
        self._pending_plans[token] = _PendingPlan(
            objective=objective,
            plan=plan,
            validation_result=validation,
        )
        return token

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
        )

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
        lines.append(f"Token de confirmacion: {token}")
        lines.append(
            "La ejecucion no se realizo porque requiere confirmacion explicita."
        )
        if validation.plan_signature:
            lines.append(f"Firma del plan: {validation.plan_signature}")
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

"""Deterministic user-facing reports for supervised Atlas executions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Mapping

from core.goal_verifier import (
    GoalVerificationStatus,
    goal_verification_result_from_dict,
)
from core.planner import ExecutionStep
from core.execution_supervisor import (
    ExecutionSession,
    ExecutionState,
    ExecutionSummary,
    ReplanRecoveryStatus,
    StepExecutionSnapshot,
    StepExecutionState,
)


_MAX_VISIBLE_TEXT = 240
_SENSITIVE_MARKERS = (
    "api_key",
    "apikey",
    "api key",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
)


class OperationalExecutionStatus(str, Enum):
    """Small closed status set intended for user-visible execution results."""

    COMPLETED = "COMPLETED"
    COMPLETED_WITH_RECOVERY = "COMPLETED_WITH_RECOVERY"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    USER_ACTION_REQUIRED = "USER_ACTION_REQUIRED"


@dataclass(frozen=True, slots=True)
class OperationalStepReport:
    """Safe user-facing projection of one supervised execution step."""

    step_id: str
    description: str
    state: str
    attempts: int
    duration_seconds: float | None
    error: str | None
    result: str | None = None
    tool_name: str | None = None
    resolved_references: tuple[str, ...] = ()
    produced_resource: str | None = None
    replaced: bool = False
    omitted: bool = False
    cancelled: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "state": self.state,
            "attempts": self.attempts,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "result": self.result,
            "tool_name": self.tool_name,
            "resolved_references": list(self.resolved_references),
            "produced_resource": self.produced_resource,
            "replaced": self.replaced,
            "omitted": self.omitted,
            "cancelled": self.cancelled,
        }


@dataclass(frozen=True, slots=True)
class OperationalExecutionReport:
    """Serializable and deterministic execution report for Atlas users."""

    session_id: str
    objective: str
    status: OperationalExecutionStatus
    title: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    skipped_steps: int
    cancelled_steps: int
    progress_percent: float
    duration_seconds: float
    retry_count: int
    retried_step_ids: tuple[str, ...]
    replan_status: str
    replan_count: int
    warnings: tuple[str, ...]
    pending_user_actions: tuple[str, ...]
    steps: tuple[OperationalStepReport, ...]
    final_message: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    execution_strategy: str = "STANDARD"
    strategy_reason: str = "Previous execution behavior applies."
    strategy_controls: tuple[str, ...] = ()
    strategy_blocked_execution: bool = False
    strategy_required_confirmation: bool = False
    strategy_allowed_recovery: bool = False
    strategy_reinforced_supervision: bool = False
    authorization_status: str = "LEGACY_NOT_RECORDED"
    authorization_ready: bool = False
    authorization_pending_confirmations: tuple[str, ...] = ()
    authorization_manual_review_pending: bool = False
    authorization_block_reason: str | None = None
    authorized_strategy: str | None = None
    dispatch_completed: bool = False
    authorization_session_id: str | None = None
    goal_verification_status: str = "NOT_APPLICABLE"
    goal_verification_satisfied_criteria: int = 0
    goal_verification_total_criteria: int = 0
    goal_verification_failed_criteria: int = 0
    goal_verification_unevaluable_criteria: int = 0
    goal_verification_evidence: tuple[str, ...] = ()
    goal_verification_resources: tuple[str, ...] = ()
    goal_verification_action: str | None = None
    goal_verification_message: str = (
        "No hay verificación del objetivo registrada."
    )
    objective_correction: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.progress_percent <= 100.0:
            raise ValueError("progress_percent must be between zero and one hundred.")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative.")
        object.__setattr__(self, "retried_step_ids", tuple(self.retried_step_ids))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(
            self,
            "pending_user_actions",
            tuple(self.pending_user_actions),
        )
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(
            self,
            "strategy_controls",
            tuple(self.strategy_controls),
        )
        object.__setattr__(
            self,
            "authorization_pending_confirmations",
            tuple(self.authorization_pending_confirmations),
        )
        object.__setattr__(
            self,
            "goal_verification_evidence",
            tuple(self.goal_verification_evidence),
        )
        object.__setattr__(
            self,
            "goal_verification_resources",
            tuple(self.goal_verification_resources),
        )
        if self.authorization_block_reason is not None:
            object.__setattr__(
                self,
                "authorization_block_reason",
                _safe_text(self.authorization_block_reason),
            )
        object.__setattr__(
            self,
            "strategy_reason",
            _safe_text(self.strategy_reason),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(
            self,
            "objective_correction",
            MappingProxyType(dict(self.objective_correction)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation without technical objects."""
        return {
            "session_id": self.session_id,
            "objective": self.objective,
            "status": self.status.value,
            "title": self.title,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "skipped_steps": self.skipped_steps,
            "cancelled_steps": self.cancelled_steps,
            "progress_percent": self.progress_percent,
            "duration_seconds": self.duration_seconds,
            "retry_count": self.retry_count,
            "retried_step_ids": list(self.retried_step_ids),
            "replan_status": self.replan_status,
            "replan_count": self.replan_count,
            "warnings": list(self.warnings),
            "pending_user_actions": list(self.pending_user_actions),
            "steps": [step.to_dict() for step in self.steps],
            "final_message": self.final_message,
            "metadata": dict(self.metadata),
            "execution_strategy": self.execution_strategy,
            "strategy_reason": self.strategy_reason,
            "strategy_controls": list(self.strategy_controls),
            "strategy_blocked_execution": self.strategy_blocked_execution,
            "strategy_required_confirmation": self.strategy_required_confirmation,
            "strategy_allowed_recovery": self.strategy_allowed_recovery,
            "strategy_reinforced_supervision": (
                self.strategy_reinforced_supervision
            ),
            "authorization_status": self.authorization_status,
            "authorization_ready": self.authorization_ready,
            "authorization_pending_confirmations": list(
                self.authorization_pending_confirmations
            ),
            "authorization_manual_review_pending": (
                self.authorization_manual_review_pending
            ),
            "authorization_block_reason": self.authorization_block_reason,
            "authorized_strategy": self.authorized_strategy,
            "dispatch_completed": self.dispatch_completed,
            "authorization_session_id": self.authorization_session_id,
            "goal_verification_status": self.goal_verification_status,
            "goal_verification_satisfied_criteria": (
                self.goal_verification_satisfied_criteria
            ),
            "goal_verification_total_criteria": (
                self.goal_verification_total_criteria
            ),
            "goal_verification_failed_criteria": (
                self.goal_verification_failed_criteria
            ),
            "goal_verification_unevaluable_criteria": (
                self.goal_verification_unevaluable_criteria
            ),
            "goal_verification_evidence": list(
                self.goal_verification_evidence
            ),
            "goal_verification_resources": list(
                self.goal_verification_resources
            ),
            "goal_verification_action": self.goal_verification_action,
            "goal_verification_message": self.goal_verification_message,
            "objective_correction": dict(self.objective_correction),
        }

    def to_text(self) -> str:
        """Render a stable Spanish report without using an LLM."""
        processed = (
            self.completed_steps
            + self.failed_steps
            + self.skipped_steps
            + self.cancelled_steps
        )
        lines = [
            f"Objetivo: {self.objective}",
            f"Resultado: {self.title}",
            (
                f"Progreso: {processed}/{self.total_steps} pasos "
                f"({self.progress_percent:.1f}%)."
            ),
            f"Duración: {self.duration_seconds:.2f} s.",
            "Ejecución:",
        ]
        lines.extend(_step_line(step) for step in self.steps)
        lines.append(_retry_text(self))
        lines.append(_replan_text(self))
        lines.append(f"Estrategia: {self.execution_strategy}.")
        lines.append(f"Motivo de estrategia: {self.strategy_reason}")
        if self.strategy_controls:
            lines.append(
                "Controles de estrategia: "
                + ", ".join(self.strategy_controls)
                + "."
            )
        lines.append("Autorización:")
        lines.append(f"- Estado: {self.authorization_status}.")
        if self.authorized_strategy is not None:
            lines.append(f"- Estrategia: {self.authorized_strategy}.")
        if self.authorization_pending_confirmations:
            lines.append(
                "- Confirmaciones pendientes: "
                + ", ".join(self.authorization_pending_confirmations)
                + "."
            )
        if self.authorization_block_reason is not None:
            lines.append(f"- Motivo: {self.authorization_block_reason}")
        lines.append(
            "- Despacho: "
            + ("completado." if self.dispatch_completed else "no realizado.")
        )
        if self.warnings:
            lines.append("Advertencias:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        if self.pending_user_actions:
            lines.append("Acción necesaria:")
            lines.extend(f"- {action}" for action in self.pending_user_actions)
        lines.append("Verificación del objetivo:")
        lines.append(f"- Estado: {self.goal_verification_status}.")
        lines.append(
            "- Criterios satisfechos: "
            f"{self.goal_verification_satisfied_criteria}/"
            f"{self.goal_verification_total_criteria}."
        )
        if self.goal_verification_failed_criteria:
            lines.append(
                "- Criterios fallidos: "
                f"{self.goal_verification_failed_criteria}."
            )
        if self.goal_verification_unevaluable_criteria:
            lines.append(
                "- Criterios no evaluables: "
                f"{self.goal_verification_unevaluable_criteria}."
            )
        for resource in self.goal_verification_resources:
            lines.append(f"- Recurso comprobado: {resource}")
        lines.append(f"- Resultado: {self.goal_verification_message}")
        if self.goal_verification_action is not None:
            lines.append(f"- Acción recomendada: {self.goal_verification_action}")
        if self.objective_correction:
            correction = self.objective_correction
            lines.append("Corrección del objetivo:")
            lines.append(
                "- Estado inicial: "
                + str(correction.get("initial_verification_status", "desconocido"))
                + "."
            )
            lines.append(
                "- Clasificación: "
                + str(correction.get("classification", "desconocida"))
                + "."
            )
            lines.append(
                "- Corrección seleccionada: "
                + str(correction.get("correction_type", "NO_SAFE_CORRECTION"))
                + "."
            )
            lines.append(
                "- Pasos correctivos: "
                + str(len(correction.get("fragment_step_ids", ())))
                + "."
            )
            lines.append(
                "- Confirmación: "
                + str(correction.get("confirmation", "pendiente o no aplicable"))
                + "."
            )
            lines.append(
                "- Resultado de la corrección: "
                + str(correction.get("status", "NOT_STARTED"))
                + "."
            )
            final_verification = correction.get("final_verification_status")
            if final_verification is not None:
                lines.append(f"- Nueva verificación: {final_verification}.")
            lines.append(
                "- Ciclos utilizados: "
                + str(correction.get("cycle", 0))
                + "/"
                + str(correction.get("cycle_limit", 1))
                + "."
            )
            if correction.get("rejection_reason") is not None:
                lines.append("- Motivo: " + str(correction["rejection_reason"]))
        lines.append(f"Mensaje final: {self.final_message}")
        return "\n".join(lines)


class ExecutionReportGenerator:
    """Transform existing supervised state into a safe operational report."""

    def generate(
        self,
        session: ExecutionSession,
        summary: ExecutionSummary,
    ) -> OperationalExecutionReport:
        if session.session_id != summary.session_id:
            raise ValueError("session and summary must identify the same execution.")

        descriptions = {
            step.id: _safe_text(step.description)
            for step in session.original_plan.ordered_steps
        }
        descriptions.update(
            {
                step.id: _safe_text(step.description)
                for step in session.active_plan.ordered_steps
            }
        )
        active_steps = {
            step.id: step
            for step in session.active_plan.ordered_steps
        }
        raw_outputs = session.results.get("step_outputs")
        step_outputs = raw_outputs if isinstance(raw_outputs, Mapping) else {}
        raw_resolution = session.results.get("step_resolution")
        step_resolution = (
            raw_resolution
            if isinstance(raw_resolution, Mapping)
            else {}
        )
        replaced_ids = {
            record.failed_step
            for record in session.replan_history
            if record.failed_step is not None
        }
        steps = tuple(
            self._step_report(
                snapshot,
                descriptions.get(step_id, step_id),
                result=step_outputs.get(step_id),
                step=active_steps.get(step_id),
                resolution=step_resolution.get(step_id),
                replaced=step_id in replaced_ids,
            )
            for step_id, snapshot in session.step_states.items()
        )
        warnings = self._warnings(session, summary, steps)
        actions = self._pending_actions(session, summary)
        status = self._status(session, summary, bool(actions))
        title = _title(status)
        final_message = _final_message(status, actions)
        retried_step_ids = tuple(
            step.step_id for step in steps if step.attempts > 1
        )
        strategy = _strategy_report_fields(session.execution_strategy)
        authorization = _authorization_report_fields(
            session.execution_authorization
        )
        verification = _verification_report_fields(session)
        correction = session.results.get("objective_correction")
        if (
            status
            in {
                OperationalExecutionStatus.COMPLETED,
                OperationalExecutionStatus.COMPLETED_WITH_RECOVERY,
            }
            and verification["goal_verification_status"]
            != GoalVerificationStatus.NOT_APPLICABLE.value
        ):
            final_message = str(
                verification["goal_verification_message"]
            )
        return OperationalExecutionReport(
            session_id=session.session_id,
            objective=_safe_text(session.original_plan.goal),
            status=status,
            title=title,
            total_steps=summary.total_steps,
            completed_steps=summary.successful_steps,
            failed_steps=summary.failed_steps,
            skipped_steps=summary.skipped_steps,
            cancelled_steps=summary.cancelled_steps,
            progress_percent=round(summary.progress * 100.0, 1),
            duration_seconds=round(summary.duration_seconds, 3),
            retry_count=summary.retry_count,
            retried_step_ids=retried_step_ids,
            replan_status=summary.replan_status.value,
            replan_count=summary.replan_count,
            warnings=warnings,
            pending_user_actions=actions,
            steps=steps,
            final_message=final_message,
            metadata={
                "source": "execution_session",
                "deterministic": True,
            },
            **strategy,
            **authorization,
            **verification,
            objective_correction=(
                correction if isinstance(correction, Mapping) else {}
            ),
        )

    @staticmethod
    def _step_report(
        snapshot: StepExecutionSnapshot,
        description: str,
        *,
        result: object | None,
        step: ExecutionStep | None,
        resolution: object | None,
        replaced: bool,
    ) -> OperationalStepReport:
        duration = None
        if snapshot.started_at is not None and snapshot.finished_at is not None:
            duration = round(
                max(
                    0.0,
                    (snapshot.finished_at - snapshot.started_at).total_seconds(),
                ),
                3,
            )
        resolution_mapping = (
            resolution
            if isinstance(resolution, Mapping)
            else {}
        )
        raw_references = resolution_mapping.get("references", ())
        references = (
            tuple(
                _safe_text(reference)
                for reference in raw_references
                if isinstance(reference, str)
            )
            if isinstance(raw_references, (list, tuple))
            else ()
        )
        return OperationalStepReport(
            step_id=_safe_text(snapshot.step_id),
            description=description,
            state=snapshot.state.value.upper(),
            attempts=snapshot.attempt_count,
            duration_seconds=duration,
            error=_safe_text(snapshot.error) if snapshot.error is not None else None,
            result=_safe_text(result) if result is not None else None,
            tool_name=(
                _safe_text(step.tool)
                if step is not None and step.tool is not None
                else None
            ),
            resolved_references=references,
            produced_resource=_produced_resource(
                step,
                completed=snapshot.state is StepExecutionState.COMPLETED,
            ),
            replaced=replaced,
            omitted=snapshot.state is StepExecutionState.SKIPPED,
            cancelled=snapshot.state is StepExecutionState.CANCELLED,
        )

    @staticmethod
    def _status(
        session: ExecutionSession,
        summary: ExecutionSummary,
        has_actions: bool,
    ) -> OperationalExecutionStatus:
        if session.state is ExecutionState.COMPLETED:
            if summary.replan_status is ReplanRecoveryStatus.SUCCEEDED:
                return OperationalExecutionStatus.COMPLETED_WITH_RECOVERY
            return OperationalExecutionStatus.COMPLETED
        if session.state is ExecutionState.CANCELLED:
            return OperationalExecutionStatus.CANCELLED
        if has_actions:
            return OperationalExecutionStatus.USER_ACTION_REQUIRED
        if summary.successful_steps > 0:
            return OperationalExecutionStatus.PARTIALLY_COMPLETED
        return OperationalExecutionStatus.FAILED

    @staticmethod
    def _warnings(
        session: ExecutionSession,
        summary: ExecutionSummary,
        steps: tuple[OperationalStepReport, ...],
    ) -> tuple[str, ...]:
        warnings = [
            f"{step.step_id}: {step.error}"
            for step in steps
            if step.error is not None
        ]
        if summary.critical_failure_step is not None:
            warnings.append(
                f"El paso crítico {summary.critical_failure_step} no se completó."
            )
        if summary.replan_status is ReplanRecoveryStatus.VALIDATION_REJECTED:
            warnings.append("La alternativa de recuperación no superó la validación.")
        elif summary.replan_status is ReplanRecoveryStatus.LIMIT_REACHED:
            warnings.append("Se alcanzó el límite de replanificaciones.")
        elif summary.replan_status is ReplanRecoveryStatus.NO_SAFE_ALTERNATIVE:
            warnings.append("No se encontró una alternativa segura.")
        if not warnings and session.last_error is not None:
            warnings.append(_safe_text(session.last_error))
        return _unique(warnings)

    @staticmethod
    def _pending_actions(
        session: ExecutionSession,
        summary: ExecutionSummary,
    ) -> tuple[str, ...]:
        actions: list[str] = []
        error_code = str(session.results.get("error_code") or "").upper()
        event_types = {event.event_type for event in session.events}
        authorization = session.execution_authorization or {}
        manual_review = (
            authorization.get("decision") == "MANUAL_REVIEW_PENDING"
        )
        if manual_review:
            actions.append(
                "Completa la revisión manual antes de solicitar la ejecución."
            )
        if (
            (
                session.state is ExecutionState.WAITING_CONFIRMATION
                and not manual_review
            )
            or error_code == "CONFIRMATION_REQUIRED"
        ):
            actions.append("Confirma o cancela el plan pendiente.")
        if "replan_validation_rejected" in event_types:
            actions.append("Corrige o sustituye la alternativa rechazada.")
        if "replan_no_safe_alternative" in event_types:
            actions.append("Proporciona una alternativa segura para continuar.")
        if "replan_limit_reached" in event_types:
            actions.append("Revisa el fallo antes de solicitar otra ejecución.")
        if "resource_selection_failed" in event_types or error_code == "NO_COMPATIBLE_RESOURCE":
            actions.append("Habilita un recurso compatible para continuar.")
        if "PERMISSION" in error_code or error_code in {
            "AUTHORIZATION_REQUIRED",
            "ACCESS_DENIED",
        }:
            actions.append("Concede el permiso requerido para continuar.")
        if error_code in {
            "PARAMETER_RESOLUTION_FAILED",
            "TOOL_SCHEMA_VALIDATION_FAILED",
            "MISSING_REQUIRED_DATA",
        }:
            actions.append("Proporciona los datos obligatorios que faltan.")
        if summary.replan_status is ReplanRecoveryStatus.VALIDATION_REJECTED:
            actions.append("Corrige o sustituye la alternativa rechazada.")
        return _unique(actions)


def _step_line(step: OperationalStepReport) -> str:
    marker = {
        "COMPLETED": "✓",
        "FAILED": "✗",
        "SKIPPED": "–",
        "CANCELLED": "×",
        "BLOCKED": "!",
        "INTERRUPTED": "!",
    }.get(step.state, "·")
    flags = []
    if step.replaced:
        flags.append("sustituido")
    if step.omitted:
        flags.append("omitido")
    if step.cancelled:
        flags.append("cancelado")
    suffix = f" [{', '.join(flags)}]" if flags else ""
    attempts = f"{step.attempts} intento" if step.attempts == 1 else f"{step.attempts} intentos"
    duration = (
        f", {step.duration_seconds:.2f} s"
        if step.duration_seconds is not None
        else ""
    )
    error = f" — {step.error}" if step.error is not None else ""
    tool = f" [{step.tool_name}]" if step.tool_name is not None else ""
    line = (
        f"{marker} {step.step_id}: {step.description}{tool} "
        f"({attempts}{duration}){suffix}{error}"
    )
    if step.result is not None:
        line += f"\n  Resultado: {step.result}"
    if step.resolved_references:
        line += "\n  Referencias resueltas: " + ", ".join(
            step.resolved_references
        )
    if step.produced_resource is not None:
        line += f"\n  Recurso producido: {step.produced_resource}"
    return line


def _produced_resource(
    step: ExecutionStep | None,
    *,
    completed: bool,
) -> str | None:
    if not completed or step is None or step.tool != "write_file":
        return None
    path = step.arguments.get("path")
    if not isinstance(path, str):
        return None
    return _safe_text(path)


def _retry_text(report: OperationalExecutionReport) -> str:
    if report.retry_count == 0:
        return "Reintentos: no fueron necesarios."
    step_ids = ", ".join(report.retried_step_ids)
    return f"Reintentos: {report.retry_count} en {step_ids}."


def _replan_text(report: OperationalExecutionReport) -> str:
    messages = {
        ReplanRecoveryStatus.NOT_NEEDED.value: "no fue necesaria.",
        ReplanRecoveryStatus.SUCCEEDED.value: "aplicada con éxito.",
        ReplanRecoveryStatus.VALIDATION_REJECTED.value: "rechazada por validación.",
        ReplanRecoveryStatus.LIMIT_REACHED.value: "límite alcanzado.",
        ReplanRecoveryStatus.NO_SAFE_ALTERNATIVE.value: "sin alternativa segura.",
        ReplanRecoveryStatus.FAILED.value: "falló.",
        ReplanRecoveryStatus.IN_PROGRESS.value: "en curso.",
    }
    return f"Replanificación: {messages.get(report.replan_status, report.replan_status)}"


def _verification_report_fields(
    session: ExecutionSession,
) -> dict[str, object]:
    raw = session.results.get("goal_verification")
    verification = None
    if isinstance(raw, Mapping):
        try:
            verification = goal_verification_result_from_dict(raw)
        except (TypeError, ValueError):
            verification = None
    if verification is not None:
        return {
            "goal_verification_status": verification.verification_status.value,
            "goal_verification_satisfied_criteria": (
                verification.satisfied_criteria
            ),
            "goal_verification_total_criteria": len(verification.criteria),
            "goal_verification_failed_criteria": verification.failed_criteria,
            "goal_verification_unevaluable_criteria": (
                verification.unevaluable_criteria
            ),
            "goal_verification_evidence": tuple(
                _safe_text(item)
                for item in verification.evidence
            ),
            "goal_verification_resources": tuple(
                _safe_text(item)
                for item in verification.resources_checked
            ),
            "goal_verification_action": (
                None
                if verification.required_action is None
                else _safe_text(verification.required_action)
            ),
            "goal_verification_message": _safe_text(
                verification.message
                or "No hay una conclusión de verificación disponible."
            ),
        }
    if session.state is ExecutionState.WAITING_CONFIRMATION:
        return {
            "goal_verification_status": (
                GoalVerificationStatus.USER_ACTION_REQUIRED.value
            ),
            "goal_verification_action": (
                "Confirma o cancela el plan pendiente."
            ),
            "goal_verification_message": (
                "El objetivo no se ha verificado porque la ejecución "
                "requiere confirmación."
            ),
        }
    return {
        "goal_verification_status": GoalVerificationStatus.NOT_APPLICABLE.value,
        "goal_verification_message": (
            "La sesión no contiene una verificación del objetivo registrada."
        ),
    }


def _strategy_report_fields(
    snapshot: Mapping[str, object] | None,
) -> dict[str, object]:
    if snapshot is None:
        return {
            "execution_strategy": "STANDARD",
            "strategy_reason": (
                "Strategy was not recorded; previous execution behavior applies."
            ),
        }
    configuration = snapshot.get("configuration")
    safe_configuration = (
        configuration if isinstance(configuration, Mapping) else {}
    )
    controls = []
    if safe_configuration.get("progress_required") is True:
        controls.append("progress_required")
    if safe_configuration.get("record_all_transitions") is True:
        controls.append("transition_trace")
    if safe_configuration.get("cancel_on_critical_failure") is True:
        controls.append("critical_failure_stop")
    supervision_mode = _safe_text(
        safe_configuration.get("supervision_mode", "NORMAL")
    )
    return {
        "execution_strategy": _safe_text(snapshot.get("strategy", "STANDARD")),
        "strategy_reason": _safe_text(
            snapshot.get("reason", "Previous execution behavior applies.")
        ),
        "strategy_controls": tuple(controls),
        "strategy_blocked_execution": (
            safe_configuration.get("execution_allowed") is False
        ),
        "strategy_required_confirmation": (
            safe_configuration.get("requires_confirmation") is True
        ),
        "strategy_allowed_recovery": (
            safe_configuration.get("allow_replanning") is True
        ),
        "strategy_reinforced_supervision": supervision_mode == "REINFORCED",
    }


def _authorization_report_fields(
    snapshot: Mapping[str, object] | None,
) -> dict[str, object]:
    if snapshot is None:
        return {
            "authorization_status": "LEGACY_NOT_RECORDED",
            "authorization_ready": False,
            "authorization_block_reason": (
                "An explicit dispatch permit was not recorded for this legacy session."
            ),
        }
    decision = _safe_text(snapshot.get("decision", "UNKNOWN"))
    pending = snapshot.get("pending_confirmation_ids", ())
    pending_ids = (
        tuple(_safe_text(value) for value in pending[:20])
        if isinstance(pending, (tuple, list))
        else ()
    )
    dispatch = snapshot.get("dispatch")
    safe_dispatch = dispatch if isinstance(dispatch, Mapping) else {}
    dispatch_completed = safe_dispatch.get("dispatched") is True
    reason = _safe_text(snapshot.get("reason", ""))
    blocked = decision in {
        "BLOCKED",
        "REJECTED",
        "MANUAL_REVIEW_PENDING",
        "CONFIRMATION_PENDING",
    }
    return {
        "authorization_status": decision,
        "authorization_ready": (
            decision == "AUTHORIZED"
            and snapshot.get("dispatch_allowed") is True
            and snapshot.get("consumed") is not True
        ),
        "authorization_pending_confirmations": pending_ids,
        "authorization_manual_review_pending": (
            decision == "MANUAL_REVIEW_PENDING"
        ),
        "authorization_block_reason": reason if blocked else None,
        "authorized_strategy": (
            _safe_text(snapshot.get("strategy"))
            if snapshot.get("strategy") is not None
            else None
        ),
        "dispatch_completed": dispatch_completed,
        "authorization_session_id": (
            _safe_text(snapshot.get("session_id"))
            if snapshot.get("session_id") is not None
            else None
        ),
    }


def _title(status: OperationalExecutionStatus) -> str:
    return {
        OperationalExecutionStatus.COMPLETED: "Ejecución completada",
        OperationalExecutionStatus.COMPLETED_WITH_RECOVERY: (
            "Ejecución completada con recuperación"
        ),
        OperationalExecutionStatus.PARTIALLY_COMPLETED: (
            "Ejecución parcialmente completada"
        ),
        OperationalExecutionStatus.CANCELLED: "Ejecución cancelada",
        OperationalExecutionStatus.FAILED: "Ejecución fallida",
        OperationalExecutionStatus.USER_ACTION_REQUIRED: (
            "Se requiere una acción del usuario"
        ),
    }[status]


def _final_message(
    status: OperationalExecutionStatus,
    actions: tuple[str, ...],
) -> str:
    if actions:
        return actions[0]
    return {
        OperationalExecutionStatus.COMPLETED: "El objetivo se completó correctamente.",
        OperationalExecutionStatus.COMPLETED_WITH_RECOVERY: (
            "El objetivo se completó después de una recuperación controlada."
        ),
        OperationalExecutionStatus.PARTIALLY_COMPLETED: (
            "La ejecución terminó con resultados parciales."
        ),
        OperationalExecutionStatus.CANCELLED: "La ejecución fue cancelada.",
        OperationalExecutionStatus.FAILED: "La ejecución no pudo completarse.",
        OperationalExecutionStatus.USER_ACTION_REQUIRED: (
            "La ejecución necesita una acción concreta para continuar."
        ),
    }[status]


def _safe_text(value: object) -> str:
    text = " ".join(str(value).replace("\ufeff", "").split())
    lowered = text.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return "[redacted]"
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", text)
    return text[:_MAX_VISIBLE_TEXT]


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))

"""Deterministic user-facing reports for supervised Atlas executions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Mapping

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
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

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
        if self.warnings:
            lines.append("Advertencias:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        if self.pending_user_actions:
            lines.append("Acción necesaria:")
            lines.extend(f"- {action}" for action in self.pending_user_actions)
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
        replaced_ids = {
            record.failed_step
            for record in session.replan_history
            if record.failed_step is not None
        }
        steps = tuple(
            self._step_report(
                snapshot,
                descriptions.get(step_id, step_id),
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
        )

    @staticmethod
    def _step_report(
        snapshot: StepExecutionSnapshot,
        description: str,
        *,
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
        return OperationalStepReport(
            step_id=_safe_text(snapshot.step_id),
            description=description,
            state=snapshot.state.value.upper(),
            attempts=snapshot.attempt_count,
            duration_seconds=duration,
            error=_safe_text(snapshot.error) if snapshot.error is not None else None,
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
        if (
            session.state is ExecutionState.WAITING_CONFIRMATION
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
    return (
        f"{marker} {step.step_id}: {step.description} "
        f"({attempts}{duration}){suffix}{error}"
    )


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
    text = " ".join(str(value).split())
    lowered = text.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return "[redacted]"
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", text)
    return text[:_MAX_VISIBLE_TEXT]


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))

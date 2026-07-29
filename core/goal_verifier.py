"""Deterministic, evidence-based verification of execution objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from core.acceptance_criteria import (
    AcceptanceCriterion,
    AcceptanceCriterionKind,
)
from core.execution_context import ExecutionContextSnapshot
from core.execution_trace import ExecutionTrace, TraceEventStatus

if TYPE_CHECKING:
    from core.execution_plan_executor import PlanExecutionResult, StepExecutionResult
    from core.planner import ExecutionPlan


MAX_VERIFICATION_EVIDENCE_ITEMS = 64
MAX_VERIFICATION_RESOURCE_BYTES = 1_048_576
MAX_VISIBLE_EVIDENCE_TEXT = 240
_SENSITIVE_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
)


class GoalVerificationReason(str, Enum):
    """Stable legacy-compatible reasons for deterministic verification."""

    SUCCESS = "SUCCESS"
    MISSING_REQUIRED_OUTPUTS = "MISSING_REQUIRED_OUTPUTS"
    OUTPUT_VALIDATION_FAILED = "OUTPUT_VALIDATION_FAILED"
    PLAN_FAILED = "PLAN_FAILED"
    PLAN_CANCELLED = "PLAN_CANCELLED"
    PLAN_BLOCKED = "PLAN_BLOCKED"
    INVALID_PLAN = "INVALID_PLAN"
    INVALID_OUTPUT_BINDING = "INVALID_OUTPUT_BINDING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    USER_ACTION_REQUIRED = "USER_ACTION_REQUIRED"
    UNKNOWN = "UNKNOWN"


class GoalVerificationStatus(str, Enum):
    """Closed objective-verification status independent from execution status."""

    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"
    USER_ACTION_REQUIRED = "USER_ACTION_REQUIRED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CriterionEvaluationStatus(str, Enum):
    """Closed status for one evaluated acceptance criterion."""

    SATISFIED = "SATISFIED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"
    USER_ACTION_REQUIRED = "USER_ACTION_REQUIRED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class OutputValidatorKind(str, Enum):
    """Declarative validators supported for legacy plan outputs."""

    EXISTS = "exists"
    NOT_NULL = "not_null"
    NON_EMPTY_COLLECTION = "non_empty_collection"
    NON_EMPTY_STRING = "non_empty_string"
    BOOLEAN_TRUE = "boolean_true"
    NON_EMPTY = "non_empty"


@dataclass(frozen=True, slots=True)
class CriterionEvaluation:
    """Sanitized evidence and outcome for one declared criterion."""

    criterion_id: str
    kind: str
    description: str
    source: str
    expected_value: str | None
    observed_value: str | None
    required: bool
    status: CriterionEvaluationStatus
    reason: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.evidence) > MAX_VERIFICATION_EVIDENCE_ITEMS:
            raise ValueError("criterion evidence exceeds the safe item limit.")
        object.__setattr__(
            self,
            "evidence",
            tuple(_safe_text(item) for item in self.evidence),
        )
        for field_name in ("description", "source", "reason"):
            object.__setattr__(
                self,
                field_name,
                _safe_text(getattr(self, field_name)),
            )
        for field_name in ("expected_value", "observed_value"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _safe_text(value))

    def to_dict(self) -> dict[str, object]:
        return {
            "criterion_id": self.criterion_id,
            "kind": self.kind,
            "description": self.description,
            "source": self.source,
            "expected_value": self.expected_value,
            "observed_value": self.observed_value,
            "required": self.required,
            "status": self.status.value,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class GoalVerificationResult:
    """Serializable objective outcome while retaining legacy fields."""

    satisfied: bool
    reason: GoalVerificationReason
    missing_outputs: tuple[str, ...] = ()
    verified_outputs: tuple[str, ...] = ()
    verification_status: GoalVerificationStatus | None = None
    criteria: tuple[CriterionEvaluation, ...] = ()
    plan_id: str | None = None
    session_id: str | None = None
    objective: str | None = None
    execution_status: str | None = None
    evidence: tuple[str, ...] = ()
    resources_checked: tuple[str, ...] = ()
    verified_value: str | None = None
    required_action: str | None = None
    message: str | None = None
    verified_at: str | None = None

    def __post_init__(self) -> None:
        if self.verification_status is None:
            object.__setattr__(
                self,
                "verification_status",
                (
                    GoalVerificationStatus.VERIFIED
                    if self.satisfied
                    else GoalVerificationStatus.NOT_VERIFIED
                ),
            )
        elif not isinstance(self.verification_status, GoalVerificationStatus):
            object.__setattr__(
                self,
                "verification_status",
                GoalVerificationStatus(self.verification_status),
            )
        object.__setattr__(self, "missing_outputs", tuple(self.missing_outputs))
        object.__setattr__(self, "verified_outputs", tuple(self.verified_outputs))
        object.__setattr__(self, "criteria", tuple(self.criteria))
        if len(self.criteria) > MAX_VERIFICATION_EVIDENCE_ITEMS:
            raise ValueError("verification criteria exceed the safe item limit.")
        if len(self.evidence) > MAX_VERIFICATION_EVIDENCE_ITEMS:
            raise ValueError("verification evidence exceeds the safe item limit.")
        object.__setattr__(
            self,
            "evidence",
            tuple(_safe_text(item) for item in self.evidence),
        )
        object.__setattr__(
            self,
            "resources_checked",
            tuple(
                _safe_text(item)
                for item in self.resources_checked[
                    :MAX_VERIFICATION_EVIDENCE_ITEMS
                ]
            ),
        )
        for field_name in (
            "objective",
            "verified_value",
            "required_action",
            "message",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _safe_text(value))

    @property
    def satisfied_criteria(self) -> int:
        return sum(
            item.status is CriterionEvaluationStatus.SATISFIED
            for item in self.criteria
        )

    @property
    def failed_criteria(self) -> int:
        return sum(
            item.status is CriterionEvaluationStatus.FAILED
            for item in self.criteria
        )

    @property
    def unevaluable_criteria(self) -> int:
        return sum(
            item.status
            in {
                CriterionEvaluationStatus.INCONCLUSIVE,
                CriterionEvaluationStatus.USER_ACTION_REQUIRED,
            }
            for item in self.criteria
        )


class GoalVerifier:
    """Verify declared criteria from plan and real execution evidence only."""

    def verify(
        self,
        plan: "ExecutionPlan",
        execution_result: "PlanExecutionResult",
        *,
        trace: ExecutionTrace | None = None,
    ) -> GoalVerificationResult:
        """Return a deterministic verification result without executing tools."""
        _trace(trace, "goal_verification_started", TraceEventStatus.STARTED.value)
        result = self._verify(plan, execution_result)
        trace_action, trace_status = _trace_outcome(result.verification_status)
        _trace(
            trace,
            trace_action,
            trace_status,
            {
                "reason": result.reason.value,
                "verification_status": result.verification_status.value,
                "criterion_count": len(result.criteria),
                "satisfied_criterion_count": result.satisfied_criteria,
                "failed_criterion_count": result.failed_criteria,
            },
        )
        return result

    def _verify(
        self,
        plan: "ExecutionPlan",
        execution_result: "PlanExecutionResult",
    ) -> GoalVerificationResult:
        from core.planner import ExecutionPlan

        if not isinstance(plan, ExecutionPlan):
            return _terminal_result(
                plan=None,
                execution_result=execution_result,
                status=GoalVerificationStatus.NOT_VERIFIED,
                reason=GoalVerificationReason.INVALID_PLAN,
                message="El objetivo no se verificó porque el plan es inválido.",
            )

        status = execution_result.plan_status
        if status == "cancelled" or execution_result.cancelled:
            return _terminal_result(
                plan=plan,
                execution_result=execution_result,
                status=GoalVerificationStatus.NOT_VERIFIED,
                reason=GoalVerificationReason.PLAN_CANCELLED,
                message="La ejecución fue cancelada; el objetivo no está verificado.",
            )
        if status in {"blocked", "interrupted"} or execution_result.blocked:
            return _terminal_result(
                plan=plan,
                execution_result=execution_result,
                status=GoalVerificationStatus.NOT_VERIFIED,
                reason=GoalVerificationReason.PLAN_BLOCKED,
                message="La ejecución quedó bloqueada; el objetivo no está verificado.",
            )
        if status == "blocked_confirmation":
            return _terminal_result(
                plan=plan,
                execution_result=execution_result,
                status=GoalVerificationStatus.USER_ACTION_REQUIRED,
                reason=GoalVerificationReason.USER_ACTION_REQUIRED,
                message="La verificación requiere confirmar primero la ejecución.",
                required_action="Confirma o cancela el plan pendiente.",
            )
        if status == "rejected":
            return _terminal_result(
                plan=plan,
                execution_result=execution_result,
                status=GoalVerificationStatus.NOT_VERIFIED,
                reason=GoalVerificationReason.INVALID_PLAN,
                message="El plan fue rechazado; el objetivo no está verificado.",
            )
        if execution_result.error_code == "EXECUTION_VARIABLE_BINDING_FAILED":
            return _terminal_result(
                plan=plan,
                execution_result=execution_result,
                status=GoalVerificationStatus.NOT_VERIFIED,
                reason=GoalVerificationReason.INVALID_OUTPUT_BINDING,
                message="Falló un enlace de salida requerido.",
            )
        if not execution_result.success or status != "completed":
            return _terminal_result(
                plan=plan,
                execution_result=execution_result,
                status=GoalVerificationStatus.NOT_VERIFIED,
                reason=GoalVerificationReason.PLAN_FAILED,
                message="La ejecución técnica falló; el objetivo no está verificado.",
            )

        criteria = tuple(plan.acceptance_criteria)
        evaluations = [
            _evaluate_criterion(criterion, plan, execution_result)
            for criterion in criteria
        ]
        evaluations.extend(_evaluate_legacy_outputs(plan, execution_result))

        if not evaluations:
            return _terminal_result(
                plan=plan,
                execution_result=execution_result,
                status=GoalVerificationStatus.INCONCLUSIVE,
                reason=GoalVerificationReason.INSUFFICIENT_EVIDENCE,
                message=(
                    "La ejecución terminó, pero no existen criterios suficientes "
                    "para verificar el objetivo."
                ),
            )

        verification_status = _overall_status(tuple(evaluations))
        reason = _overall_reason(verification_status, tuple(evaluations))
        evidence = tuple(
            dict.fromkeys(
                item
                for evaluation in evaluations
                for item in evaluation.evidence
            )
        )[:MAX_VERIFICATION_EVIDENCE_ITEMS]
        resources = tuple(
            dict.fromkeys(
                criterion.resource_path
                for criterion in criteria
                if criterion.resource_path is not None
                and any(
                    evaluation.criterion_id == criterion.criterion_id
                    and evaluation.status
                    is CriterionEvaluationStatus.SATISFIED
                    for evaluation in evaluations
                )
            )
        )
        required_action = (
            "Confirma o revisa la acción pendiente."
            if verification_status is GoalVerificationStatus.USER_ACTION_REQUIRED
            else (
                "Revisa los criterios fallidos y el recurso producido."
                if verification_status is GoalVerificationStatus.NOT_VERIFIED
                else None
            )
        )
        return GoalVerificationResult(
            satisfied=verification_status is GoalVerificationStatus.VERIFIED,
            reason=reason,
            missing_outputs=tuple(
                _legacy_output_name(item.criterion_id)
                for item in evaluations
                if item.required
                and item.status is CriterionEvaluationStatus.FAILED
            ),
            verified_outputs=tuple(
                _legacy_output_name(item.criterion_id)
                for item in evaluations
                if item.status is CriterionEvaluationStatus.SATISFIED
            ),
            verification_status=verification_status,
            criteria=tuple(evaluations),
            plan_id=_optional_metadata_text(
                execution_result.metadata.get("plan_signature")
            ),
            objective=plan.goal,
            execution_status=execution_result.plan_status,
            evidence=evidence,
            resources_checked=resources,
            verified_value=(
                _safe_observed(execution_result.output)
                if verification_status is GoalVerificationStatus.VERIFIED
                else None
            ),
            required_action=required_action,
            message=_verification_message(verification_status),
            verified_at=_verification_timestamp(execution_result),
        )


def normalize_required_outputs(values: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize required output names at model boundaries."""
    if values is None:
        return ()
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("required_outputs must contain non-empty strings.")
        name = value.strip()
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


def normalize_output_validators(
    values: Mapping[str, Sequence[OutputValidatorKind | str]] | None,
) -> Mapping[str, tuple[OutputValidatorKind, ...]]:
    """Normalize declarative output validators at model boundaries."""
    if values is None:
        return {}
    normalized: dict[str, tuple[OutputValidatorKind, ...]] = {}
    for raw_name, raw_validators in values.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("output validator names must be non-empty strings.")
        if isinstance(raw_validators, (str, bytes)) or not isinstance(
            raw_validators,
            Sequence,
        ):
            raise ValueError("output validators must be sequences.")
        validator_tuple = tuple(_validator_kind(item) for item in raw_validators)
        if not validator_tuple:
            raise ValueError("output validators cannot be empty.")
        normalized[raw_name.strip()] = validator_tuple
    return normalized


def goal_verification_result_to_dict(
    result: GoalVerificationResult | None,
) -> dict[str, Any] | None:
    """Serialize verification while preserving compatibility fields."""
    if result is None:
        return None
    return {
        "satisfied": result.satisfied,
        "reason": result.reason.value,
        "missing_outputs": list(result.missing_outputs),
        "verified_outputs": list(result.verified_outputs),
        "verification_status": result.verification_status.value,
        "criteria": [item.to_dict() for item in result.criteria],
        "plan_id": result.plan_id,
        "session_id": result.session_id,
        "objective": result.objective,
        "execution_status": result.execution_status,
        "evidence": list(result.evidence),
        "resources_checked": list(result.resources_checked),
        "verified_value": result.verified_value,
        "required_action": result.required_action,
        "message": result.message,
        "verified_at": result.verified_at,
    }


def goal_verification_result_from_dict(
    payload: Mapping[str, Any] | None,
) -> GoalVerificationResult | None:
    """Load new or legacy persisted verification results."""
    if payload is None:
        return None
    satisfied = _bool(payload, "satisfied")
    criteria_payload = payload.get("criteria", [])
    if not isinstance(criteria_payload, list):
        raise ValueError("criteria must be a list.")
    return GoalVerificationResult(
        satisfied=satisfied,
        reason=GoalVerificationReason(_str(payload, "reason")),
        missing_outputs=_str_tuple(payload, "missing_outputs"),
        verified_outputs=_str_tuple(payload, "verified_outputs"),
        verification_status=GoalVerificationStatus(
            payload.get(
                "verification_status",
                (
                    GoalVerificationStatus.VERIFIED.value
                    if satisfied
                    else GoalVerificationStatus.NOT_VERIFIED.value
                ),
            )
        ),
        criteria=tuple(
            _criterion_evaluation_from_dict(item)
            for item in criteria_payload
        ),
        plan_id=_optional_str(payload, "plan_id"),
        session_id=_optional_str(payload, "session_id"),
        objective=_optional_str(payload, "objective"),
        execution_status=_optional_str(payload, "execution_status"),
        evidence=_optional_str_tuple(payload, "evidence"),
        resources_checked=_optional_str_tuple(payload, "resources_checked"),
        verified_value=_optional_str(payload, "verified_value"),
        required_action=_optional_str(payload, "required_action"),
        message=_optional_str(payload, "message"),
        verified_at=_optional_str(payload, "verified_at"),
    )


def _evaluate_criterion(
    criterion: AcceptanceCriterion,
    plan: "ExecutionPlan",
    execution_result: "PlanExecutionResult",
) -> CriterionEvaluation:
    outputs = _completed_outputs(execution_result)
    completed = _completed_step_ids(execution_result)
    source = _criterion_source(criterion)
    kind = criterion.kind

    if kind is AcceptanceCriterionKind.STEP_COMPLETED:
        observed = criterion.source_step_id in completed
        return _boolean_evaluation(
            criterion,
            source,
            observed,
            f"step:{criterion.source_step_id}:completed",
            "El paso requerido terminó correctamente.",
            "El paso requerido no terminó correctamente.",
        )

    if kind in {
        AcceptanceCriterionKind.OUTPUT_EXISTS,
        AcceptanceCriterionKind.OUTPUT_EQUALS,
        AcceptanceCriterionKind.OUTPUT_CONTAINS,
    }:
        source_state = _step_value(
            outputs,
            criterion.source_step_id,
            criterion.source_path,
        )
        if not source_state[0]:
            return _evaluation(
                criterion,
                source,
                CriterionEvaluationStatus.INCONCLUSIVE,
                "No existe la salida declarada.",
                observed=None,
                evidence=(),
            )
        source_value = source_state[1]
        if kind is AcceptanceCriterionKind.OUTPUT_EXISTS:
            return _evaluation(
                criterion,
                source,
                CriterionEvaluationStatus.SATISFIED,
                "La salida declarada existe.",
                observed=source_value,
                evidence=(f"output:{criterion.source_step_id}:exists",),
            )
        if kind is AcceptanceCriterionKind.OUTPUT_CONTAINS:
            expected = criterion.expected_value
            matches = (
                isinstance(source_value, str)
                and isinstance(expected, str)
                and expected in source_value
            ) or (
                isinstance(source_value, (list, tuple, set, Mapping))
                and expected in source_value
            )
            return _comparison_evaluation(
                criterion,
                source,
                matches,
                source_value,
                expected,
                "La salida contiene el valor esperado.",
                "La salida no contiene el valor esperado.",
            )
        comparison_state = (
            (criterion.expected_value is not None, criterion.expected_value)
            if criterion.comparison_step_id is None
            else _step_value(
                outputs,
                criterion.comparison_step_id,
                criterion.comparison_path,
            )
        )
        if not comparison_state[0]:
            return _evaluation(
                criterion,
                source,
                CriterionEvaluationStatus.INCONCLUSIVE,
                "No existe la salida de comparación declarada.",
                observed=source_value,
                evidence=(),
            )
        return _comparison_evaluation(
            criterion,
            source,
            source_value == comparison_state[1],
            source_value,
            comparison_state[1],
            "Las salidas estructuradas coinciden.",
            "Las salidas estructuradas no coinciden.",
        )

    if kind in {
        AcceptanceCriterionKind.RESOURCE_EXISTS,
        AcceptanceCriterionKind.RESOURCE_READABLE,
        AcceptanceCriterionKind.RESOURCE_CONTENT_EQUALS,
    }:
        resource = _declared_resource(
            criterion,
            plan,
            execution_result,
            completed_step_ids=completed,
        )
        if resource is None:
            return _evaluation(
                criterion,
                source,
                CriterionEvaluationStatus.INCONCLUSIVE,
                "El recurso no está declarado por un paso de escritura completado.",
                observed=None,
                evidence=(),
            )
        if kind is AcceptanceCriterionKind.RESOURCE_EXISTS:
            exists = resource.exists() and resource.is_file()
            return _boolean_evaluation(
                criterion,
                source,
                exists,
                f"resource:{criterion.resource_path}:exists",
                "El recurso declarado existe.",
                "El recurso declarado no existe.",
            )
        readable = _read_declared_text(resource)
        if isinstance(readable, _ResourceReadFailure):
            return _evaluation(
                criterion,
                source,
                (
                    CriterionEvaluationStatus.FAILED
                    if readable.definitive
                    else CriterionEvaluationStatus.INCONCLUSIVE
                ),
                readable.reason,
                observed=None,
                evidence=(f"resource:{criterion.resource_path}:unreadable",),
            )
        if kind is AcceptanceCriterionKind.RESOURCE_READABLE:
            return _evaluation(
                criterion,
                source,
                CriterionEvaluationStatus.SATISFIED,
                "El recurso declarado puede leerse de forma segura.",
                observed=f"<text length={len(readable)}>",
                evidence=(f"resource:{criterion.resource_path}:readable",),
            )
        comparison_state = _step_value(
            outputs,
            criterion.comparison_step_id,
            criterion.comparison_path,
        )
        if not comparison_state[0]:
            return _evaluation(
                criterion,
                source,
                CriterionEvaluationStatus.INCONCLUSIVE,
                "No existe el valor esperado para comparar el recurso.",
                observed=f"<text length={len(readable)}>",
                evidence=(),
            )
        return _comparison_evaluation(
            criterion,
            source,
            readable == comparison_state[1],
            readable,
            comparison_state[1],
            "El contenido del recurso coincide con el resultado esperado.",
            "El contenido del recurso no coincide con el resultado esperado.",
            evidence=(f"resource:{criterion.resource_path}:content_compared",),
        )

    if kind is AcceptanceCriterionKind.EXPECTED_TOOL_USED:
        step = next(
            (
                item
                for item in plan.ordered_steps
                if item.id == criterion.source_step_id
            ),
            None,
        )
        used = (
            step is not None
            and step.id in completed
            and step.tool == criterion.tool_name
        )
        return _boolean_evaluation(
            criterion,
            source,
            used,
            f"tool:{criterion.tool_name}:step:{criterion.source_step_id}",
            "La herramienta prevista se ejecutó correctamente.",
            "La herramienta prevista no se ejecutó correctamente.",
        )

    if kind is AcceptanceCriterionKind.EXPECTED_STEP_COUNT:
        observed = len(completed)
        expected = criterion.expected_count
        return _comparison_evaluation(
            criterion,
            source,
            expected is not None and observed == expected,
            observed,
            expected,
            "Se completó el número esperado de pasos.",
            "No se completó el número esperado de pasos.",
        )

    if kind in {
        AcceptanceCriterionKind.NO_PENDING_CONFIRMATIONS,
        AcceptanceCriterionKind.USER_CONFIRMATION_REQUIRED,
    }:
        granted = (
            not plan.requires_confirmation
            or execution_result.metadata.get("confirmation_granted") is True
        )
        if granted:
            return _evaluation(
                criterion,
                source,
                CriterionEvaluationStatus.SATISFIED,
                "No quedan confirmaciones pendientes.",
                observed=True,
                evidence=("confirmation:satisfied",),
            )
        return _evaluation(
            criterion,
            source,
            CriterionEvaluationStatus.USER_ACTION_REQUIRED,
            "La confirmación requerida continúa pendiente.",
            observed=False,
            evidence=("confirmation:pending",),
        )

    if kind is AcceptanceCriterionKind.NO_CRITICAL_FAILURES:
        failed = set(execution_result.failed_steps)
        critical_failed = any(
            step.id in failed and getattr(step, "criticality", None) is not None
            for step in plan.ordered_steps
        )
        passed = execution_result.success and not critical_failed
        return _boolean_evaluation(
            criterion,
            source,
            passed,
            "execution:no_critical_failures",
            "No existen fallos críticos.",
            "Existe al menos un fallo crítico.",
        )

    return _evaluation(
        criterion,
        source,
        CriterionEvaluationStatus.INCONCLUSIVE,
        "El criterio no pudo evaluarse.",
        observed=None,
        evidence=(),
    )


def _completed_outputs(
    execution_result: "PlanExecutionResult",
) -> dict[str, object]:
    outputs: dict[str, object] = {}
    snapshot = execution_result.metadata.get("execution_context_snapshot")
    if isinstance(snapshot, ExecutionContextSnapshot):
        outputs.update(snapshot.results_by_step_id)
    outputs.update({
        item.step_id: item.output
        for item in execution_result.step_results
        if item.success and item.status == "completed"
    })
    return outputs


def _completed_step_ids(
    execution_result: "PlanExecutionResult",
) -> set[str]:
    completed = set(execution_result.completed_steps)
    snapshot = execution_result.metadata.get("execution_context_snapshot")
    if isinstance(snapshot, ExecutionContextSnapshot):
        completed.update(
            step_id
            for step_id, state in snapshot.step_states.items()
            if state == "SUCCEEDED"
        )
    return completed


def _evaluate_legacy_outputs(
    plan: "ExecutionPlan",
    execution_result: "PlanExecutionResult",
) -> tuple[CriterionEvaluation, ...]:
    output = execution_result.output
    evaluations: list[CriterionEvaluation] = []
    for name in plan.required_outputs:
        exists, value = _output_value(output, name)
        criterion = AcceptanceCriterion(
            criterion_id=f"legacy.required.{name}",
            kind=AcceptanceCriterionKind.OUTPUT_EXISTS,
            description=f"Required plan output '{name}' exists.",
            source_step_id="plan_output",
        )
        evaluations.append(
            _evaluation(
                criterion,
                f"plan.output.{name}",
                (
                    CriterionEvaluationStatus.SATISFIED
                    if exists
                    else CriterionEvaluationStatus.FAILED
                ),
                (
                    "La salida requerida existe."
                    if exists
                    else "Falta la salida requerida."
                ),
                observed=value if exists else None,
                evidence=((f"plan_output:{name}:exists",) if exists else ()),
            )
        )
    for name, validators in plan.output_validators.items():
        try:
            valid = _validators_satisfied(
                _output_value(output, name),
                tuple(_validator_kind(item) for item in validators),
            )
        except ValueError:
            valid = False
        criterion = AcceptanceCriterion(
            criterion_id=f"legacy.validator.{name}",
            kind=AcceptanceCriterionKind.OUTPUT_EXISTS,
            description=f"Declared validators for '{name}' are satisfied.",
            source_step_id="plan_output",
        )
        evaluations.append(
            _evaluation(
                criterion,
                f"plan.output.{name}",
                (
                    CriterionEvaluationStatus.SATISFIED
                    if valid
                    else CriterionEvaluationStatus.FAILED
                ),
                (
                    "Los validadores declarados se cumplen."
                    if valid
                    else "Los validadores declarados no se cumplen."
                ),
                observed=_output_value(output, name)[1],
                evidence=((f"plan_output:{name}:validated",) if valid else ()),
            )
        )
    return tuple(evaluations)


def _declared_resource(
    criterion: AcceptanceCriterion,
    plan: "ExecutionPlan",
    execution_result: "PlanExecutionResult",
    *,
    completed_step_ids: set[str] | None = None,
) -> Path | None:
    if criterion.resource_path is None:
        return None
    completed = (
        _completed_step_ids(execution_result)
        if completed_step_ids is None
        else completed_step_ids
    )
    for step in plan.ordered_steps:
        if step.id not in completed or step.tool != "write_file":
            continue
        path = step.arguments.get("path")
        if isinstance(path, str) and path == criterion.resource_path:
            return Path(path)
    return None


@dataclass(frozen=True, slots=True)
class _ResourceReadFailure:
    reason: str
    definitive: bool


def _read_declared_text(path: Path) -> str | _ResourceReadFailure:
    try:
        if not path.exists() or not path.is_file():
            return _ResourceReadFailure("El recurso declarado no existe.", True)
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            return _ResourceReadFailure(
                "No se verifica contenido mediante enlaces simbólicos.",
                False,
            )
        if path.stat().st_size > MAX_VERIFICATION_RESOURCE_BYTES:
            return _ResourceReadFailure(
                "El recurso supera el límite seguro de verificación.",
                False,
            )
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return _ResourceReadFailure(
            "El recurso declarado no pudo leerse de forma segura.",
            True,
        )


def _step_value(
    outputs: Mapping[str, object],
    step_id: str | None,
    path: tuple[str | int, ...],
) -> tuple[bool, object | None]:
    if step_id is None or step_id not in outputs:
        return False, None
    value: object = outputs[step_id]
    for segment in path:
        if isinstance(value, Mapping) and isinstance(segment, str):
            if segment not in value:
                return False, None
            value = value[segment]
            continue
        if (
            isinstance(value, (list, tuple))
            and isinstance(segment, int)
            and segment < len(value)
        ):
            value = value[segment]
            continue
        return False, None
    return True, value


def _overall_status(
    evaluations: tuple[CriterionEvaluation, ...],
) -> GoalVerificationStatus:
    required = tuple(item for item in evaluations if item.required)
    considered = required or evaluations
    if any(
        item.status is CriterionEvaluationStatus.USER_ACTION_REQUIRED
        for item in considered
    ):
        return GoalVerificationStatus.USER_ACTION_REQUIRED
    if any(
        item.status is CriterionEvaluationStatus.FAILED
        for item in considered
    ):
        return GoalVerificationStatus.NOT_VERIFIED
    if any(
        item.status is CriterionEvaluationStatus.INCONCLUSIVE
        for item in considered
    ):
        if any(
            item.status is CriterionEvaluationStatus.SATISFIED
            for item in considered
        ):
            return GoalVerificationStatus.PARTIALLY_VERIFIED
        return GoalVerificationStatus.INCONCLUSIVE
    if considered and all(
        item.status
        in {
            CriterionEvaluationStatus.SATISFIED,
            CriterionEvaluationStatus.NOT_APPLICABLE,
        }
        for item in considered
    ):
        return GoalVerificationStatus.VERIFIED
    return GoalVerificationStatus.INCONCLUSIVE


def _overall_reason(
    status: GoalVerificationStatus,
    evaluations: tuple[CriterionEvaluation, ...],
) -> GoalVerificationReason:
    if status is GoalVerificationStatus.VERIFIED:
        return GoalVerificationReason.SUCCESS
    if status is GoalVerificationStatus.USER_ACTION_REQUIRED:
        return GoalVerificationReason.USER_ACTION_REQUIRED
    if status in {
        GoalVerificationStatus.INCONCLUSIVE,
        GoalVerificationStatus.PARTIALLY_VERIFIED,
    }:
        return GoalVerificationReason.INSUFFICIENT_EVIDENCE
    if any(item.criterion_id.startswith("legacy.required.") for item in evaluations):
        return GoalVerificationReason.MISSING_REQUIRED_OUTPUTS
    return GoalVerificationReason.OUTPUT_VALIDATION_FAILED


def _legacy_output_name(criterion_id: str) -> str:
    for prefix in ("legacy.required.", "legacy.validator."):
        if criterion_id.startswith(prefix):
            return criterion_id[len(prefix) :]
    return criterion_id


def _terminal_result(
    *,
    plan: "ExecutionPlan | None",
    execution_result: "PlanExecutionResult",
    status: GoalVerificationStatus,
    reason: GoalVerificationReason,
    message: str,
    required_action: str | None = None,
) -> GoalVerificationResult:
    evaluations = (
        ()
        if plan is None
        else tuple(
            _evaluate_criterion(
                criterion,
                plan,
                execution_result,
            )
            for criterion in plan.acceptance_criteria
        )
    )
    return GoalVerificationResult(
        satisfied=False,
        reason=reason,
        verification_status=status,
        plan_id=_optional_metadata_text(
            execution_result.metadata.get("plan_signature")
        ),
        objective=None if plan is None else plan.goal,
        execution_status=execution_result.plan_status,
        criteria=evaluations,
        evidence=tuple(
            dict.fromkeys(
                evidence
                for evaluation in evaluations
                for evidence in evaluation.evidence
            )
        )[:MAX_VERIFICATION_EVIDENCE_ITEMS],
        resources_checked=tuple(
            dict.fromkeys(
                criterion.resource_path
                for criterion in (() if plan is None else plan.acceptance_criteria)
                if criterion.resource_path is not None
                and any(
                    evaluation.criterion_id == criterion.criterion_id
                    and evaluation.status
                    is CriterionEvaluationStatus.SATISFIED
                    for evaluation in evaluations
                )
            )
        ),
        required_action=required_action,
        message=message,
        verified_at=_verification_timestamp(execution_result),
    )


def _evaluation(
    criterion: AcceptanceCriterion,
    source: str,
    status: CriterionEvaluationStatus,
    reason: str,
    *,
    observed: object | None,
    evidence: tuple[str, ...],
    expected: object | None = None,
) -> CriterionEvaluation:
    return CriterionEvaluation(
        criterion_id=criterion.criterion_id,
        kind=criterion.kind.value,
        description=criterion.description,
        source=source,
        expected_value=_safe_observed(
            criterion.expected_value if expected is None else expected
        ),
        observed_value=_safe_observed(observed),
        required=criterion.required,
        status=status,
        reason=reason,
        evidence=tuple(_safe_text(item) for item in evidence),
    )


def _boolean_evaluation(
    criterion: AcceptanceCriterion,
    source: str,
    passed: bool,
    evidence: str,
    success_reason: str,
    failure_reason: str,
) -> CriterionEvaluation:
    return _evaluation(
        criterion,
        source,
        (
            CriterionEvaluationStatus.SATISFIED
            if passed
            else CriterionEvaluationStatus.FAILED
        ),
        success_reason if passed else failure_reason,
        observed=passed,
        evidence=((evidence,) if passed else ()),
        expected=True,
    )


def _comparison_evaluation(
    criterion: AcceptanceCriterion,
    source: str,
    passed: bool,
    observed: object,
    expected: object,
    success_reason: str,
    failure_reason: str,
    *,
    evidence: tuple[str, ...] = ("comparison:performed",),
) -> CriterionEvaluation:
    return _evaluation(
        criterion,
        source,
        (
            CriterionEvaluationStatus.SATISFIED
            if passed
            else CriterionEvaluationStatus.FAILED
        ),
        success_reason if passed else failure_reason,
        observed=observed,
        expected=expected,
        evidence=evidence,
    )


def _criterion_source(criterion: AcceptanceCriterion) -> str:
    if criterion.resource_path is not None:
        return f"resource:{criterion.resource_path}"
    if criterion.source_step_id is not None:
        path = ".".join(str(item) for item in criterion.source_path)
        return (
            f"step:{criterion.source_step_id}.output"
            + (f".{path}" if path else "")
        )
    if criterion.tool_name is not None:
        return f"tool:{criterion.tool_name}"
    return "execution"


def _verification_message(status: GoalVerificationStatus) -> str:
    return {
        GoalVerificationStatus.VERIFIED: (
            "El objetivo solicitado se cumplió y fue verificado con evidencia."
        ),
        GoalVerificationStatus.PARTIALLY_VERIFIED: (
            "El objetivo solo pudo verificarse parcialmente."
        ),
        GoalVerificationStatus.NOT_VERIFIED: (
            "La ejecución terminó, pero el objetivo no quedó verificado."
        ),
        GoalVerificationStatus.INCONCLUSIVE: (
            "No existe evidencia suficiente para verificar el objetivo."
        ),
        GoalVerificationStatus.USER_ACTION_REQUIRED: (
            "La verificación requiere una acción del usuario."
        ),
        GoalVerificationStatus.NOT_APPLICABLE: (
            "La verificación no es aplicable a esta ejecución."
        ),
    }[status]


def _trace_outcome(
    status: GoalVerificationStatus,
) -> tuple[str, str]:
    if status is GoalVerificationStatus.VERIFIED:
        return (
            "goal_verification_succeeded",
            TraceEventStatus.FINISHED.value,
        )
    if status is GoalVerificationStatus.NOT_VERIFIED:
        return (
            "goal_verification_failed",
            TraceEventStatus.FAILED.value,
        )
    if status is GoalVerificationStatus.USER_ACTION_REQUIRED:
        return (
            "goal_verification_user_action_required",
            TraceEventStatus.FINISHED.value,
        )
    return (
        "goal_verification_inconclusive",
        TraceEventStatus.FINISHED.value,
    )


def _verification_timestamp(execution_result: "PlanExecutionResult") -> str:
    if execution_result.finished_at is not None:
        return execution_result.finished_at
    return datetime.now(timezone.utc).isoformat()


def _safe_observed(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return f"<text length={len(value)}>"
    if isinstance(value, Mapping):
        return f"<mapping size={len(value)}>"
    if isinstance(value, (list, tuple, set)):
        return f"<{type(value).__name__} length={len(value)}>"
    return f"<{type(value).__name__}>"


def _safe_text(value: object) -> str:
    normalized = " ".join(str(value).split())
    lowered = normalized.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return "[redacted]"
    normalized = re.sub(
        r"\bsk-[A-Za-z0-9_-]{8,}\b",
        "[redacted]",
        normalized,
    )
    return normalized[:MAX_VISIBLE_EVIDENCE_TEXT]


def _optional_metadata_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _validator_kind(value: OutputValidatorKind | str) -> OutputValidatorKind:
    if isinstance(value, OutputValidatorKind):
        return value
    if isinstance(value, str):
        return OutputValidatorKind(value.strip())
    raise ValueError("output validator must be OutputValidatorKind or str.")


def _output_value(output: object, name: str) -> tuple[bool, object | None]:
    if isinstance(output, Mapping) and name in output:
        return True, output[name]
    return False, None


def _validators_satisfied(
    value_state: tuple[bool, object | None],
    validators: tuple[OutputValidatorKind, ...],
) -> bool:
    exists, value = value_state
    for validator in validators:
        if validator is OutputValidatorKind.EXISTS and not exists:
            return False
        if validator is OutputValidatorKind.NOT_NULL and (not exists or value is None):
            return False
        if validator is OutputValidatorKind.NON_EMPTY_STRING and (
            not exists or not isinstance(value, str) or not value
        ):
            return False
        if validator is OutputValidatorKind.NON_EMPTY_COLLECTION and (
            not exists
            or not isinstance(value, (Mapping, list, tuple, set))
            or len(value) == 0
        ):
            return False
        if validator is OutputValidatorKind.BOOLEAN_TRUE and (
            not exists or value is not True
        ):
            return False
        if validator is OutputValidatorKind.NON_EMPTY:
            if not exists or value is None:
                return False
            if isinstance(value, str) and not value:
                return False
            if isinstance(value, (Mapping, list, tuple, set)) and len(value) == 0:
                return False
    return True


def _criterion_evaluation_from_dict(
    payload: object,
) -> CriterionEvaluation:
    if not isinstance(payload, Mapping):
        raise ValueError("criterion evaluation must be an object.")
    return CriterionEvaluation(
        criterion_id=_str(payload, "criterion_id"),
        kind=_str(payload, "kind"),
        description=_str(payload, "description"),
        source=_str(payload, "source"),
        expected_value=_optional_str(payload, "expected_value"),
        observed_value=_optional_str(payload, "observed_value"),
        required=_bool(payload, "required"),
        status=CriterionEvaluationStatus(_str(payload, "status")),
        reason=_str(payload, "reason"),
        evidence=_optional_str_tuple(payload, "evidence"),
    )


def _trace(
    trace: ExecutionTrace | None,
    action: str,
    status: str,
    details: dict[str, object] | None = None,
) -> None:
    if trace is None:
        return
    trace.add_event(
        component="GoalVerifier",
        action=action,
        status=status,
        details={} if details is None else details,
    )


def _bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a bool.")
    return value


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


def _str_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{key} must be a list of strings.")
    return tuple(value)


def _optional_str_tuple(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[str, ...]:
    if key not in payload:
        return ()
    return _str_tuple(payload, key)

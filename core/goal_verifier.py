"""Deterministic goal verification for completed execution plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from core.execution_trace import ExecutionTrace, TraceEventStatus
from core.planner import ExecutionPlan

if TYPE_CHECKING:
    from core.execution_plan_executor import PlanExecutionResult


class GoalVerificationReason(str, Enum):
    """Stable reasons produced by deterministic goal verification."""

    SUCCESS = "SUCCESS"
    MISSING_REQUIRED_OUTPUTS = "MISSING_REQUIRED_OUTPUTS"
    OUTPUT_VALIDATION_FAILED = "OUTPUT_VALIDATION_FAILED"
    PLAN_FAILED = "PLAN_FAILED"
    PLAN_CANCELLED = "PLAN_CANCELLED"
    PLAN_BLOCKED = "PLAN_BLOCKED"
    INVALID_PLAN = "INVALID_PLAN"
    INVALID_OUTPUT_BINDING = "INVALID_OUTPUT_BINDING"
    UNKNOWN = "UNKNOWN"


class OutputValidatorKind(str, Enum):
    """Declarative validators supported for plan outputs."""

    EXISTS = "exists"
    NOT_NULL = "not_null"
    NON_EMPTY_COLLECTION = "non_empty_collection"
    NON_EMPTY_STRING = "non_empty_string"
    BOOLEAN_TRUE = "boolean_true"
    NON_EMPTY = "non_empty"


@dataclass(frozen=True, slots=True)
class GoalVerificationResult:
    """Structured result for deterministic goal verification."""

    satisfied: bool
    reason: GoalVerificationReason
    missing_outputs: tuple[str, ...] = ()
    verified_outputs: tuple[str, ...] = ()


class GoalVerifier:
    """Verify whether a plan goal was satisfied using structured data only."""

    def verify(
        self,
        plan: ExecutionPlan,
        execution_result: "PlanExecutionResult",
        *,
        trace: ExecutionTrace | None = None,
    ) -> GoalVerificationResult:
        """Return a deterministic goal-verification result."""
        _trace(trace, "goal_verification_started", TraceEventStatus.STARTED.value)
        result = self._verify(plan, execution_result, trace=trace)
        _trace(
            trace,
            "goal_verification_succeeded" if result.satisfied else "goal_verification_failed",
            TraceEventStatus.FINISHED.value if result.satisfied else TraceEventStatus.FAILED.value,
            {
                "reason": result.reason.value,
                "missing_output_count": len(result.missing_outputs),
                "verified_output_count": len(result.verified_outputs),
            },
        )
        return result

    def _verify(
        self,
        plan: ExecutionPlan,
        execution_result: "PlanExecutionResult",
        *,
        trace: ExecutionTrace | None,
    ) -> GoalVerificationResult:
        if not isinstance(plan, ExecutionPlan):
            return GoalVerificationResult(False, GoalVerificationReason.INVALID_PLAN)
        status = execution_result.plan_status
        if status == "cancelled" or execution_result.cancelled:
            return GoalVerificationResult(False, GoalVerificationReason.PLAN_CANCELLED)
        if status in {
            "blocked",
            "blocked_confirmation",
            "interrupted",
        } or execution_result.blocked:
            return GoalVerificationResult(False, GoalVerificationReason.PLAN_BLOCKED)
        if status == "rejected":
            return GoalVerificationResult(False, GoalVerificationReason.INVALID_PLAN)
        if execution_result.error_code == "EXECUTION_VARIABLE_BINDING_FAILED":
            return GoalVerificationResult(False, GoalVerificationReason.INVALID_OUTPUT_BINDING)
        if not execution_result.success or status != "completed":
            return GoalVerificationResult(False, GoalVerificationReason.PLAN_FAILED)

        output = execution_result.output
        if plan.required_outputs:
            if not isinstance(output, Mapping):
                for name in plan.required_outputs:
                    _trace_missing(trace, name)
                return GoalVerificationResult(
                    False,
                    GoalVerificationReason.MISSING_REQUIRED_OUTPUTS,
                    missing_outputs=tuple(plan.required_outputs),
                )
            missing = tuple(name for name in plan.required_outputs if name not in output)
            if missing:
                for name in missing:
                    _trace_missing(trace, name)
                return GoalVerificationResult(
                    False,
                    GoalVerificationReason.MISSING_REQUIRED_OUTPUTS,
                    missing_outputs=missing,
                )

        try:
            invalid = tuple(
                name
                for name, validators in plan.output_validators.items()
                if not _validators_satisfied(
                    _output_value(output, name),
                    tuple(_validator_kind(item) for item in validators),
                )
            )
        except ValueError:
            invalid = tuple(plan.output_validators)
        if invalid:
            for name in invalid:
                _trace_invalid(trace, name)
            return GoalVerificationResult(
                False,
                GoalVerificationReason.OUTPUT_VALIDATION_FAILED,
                missing_outputs=invalid,
            )

        verified = tuple(
            dict.fromkeys(tuple(plan.required_outputs) + tuple(plan.output_validators))
        )
        return GoalVerificationResult(
            True,
            GoalVerificationReason.SUCCESS,
            verified_outputs=verified,
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
        if isinstance(raw_validators, (str, bytes)) or not isinstance(raw_validators, Sequence):
            raise ValueError("output validators must be sequences.")
        validator_tuple = tuple(_validator_kind(item) for item in raw_validators)
        if not validator_tuple:
            raise ValueError("output validators cannot be empty.")
        normalized[raw_name.strip()] = validator_tuple
    return normalized


def goal_verification_result_to_dict(
    result: GoalVerificationResult | None,
) -> dict[str, Any] | None:
    """Serialize a goal verification result to a JSON-compatible object."""
    if result is None:
        return None
    return {
        "satisfied": result.satisfied,
        "reason": result.reason.value,
        "missing_outputs": list(result.missing_outputs),
        "verified_outputs": list(result.verified_outputs),
    }


def goal_verification_result_from_dict(
    payload: Mapping[str, Any] | None,
) -> GoalVerificationResult | None:
    """Load a goal verification result from a persisted object."""
    if payload is None:
        return None
    return GoalVerificationResult(
        satisfied=_bool(payload, "satisfied"),
        reason=GoalVerificationReason(_str(payload, "reason")),
        missing_outputs=_str_tuple(payload, "missing_outputs"),
        verified_outputs=_str_tuple(payload, "verified_outputs"),
    )


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
            not exists or not isinstance(value, (Mapping, list, tuple, set)) or len(value) == 0
        ):
            return False
        if validator is OutputValidatorKind.BOOLEAN_TRUE and (not exists or value is not True):
            return False
        if validator is OutputValidatorKind.NON_EMPTY:
            if not exists or value is None:
                return False
            if isinstance(value, str) and not value:
                return False
            if isinstance(value, (Mapping, list, tuple, set)) and len(value) == 0:
                return False
    return True


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


def _trace_missing(trace: ExecutionTrace | None, output_name: str) -> None:
    _trace(
        trace,
        "goal_missing_output",
        TraceEventStatus.FAILED.value,
        {"output_name": output_name},
    )


def _trace_invalid(trace: ExecutionTrace | None, output_name: str) -> None:
    _trace(
        trace,
        "goal_output_invalid",
        TraceEventStatus.FAILED.value,
        {"output_name": output_name},
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


def _str_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings.")
    return tuple(value)

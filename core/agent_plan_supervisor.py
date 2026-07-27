"""Deterministic supervision for already produced cooperation-plan results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType

from core.agent_cooperation_automatic_execution import AgentCooperationAutomaticExecutionResult
from core.agent_cooperation_plan import (
    AgentCooperationDependency,
    AgentCooperationPlan,
    AgentCooperationPlanResult,
    AgentCooperationPlanStatus,
    AgentCooperationTaskResult,
    AgentCooperationTaskStatus,
    agent_cooperation_plan_signature,
)


MAX_SUPERVISION_TASKS = 100
MAX_SUPERVISION_RESULTS = 100
MAX_SUPERVISION_OUTPUTS = 512
MAX_SUPERVISION_OUTPUT_ITEMS = 1_024
MAX_SUPERVISION_DEPTH = 12
MAX_SUPERVISION_STRING_LENGTH = 2_000
MAX_SUPERVISION_LOGICAL_TIME = 1_000_000
MAX_SUPERVISION_METADATA_ITEMS = 32
MAX_SUPERVISION_EVENTS = 1_024
_SIGNATURE_HEX = set("0123456789abcdef")
_SENSITIVE_KEY_PARTS = (
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "private_key",
    "credential",
    "prompt",
)
_DYNAMIC_CODE_KEY_PARTS = (
    "python_path",
    "module_path",
    "import_path",
    "handler_path",
    "callback",
    "callable",
)


class AgentPlanSupervisorError(RuntimeError):
    """Base error for deterministic plan-result supervision."""


class InvalidAgentPlanSupervisorRequestError(AgentPlanSupervisorError):
    """Raised when a supervision request or policy is malformed."""


class AgentPlanSupervisorStatus(str, Enum):
    """Terminal statuses for one supervision pass."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_PLAN = "INVALID_PLAN"
    INVALID_RESULT = "INVALID_RESULT"
    INVALID_SIGNATURE = "INVALID_SIGNATURE"
    INCONSISTENT_RESULT = "INCONSISTENT_RESULT"
    LIMIT_REACHED = "LIMIT_REACHED"
    SUPERVISION_DISABLED = "SUPERVISION_DISABLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentPlanSupervisorDecisionType(str, Enum):
    """Final structured supervision decision."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class AgentPlanSupervisorPolicy:
    """Conservative opt-in supervision policy with hard upper bounds."""

    enabled: bool = False
    require_plan_signature: bool = True
    require_result_signature: bool = True
    require_all_tasks_accounted_for: bool = True
    require_dependencies_satisfied: bool = True
    require_consistent_aggregate_status: bool = True
    require_non_empty_required_outputs: bool = True
    reject_duplicate_task_results: bool = True
    reject_unknown_task_results: bool = True
    reject_sensitive_keys: bool = True
    max_tasks: int = 64
    max_results: int = 64
    max_outputs: int = 256
    max_output_items: int = 512
    max_depth: int = 8
    max_string_length: int = 1_000
    max_logical_time: int = 10_000
    minimum_success_ratio: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "require_plan_signature",
            "require_result_signature",
            "require_all_tasks_accounted_for",
            "require_dependencies_satisfied",
            "require_consistent_aggregate_status",
            "require_non_empty_required_outputs",
            "reject_duplicate_task_results",
            "reject_unknown_task_results",
            "reject_sensitive_keys",
        ):
            if type(getattr(self, name)) is not bool:
                raise InvalidAgentPlanSupervisorRequestError(f"{name} must be a bool.")
        limits = (
            ("max_tasks", MAX_SUPERVISION_TASKS),
            ("max_results", MAX_SUPERVISION_RESULTS),
            ("max_outputs", MAX_SUPERVISION_OUTPUTS),
            ("max_output_items", MAX_SUPERVISION_OUTPUT_ITEMS),
            ("max_depth", MAX_SUPERVISION_DEPTH),
            ("max_string_length", MAX_SUPERVISION_STRING_LENGTH),
            ("max_logical_time", MAX_SUPERVISION_LOGICAL_TIME),
        )
        for name, maximum in limits:
            object.__setattr__(self, name, _bounded_int(getattr(self, name), name, maximum))
        ratio = self.minimum_success_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio):
            raise InvalidAgentPlanSupervisorRequestError("minimum_success_ratio must be finite.")
        if ratio < 0.0 or ratio > 1.0:
            raise InvalidAgentPlanSupervisorRequestError("minimum_success_ratio must be between 0 and 1.")
        object.__setattr__(self, "minimum_success_ratio", float(ratio))


@dataclass(frozen=True, slots=True)
class AgentPlanSupervisorRequest:
    """Input for observing one already produced cooperation result."""

    plan: AgentCooperationPlan
    execution_result: AgentCooperationPlanResult | AgentCooperationAutomaticExecutionResult
    policy: AgentPlanSupervisorPolicy = field(default_factory=AgentPlanSupervisorPolicy)
    execution_id: str | None = None
    correlation_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, AgentCooperationPlan):
            raise InvalidAgentPlanSupervisorRequestError("plan must be AgentCooperationPlan.")
        if not isinstance(
            self.execution_result,
            (AgentCooperationPlanResult, AgentCooperationAutomaticExecutionResult),
        ):
            raise InvalidAgentPlanSupervisorRequestError(
                "execution_result must be AgentCooperationPlanResult or "
                "AgentCooperationAutomaticExecutionResult."
            )
        if not isinstance(self.policy, AgentPlanSupervisorPolicy):
            raise InvalidAgentPlanSupervisorRequestError("policy must be AgentPlanSupervisorPolicy.")
        for name in ("execution_id", "correlation_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, self.policy, "metadata")))


@dataclass(frozen=True, slots=True)
class AgentPlanSupervisorDecision:
    """Bounded explanation of a supervision decision."""

    decision: AgentPlanSupervisorDecisionType
    reasons: tuple[str, ...] = ()
    checks_passed: tuple[str, ...] = ()
    checks_failed: tuple[str, ...] = ()
    expected_tasks: tuple[str, ...] = ()
    observed_tasks: tuple[str, ...] = ()
    inconsistencies: tuple[str, ...] = ()
    limits_reached: tuple[str, ...] = ()
    safe_summary: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", _decision_type(self.decision))
        for name in ("reasons", "checks_passed", "checks_failed", "expected_tasks", "observed_tasks"):
            object.__setattr__(self, name, _identifier_tuple(getattr(self, name), name))
        object.__setattr__(self, "inconsistencies", tuple(_safe_code(item, "inconsistency") for item in self.inconsistencies))
        object.__setattr__(self, "limits_reached", _identifier_tuple(self.limits_reached, "limits_reached"))
        object.__setattr__(
            self,
            "safe_summary",
            MappingProxyType(_safe_primitive_mapping(self.safe_summary, "safe_summary")),
        )


@dataclass(frozen=True, slots=True)
class AgentPlanSupervisorEvent:
    """Safe event emitted by the supervisor."""

    name: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "event name"))
        object.__setattr__(self, "status", _identifier(self.status, "event status"))
        object.__setattr__(
            self,
            "details",
            MappingProxyType(_safe_primitive_mapping(self.details, "event details")),
        )


@dataclass(frozen=True, slots=True)
class AgentPlanSupervisorResult:
    """Immutable sanitized result of supervision."""

    status: AgentPlanSupervisorStatus
    decision: AgentPlanSupervisorDecision
    request_signature: str
    plan_signature: str | None = None
    execution_signature: str | None = None
    total_tasks: int = 0
    succeeded_tasks: tuple[str, ...] = ()
    failed_tasks: tuple[str, ...] = ()
    skipped_tasks: tuple[str, ...] = ()
    blocked_tasks: tuple[str, ...] = ()
    unknown_tasks: tuple[str, ...] = ()
    duplicate_tasks: tuple[str, ...] = ()
    missing_tasks: tuple[str, ...] = ()
    valid_outputs: tuple[str, ...] = ()
    invalid_outputs: tuple[str, ...] = ()
    success_ratio: float = 0.0
    limits_reached: tuple[str, ...] = ()
    inconsistencies: tuple[str, ...] = ()
    events: tuple[AgentPlanSupervisorEvent, ...] = ()
    metrics: Mapping[str, int | float] = field(default_factory=dict)
    safe_summary: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        if not isinstance(self.decision, AgentPlanSupervisorDecision):
            raise InvalidAgentPlanSupervisorRequestError("decision must be AgentPlanSupervisorDecision.")
        _validate_signature(self.request_signature, "request_signature", allow_empty=True)
        if self.plan_signature is not None:
            _validate_signature(self.plan_signature, "plan_signature")
        if self.execution_signature is not None:
            _validate_signature(self.execution_signature, "execution_signature")
        if isinstance(self.total_tasks, bool) or not isinstance(self.total_tasks, int) or self.total_tasks < 0:
            raise InvalidAgentPlanSupervisorRequestError("total_tasks must be a non-negative integer.")
        for name in (
            "succeeded_tasks",
            "failed_tasks",
            "skipped_tasks",
            "blocked_tasks",
            "unknown_tasks",
            "duplicate_tasks",
            "missing_tasks",
            "valid_outputs",
            "invalid_outputs",
            "limits_reached",
        ):
            object.__setattr__(self, name, _identifier_tuple(getattr(self, name), name))
        object.__setattr__(self, "inconsistencies", tuple(_safe_code(item, "inconsistency") for item in self.inconsistencies))
        ratio = self.success_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio):
            raise InvalidAgentPlanSupervisorRequestError("success_ratio must be finite.")
        object.__setattr__(self, "success_ratio", float(ratio))
        object.__setattr__(self, "events", tuple(self.events)[-MAX_SUPERVISION_EVENTS:])
        if not all(isinstance(item, AgentPlanSupervisorEvent) for item in self.events):
            raise InvalidAgentPlanSupervisorRequestError("events must contain AgentPlanSupervisorEvent values.")
        object.__setattr__(self, "metrics", MappingProxyType(_metric_mapping(self.metrics)))
        object.__setattr__(
            self,
            "safe_summary",
            MappingProxyType(_safe_primitive_mapping(self.safe_summary, "safe_summary")),
        )


class AgentPlanSupervisor:
    """Observe a plan and its already produced result without side effects."""

    def supervise(self, request: AgentPlanSupervisorRequest) -> AgentPlanSupervisorResult:
        """Return a deterministic supervision result for a finished execution."""

        events: list[AgentPlanSupervisorEvent] = []
        metrics = _base_metrics()
        metrics["agent_plan_supervisions_requested"] = 1
        _event(events, "agent_plan_supervision_requested", "requested")
        _event(events, "agent_plan_supervision_validation_started", "started")
        if not isinstance(request, AgentPlanSupervisorRequest):
            metrics["agent_plan_supervisions_invalid"] = 1
            _event(events, "agent_plan_supervision_validation_failed", "failed")
            return _build_result(
                status=AgentPlanSupervisorStatus.INVALID_REQUEST,
                decision_type=AgentPlanSupervisorDecisionType.REJECTED,
                reasons=("INVALID_REQUEST",),
                checks_failed=("REQUEST_TYPE",),
                events=events,
                metrics=metrics,
            )

        request_signature = agent_plan_supervisor_request_signature(request)
        policy = request.policy
        if not policy.enabled:
            _event(events, "agent_plan_supervision_completed", "disabled")
            return _build_result(
                status=AgentPlanSupervisorStatus.SUPERVISION_DISABLED,
                decision_type=AgentPlanSupervisorDecisionType.SKIPPED,
                request_signature=request_signature,
                plan_signature=_observed_plan_signature(request),
                execution_signature=_observed_execution_signature(request.execution_result),
                reasons=("POLICY_DISABLED",),
                checks_passed=("REQUEST_STRUCTURE",),
                events=events,
                metrics=metrics,
                total_tasks=len(request.plan.tasks),
            )

        try:
            return self._supervise_enabled(request, request_signature, events, metrics)
        except AgentPlanSupervisorError as error:
            metrics["agent_plan_supervisions_invalid"] = 1
            _event(events, "agent_plan_supervision_failed", "failed")
            return _build_result(
                status=AgentPlanSupervisorStatus.INTERNAL_ERROR,
                decision_type=AgentPlanSupervisorDecisionType.REJECTED,
                request_signature=request_signature,
                plan_signature=_observed_plan_signature(request),
                execution_signature=_observed_execution_signature(request.execution_result),
                reasons=("INTERNAL_ERROR",),
                checks_failed=("INTERNAL_ERROR",),
                inconsistencies=(_safe_code(str(error), "error"),),
                events=events,
                metrics=metrics,
                total_tasks=len(request.plan.tasks),
            )

    def _supervise_enabled(
        self,
        request: AgentPlanSupervisorRequest,
        request_signature: str,
        events: list[AgentPlanSupervisorEvent],
        metrics: dict[str, int | float],
    ) -> AgentPlanSupervisorResult:
        plan = request.plan
        result = _cooperation_result(request.execution_result)
        expected_ids = tuple(task.task_id for task in plan.tasks)
        expected_set = set(expected_ids)
        plan_signature = _observed_plan_signature(request)
        execution_signature = _observed_execution_signature(request.execution_result)
        checks_passed: list[str] = ["REQUEST_STRUCTURE"]
        checks_failed: list[str] = []
        inconsistencies: list[str] = []
        limits_reached: list[str] = []
        invalid_outputs: list[str] = []
        valid_outputs: list[str] = []

        _event(events, "agent_plan_supervision_signature_checked", "started")
        if policy_requires(request.policy.require_plan_signature) and plan_signature is None:
            checks_failed.append("PLAN_SIGNATURE_PRESENT")
            inconsistencies.append("PLAN_SIGNATURE_MISSING")
        if result is None:
            checks_failed.append("RESULT_PRESENT")
            inconsistencies.append("EXECUTION_RESULT_MISSING")
            return self._finalize(
                AgentPlanSupervisorStatus.INVALID_RESULT,
                request_signature,
                plan_signature,
                execution_signature,
                expected_ids,
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                valid_outputs,
                invalid_outputs,
                checks_passed,
                checks_failed,
                inconsistencies,
                limits_reached,
                events,
                metrics,
            )
        checks_passed.append("RESULT_PRESENT")

        if request.policy.require_plan_signature and result.plan_signature != plan_signature:
            checks_failed.append("PLAN_SIGNATURE_MATCH")
            inconsistencies.append("PLAN_SIGNATURE_MISMATCH")
        else:
            checks_passed.append("PLAN_SIGNATURE_MATCH")
        if request.policy.require_result_signature and not _valid_optional_signature(execution_signature):
            checks_failed.append("RESULT_SIGNATURE_PRESENT")
            inconsistencies.append("RESULT_SIGNATURE_MISSING")
        else:
            checks_passed.append("RESULT_SIGNATURE_PRESENT")

        if isinstance(request.execution_result, AgentCooperationAutomaticExecutionResult):
            automatic = request.execution_result
            if automatic.generated_plan is not None and automatic.generated_plan is not plan:
                generated_signature = agent_cooperation_plan_signature(automatic.generated_plan)
                if generated_signature != plan_signature:
                    checks_failed.append("AUTOMATIC_PLAN_MATCH")
                    inconsistencies.append("RESULT_BELONGS_TO_DIFFERENT_PLAN")
                else:
                    checks_passed.append("AUTOMATIC_PLAN_MATCH")
            if automatic.plan_signature is not None and automatic.plan_signature != plan_signature:
                checks_failed.append("AUTOMATIC_PLAN_SIGNATURE_MATCH")
                inconsistencies.append("AUTOMATIC_PLAN_SIGNATURE_MISMATCH")
            else:
                checks_passed.append("AUTOMATIC_PLAN_SIGNATURE_MATCH")

        if result.plan_id != plan.plan_id:
            checks_failed.append("PLAN_ID_MATCH")
            inconsistencies.append("PLAN_ID_MISMATCH")
        else:
            checks_passed.append("PLAN_ID_MATCH")

        task_results = tuple(result.task_results)
        metrics["agent_plan_supervision_tasks_expected"] = len(expected_ids)
        _limit(len(expected_ids), request.policy.max_tasks, "MAX_TASKS", limits_reached, events)
        _limit(len(task_results), request.policy.max_results, "MAX_RESULTS", limits_reached, events)
        _limit(len(result.outputs), request.policy.max_outputs, "MAX_OUTPUTS", limits_reached, events)
        logical_time = sum(task.logical_timeout_limit for task in plan.tasks)
        _limit(logical_time, request.policy.max_logical_time, "MAX_LOGICAL_TIME", limits_reached, events)

        by_task: dict[str, list[AgentCooperationTaskResult]] = {}
        observed_from_results: list[str] = []
        for item in task_results:
            _event(events, "agent_plan_supervision_task_checked", "checked", task_known=item.task_id in expected_set)
            by_task.setdefault(item.task_id, []).append(item)
            observed_from_results.append(item.task_id)
        observed_ids = tuple(sorted(set(observed_from_results).union(result.execution_order)))
        unknown = tuple(sorted(task_id for task_id in observed_ids if task_id not in expected_set))
        duplicate = tuple(sorted(task_id for task_id, items in by_task.items() if len(items) > 1))
        missing = tuple(sorted(task_id for task_id in expected_ids if task_id not in by_task))
        if result.execution_order != tuple(dict.fromkeys(result.execution_order)):
            duplicate = tuple(sorted(set(duplicate).union(_duplicates(result.execution_order))))
        if request.policy.reject_unknown_task_results and unknown:
            checks_failed.append("NO_UNKNOWN_TASKS")
            inconsistencies.append("UNKNOWN_TASK_RESULTS")
        else:
            checks_passed.append("NO_UNKNOWN_TASKS")
        if request.policy.reject_duplicate_task_results and duplicate:
            checks_failed.append("NO_DUPLICATE_TASKS")
            inconsistencies.append("DUPLICATE_TASK_RESULTS")
        else:
            checks_passed.append("NO_DUPLICATE_TASKS")
        if request.policy.require_all_tasks_accounted_for and missing:
            checks_failed.append("ALL_TASKS_ACCOUNTED_FOR")
            inconsistencies.append("MISSING_TASK_RESULTS")
        else:
            checks_passed.append("ALL_TASKS_ACCOUNTED_FOR")

        succeeded = tuple(sorted(task_id for task_id, items in by_task.items() if task_id in expected_set and items[-1].status is AgentCooperationTaskStatus.SUCCESS))
        failed = tuple(sorted(task_id for task_id, items in by_task.items() if task_id in expected_set and _is_failed_status(items[-1].status)))
        skipped = tuple(sorted(task_id for task_id, items in by_task.items() if task_id in expected_set and items[-1].status is AgentCooperationTaskStatus.SKIPPED))
        blocked = tuple(sorted(task_id for task_id, items in by_task.items() if task_id in expected_set and _is_blocked_status(items[-1].status)))

        _check_dependencies(plan.dependencies, by_task, request.policy, checks_passed, checks_failed, inconsistencies, events)
        _check_aggregate(result, len(expected_ids), succeeded, failed, skipped, blocked, request.policy, checks_passed, checks_failed, inconsistencies)
        _check_metrics(result, succeeded, failed, skipped, blocked, missing, unknown, duplicate, checks_passed, checks_failed, inconsistencies)
        _check_events(result, checks_passed, checks_failed, inconsistencies)
        _check_outputs(plan, result, by_task, request.policy, valid_outputs, invalid_outputs, inconsistencies, limits_reached, events)
        if invalid_outputs:
            checks_failed.append("OUTPUTS_SAFE")
        else:
            checks_passed.append("OUTPUTS_SAFE")

        metrics["agent_plan_supervision_tasks_succeeded"] = len(succeeded)
        metrics["agent_plan_supervision_tasks_failed"] = len(failed)
        metrics["agent_plan_supervision_tasks_skipped"] = len(skipped)
        metrics["agent_plan_supervision_tasks_blocked"] = len(blocked)
        metrics["agent_plan_supervision_tasks_missing"] = len(missing)
        metrics["agent_plan_supervision_tasks_unknown"] = len(unknown)
        metrics["agent_plan_supervision_duplicate_results"] = len(duplicate)
        metrics["agent_plan_supervision_valid_outputs"] = len(valid_outputs)
        metrics["agent_plan_supervision_invalid_outputs"] = len(invalid_outputs)
        metrics["agent_plan_supervision_inconsistencies"] = len(set(inconsistencies))
        metrics["agent_plan_supervision_limits_reached"] = len(set(limits_reached))
        success_ratio = _success_ratio(len(expected_ids), len(succeeded))
        if success_ratio < request.policy.minimum_success_ratio:
            checks_failed.append("MINIMUM_SUCCESS_RATIO")
            inconsistencies.append("MINIMUM_SUCCESS_RATIO_NOT_REACHED")
        else:
            checks_passed.append("MINIMUM_SUCCESS_RATIO")

        return self._finalize(
            _status_from_findings(result.status, checks_failed, inconsistencies, limits_reached, success_ratio),
            request_signature,
            plan_signature,
            execution_signature,
            expected_ids,
            observed_ids,
            succeeded,
            failed,
            skipped,
            blocked,
            unknown,
            duplicate,
            valid_outputs,
            invalid_outputs,
            checks_passed,
            checks_failed,
            inconsistencies,
            limits_reached,
            events,
            metrics,
            missing=missing,
        )

    def _finalize(
        self,
        status: AgentPlanSupervisorStatus,
        request_signature: str,
        plan_signature: str | None,
        execution_signature: str | None,
        expected_tasks: tuple[str, ...],
        observed_tasks: tuple[str, ...],
        succeeded_tasks: tuple[str, ...],
        failed_tasks: tuple[str, ...],
        skipped_tasks: tuple[str, ...],
        blocked_tasks: tuple[str, ...],
        unknown_tasks: tuple[str, ...],
        duplicate_tasks: tuple[str, ...],
        valid_outputs: Sequence[str],
        invalid_outputs: Sequence[str],
        checks_passed: Sequence[str],
        checks_failed: Sequence[str],
        inconsistencies: Sequence[str],
        limits_reached: Sequence[str],
        events: list[AgentPlanSupervisorEvent],
        metrics: dict[str, int | float],
        *,
        missing: tuple[str, ...] = (),
    ) -> AgentPlanSupervisorResult:
        _event(
            events,
            "agent_plan_supervision_validation_succeeded" if not checks_failed else "agent_plan_supervision_validation_failed",
            "succeeded" if not checks_failed else "failed",
        )
        if inconsistencies:
            _event(events, "agent_plan_supervision_inconsistency_detected", "detected", count=len(set(inconsistencies)))
        _event(events, "agent_plan_supervision_completed", "completed")
        if status is AgentPlanSupervisorStatus.SUCCESS:
            metrics["agent_plan_supervisions_succeeded"] = 1
        elif status is AgentPlanSupervisorStatus.PARTIAL_SUCCESS:
            metrics["agent_plan_supervisions_partial"] = 1
        elif status in (
            AgentPlanSupervisorStatus.INVALID_REQUEST,
            AgentPlanSupervisorStatus.INVALID_PLAN,
            AgentPlanSupervisorStatus.INVALID_RESULT,
            AgentPlanSupervisorStatus.INVALID_SIGNATURE,
        ):
            metrics["agent_plan_supervisions_invalid"] = 1
        else:
            metrics["agent_plan_supervisions_failed"] = 1
        unique_inconsistencies = tuple(sorted(set(inconsistencies)))
        unique_limits = tuple(sorted(set(limits_reached)))
        safe_summary = {
            "status": status.value,
            "total_tasks": len(expected_tasks),
            "observed_tasks": len(observed_tasks),
            "succeeded_tasks": len(succeeded_tasks),
            "failed_tasks": len(failed_tasks),
            "skipped_tasks": len(skipped_tasks),
            "blocked_tasks": len(blocked_tasks),
            "missing_tasks": len(missing),
            "unknown_tasks": len(unknown_tasks),
            "duplicate_tasks": len(duplicate_tasks),
            "invalid_outputs": len(tuple(invalid_outputs)),
            "inconsistencies": len(unique_inconsistencies),
            "limits_reached": len(unique_limits),
        }
        decision_type = (
            AgentPlanSupervisorDecisionType.ACCEPTED
            if status in (AgentPlanSupervisorStatus.SUCCESS, AgentPlanSupervisorStatus.PARTIAL_SUCCESS)
            else AgentPlanSupervisorDecisionType.REJECTED
        )
        decision = AgentPlanSupervisorDecision(
            decision=decision_type,
            reasons=tuple(sorted(set(unique_inconsistencies or ("RESULT_COHERENT",)))),
            checks_passed=tuple(sorted(set(checks_passed))),
            checks_failed=tuple(sorted(set(checks_failed))),
            expected_tasks=expected_tasks,
            observed_tasks=observed_tasks,
            inconsistencies=unique_inconsistencies,
            limits_reached=unique_limits,
            safe_summary=safe_summary,
        )
        return AgentPlanSupervisorResult(
            status=status,
            decision=decision,
            request_signature=request_signature,
            plan_signature=plan_signature,
            execution_signature=execution_signature if _valid_optional_signature(execution_signature) else None,
            total_tasks=len(expected_tasks),
            succeeded_tasks=succeeded_tasks,
            failed_tasks=failed_tasks,
            skipped_tasks=skipped_tasks,
            blocked_tasks=blocked_tasks,
            unknown_tasks=unknown_tasks,
            duplicate_tasks=duplicate_tasks,
            missing_tasks=missing,
            valid_outputs=tuple(sorted(set(valid_outputs))),
            invalid_outputs=tuple(sorted(set(invalid_outputs))),
            success_ratio=_success_ratio(len(expected_tasks), len(succeeded_tasks)),
            limits_reached=unique_limits,
            inconsistencies=unique_inconsistencies,
            events=tuple(events),
            metrics=metrics,
            safe_summary=safe_summary,
        )


def agent_plan_supervisor_request_signature(request: AgentPlanSupervisorRequest) -> str:
    """Return a canonical SHA-256 signature for one supervision request."""

    if not isinstance(request, AgentPlanSupervisorRequest):
        raise InvalidAgentPlanSupervisorRequestError("request must be AgentPlanSupervisorRequest.")
    payload = {
        "plan_signature": agent_cooperation_plan_signature(request.plan),
        "execution_signature": _observed_execution_signature(request.execution_result),
        "execution_result_plan_id": _observed_result_plan_id(request.execution_result),
        "policy": _policy_payload(request.policy),
        "execution_id": request.execution_id,
        "correlation_id": request.correlation_id,
        "metadata": request.metadata,
    }
    return _signature(payload)


def build_core_agent_plan_supervisor() -> AgentPlanSupervisor:
    """Build the pure supervisor without executing plans or agents."""

    return AgentPlanSupervisor()


def _cooperation_result(
    result: AgentCooperationPlanResult | AgentCooperationAutomaticExecutionResult,
) -> AgentCooperationPlanResult | None:
    if isinstance(result, AgentCooperationPlanResult):
        return result
    return result.cooperation_result


def _observed_plan_signature(request: AgentPlanSupervisorRequest) -> str | None:
    try:
        return agent_cooperation_plan_signature(request.plan)
    except (RuntimeError, TypeError, ValueError):
        return None


def _observed_execution_signature(
    result: AgentCooperationPlanResult | AgentCooperationAutomaticExecutionResult,
) -> str | None:
    if isinstance(result, AgentCooperationAutomaticExecutionResult):
        return result.signature or None
    if isinstance(result, AgentCooperationPlanResult):
        return result.request_signature or None
    return None


def _observed_result_plan_id(
    result: AgentCooperationPlanResult | AgentCooperationAutomaticExecutionResult,
) -> str | None:
    cooperation_result = _cooperation_result(result)
    return None if cooperation_result is None else cooperation_result.plan_id


def _check_dependencies(
    dependencies: Sequence[AgentCooperationDependency],
    by_task: Mapping[str, list[AgentCooperationTaskResult]],
    policy: AgentPlanSupervisorPolicy,
    checks_passed: list[str],
    checks_failed: list[str],
    inconsistencies: list[str],
    events: list[AgentPlanSupervisorEvent],
) -> None:
    failures = 0
    for dependency in dependencies:
        prerequisite = by_task.get(dependency.prerequisite_task_id, ())
        dependent = by_task.get(dependency.dependent_task_id, ())
        prerequisite_success = bool(prerequisite and prerequisite[-1].status is AgentCooperationTaskStatus.SUCCESS)
        dependent_executed = bool(dependent and dependent[-1].status is AgentCooperationTaskStatus.SUCCESS)
        satisfied = prerequisite_success or not dependent_executed or not dependency.required
        _event(events, "agent_plan_supervision_dependency_checked", "checked", satisfied=satisfied)
        if not satisfied:
            failures += 1
    if policy.require_dependencies_satisfied and failures:
        checks_failed.append("DEPENDENCIES_SATISFIED")
        inconsistencies.append("DEPENDENCIES_UNSATISFIED")
    else:
        checks_passed.append("DEPENDENCIES_SATISFIED")


def _check_aggregate(
    result: AgentCooperationPlanResult,
    total: int,
    succeeded: tuple[str, ...],
    failed: tuple[str, ...],
    skipped: tuple[str, ...],
    blocked: tuple[str, ...],
    policy: AgentPlanSupervisorPolicy,
    checks_passed: list[str],
    checks_failed: list[str],
    inconsistencies: list[str],
) -> None:
    expected = _expected_plan_status(total, succeeded, failed, skipped, blocked)
    acceptable = {expected}
    if expected is AgentCooperationPlanStatus.PARTIAL_SUCCESS:
        acceptable.add(AgentCooperationPlanStatus.FAILED)
    if policy.require_consistent_aggregate_status and result.status not in acceptable:
        checks_failed.append("AGGREGATE_STATUS_CONSISTENT")
        inconsistencies.append("AGGREGATE_STATUS_INCONSISTENT")
    else:
        checks_passed.append("AGGREGATE_STATUS_CONSISTENT")


def _check_metrics(
    result: AgentCooperationPlanResult,
    succeeded: tuple[str, ...],
    failed: tuple[str, ...],
    skipped: tuple[str, ...],
    blocked: tuple[str, ...],
    missing: tuple[str, ...],
    unknown: tuple[str, ...],
    duplicate: tuple[str, ...],
    checks_passed: list[str],
    checks_failed: list[str],
    inconsistencies: list[str],
) -> None:
    expected = {
        "cooperation_tasks_succeeded": len(succeeded),
        "cooperation_tasks_failed": len(failed),
        "cooperation_tasks_skipped": len(skipped),
        "cooperation_tasks_blocked": len(blocked),
    }
    mismatches = sum(1 for key, value in expected.items() if key in result.metrics and result.metrics[key] != value)
    if mismatches:
        checks_failed.append("METRICS_CONSISTENT")
        inconsistencies.append("METRICS_INCONSISTENT")
    else:
        checks_passed.append("METRICS_CONSISTENT")
    if missing or unknown or duplicate:
        return


def _check_events(
    result: AgentCooperationPlanResult,
    checks_passed: list[str],
    checks_failed: list[str],
    inconsistencies: list[str],
) -> None:
    invalid = sum(
        1
        for event in result.events
        if not hasattr(event, "name") or not hasattr(event, "status")
    )
    if invalid:
        checks_failed.append("EVENTS_CONSISTENT")
        inconsistencies.append("EVENTS_INCONSISTENT")
    else:
        checks_passed.append("EVENTS_CONSISTENT")


def _check_outputs(
    plan: AgentCooperationPlan,
    result: AgentCooperationPlanResult,
    by_task: Mapping[str, list[AgentCooperationTaskResult]],
    policy: AgentPlanSupervisorPolicy,
    valid_outputs: list[str],
    invalid_outputs: list[str],
    inconsistencies: list[str],
    limits_reached: list[str],
    events: list[AgentPlanSupervisorEvent],
) -> None:
    seen_payload_signatures: dict[str, str] = {}
    for task in plan.tasks:
        item = by_task.get(task.task_id, [None])[-1]
        output = None if item is None else item.output
        if policy.require_non_empty_required_outputs and item is not None and item.status is AgentCooperationTaskStatus.SUCCESS and not output:
            invalid_outputs.append(task.task_id)
            inconsistencies.append("REQUIRED_OUTPUT_EMPTY")
        if output is None:
            continue
        try:
            _inspect_safe_value(output, policy, depth=0, counter={"items": 0})
            output_signature = _signature(output)
        except InvalidAgentPlanSupervisorRequestError as error:
            invalid_outputs.append(task.task_id)
            inconsistencies.append("INVALID_OUTPUT_STRUCTURE")
            _record_output_limit(error, limits_reached)
            _event(events, "agent_plan_supervision_output_checked", "invalid", task_id=task.task_id)
            continue
        _event(events, "agent_plan_supervision_output_checked", "checked", task_id=task.task_id)
        if output_signature in seen_payload_signatures:
            invalid_outputs.append(task.task_id)
            inconsistencies.append("DUPLICATE_OUTPUTS")
        else:
            seen_payload_signatures[output_signature] = task.task_id
            valid_outputs.append(task.task_id)
    output_counter = {"items": 0}
    try:
        _inspect_safe_value(result.outputs, policy, depth=0, counter=output_counter)
    except InvalidAgentPlanSupervisorRequestError as error:
        invalid_outputs.append("outputs")
        inconsistencies.append("INVALID_OUTPUT_STRUCTURE")
        _record_output_limit(error, limits_reached)
    if output_counter["items"] > policy.max_output_items:
        limits_reached.append("MAX_OUTPUT_ITEMS")


def _expected_plan_status(
    total: int,
    succeeded: tuple[str, ...],
    failed: tuple[str, ...],
    skipped: tuple[str, ...],
    blocked: tuple[str, ...],
) -> AgentCooperationPlanStatus:
    if blocked:
        return AgentCooperationPlanStatus.BLOCKED
    if failed:
        return AgentCooperationPlanStatus.PARTIAL_SUCCESS if succeeded else AgentCooperationPlanStatus.FAILED
    if skipped and len(succeeded) + len(skipped) >= total:
        return AgentCooperationPlanStatus.PARTIAL_SUCCESS
    if total and len(succeeded) == total:
        return AgentCooperationPlanStatus.SUCCESS
    if succeeded:
        return AgentCooperationPlanStatus.PARTIAL_SUCCESS
    return AgentCooperationPlanStatus.FAILED


def _status_from_findings(
    result_status: AgentCooperationPlanStatus,
    checks_failed: Sequence[str],
    inconsistencies: Sequence[str],
    limits_reached: Sequence[str],
    success_ratio: float,
) -> AgentPlanSupervisorStatus:
    if limits_reached:
        return AgentPlanSupervisorStatus.LIMIT_REACHED
    if "PLAN_SIGNATURE_MATCH" in checks_failed or "RESULT_SIGNATURE_PRESENT" in checks_failed:
        return AgentPlanSupervisorStatus.INVALID_SIGNATURE
    if "PLAN_ID_MATCH" in checks_failed or "AUTOMATIC_PLAN_MATCH" in checks_failed:
        return AgentPlanSupervisorStatus.INCONSISTENT_RESULT
    if checks_failed or inconsistencies:
        return AgentPlanSupervisorStatus.INCONSISTENT_RESULT
    if result_status is AgentCooperationPlanStatus.SUCCESS:
        return AgentPlanSupervisorStatus.SUCCESS
    if result_status is AgentCooperationPlanStatus.PARTIAL_SUCCESS or 0.0 < success_ratio < 1.0:
        return AgentPlanSupervisorStatus.PARTIAL_SUCCESS
    return AgentPlanSupervisorStatus.FAILED


def _is_failed_status(status: AgentCooperationTaskStatus) -> bool:
    return status in (
        AgentCooperationTaskStatus.FAILED,
        AgentCooperationTaskStatus.DEPENDENCY_FAILED,
        AgentCooperationTaskStatus.AGENT_RESOLUTION_FAILED,
        AgentCooperationTaskStatus.AGENT_EXECUTION_FAILED,
        AgentCooperationTaskStatus.SKILL_AUTHORIZATION_FAILED,
        AgentCooperationTaskStatus.LIMIT_REACHED,
        AgentCooperationTaskStatus.OUTPUT_BINDING_FAILED,
    )


def _is_blocked_status(status: AgentCooperationTaskStatus) -> bool:
    return status in (
        AgentCooperationTaskStatus.BLOCKED,
        AgentCooperationTaskStatus.DEPENDENCY_BLOCKED,
    )


def _success_ratio(total: int, succeeded: int) -> float:
    return 1.0 if total == 0 else succeeded / total


def _limit(
    actual: int,
    maximum: int,
    name: str,
    limits_reached: list[str],
    events: list[AgentPlanSupervisorEvent],
) -> None:
    if actual > maximum:
        limits_reached.append(name)
        _event(events, "agent_plan_supervision_limit_reached", "reached", limit_name=name)


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _record_output_limit(
    error: InvalidAgentPlanSupervisorRequestError,
    limits_reached: list[str],
) -> None:
    message = str(error)
    if "depth" in message:
        limits_reached.append("MAX_DEPTH")
    if "item limit" in message:
        limits_reached.append("MAX_OUTPUT_ITEMS")
    if "string exceeds" in message:
        limits_reached.append("MAX_STRING_LENGTH")


def policy_requires(value: bool) -> bool:
    return bool(value)


def _build_result(
    *,
    status: AgentPlanSupervisorStatus,
    decision_type: AgentPlanSupervisorDecisionType,
    request_signature: str = "",
    plan_signature: str | None = None,
        execution_signature: str | None = None,
    reasons: tuple[str, ...] = (),
    checks_passed: tuple[str, ...] = (),
    checks_failed: tuple[str, ...] = (),
    inconsistencies: tuple[str, ...] = (),
    events: Sequence[AgentPlanSupervisorEvent] = (),
    metrics: Mapping[str, int | float] | None = None,
    total_tasks: int = 0,
) -> AgentPlanSupervisorResult:
    summary = {"status": status.value, "total_tasks": total_tasks}
    decision = AgentPlanSupervisorDecision(
        decision=decision_type,
        reasons=reasons,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        inconsistencies=inconsistencies,
        safe_summary=summary,
    )
    return AgentPlanSupervisorResult(
        status=status,
        decision=decision,
        request_signature=request_signature,
        plan_signature=plan_signature,
        execution_signature=execution_signature if _valid_optional_signature(execution_signature) else None,
        total_tasks=total_tasks,
        inconsistencies=inconsistencies,
        events=tuple(events),
        metrics=_base_metrics() if metrics is None else metrics,
        safe_summary=summary,
    )


def _event(
    events: list[AgentPlanSupervisorEvent],
    name: str,
    status: str,
    **details: object,
) -> None:
    if len(events) < MAX_SUPERVISION_EVENTS:
        events.append(AgentPlanSupervisorEvent(name, status, details))


def _base_metrics() -> dict[str, int | float]:
    return {
        "agent_plan_supervisions_requested": 0,
        "agent_plan_supervisions_succeeded": 0,
        "agent_plan_supervisions_partial": 0,
        "agent_plan_supervisions_failed": 0,
        "agent_plan_supervisions_invalid": 0,
        "agent_plan_supervision_tasks_expected": 0,
        "agent_plan_supervision_tasks_succeeded": 0,
        "agent_plan_supervision_tasks_failed": 0,
        "agent_plan_supervision_tasks_skipped": 0,
        "agent_plan_supervision_tasks_blocked": 0,
        "agent_plan_supervision_tasks_missing": 0,
        "agent_plan_supervision_tasks_unknown": 0,
        "agent_plan_supervision_duplicate_results": 0,
        "agent_plan_supervision_valid_outputs": 0,
        "agent_plan_supervision_invalid_outputs": 0,
        "agent_plan_supervision_inconsistencies": 0,
        "agent_plan_supervision_limits_reached": 0,
    }


def _policy_payload(policy: AgentPlanSupervisorPolicy) -> Mapping[str, object]:
    return {
        name: getattr(policy, name)
        for name in sorted(policy.__dataclass_fields__)
    }


def _safe_mapping(
    value: Mapping[str, object],
    policy: AgentPlanSupervisorPolicy,
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentPlanSupervisorRequestError(f"{field_name} must be a mapping.")
    if len(value) > MAX_SUPERVISION_METADATA_ITEMS:
        raise InvalidAgentPlanSupervisorRequestError(f"{field_name} has too many items.")
    safe: dict[str, object] = {}
    counter = {"items": 0}
    for raw_key in sorted(value, key=lambda item: str(item)):
        key = _identifier(raw_key, f"{field_name} key")
        if _forbidden_key(key, policy):
            raise InvalidAgentPlanSupervisorRequestError(f"{field_name} contains a forbidden key.")
        safe[key] = _inspect_safe_value(value[raw_key], policy, depth=0, counter=counter)
    return safe


def _inspect_safe_value(
    value: object,
    policy: AgentPlanSupervisorPolicy,
    *,
    depth: int,
    counter: dict[str, int],
) -> object:
    if depth > policy.max_depth:
        raise InvalidAgentPlanSupervisorRequestError("value exceeds maximum depth.")
    counter["items"] += 1
    if counter["items"] > policy.max_output_items:
        raise InvalidAgentPlanSupervisorRequestError("value exceeds item limit.")
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentPlanSupervisorRequestError("floats must be finite.")
        return value
    if isinstance(value, str):
        if len(value) > policy.max_string_length:
            raise InvalidAgentPlanSupervisorRequestError("string exceeds length limit.")
        return value
    if isinstance(value, Mapping):
        safe: dict[str, object] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = _identifier(raw_key, "value key")
            if _forbidden_key(key, policy):
                raise InvalidAgentPlanSupervisorRequestError("value contains a forbidden key.")
            safe[key] = _inspect_safe_value(value[raw_key], policy, depth=depth + 1, counter=counter)
        return MappingProxyType(safe)
    if isinstance(value, (tuple, list)):
        if len(value) > policy.max_output_items:
            raise InvalidAgentPlanSupervisorRequestError("sequence exceeds item limit.")
        return tuple(_inspect_safe_value(item, policy, depth=depth + 1, counter=counter) for item in value)
    raise InvalidAgentPlanSupervisorRequestError("value contains an unsupported object.")


def _safe_primitive_mapping(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentPlanSupervisorRequestError(f"{field_name} must be a mapping.")
    safe: dict[str, object] = {}
    for raw_key in sorted(value, key=lambda item: str(item)):
        key = _identifier(raw_key, f"{field_name} key")
        item = value[raw_key]
        if item is None or type(item) in (bool, int, str):
            safe[key] = item
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise InvalidAgentPlanSupervisorRequestError(f"{field_name} floats must be finite.")
            safe[key] = item
        else:
            raise InvalidAgentPlanSupervisorRequestError(f"{field_name} values must be primitive.")
    return safe


def _metric_mapping(value: Mapping[str, int | float]) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise InvalidAgentPlanSupervisorRequestError("metrics must be a mapping.")
    result: dict[str, int | float] = {}
    for key, item in value.items():
        normalized = _identifier(key, "metric name")
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
            raise InvalidAgentPlanSupervisorRequestError("metric values must be non-negative finite numbers.")
        result[normalized] = item
    return result


def _forbidden_key(key: str, policy: AgentPlanSupervisorPolicy) -> bool:
    normalized = key.replace("-", "_").lower()
    if policy.reject_sensitive_keys and any(part in normalized for part in _SENSITIVE_KEY_PARTS):
        return True
    return any(part in normalized for part in _DYNAMIC_CODE_KEY_PARTS)


def _signature(value: object) -> str:
    encoded = json.dumps(_jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise InvalidAgentPlanSupervisorRequestError("value is not deterministically serializable.")


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidAgentPlanSupervisorRequestError(f"{field_name} must be a string.")
    if value != value.strip() or not value or len(value) > 128:
        raise InvalidAgentPlanSupervisorRequestError(f"{field_name} contains unsupported characters.")
    if not all(character.isalnum() or character in "_.-" for character in value):
        raise InvalidAgentPlanSupervisorRequestError(f"{field_name} contains unsupported characters.")
    return value


def _identifier_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise InvalidAgentPlanSupervisorRequestError(f"{field_name} must be a sequence.")
    return tuple(dict.fromkeys(_identifier(value, field_name) for value in tuple(values)))


def _safe_code(value: str, field_name: str) -> str:
    normalized = "_".join(str(value).upper().replace("-", "_").split())
    if not normalized:
        normalized = "UNKNOWN"
    return _identifier(normalized[:128], field_name)


def _bounded_int(value: int, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise InvalidAgentPlanSupervisorRequestError(f"{field_name} is outside the allowed range.")
    return value


def _validate_signature(value: str, field_name: str, *, allow_empty: bool = False) -> None:
    if allow_empty and value == "":
        return
    if not isinstance(value, str) or len(value) != 64 or any(character not in _SIGNATURE_HEX for character in value):
        raise InvalidAgentPlanSupervisorRequestError(f"{field_name} must be a SHA-256 hex digest.")


def _valid_optional_signature(value: str | None) -> bool:
    if value is None:
        return False
    try:
        _validate_signature(value, "signature")
    except InvalidAgentPlanSupervisorRequestError:
        return False
    return True


def _status(value: AgentPlanSupervisorStatus | str) -> AgentPlanSupervisorStatus:
    if isinstance(value, AgentPlanSupervisorStatus):
        return value
    try:
        return AgentPlanSupervisorStatus(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentPlanSupervisorRequestError("invalid supervision status.") from error


def _decision_type(
    value: AgentPlanSupervisorDecisionType | str,
) -> AgentPlanSupervisorDecisionType:
    if isinstance(value, AgentPlanSupervisorDecisionType):
        return value
    try:
        return AgentPlanSupervisorDecisionType(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentPlanSupervisorRequestError("invalid supervision decision.") from error

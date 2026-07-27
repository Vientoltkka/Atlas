"""Controlled one-shot execution for accepted replanning proposals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType

from core.agent_cooperation_plan import (
    AgentCooperationPlan,
    AgentCooperationPlanPolicy,
    AgentCooperationPlanRequest,
    AgentCooperationPlanResult,
    AgentCooperationPlanStatus,
    AgentCooperationPlanner,
    AgentCooperationTaskStatus,
    agent_cooperation_plan_signature,
)
from core.agent_plan_replanner import (
    AgentPlanReplanner,
    AgentPlanReplanningResult,
    AgentPlanReplanningStatus,
)
from core.agent_plan_supervisor import AgentPlanSupervisor
from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolver


MAX_REPLANNED_EXECUTIONS = 1
MAX_TOTAL_ATTEMPTS = 4
MAX_TOTAL_FAILURES = 4
MAX_REPLANNED_TASKS = 100
MAX_REPLANNED_OUTPUTS = 512
MAX_REPLANNED_DEPTH = 32
MAX_REPLANNED_LOGICAL_TIME = 1_000_000
MAX_REPLANNED_METADATA_ITEMS = 32
MAX_REPLANNED_VALUE_DEPTH = 8
MAX_REPLANNED_TOTAL_ITEMS = 1_024
MAX_REPLANNED_STRING_LENGTH = 1_000
MAX_REPLANNED_EVENTS = 1_024
_SIGNATURE_HEX = set("0123456789abcdef")
_SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "token",
    "authorization",
    "cookie",
    "credential",
    "private_key",
    "api_key",
    "refresh_token",
    "access_token",
)
_DYNAMIC_KEY_PARTS = (
    "python_path",
    "module_path",
    "import_path",
    "handler_path",
    "callback",
    "callable",
)


class AgentReplannedExecutionError(RuntimeError):
    """Base error for controlled replanned execution."""


class InvalidAgentReplannedExecutionRequestError(AgentReplannedExecutionError):
    """Raised when a replanned execution request or policy is malformed."""


class AgentReplannedExecutionStatus(str, Enum):
    """Terminal statuses for one authorized replanned execution request."""

    DISABLED = "DISABLED"
    REJECTED = "REJECTED"
    SIGNATURE_ERROR = "SIGNATURE_ERROR"
    POLICY_ERROR = "POLICY_ERROR"
    LIMIT_REACHED = "LIMIT_REACHED"
    DUPLICATE = "DUPLICATE"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    INVALID_REQUEST = "INVALID_REQUEST"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class AgentReplannedExecutionPolicy:
    """Explicit opt-in policy for a single replanned execution."""

    enabled: bool = False
    execute_replanned_plan: bool = False
    max_replanned_executions: int = 1
    max_total_attempts: int = 2
    max_total_failures: int = 2
    max_tasks: int = 64
    max_outputs: int = 256
    max_depth: int = 12
    max_logical_time: int = 10_000
    allow_partial_success: bool = False
    require_progress: bool = True
    require_supervision: bool = True
    require_replanning_signature: bool = True
    fail_closed: bool = True

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "execute_replanned_plan",
            "allow_partial_success",
            "require_progress",
            "require_supervision",
            "require_replanning_signature",
            "fail_closed",
        ):
            if type(getattr(self, name)) is not bool:
                raise InvalidAgentReplannedExecutionRequestError(f"{name} must be a bool.")
        for name, maximum in (
            ("max_replanned_executions", MAX_REPLANNED_EXECUTIONS),
            ("max_total_attempts", MAX_TOTAL_ATTEMPTS),
            ("max_total_failures", MAX_TOTAL_FAILURES),
            ("max_tasks", MAX_REPLANNED_TASKS),
            ("max_outputs", MAX_REPLANNED_OUTPUTS),
            ("max_depth", MAX_REPLANNED_DEPTH),
            ("max_logical_time", MAX_REPLANNED_LOGICAL_TIME),
        ):
            object.__setattr__(self, name, _bounded_int(getattr(self, name), name, maximum))


@dataclass(frozen=True, slots=True)
class AgentReplannedExecutionEvent:
    """Safe structured event for replanned execution."""

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
class AgentReplannedExecutionDecision:
    """Bounded decision for a replanned execution request."""

    status: AgentReplannedExecutionStatus
    reason_code: str
    accepted: bool = False
    duplicate: bool = False
    limits_reached: tuple[str, ...] = ()
    safe_summary: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        for name in ("accepted", "duplicate"):
            if type(getattr(self, name)) is not bool:
                raise InvalidAgentReplannedExecutionRequestError(f"{name} must be a bool.")
        object.__setattr__(self, "limits_reached", _identifier_tuple(self.limits_reached, "limits_reached"))
        object.__setattr__(
            self,
            "safe_summary",
            MappingProxyType(_safe_primitive_mapping(self.safe_summary, "safe_summary")),
        )


@dataclass(frozen=True, slots=True)
class AgentReplannedExecutionRequest:
    """Request to execute exactly one accepted replanning proposal."""

    replanning_result: AgentPlanReplanningResult
    policy: AgentReplannedExecutionPolicy = field(default_factory=AgentReplannedExecutionPolicy)
    execution_id: str | None = None
    correlation_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.replanning_result, AgentPlanReplanningResult):
            raise InvalidAgentReplannedExecutionRequestError(
                "replanning_result must be AgentPlanReplanningResult."
            )
        if not isinstance(self.policy, AgentReplannedExecutionPolicy):
            raise InvalidAgentReplannedExecutionRequestError("policy must be AgentReplannedExecutionPolicy.")
        for name in ("execution_id", "correlation_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))


@dataclass(frozen=True, slots=True)
class AgentReplannedExecutionResult:
    """Immutable outcome of one controlled replanned execution request."""

    status: AgentReplannedExecutionStatus
    decision: AgentReplannedExecutionDecision
    request_signature: str
    replanning_signature: str | None = None
    proposed_plan_signature: str | None = None
    replanning_result: AgentPlanReplanningResult | None = None
    execution_result: AgentCooperationPlanResult | None = None
    events: tuple[AgentReplannedExecutionEvent, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    safe_summary: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        if not isinstance(self.decision, AgentReplannedExecutionDecision):
            raise InvalidAgentReplannedExecutionRequestError(
                "decision must be AgentReplannedExecutionDecision."
            )
        _validate_signature(self.request_signature, "request_signature", allow_empty=True)
        for name in ("replanning_signature", "proposed_plan_signature"):
            value = getattr(self, name)
            if value is not None:
                _validate_signature(value, name)
        if self.replanning_result is not None and not isinstance(self.replanning_result, AgentPlanReplanningResult):
            raise InvalidAgentReplannedExecutionRequestError(
                "replanning_result must be AgentPlanReplanningResult or None."
            )
        if self.execution_result is not None and not isinstance(self.execution_result, AgentCooperationPlanResult):
            raise InvalidAgentReplannedExecutionRequestError(
                "execution_result must be AgentCooperationPlanResult or None."
            )
        events = tuple(self.events)[-MAX_REPLANNED_EVENTS:]
        if not all(isinstance(item, AgentReplannedExecutionEvent) for item in events):
            raise InvalidAgentReplannedExecutionRequestError(
                "events must contain AgentReplannedExecutionEvent values."
            )
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "metrics", MappingProxyType(_metric_mapping(self.metrics)))
        object.__setattr__(
            self,
            "safe_summary",
            MappingProxyType(_safe_primitive_mapping(self.safe_summary, "safe_summary")),
        )


class AgentReplannedExecutionService:
    """Execute at most one accepted replanning proposal per request signature."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        agent_resolver: AgentResolver,
        agent_cooperation_planner: AgentCooperationPlanner,
        agent_plan_supervisor: AgentPlanSupervisor,
        agent_plan_replanner: AgentPlanReplanner,
    ) -> None:
        if not isinstance(agent_registry, AgentRegistry):
            raise AgentReplannedExecutionError("agent_registry must be AgentRegistry.")
        if not isinstance(agent_resolver, AgentResolver):
            raise AgentReplannedExecutionError("agent_resolver must be AgentResolver.")
        if not isinstance(agent_cooperation_planner, AgentCooperationPlanner):
            raise AgentReplannedExecutionError("agent_cooperation_planner must be AgentCooperationPlanner.")
        if not isinstance(agent_plan_supervisor, AgentPlanSupervisor):
            raise AgentReplannedExecutionError("agent_plan_supervisor must be AgentPlanSupervisor.")
        if not isinstance(agent_plan_replanner, AgentPlanReplanner):
            raise AgentReplannedExecutionError("agent_plan_replanner must be AgentPlanReplanner.")
        self.agent_registry = agent_registry
        self.agent_resolver = agent_resolver
        self.agent_cooperation_planner = agent_cooperation_planner
        self.agent_plan_supervisor = agent_plan_supervisor
        self.agent_plan_replanner = agent_plan_replanner
        self._results_by_signature: dict[str, AgentReplannedExecutionResult] = {}

    def execute(
        self,
        request: AgentReplannedExecutionRequest,
    ) -> AgentReplannedExecutionResult:
        """Execute one proposed plan through the existing cooperation planner."""

        events: list[AgentReplannedExecutionEvent] = []
        metrics = _base_metrics()
        metrics["agent_replanned_execution_requests"] = 1
        _event(events, "agent_replanned_execution_requested", "requested")
        if not isinstance(request, AgentReplannedExecutionRequest):
            metrics["agent_replanned_execution_rejected"] = 1
            _event(events, "agent_replanned_execution_rejected", "rejected", reason_code="INVALID_REQUEST")
            return _result(
                status=AgentReplannedExecutionStatus.INVALID_REQUEST,
                reason_code="INVALID_REQUEST",
                request_signature="",
                events=events,
                metrics=metrics,
            )
        request_signature = agent_replanned_execution_request_signature(request)
        if request_signature in self._results_by_signature:
            cached = self._results_by_signature[request_signature]
            duplicate_events = tuple(cached.events) + (
                AgentReplannedExecutionEvent("agent_replanned_execution_duplicate", "duplicate"),
            )
            duplicate_metrics = dict(cached.metrics)
            duplicate_metrics["agent_replanned_execution_duplicates"] = 1
            return replace(
                cached,
                status=AgentReplannedExecutionStatus.DUPLICATE,
                decision=replace(cached.decision, duplicate=True, status=AgentReplannedExecutionStatus.DUPLICATE),
                events=duplicate_events,
                metrics=duplicate_metrics,
            )

        validation = _validate_request(request)
        if validation is not None:
            status, reason_code, event_name, limits = validation
            if status is AgentReplannedExecutionStatus.LIMIT_REACHED:
                metrics["agent_replanned_execution_limits"] = len(limits) or 1
            else:
                metrics["agent_replanned_execution_rejected"] = 1
            _event(events, event_name, "rejected", reason_code=reason_code)
            result = _result(
                status=status,
                reason_code=reason_code,
                request_signature=request_signature,
                replanning_signature=request.replanning_result.request_signature,
                proposed_plan_signature=request.replanning_result.proposed_plan_signature,
                replanning_result=request.replanning_result,
                events=events,
                metrics=metrics,
                limits_reached=limits,
            )
            if status is not AgentReplannedExecutionStatus.DUPLICATE:
                self._results_by_signature[request_signature] = result
            return result

        _event(events, "agent_replanned_execution_validated", "validated")
        _event(events, "agent_replanned_execution_started", "started")
        plan = request.replanning_result.proposed_plan
        assert plan is not None
        try:
            cooperation_result = self.agent_cooperation_planner.execute(
                AgentCooperationPlanRequest(
                    plan=plan,
                    policy=plan.policy,
                    execution_id=request.execution_id,
                    correlation_id=request.correlation_id,
                    metadata=request.metadata,
                )
            )
        except (RuntimeError, TypeError, ValueError):
            metrics["agent_replanned_execution_failures"] = 1
            _event(events, "agent_replanned_execution_failed", "failed")
            result = _result(
                status=AgentReplannedExecutionStatus.FAILED,
                reason_code="PLANNER_EXECUTION_FAILED",
                request_signature=request_signature,
                replanning_signature=request.replanning_result.request_signature,
                proposed_plan_signature=request.replanning_result.proposed_plan_signature,
                replanning_result=request.replanning_result,
                events=events,
                metrics=metrics,
            )
            self._results_by_signature[request_signature] = result
            return result
        safe_execution_result = _sanitize_execution_result(cooperation_result)
        post_limits = _result_limits_reached(safe_execution_result, request.policy)
        if post_limits:
            metrics["agent_replanned_execution_limits"] = len(post_limits)
            _event(events, "agent_replanned_execution_limit", "rejected", limit_name=post_limits[0])
            result = _result(
                status=AgentReplannedExecutionStatus.LIMIT_REACHED,
                reason_code="LIMIT_REACHED",
                request_signature=request_signature,
                replanning_signature=request.replanning_result.request_signature,
                proposed_plan_signature=request.replanning_result.proposed_plan_signature,
                replanning_result=request.replanning_result,
                execution_result=safe_execution_result,
                events=events,
                metrics=metrics,
                limits_reached=post_limits,
            )
            self._results_by_signature[request_signature] = result
            return result
        status = _execution_status(safe_execution_result, request.policy)
        if status is AgentReplannedExecutionStatus.SUCCESS:
            metrics["agent_replanned_execution_success"] = 1
        elif status is AgentReplannedExecutionStatus.PARTIAL_SUCCESS:
            metrics["agent_replanned_execution_partial"] = 1
        else:
            metrics["agent_replanned_execution_failures"] = 1
        metrics["agent_replanned_execution_tasks"] = len(safe_execution_result.task_results)
        metrics["agent_replanned_execution_outputs"] = len(safe_execution_result.outputs)
        metrics["agent_replanned_execution_duration"] = len(safe_execution_result.execution_order)
        _event(
            events,
            "agent_replanned_execution_completed" if status is not AgentReplannedExecutionStatus.FAILED else "agent_replanned_execution_failed",
            "completed" if status is not AgentReplannedExecutionStatus.FAILED else "failed",
        )
        result = _result(
            status=status,
            reason_code=status.value,
            request_signature=request_signature,
            replanning_signature=request.replanning_result.request_signature,
            proposed_plan_signature=request.replanning_result.proposed_plan_signature,
            replanning_result=request.replanning_result,
            execution_result=safe_execution_result,
            events=events,
            metrics=metrics,
            accepted=True,
        )
        self._results_by_signature[request_signature] = result
        return result


def agent_replanned_execution_request_signature(
    request: AgentReplannedExecutionRequest,
) -> str:
    """Return a canonical SHA-256 signature for one replanned execution request."""

    if not isinstance(request, AgentReplannedExecutionRequest):
        raise InvalidAgentReplannedExecutionRequestError(
            "request must be AgentReplannedExecutionRequest."
        )
    payload = {
        "replanning_signature": request.replanning_result.request_signature,
        "proposed_plan_signature": request.replanning_result.proposed_plan_signature,
        "policy": _policy_payload(request.policy),
        "execution_id": request.execution_id,
        "correlation_id": request.correlation_id,
        "metadata": request.metadata,
    }
    return _signature(payload)


def build_core_agent_replanned_execution_service(
    *,
    agent_registry: AgentRegistry,
    agent_resolver: AgentResolver,
    agent_cooperation_planner: AgentCooperationPlanner,
    agent_plan_supervisor: AgentPlanSupervisor,
    agent_plan_replanner: AgentPlanReplanner,
) -> AgentReplannedExecutionService:
    """Build the controlled service with shared existing components."""

    return AgentReplannedExecutionService(
        agent_registry=agent_registry,
        agent_resolver=agent_resolver,
        agent_cooperation_planner=agent_cooperation_planner,
        agent_plan_supervisor=agent_plan_supervisor,
        agent_plan_replanner=agent_plan_replanner,
    )


def _validate_request(
    request: AgentReplannedExecutionRequest,
) -> tuple[AgentReplannedExecutionStatus, str, str, tuple[str, ...]] | None:
    policy = request.policy
    if not policy.enabled:
        return AgentReplannedExecutionStatus.DISABLED, "POLICY_DISABLED", "agent_replanned_execution_policy_error", ()
    if not policy.execute_replanned_plan:
        return AgentReplannedExecutionStatus.POLICY_ERROR, "EXECUTION_NOT_AUTHORIZED", "agent_replanned_execution_policy_error", ()
    replanning = request.replanning_result
    if replanning.status is not AgentPlanReplanningStatus.REPLAN_PROPOSED:
        return AgentReplannedExecutionStatus.REJECTED, "REPLANNING_NOT_ACCEPTED", "agent_replanned_execution_rejected", ()
    if replanning.proposed_plan is None:
        return AgentReplannedExecutionStatus.REJECTED, "MISSING_PROPOSED_PLAN", "agent_replanned_execution_rejected", ()
    if policy.require_replanning_signature and not _valid_signature(replanning.request_signature):
        return AgentReplannedExecutionStatus.SIGNATURE_ERROR, "INVALID_REPLANNING_SIGNATURE", "agent_replanned_execution_signature_error", ()
    actual_plan_signature = agent_cooperation_plan_signature(replanning.proposed_plan)
    if replanning.proposed_plan_signature != actual_plan_signature:
        return AgentReplannedExecutionStatus.SIGNATURE_ERROR, "PROPOSED_PLAN_SIGNATURE_MISMATCH", "agent_replanned_execution_signature_error", ()
    if policy.require_progress and not replanning.progress_reasons:
        return AgentReplannedExecutionStatus.REJECTED, "NO_PROGRESS", "agent_replanned_execution_rejected", ()
    if policy.require_supervision and replanning.supervision_signature is None:
        return AgentReplannedExecutionStatus.REJECTED, "MISSING_SUPERVISION_SIGNATURE", "agent_replanned_execution_rejected", ()
    limits = _limits_reached(replanning.proposed_plan, policy)
    if limits:
        return AgentReplannedExecutionStatus.LIMIT_REACHED, "LIMIT_REACHED", "agent_replanned_execution_limit", limits
    return None


def _limits_reached(
    plan: AgentCooperationPlan,
    policy: AgentReplannedExecutionPolicy,
) -> tuple[str, ...]:
    limits: list[str] = []
    checks = (
        (len(plan.tasks), policy.max_tasks, "MAX_TASKS"),
        (len(plan.dependencies), policy.max_tasks, "MAX_DEPENDENCIES"),
        (_plan_depth(plan), policy.max_depth, "MAX_DEPTH"),
        (sum(task.logical_timeout_limit for task in plan.tasks), policy.max_logical_time, "MAX_LOGICAL_TIME"),
    )
    for actual, maximum, name in checks:
        if actual > maximum:
            limits.append(name)
    return tuple(limits)


def _result_limits_reached(
    result: AgentCooperationPlanResult,
    policy: AgentReplannedExecutionPolicy,
) -> tuple[str, ...]:
    limits: list[str] = []
    if len(result.outputs) > policy.max_outputs:
        limits.append("MAX_OUTPUTS")
    failures = sum(item.status is not AgentCooperationTaskStatus.SUCCESS for item in result.task_results)
    if failures > policy.max_total_failures:
        limits.append("MAX_TOTAL_FAILURES")
    if len(result.task_results) > policy.max_total_attempts:
        limits.append("MAX_TOTAL_ATTEMPTS")
    return tuple(limits)


def _execution_status(
    result: AgentCooperationPlanResult,
    policy: AgentReplannedExecutionPolicy,
) -> AgentReplannedExecutionStatus:
    if result.status is AgentCooperationPlanStatus.SUCCESS:
        return AgentReplannedExecutionStatus.SUCCESS
    if result.status is AgentCooperationPlanStatus.PARTIAL_SUCCESS and policy.allow_partial_success:
        return AgentReplannedExecutionStatus.PARTIAL_SUCCESS
    if (
        policy.allow_partial_success
        and any(item.status is AgentCooperationTaskStatus.SUCCESS for item in result.task_results)
    ):
        return AgentReplannedExecutionStatus.PARTIAL_SUCCESS
    return AgentReplannedExecutionStatus.FAILED


def _sanitize_execution_result(
    result: AgentCooperationPlanResult,
) -> AgentCooperationPlanResult:
    task_results = tuple(replace(item, execution_result=None) for item in result.task_results)
    return replace(result, task_results=task_results)


def _result(
    *,
    status: AgentReplannedExecutionStatus,
    reason_code: str,
    request_signature: str,
    replanning_signature: str | None = None,
    proposed_plan_signature: str | None = None,
    replanning_result: AgentPlanReplanningResult | None = None,
    execution_result: AgentCooperationPlanResult | None = None,
    events: Sequence[AgentReplannedExecutionEvent] = (),
    metrics: Mapping[str, int] | None = None,
    limits_reached: tuple[str, ...] = (),
    accepted: bool = False,
) -> AgentReplannedExecutionResult:
    safe_summary = {
        "status": status.value,
        "accepted": accepted,
        "has_replanning": replanning_result is not None,
        "has_execution_result": execution_result is not None,
        "task_count": 0 if execution_result is None else len(execution_result.task_results),
        "output_count": 0 if execution_result is None else len(execution_result.outputs),
    }
    decision = AgentReplannedExecutionDecision(
        status=status,
        reason_code=reason_code,
        accepted=accepted,
        limits_reached=limits_reached,
        safe_summary=safe_summary,
    )
    return AgentReplannedExecutionResult(
        status=status,
        decision=decision,
        request_signature=request_signature,
        replanning_signature=replanning_signature,
        proposed_plan_signature=proposed_plan_signature,
        replanning_result=replanning_result,
        execution_result=execution_result,
        events=tuple(events),
        metrics=_base_metrics() if metrics is None else metrics,
        safe_summary=safe_summary,
    )


def _base_metrics() -> dict[str, int]:
    return {
        "agent_replanned_execution_requests": 0,
        "agent_replanned_execution_success": 0,
        "agent_replanned_execution_failures": 0,
        "agent_replanned_execution_rejected": 0,
        "agent_replanned_execution_duplicates": 0,
        "agent_replanned_execution_limits": 0,
        "agent_replanned_execution_tasks": 0,
        "agent_replanned_execution_outputs": 0,
        "agent_replanned_execution_duration": 0,
        "agent_replanned_execution_partial": 0,
    }


def _event(
    events: list[AgentReplannedExecutionEvent],
    name: str,
    status: str,
    **details: object,
) -> None:
    if len(events) < MAX_REPLANNED_EVENTS:
        events.append(AgentReplannedExecutionEvent(name, status, details))


def _policy_payload(policy: AgentReplannedExecutionPolicy) -> Mapping[str, object]:
    return {name: getattr(policy, name) for name in sorted(policy.__dataclass_fields__)}


def _plan_depth(plan: AgentCooperationPlan) -> int:
    dependencies: dict[str, list[str]] = {task.task_id: [] for task in plan.tasks}
    for dependency in plan.dependencies:
        dependencies.setdefault(dependency.dependent_task_id, []).append(dependency.prerequisite_task_id)
    depths: dict[str, int] = {}
    visiting: set[str] = set()

    def visit(task_id: str) -> int:
        if task_id in visiting:
            return MAX_REPLANNED_DEPTH + 1
        if task_id in depths:
            return depths[task_id]
        visiting.add(task_id)
        depth = 1 + max((visit(source) for source in dependencies.get(task_id, ())), default=0)
        visiting.remove(task_id)
        depths[task_id] = depth
        return depth

    return max((visit(task.task_id) for task in plan.tasks), default=0)


def _safe_mapping(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentReplannedExecutionRequestError(f"{field_name} must be a mapping.")
    if len(value) > MAX_REPLANNED_METADATA_ITEMS:
        raise InvalidAgentReplannedExecutionRequestError(f"{field_name} has too many items.")
    counter = {"items": 0}
    return _safe_mapping_inner(value, field_name, 0, counter)


def _safe_mapping_inner(
    value: Mapping[str, object],
    field_name: str,
    depth: int,
    counter: dict[str, int],
) -> dict[str, object]:
    if depth > MAX_REPLANNED_VALUE_DEPTH:
        raise InvalidAgentReplannedExecutionRequestError(f"{field_name} exceeds maximum depth.")
    result: dict[str, object] = {}
    for raw_key in sorted(value, key=lambda item: str(item)):
        key = _identifier(raw_key, f"{field_name} key")
        if _forbidden_key(key):
            raise InvalidAgentReplannedExecutionRequestError(f"{field_name} contains a forbidden key.")
        result[key] = _safe_value(value[raw_key], field_name, depth + 1, counter)
    return result


def _safe_value(value: object, field_name: str, depth: int, counter: dict[str, int]) -> object:
    if depth > MAX_REPLANNED_VALUE_DEPTH:
        raise InvalidAgentReplannedExecutionRequestError(f"{field_name} exceeds maximum depth.")
    counter["items"] += 1
    if counter["items"] > MAX_REPLANNED_TOTAL_ITEMS:
        raise InvalidAgentReplannedExecutionRequestError(f"{field_name} exceeds item limit.")
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentReplannedExecutionRequestError(f"{field_name} floats must be finite.")
        return value
    if isinstance(value, str):
        if len(value) > MAX_REPLANNED_STRING_LENGTH:
            raise InvalidAgentReplannedExecutionRequestError(f"{field_name} string exceeds length limit.")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_mapping_inner(value, field_name, depth + 1, counter))
    if isinstance(value, (tuple, list)):
        return tuple(_safe_value(item, field_name, depth + 1, counter) for item in value)
    raise InvalidAgentReplannedExecutionRequestError(f"{field_name} contains unsupported values.")


def _safe_primitive_mapping(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentReplannedExecutionRequestError(f"{field_name} must be a mapping.")
    result: dict[str, object] = {}
    for raw_key in sorted(value, key=lambda item: str(item)):
        key = _identifier(raw_key, f"{field_name} key")
        item = value[raw_key]
        if item is None or type(item) in (bool, int, str):
            result[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[key] = item
        else:
            raise InvalidAgentReplannedExecutionRequestError(f"{field_name} values must be primitive.")
    return result


def _metric_mapping(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise InvalidAgentReplannedExecutionRequestError("metrics must be a mapping.")
    result: dict[str, int] = {}
    for key, item in value.items():
        normalized = _identifier(key, "metric name")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise InvalidAgentReplannedExecutionRequestError("metrics must be non-negative integers.")
        result[normalized] = item
    return result


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
    raise InvalidAgentReplannedExecutionRequestError("value is not serializable.")


def _forbidden_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS + _DYNAMIC_KEY_PARTS)


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 128:
        raise InvalidAgentReplannedExecutionRequestError(f"{field_name} contains unsupported characters.")
    if not all(character.isalnum() or character in "_.-" for character in value):
        raise InvalidAgentReplannedExecutionRequestError(f"{field_name} contains unsupported characters.")
    return value


def _identifier_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise InvalidAgentReplannedExecutionRequestError(f"{field_name} must be a sequence.")
    return tuple(dict.fromkeys(_identifier(value, field_name) for value in tuple(values)))


def _bounded_int(value: int, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise InvalidAgentReplannedExecutionRequestError(f"{field_name} is outside the allowed range.")
    return value


def _validate_signature(value: str, field_name: str, *, allow_empty: bool = False) -> None:
    if allow_empty and value == "":
        return
    if not _valid_signature(value):
        raise InvalidAgentReplannedExecutionRequestError(f"{field_name} must be a SHA-256 hex digest.")


def _valid_signature(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in _SIGNATURE_HEX for character in value)


def _status(value: AgentReplannedExecutionStatus | str) -> AgentReplannedExecutionStatus:
    try:
        return value if isinstance(value, AgentReplannedExecutionStatus) else AgentReplannedExecutionStatus(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentReplannedExecutionRequestError("invalid replanned execution status.") from error

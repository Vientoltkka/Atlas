"""Controlled deterministic replanning proposals for cooperation plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType

from core.agent_cooperation_plan import (
    AgentCooperationDependency,
    AgentCooperationPlan,
    AgentCooperationTask,
    AgentCooperationTaskStatus,
    agent_cooperation_plan_signature,
)
from core.agent_plan_supervisor import (
    AgentPlanSupervisor,
    AgentPlanSupervisorDecisionType,
    AgentPlanSupervisorResult,
    AgentPlanSupervisorStatus,
)
from core.agent_registry import AgentDefinition, AgentRegistry
from core.agent_resolver import (
    AgentResolutionRequest,
    AgentResolutionStatus,
    AgentResolver,
)


MAX_REPLANNING_ATTEMPTS = 16
MAX_REPLANNING_TASKS = 100
MAX_REPLANNING_DEPENDENCIES = 400
MAX_REPLANNING_PLAN_DEPTH = 32
MAX_REPLANNING_ACTIONS = 128
MAX_REPLANNING_RETRY_TASKS = 64
MAX_REPLANNING_SKIPPED_TASKS = 64
MAX_REPLANNING_REPLACEMENT_AGENTS = 32
MAX_REPLANNING_OUTPUT_ITEMS = 1_024
MAX_REPLANNING_LOGICAL_TIME = 1_000_000
MAX_REPLANNING_METADATA_ITEMS = 32
MAX_REPLANNING_VALUE_DEPTH = 8
MAX_REPLANNING_STRING_LENGTH = 1_000
MAX_REPLANNING_EVENTS = 1_024
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
_DYNAMIC_KEY_PARTS = (
    "python_path",
    "module_path",
    "import_path",
    "handler_path",
    "callback",
    "callable",
    "coroutine",
)


class AgentPlanReplanningError(RuntimeError):
    """Base error for deterministic cooperation-plan replanning."""


class InvalidAgentPlanReplanningRequestError(AgentPlanReplanningError):
    """Raised when a replanning request or policy is malformed."""


class AgentPlanReplanningStatus(str, Enum):
    """Terminal statuses for one replanning attempt."""

    DISABLED = "DISABLED"
    VALID = "VALID"
    INVALID_REQUEST = "INVALID_REQUEST"
    KEEP_ORIGINAL_PLAN = "KEEP_ORIGINAL_PLAN"
    REPLAN_PROPOSED = "REPLAN_PROPOSED"
    REPLAN_BLOCKED = "REPLAN_BLOCKED"
    REPLAN_REJECTED = "REPLAN_REJECTED"
    PLAN_SIGNATURE_MISMATCH = "PLAN_SIGNATURE_MISMATCH"
    SUPERVISION_REJECTED = "SUPERVISION_REJECTED"
    NO_RECOVERABLE_ACTION = "NO_RECOVERABLE_ACTION"
    LIMIT_REACHED = "LIMIT_REACHED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentPlanReplanningDecisionType(str, Enum):
    """Structured decision emitted by the replanner."""

    KEEP_ORIGINAL_PLAN = "KEEP_ORIGINAL_PLAN"
    REPLAN_REQUIRED = "REPLAN_REQUIRED"
    REPLAN_NOT_REQUIRED = "REPLAN_NOT_REQUIRED"
    REPLAN_BLOCKED = "REPLAN_BLOCKED"
    INVALID_REQUEST = "INVALID_REQUEST"
    PLAN_SIGNATURE_MISMATCH = "PLAN_SIGNATURE_MISMATCH"
    SUPERVISION_REJECTED = "SUPERVISION_REJECTED"
    NO_RECOVERABLE_ACTION = "NO_RECOVERABLE_ACTION"
    LIMIT_REACHED = "LIMIT_REACHED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentPlanReplanningActionType(str, Enum):
    """Allowed structural actions in a proposal."""

    RETRY_TASK = "RETRY_TASK"
    SKIP_TASK = "SKIP_TASK"
    REMOVE_TASK = "REMOVE_TASK"
    REPLACE_AGENT = "REPLACE_AGENT"
    REBUILD_DEPENDENCIES = "REBUILD_DEPENDENCIES"
    PRESERVE_TASK = "PRESERVE_TASK"
    BLOCK_DEPENDENT_TASK = "BLOCK_DEPENDENT_TASK"
    REJECT_REPLAN = "REJECT_REPLAN"


@dataclass(frozen=True, slots=True)
class AgentPlanReplanningReason:
    """One safe reason for a decision or action."""

    code: str
    detail: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _safe_identifier(self.code, "reason code"))
        if self.detail is not None:
            object.__setattr__(self, "detail", _safe_text(self.detail))


@dataclass(frozen=True, slots=True)
class AgentPlanReplanningPolicy:
    """Explicit opt-in policy and conservative bounds."""

    enabled: bool = False
    max_replanning_attempts: int = 3
    max_tasks: int = 64
    max_dependencies: int = 128
    max_plan_depth: int = 12
    max_actions: int = 64
    max_retry_tasks: int = 16
    max_skipped_tasks: int = 16
    max_replacement_agents: int = 8
    max_output_items: int = 512
    max_logical_time: int = 10_000
    allow_retry_failed_tasks: bool = False
    allow_skip_unrecoverable_tasks: bool = False
    allow_remove_blocked_tasks: bool = False
    allow_dependency_rebuild: bool = False
    allow_agent_reselection: bool = False
    allow_partial_plan: bool = False
    require_plan_signature_match: bool = True
    require_supervision_success: bool = False
    require_progress: bool = True
    fail_closed: bool = True

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "allow_retry_failed_tasks",
            "allow_skip_unrecoverable_tasks",
            "allow_remove_blocked_tasks",
            "allow_dependency_rebuild",
            "allow_agent_reselection",
            "allow_partial_plan",
            "require_plan_signature_match",
            "require_supervision_success",
            "require_progress",
            "fail_closed",
        ):
            if type(getattr(self, name)) is not bool:
                raise InvalidAgentPlanReplanningRequestError(f"{name} must be a bool.")
        for name, maximum in (
            ("max_replanning_attempts", MAX_REPLANNING_ATTEMPTS),
            ("max_tasks", MAX_REPLANNING_TASKS),
            ("max_dependencies", MAX_REPLANNING_DEPENDENCIES),
            ("max_plan_depth", MAX_REPLANNING_PLAN_DEPTH),
            ("max_actions", MAX_REPLANNING_ACTIONS),
            ("max_retry_tasks", MAX_REPLANNING_RETRY_TASKS),
            ("max_skipped_tasks", MAX_REPLANNING_SKIPPED_TASKS),
            ("max_replacement_agents", MAX_REPLANNING_REPLACEMENT_AGENTS),
            ("max_output_items", MAX_REPLANNING_OUTPUT_ITEMS),
            ("max_logical_time", MAX_REPLANNING_LOGICAL_TIME),
        ):
            object.__setattr__(self, name, _bounded_int(getattr(self, name), name, maximum))


@dataclass(frozen=True, slots=True)
class AgentPlanReplanningAction:
    """One deterministic structural action in a replanning proposal."""

    task_id: str
    action_type: AgentPlanReplanningActionType
    reason: AgentPlanReplanningReason
    previous_agent_id: str | None = None
    replacement_agent_id: str | None = None
    affected_dependency_ids: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    order: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _safe_identifier(self.task_id, "task_id"))
        object.__setattr__(self, "action_type", _action_type(self.action_type))
        if not isinstance(self.reason, AgentPlanReplanningReason):
            raise InvalidAgentPlanReplanningRequestError("reason must be AgentPlanReplanningReason.")
        for name in ("previous_agent_id", "replacement_agent_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _safe_identifier(value, name))
        object.__setattr__(
            self,
            "affected_dependency_ids",
            _identifier_tuple(self.affected_dependency_ids, "affected_dependency_ids"),
        )
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise InvalidAgentPlanReplanningRequestError("order must be a non-negative integer.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))


@dataclass(frozen=True, slots=True)
class AgentPlanReplanningDecision:
    """Bounded explanation for the replanning result."""

    decision: AgentPlanReplanningDecisionType
    reasons: tuple[AgentPlanReplanningReason, ...] = ()
    actions: tuple[AgentPlanReplanningAction, ...] = ()
    limits_reached: tuple[str, ...] = ()
    progress_reasons: tuple[str, ...] = ()
    safe_summary: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", _decision_type(self.decision))
        reasons = tuple(self.reasons)
        if not all(isinstance(item, AgentPlanReplanningReason) for item in reasons):
            raise InvalidAgentPlanReplanningRequestError("reasons must contain AgentPlanReplanningReason values.")
        actions = tuple(sorted(self.actions, key=lambda item: (item.order, item.task_id, item.action_type.value)))
        if not all(isinstance(item, AgentPlanReplanningAction) for item in actions):
            raise InvalidAgentPlanReplanningRequestError("actions must contain AgentPlanReplanningAction values.")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "limits_reached", _identifier_tuple(self.limits_reached, "limits_reached"))
        object.__setattr__(self, "progress_reasons", _identifier_tuple(self.progress_reasons, "progress_reasons"))
        object.__setattr__(
            self,
            "safe_summary",
            MappingProxyType(_safe_primitive_mapping(self.safe_summary, "safe_summary")),
        )


@dataclass(frozen=True, slots=True)
class AgentPlanReplanningEvent:
    """Safe structured replanning event."""

    name: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _safe_identifier(self.name, "event name"))
        object.__setattr__(self, "status", _safe_identifier(self.status, "event status"))
        object.__setattr__(
            self,
            "details",
            MappingProxyType(_safe_primitive_mapping(self.details, "event details")),
        )


@dataclass(frozen=True, slots=True)
class AgentPlanReplanningRequest:
    """Input for generating one structural replanning proposal."""

    original_plan: AgentCooperationPlan
    supervision_result: AgentPlanSupervisorResult
    policy: AgentPlanReplanningPolicy = field(default_factory=AgentPlanReplanningPolicy)
    metadata: Mapping[str, object] = field(default_factory=dict)
    correlation_id: str | None = None
    causation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.original_plan, AgentCooperationPlan):
            raise InvalidAgentPlanReplanningRequestError("original_plan must be AgentCooperationPlan.")
        if not isinstance(self.supervision_result, AgentPlanSupervisorResult):
            raise InvalidAgentPlanReplanningRequestError("supervision_result must be AgentPlanSupervisorResult.")
        if not isinstance(self.policy, AgentPlanReplanningPolicy):
            raise InvalidAgentPlanReplanningRequestError("policy must be AgentPlanReplanningPolicy.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))
        for name in ("correlation_id", "causation_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _safe_identifier(value, name))


@dataclass(frozen=True, slots=True)
class AgentPlanReplanningResult:
    """Immutable proposal returned by the replanner."""

    status: AgentPlanReplanningStatus
    decision: AgentPlanReplanningDecision
    request_signature: str
    original_plan_signature: str | None = None
    supervision_signature: str | None = None
    proposed_plan: AgentCooperationPlan | None = None
    proposed_plan_signature: str | None = None
    actions: tuple[AgentPlanReplanningAction, ...] = ()
    preserved_tasks: tuple[str, ...] = ()
    retried_tasks: tuple[str, ...] = ()
    skipped_tasks: tuple[str, ...] = ()
    removed_tasks: tuple[str, ...] = ()
    replaced_agents: tuple[str, ...] = ()
    rebuilt_dependencies: tuple[str, ...] = ()
    progress_reasons: tuple[str, ...] = ()
    limits_reached: tuple[str, ...] = ()
    events: tuple[AgentPlanReplanningEvent, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    safe_summary: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        if not isinstance(self.decision, AgentPlanReplanningDecision):
            raise InvalidAgentPlanReplanningRequestError("decision must be AgentPlanReplanningDecision.")
        _validate_signature(self.request_signature, "request_signature", allow_empty=True)
        for name in ("original_plan_signature", "supervision_signature", "proposed_plan_signature"):
            value = getattr(self, name)
            if value is not None:
                _validate_signature(value, name)
        if self.proposed_plan is not None and not isinstance(self.proposed_plan, AgentCooperationPlan):
            raise InvalidAgentPlanReplanningRequestError("proposed_plan must be AgentCooperationPlan or None.")
        actions = tuple(sorted(self.actions, key=lambda item: (item.order, item.task_id, item.action_type.value)))
        if not all(isinstance(item, AgentPlanReplanningAction) for item in actions):
            raise InvalidAgentPlanReplanningRequestError("actions must contain AgentPlanReplanningAction values.")
        object.__setattr__(self, "actions", actions)
        for name in (
            "preserved_tasks",
            "retried_tasks",
            "skipped_tasks",
            "removed_tasks",
            "replaced_agents",
            "rebuilt_dependencies",
            "progress_reasons",
            "limits_reached",
        ):
            object.__setattr__(self, name, _identifier_tuple(getattr(self, name), name))
        events = tuple(self.events)[-MAX_REPLANNING_EVENTS:]
        if not all(isinstance(item, AgentPlanReplanningEvent) for item in events):
            raise InvalidAgentPlanReplanningRequestError("events must contain AgentPlanReplanningEvent values.")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "metrics", MappingProxyType(_metric_mapping(self.metrics)))
        object.__setattr__(
            self,
            "safe_summary",
            MappingProxyType(_safe_primitive_mapping(self.safe_summary, "safe_summary")),
        )


class AgentPlanReplanner:
    """Create structural replanning proposals without running the proposal."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        agent_resolver: AgentResolver,
        agent_cooperation_planner: object,
        agent_plan_supervisor: AgentPlanSupervisor,
    ) -> None:
        if not isinstance(agent_registry, AgentRegistry):
            raise AgentPlanReplanningError("agent_registry must be AgentRegistry.")
        if not isinstance(agent_resolver, AgentResolver):
            raise AgentPlanReplanningError("agent_resolver must be AgentResolver.")
        if not isinstance(agent_plan_supervisor, AgentPlanSupervisor):
            raise AgentPlanReplanningError("agent_plan_supervisor must be AgentPlanSupervisor.")
        self.agent_registry = agent_registry
        self.agent_resolver = agent_resolver
        self.agent_cooperation_planner = agent_cooperation_planner
        self.agent_plan_supervisor = agent_plan_supervisor

    def replan(self, request: AgentPlanReplanningRequest) -> AgentPlanReplanningResult:
        """Return a deterministic proposal derived from structured supervision."""

        events: list[AgentPlanReplanningEvent] = []
        metrics = _base_metrics()
        metrics["agent_plan_replanning_requests"] = 1
        _event(events, "agent_plan_replanning_requested", "requested")
        _event(events, "agent_plan_replanning_validation_started", "started")
        if not isinstance(request, AgentPlanReplanningRequest):
            metrics["agent_plan_replanning_failed"] = 1
            _event(events, "agent_plan_replanning_validation_failed", "failed")
            return _simple_result(
                AgentPlanReplanningStatus.INVALID_REQUEST,
                AgentPlanReplanningDecisionType.INVALID_REQUEST,
                "",
                (AgentPlanReplanningReason("INVALID_REQUEST"),),
                events,
                metrics,
            )
        signature = agent_plan_replanning_request_signature(request)
        original_signature = agent_cooperation_plan_signature(request.original_plan)
        supervision_signature = _supervision_signature(request.supervision_result)
        if not request.policy.enabled:
            _event(events, "agent_plan_replanning_completed", "disabled")
            metrics["agent_plan_replanning_blocked"] = 1
            return _simple_result(
                AgentPlanReplanningStatus.DISABLED,
                AgentPlanReplanningDecisionType.REPLAN_BLOCKED,
                signature,
                (AgentPlanReplanningReason("POLICY_DISABLED"),),
                events,
                metrics,
                original_signature=original_signature,
                supervision_signature=supervision_signature,
            )
        validation = _validate_request(request, original_signature)
        if validation is not None:
            status, decision, reason = validation
            _event(events, "agent_plan_replanning_validation_failed", "failed", reason=reason.code)
            metrics["agent_plan_replanning_failed"] = 1
            return _simple_result(
                status,
                decision,
                signature,
                (reason,),
                events,
                metrics,
                original_signature=original_signature,
                supervision_signature=supervision_signature,
            )
        _event(events, "agent_plan_replanning_validation_succeeded", "succeeded")
        _event(events, "agent_plan_replanning_analysis_started", "started")
        return self._replan_validated(request, signature, original_signature, supervision_signature, events, metrics)

    def _replan_validated(
        self,
        request: AgentPlanReplanningRequest,
        request_signature: str,
        original_signature: str,
        supervision_signature: str,
        events: list[AgentPlanReplanningEvent],
        metrics: dict[str, int],
    ) -> AgentPlanReplanningResult:
        plan = request.original_plan
        policy = request.policy
        task_by_id = {task.task_id: task for task in plan.tasks}
        actions: list[AgentPlanReplanningAction] = []
        changed_tasks: dict[str, AgentCooperationTask | None] = {}
        progress: list[str] = []
        limits: list[str] = []
        blocked_reasons: list[AgentPlanReplanningReason] = []
        successful = set(request.supervision_result.succeeded_tasks)
        failed = tuple(task_id for task_id in request.supervision_result.failed_tasks if task_id in task_by_id)
        blocked = tuple(task_id for task_id in request.supervision_result.blocked_tasks if task_id in task_by_id)
        skipped = tuple(task_id for task_id in request.supervision_result.skipped_tasks if task_id in task_by_id)
        unknown = tuple(task_id for task_id in request.supervision_result.unknown_tasks if task_id not in task_by_id)
        if unknown:
            blocked_reasons.append(AgentPlanReplanningReason("UNKNOWN_TASK_RESULTS"))
        for task in plan.tasks:
            if task.task_id in successful:
                actions.append(_action(task.task_id, "PRESERVE_TASK", "TASK_SUCCEEDED", len(actions)))
                _event(events, "agent_plan_replanning_task_classified", "succeeded", task_id=task.task_id)
        retry_count = 0
        skip_count = 0
        replacement_count = 0
        for task_id in failed:
            task = task_by_id[task_id]
            _event(events, "agent_plan_replanning_task_classified", "failed", task_id=task_id)
            replacement = None
            if policy.allow_agent_reselection and task.agent_id is not None:
                replacement, reason = self._replacement_agent(task, policy, events)
                if reason is not None:
                    blocked_reasons.append(reason)
                if replacement is not None:
                    replacement_count += 1
                    if replacement_count > policy.max_replacement_agents:
                        limits.append("MAX_REPLACEMENT_AGENTS")
                        _event(events, "agent_plan_replanning_limit_reached", "reached", limit_name="MAX_REPLACEMENT_AGENTS")
                    changed_tasks[task_id] = replace(task, agent_id=replacement.agent_id)
                    actions.append(
                        AgentPlanReplanningAction(
                            task_id=task_id,
                            action_type=AgentPlanReplanningActionType.REPLACE_AGENT,
                            reason=AgentPlanReplanningReason("COMPATIBLE_AGENT_RESELECTED"),
                            previous_agent_id=task.agent_id,
                            replacement_agent_id=replacement.agent_id,
                            order=len(actions),
                        )
                    )
                    _event(events, "agent_plan_replanning_action_created", "created", action_type="REPLACE_AGENT")
                    progress.append("AGENT_REPLACED")
            if policy.allow_retry_failed_tasks:
                retry_count += 1
                if retry_count > policy.max_retry_tasks:
                    limits.append("MAX_RETRY_TASKS")
                    _event(events, "agent_plan_replanning_limit_reached", "reached", limit_name="MAX_RETRY_TASKS")
                changed_tasks.setdefault(task_id, _retry_task(task))
                actions.append(_action(task_id, "RETRY_TASK", "FAILED_TASK_RECOVERABLE", len(actions)))
                progress.append("FAILED_TASK_RETRIED")
            elif policy.allow_skip_unrecoverable_tasks:
                skip_count += 1
                if skip_count > policy.max_skipped_tasks:
                    limits.append("MAX_SKIPPED_TASKS")
                    _event(events, "agent_plan_replanning_limit_reached", "reached", limit_name="MAX_SKIPPED_TASKS")
                actions.append(_action(task_id, "SKIP_TASK", "FAILED_TASK_SKIPPED", len(actions)))
                changed_tasks[task_id] = None
                progress.append("TASK_SKIPPED")
            elif replacement is None:
                blocked_reasons.append(AgentPlanReplanningReason("RETRY_NOT_ALLOWED"))
        for task_id in skipped:
            _event(events, "agent_plan_replanning_task_classified", "skipped", task_id=task_id)
            if policy.allow_skip_unrecoverable_tasks:
                skip_count += 1
                actions.append(_action(task_id, "SKIP_TASK", "SKIPPED_TASK_AUTHORIZED", len(actions)))
                changed_tasks[task_id] = None
                progress.append("TASK_SKIPPED")
            else:
                blocked_reasons.append(AgentPlanReplanningReason("SKIP_NOT_ALLOWED"))
        for task_id in blocked:
            _event(events, "agent_plan_replanning_task_classified", "blocked", task_id=task_id)
            if policy.allow_remove_blocked_tasks:
                actions.append(_action(task_id, "REMOVE_TASK", "BLOCKED_TASK_REMOVED", len(actions)))
                changed_tasks[task_id] = None
                progress.append("BLOCKED_TASK_REMOVED")
            else:
                actions.append(_action(task_id, "BLOCK_DEPENDENT_TASK", "BLOCKED_TASK_RETAINED", len(actions)))
                blocked_reasons.append(AgentPlanReplanningReason("REMOVE_BLOCKED_TASK_NOT_ALLOWED"))
        _check_limits(plan, actions, policy, limits, events)
        if limits:
            metrics["agent_plan_replanning_limits_reached"] = len(set(limits))
            return self._finish(
                AgentPlanReplanningStatus.LIMIT_REACHED,
                AgentPlanReplanningDecisionType.LIMIT_REACHED,
                request_signature,
                original_signature,
                supervision_signature,
                None,
                actions,
                (AgentPlanReplanningReason("LIMIT_REACHED"),),
                progress,
                limits,
                events,
                metrics,
            )
        if blocked_reasons and policy.fail_closed and not actions:
            _event(events, "agent_plan_replanning_plan_rejected", "blocked")
            return self._finish(
                AgentPlanReplanningStatus.REPLAN_BLOCKED,
                AgentPlanReplanningDecisionType.REPLAN_BLOCKED,
                request_signature,
                original_signature,
                supervision_signature,
                None,
                actions + [_action("plan", "REJECT_REPLAN", blocked_reasons[0].code, len(actions))],
                tuple(blocked_reasons),
                progress,
                limits,
                events,
                metrics,
            )
        if not failed and not blocked and not skipped and not unknown:
            _event(events, "agent_plan_replanning_no_progress", "not_required")
            metrics["agent_plan_replanning_no_change"] = 1
            return self._finish(
                AgentPlanReplanningStatus.KEEP_ORIGINAL_PLAN,
                AgentPlanReplanningDecisionType.KEEP_ORIGINAL_PLAN,
                request_signature,
                original_signature,
                supervision_signature,
                plan,
                actions,
                (AgentPlanReplanningReason("ALL_TASKS_SUCCEEDED"),),
                (),
                (),
                events,
                metrics,
            )
        proposed = _build_proposed_plan(plan, changed_tasks, policy, actions, events)
        if proposed is None:
            _event(events, "agent_plan_replanning_plan_rejected", "invalid")
            metrics["agent_plan_replanning_cycles_detected"] = 1
            return self._finish(
                AgentPlanReplanningStatus.REPLAN_REJECTED,
                AgentPlanReplanningDecisionType.REPLAN_BLOCKED,
                request_signature,
                original_signature,
                supervision_signature,
                None,
                actions + [_action("plan", "REJECT_REPLAN", "INVALID_PROPOSED_DAG", len(actions))],
                (AgentPlanReplanningReason("INVALID_PROPOSED_DAG"),),
                progress,
                (),
                events,
                metrics,
            )
        proposed_signature = agent_cooperation_plan_signature(proposed)
        if policy.require_progress and (not progress or proposed_signature == original_signature):
            _event(events, "agent_plan_replanning_no_progress", "no_progress")
            metrics["agent_plan_replanning_no_change"] = 1
            return self._finish(
                AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION,
                AgentPlanReplanningDecisionType.NO_RECOVERABLE_ACTION,
                request_signature,
                original_signature,
                supervision_signature,
                None,
                actions,
                (AgentPlanReplanningReason("NO_STRUCTURAL_PROGRESS"),),
                (),
                (),
                events,
                metrics,
            )
        _event(events, "agent_plan_replanning_plan_created", "created")
        return self._finish(
            AgentPlanReplanningStatus.REPLAN_PROPOSED,
            AgentPlanReplanningDecisionType.REPLAN_REQUIRED,
            request_signature,
            original_signature,
            supervision_signature,
            proposed,
            actions,
            (AgentPlanReplanningReason("STRUCTURAL_PROGRESS"),),
            tuple(sorted(set(progress))),
            (),
            events,
            metrics,
        )

    def _replacement_agent(
        self,
        task: AgentCooperationTask,
        policy: AgentPlanReplanningPolicy,
        events: list[AgentPlanReplanningEvent],
    ) -> tuple[AgentDefinition | None, AgentPlanReplanningReason | None]:
        _event(events, "agent_plan_replanning_agent_reselection_started", "started", task_id=task.task_id)
        request = AgentResolutionRequest(
            required_capability_ids=task.required_capability_ids,
            required_agent_types=task.required_agent_types,
            required_permission_ids=task.required_permission_ids,
            excluded_agent_ids=tuple(sorted(set(task.excluded_agent_ids + ((task.agent_id,) if task.agent_id else ())))),
            enabled_only=task.enabled_only,
            require_unique_top_score=True,
            maximum_candidates_considered=policy.max_replacement_agents,
        )
        result = self.agent_resolver.resolve(request)
        if result.status is not AgentResolutionStatus.RESOLVED or result.selected_agent is None:
            _event(
                events,
                "agent_plan_replanning_agent_reselection_failed",
                "failed",
                resolution_status=result.status.value,
            )
            code = "AMBIGUOUS_AGENT_RESELECTION" if result.status is AgentResolutionStatus.AMBIGUOUS else "NO_REPLACEMENT_AGENT"
            return None, AgentPlanReplanningReason(code)
        if not _agent_authorizes_required_skills(result.selected_agent, task.required_skill_ids):
            _event(
                events,
                "agent_plan_replanning_agent_reselection_failed",
                "failed",
                resolution_status="SKILL_NOT_AUTHORIZED",
            )
            return None, AgentPlanReplanningReason("REQUIRED_SKILL_NOT_AUTHORIZED")
        _event(events, "agent_plan_replanning_agent_reselection_succeeded", "succeeded", task_id=task.task_id)
        return result.selected_agent, None

    def _finish(
        self,
        status: AgentPlanReplanningStatus,
        decision_type: AgentPlanReplanningDecisionType,
        request_signature: str,
        original_signature: str | None,
        supervision_signature: str | None,
        proposed_plan: AgentCooperationPlan | None,
        actions: Sequence[AgentPlanReplanningAction],
        reasons: tuple[AgentPlanReplanningReason, ...],
        progress: Sequence[str],
        limits: Sequence[str],
        events: list[AgentPlanReplanningEvent],
        metrics: dict[str, int],
    ) -> AgentPlanReplanningResult:
        _event(events, "agent_plan_replanning_completed", "completed")
        if status is AgentPlanReplanningStatus.REPLAN_PROPOSED:
            metrics["agent_plan_replanning_succeeded"] = 1
        elif status in (AgentPlanReplanningStatus.KEEP_ORIGINAL_PLAN, AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION):
            metrics["agent_plan_replanning_no_change"] = max(metrics["agent_plan_replanning_no_change"], 1)
        elif status in (AgentPlanReplanningStatus.REPLAN_BLOCKED, AgentPlanReplanningStatus.DISABLED):
            metrics["agent_plan_replanning_blocked"] = 1
        else:
            metrics["agent_plan_replanning_failed"] = 1
        _fill_action_metrics(metrics, actions)
        proposed_signature = None if proposed_plan is None else agent_cooperation_plan_signature(proposed_plan)
        summary = {
            "status": status.value,
            "actions": len(tuple(actions)),
            "has_proposed_plan": proposed_plan is not None,
            "progress": len(tuple(progress)),
            "limits_reached": len(tuple(limits)),
        }
        decision = AgentPlanReplanningDecision(
            decision=decision_type,
            reasons=reasons,
            actions=tuple(actions),
            limits_reached=tuple(sorted(set(limits))),
            progress_reasons=tuple(sorted(set(progress))),
            safe_summary=summary,
        )
        return AgentPlanReplanningResult(
            status=status,
            decision=decision,
            request_signature=request_signature,
            original_plan_signature=original_signature,
            supervision_signature=supervision_signature,
            proposed_plan=proposed_plan,
            proposed_plan_signature=proposed_signature,
            actions=tuple(actions),
            preserved_tasks=tuple(action.task_id for action in actions if action.action_type is AgentPlanReplanningActionType.PRESERVE_TASK),
            retried_tasks=tuple(action.task_id for action in actions if action.action_type is AgentPlanReplanningActionType.RETRY_TASK),
            skipped_tasks=tuple(action.task_id for action in actions if action.action_type is AgentPlanReplanningActionType.SKIP_TASK),
            removed_tasks=tuple(action.task_id for action in actions if action.action_type is AgentPlanReplanningActionType.REMOVE_TASK),
            replaced_agents=tuple(action.task_id for action in actions if action.action_type is AgentPlanReplanningActionType.REPLACE_AGENT),
            rebuilt_dependencies=tuple(action.task_id for action in actions if action.action_type is AgentPlanReplanningActionType.REBUILD_DEPENDENCIES),
            progress_reasons=tuple(sorted(set(progress))),
            limits_reached=tuple(sorted(set(limits))),
            events=tuple(events),
            metrics=metrics,
            safe_summary=summary,
        )


def agent_plan_replanning_request_signature(request: AgentPlanReplanningRequest) -> str:
    """Return a canonical SHA-256 signature for one replanning request."""

    if not isinstance(request, AgentPlanReplanningRequest):
        raise InvalidAgentPlanReplanningRequestError("request must be AgentPlanReplanningRequest.")
    payload = {
        "original_plan_signature": agent_cooperation_plan_signature(request.original_plan),
        "supervision_signature": _supervision_signature(request.supervision_result),
        "policy": _policy_payload(request.policy),
        "supervised_tasks": {
            "succeeded": request.supervision_result.succeeded_tasks,
            "failed": request.supervision_result.failed_tasks,
            "skipped": request.supervision_result.skipped_tasks,
            "blocked": request.supervision_result.blocked_tasks,
            "unknown": request.supervision_result.unknown_tasks,
            "missing": request.supervision_result.missing_tasks,
            "limits": request.supervision_result.limits_reached,
        },
        "metadata": request.metadata,
        "correlation_id": request.correlation_id,
        "causation_id": request.causation_id,
    }
    return _signature(payload)


def build_core_agent_plan_replanner(
    *,
    agent_registry: AgentRegistry,
    agent_resolver: AgentResolver,
    agent_cooperation_planner: object,
    agent_plan_supervisor: AgentPlanSupervisor,
) -> AgentPlanReplanner:
    """Build the pure replanner from shared collaborators."""

    return AgentPlanReplanner(
        agent_registry=agent_registry,
        agent_resolver=agent_resolver,
        agent_cooperation_planner=agent_cooperation_planner,
        agent_plan_supervisor=agent_plan_supervisor,
    )


def _validate_request(
    request: AgentPlanReplanningRequest,
    original_signature: str,
) -> tuple[AgentPlanReplanningStatus, AgentPlanReplanningDecisionType, AgentPlanReplanningReason] | None:
    supervision = request.supervision_result
    policy = request.policy
    if policy.require_plan_signature_match and supervision.plan_signature != original_signature:
        return (
            AgentPlanReplanningStatus.PLAN_SIGNATURE_MISMATCH,
            AgentPlanReplanningDecisionType.PLAN_SIGNATURE_MISMATCH,
            AgentPlanReplanningReason("PLAN_SIGNATURE_MISMATCH"),
        )
    if supervision.decision.decision is AgentPlanSupervisorDecisionType.REJECTED and policy.fail_closed:
        return (
            AgentPlanReplanningStatus.SUPERVISION_REJECTED,
            AgentPlanReplanningDecisionType.SUPERVISION_REJECTED,
            AgentPlanReplanningReason("SUPERVISION_REJECTED"),
        )
    if policy.require_supervision_success and supervision.status is not AgentPlanSupervisorStatus.SUCCESS:
        return (
            AgentPlanReplanningStatus.SUPERVISION_REJECTED,
            AgentPlanReplanningDecisionType.SUPERVISION_REJECTED,
            AgentPlanReplanningReason("SUPERVISION_NOT_SUCCESSFUL"),
        )
    if len(request.original_plan.tasks) > policy.max_tasks:
        return (AgentPlanReplanningStatus.LIMIT_REACHED, AgentPlanReplanningDecisionType.LIMIT_REACHED, AgentPlanReplanningReason("MAX_TASKS"))
    if len(request.original_plan.dependencies) > policy.max_dependencies:
        return (AgentPlanReplanningStatus.LIMIT_REACHED, AgentPlanReplanningDecisionType.LIMIT_REACHED, AgentPlanReplanningReason("MAX_DEPENDENCIES"))
    if _plan_depth(request.original_plan) > policy.max_plan_depth:
        return (AgentPlanReplanningStatus.LIMIT_REACHED, AgentPlanReplanningDecisionType.LIMIT_REACHED, AgentPlanReplanningReason("MAX_PLAN_DEPTH"))
    if sum(task.logical_timeout_limit for task in request.original_plan.tasks) > policy.max_logical_time:
        return (AgentPlanReplanningStatus.LIMIT_REACHED, AgentPlanReplanningDecisionType.LIMIT_REACHED, AgentPlanReplanningReason("MAX_LOGICAL_TIME"))
    return None


def _build_proposed_plan(
    original: AgentCooperationPlan,
    changed_tasks: Mapping[str, AgentCooperationTask | None],
    policy: AgentPlanReplanningPolicy,
    actions: list[AgentPlanReplanningAction],
    events: list[AgentPlanReplanningEvent],
) -> AgentCooperationPlan | None:
    removed = {task_id for task_id, task in changed_tasks.items() if task is None}
    tasks = tuple(
        changed_tasks.get(task.task_id, task)
        for task in original.tasks
        if task.task_id not in removed
    )
    tasks = tuple(task for task in tasks if task is not None)
    dependency_ids = _dependency_ids(original.dependencies)
    dependencies = tuple(
        dependency
        for dependency in original.dependencies
        if dependency.prerequisite_task_id not in removed and dependency.dependent_task_id not in removed
    )
    if len(dependencies) != len(original.dependencies):
        if not policy.allow_dependency_rebuild:
            return None
        actions.append(
            AgentPlanReplanningAction(
                task_id="plan",
                action_type=AgentPlanReplanningActionType.REBUILD_DEPENDENCIES,
                reason=AgentPlanReplanningReason("AFFECTED_DEPENDENCIES_REBUILT"),
                affected_dependency_ids=dependency_ids,
                order=len(actions),
            )
        )
        _event(events, "agent_plan_replanning_action_created", "created", action_type="REBUILD_DEPENDENCIES")
    if not tasks:
        return None
    try:
        return AgentCooperationPlan(
            plan_id=original.plan_id,
            tasks=tasks,
            dependencies=dependencies,
            metadata=original.metadata,
            schema_version=original.schema_version,
            execution_strategy=original.execution_strategy,
            policy=original.policy,
        )
    except (RuntimeError, TypeError, ValueError):
        return None


def _retry_task(task: AgentCooperationTask) -> AgentCooperationTask:
    metadata = dict(task.metadata)
    metadata["replanning_action"] = "retry"
    return replace(task, metadata=metadata)


def _check_limits(
    plan: AgentCooperationPlan,
    actions: Sequence[AgentPlanReplanningAction],
    policy: AgentPlanReplanningPolicy,
    limits: list[str],
    events: list[AgentPlanReplanningEvent],
) -> None:
    checks = (
        (len(actions), policy.max_actions, "MAX_ACTIONS"),
        (len(plan.tasks), policy.max_tasks, "MAX_TASKS"),
        (len(plan.dependencies), policy.max_dependencies, "MAX_DEPENDENCIES"),
        (_plan_depth(plan), policy.max_plan_depth, "MAX_PLAN_DEPTH"),
        (sum(task.logical_timeout_limit for task in plan.tasks), policy.max_logical_time, "MAX_LOGICAL_TIME"),
    )
    for actual, maximum, name in checks:
        if actual > maximum:
            limits.append(name)
            _event(events, "agent_plan_replanning_limit_reached", "reached", limit_name=name)


def _agent_authorizes_required_skills(agent: AgentDefinition, required_skill_ids: Sequence[str]) -> bool:
    if not required_skill_ids:
        return True
    denied = _metadata_ids(agent.metadata.get("denied_skill_ids"))
    allowed = _metadata_ids(agent.metadata.get("allowed_skill_ids"))
    required = _metadata_ids(agent.metadata.get("required_skill_ids"))
    for skill_id in required_skill_ids:
        if skill_id in denied:
            return False
        if allowed and skill_id not in allowed:
            return False
        if required and skill_id not in required:
            return False
    return True


def _metadata_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    return tuple(_safe_identifier(part.strip(), "metadata id") for part in value.split(",") if part.strip())


def _fill_action_metrics(metrics: dict[str, int], actions: Sequence[AgentPlanReplanningAction]) -> None:
    metrics["agent_plan_replanning_actions_created"] = len(actions)
    metrics["agent_plan_replanning_tasks_evaluated"] = len({action.task_id for action in actions if action.task_id != "plan"})
    metrics["agent_plan_replanning_tasks_preserved"] = sum(action.action_type is AgentPlanReplanningActionType.PRESERVE_TASK for action in actions)
    metrics["agent_plan_replanning_tasks_retried"] = sum(action.action_type is AgentPlanReplanningActionType.RETRY_TASK for action in actions)
    metrics["agent_plan_replanning_tasks_skipped"] = sum(action.action_type is AgentPlanReplanningActionType.SKIP_TASK for action in actions)
    metrics["agent_plan_replanning_tasks_removed"] = sum(action.action_type is AgentPlanReplanningActionType.REMOVE_TASK for action in actions)
    metrics["agent_plan_replanning_agents_replaced"] = sum(action.action_type is AgentPlanReplanningActionType.REPLACE_AGENT for action in actions)
    metrics["agent_plan_replanning_dependencies_rebuilt"] = sum(action.action_type is AgentPlanReplanningActionType.REBUILD_DEPENDENCIES for action in actions)


def _action(
    task_id: str,
    action_type: AgentPlanReplanningActionType | str,
    reason: str,
    order: int,
) -> AgentPlanReplanningAction:
    value = AgentPlanReplanningAction(
        task_id=task_id,
        action_type=action_type,
        reason=AgentPlanReplanningReason(reason),
        order=order,
    )
    return value


def _simple_result(
    status: AgentPlanReplanningStatus,
    decision_type: AgentPlanReplanningDecisionType,
    request_signature: str,
    reasons: tuple[AgentPlanReplanningReason, ...],
    events: Sequence[AgentPlanReplanningEvent],
    metrics: Mapping[str, int],
    *,
    original_signature: str | None = None,
    supervision_signature: str | None = None,
) -> AgentPlanReplanningResult:
    summary = {"status": status.value, "actions": 0, "has_proposed_plan": False}
    decision = AgentPlanReplanningDecision(decision=decision_type, reasons=reasons, safe_summary=summary)
    return AgentPlanReplanningResult(
        status=status,
        decision=decision,
        request_signature=request_signature,
        original_plan_signature=original_signature,
        supervision_signature=supervision_signature,
        events=tuple(events),
        metrics=metrics,
        safe_summary=summary,
    )


def _event(
    events: list[AgentPlanReplanningEvent],
    name: str,
    status: str,
    **details: object,
) -> None:
    if len(events) < MAX_REPLANNING_EVENTS:
        events.append(AgentPlanReplanningEvent(name, status, details))


def _base_metrics() -> dict[str, int]:
    return {
        "agent_plan_replanning_requests": 0,
        "agent_plan_replanning_succeeded": 0,
        "agent_plan_replanning_failed": 0,
        "agent_plan_replanning_blocked": 0,
        "agent_plan_replanning_no_change": 0,
        "agent_plan_replanning_limits_reached": 0,
        "agent_plan_replanning_tasks_evaluated": 0,
        "agent_plan_replanning_tasks_preserved": 0,
        "agent_plan_replanning_tasks_retried": 0,
        "agent_plan_replanning_tasks_skipped": 0,
        "agent_plan_replanning_tasks_removed": 0,
        "agent_plan_replanning_agents_replaced": 0,
        "agent_plan_replanning_dependencies_rebuilt": 0,
        "agent_plan_replanning_actions_created": 0,
        "agent_plan_replanning_cycles_detected": 0,
    }


def _supervision_signature(supervision: AgentPlanSupervisorResult) -> str:
    return _signature(
        {
            "request_signature": supervision.request_signature,
            "status": supervision.status.value,
            "plan_signature": supervision.plan_signature,
            "succeeded": supervision.succeeded_tasks,
            "failed": supervision.failed_tasks,
            "skipped": supervision.skipped_tasks,
            "blocked": supervision.blocked_tasks,
            "unknown": supervision.unknown_tasks,
            "missing": supervision.missing_tasks,
            "limits": supervision.limits_reached,
            "inconsistencies": supervision.inconsistencies,
        }
    )


def _policy_payload(policy: AgentPlanReplanningPolicy) -> Mapping[str, object]:
    return {name: getattr(policy, name) for name in sorted(policy.__dataclass_fields__)}


def _dependency_ids(dependencies: Sequence[AgentCooperationDependency]) -> tuple[str, ...]:
    return tuple(
        f"{dependency.prerequisite_task_id}.{dependency.dependent_task_id}"
        for dependency in dependencies
    )


def _plan_depth(plan: AgentCooperationPlan) -> int:
    prerequisites: dict[str, list[str]] = {task.task_id: [] for task in plan.tasks}
    for dependency in plan.dependencies:
        prerequisites.setdefault(dependency.dependent_task_id, []).append(dependency.prerequisite_task_id)
    depths: dict[str, int] = {}
    visiting: set[str] = set()

    def visit(task_id: str) -> int:
        if task_id in visiting:
            return MAX_REPLANNING_PLAN_DEPTH + 1
        if task_id in depths:
            return depths[task_id]
        visiting.add(task_id)
        depth = 1 + max((visit(item) for item in prerequisites.get(task_id, ())), default=0)
        visiting.remove(task_id)
        depths[task_id] = depth
        return depth

    return max((visit(task.task_id) for task in plan.tasks), default=0)


def _safe_mapping(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentPlanReplanningRequestError(f"{field_name} must be a mapping.")
    if len(value) > MAX_REPLANNING_METADATA_ITEMS:
        raise InvalidAgentPlanReplanningRequestError(f"{field_name} has too many items.")
    counter = {"items": 0}
    return _safe_mapping_inner(value, field_name, 0, counter)


def _safe_mapping_inner(
    value: Mapping[str, object],
    field_name: str,
    depth: int,
    counter: dict[str, int],
) -> dict[str, object]:
    if depth > MAX_REPLANNING_VALUE_DEPTH:
        raise InvalidAgentPlanReplanningRequestError(f"{field_name} exceeds maximum depth.")
    result: dict[str, object] = {}
    for raw_key in sorted(value, key=lambda item: str(item)):
        key = _safe_identifier(raw_key, f"{field_name} key")
        if _forbidden_key(key):
            raise InvalidAgentPlanReplanningRequestError(f"{field_name} contains a forbidden key.")
        result[key] = _safe_value(value[raw_key], field_name, depth + 1, counter)
    return result


def _safe_value(value: object, field_name: str, depth: int, counter: dict[str, int]) -> object:
    if depth > MAX_REPLANNING_VALUE_DEPTH:
        raise InvalidAgentPlanReplanningRequestError(f"{field_name} exceeds maximum depth.")
    counter["items"] += 1
    if counter["items"] > MAX_REPLANNING_OUTPUT_ITEMS:
        raise InvalidAgentPlanReplanningRequestError(f"{field_name} exceeds item limit.")
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentPlanReplanningRequestError(f"{field_name} floats must be finite.")
        return value
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_mapping_inner(value, field_name, depth + 1, counter))
    if isinstance(value, (tuple, list)):
        return tuple(_safe_value(item, field_name, depth + 1, counter) for item in value)
    raise InvalidAgentPlanReplanningRequestError(f"{field_name} contains an unsupported object.")


def _safe_primitive_mapping(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentPlanReplanningRequestError(f"{field_name} must be a mapping.")
    result: dict[str, object] = {}
    for raw_key in sorted(value, key=lambda item: str(item)):
        key = _safe_identifier(raw_key, f"{field_name} key")
        item = value[raw_key]
        if item is None or type(item) in (bool, int, str):
            result[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[key] = item
        else:
            raise InvalidAgentPlanReplanningRequestError(f"{field_name} values must be primitive.")
    return result


def _metric_mapping(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise InvalidAgentPlanReplanningRequestError("metrics must be a mapping.")
    result: dict[str, int] = {}
    for key, item in value.items():
        normalized = _safe_identifier(key, "metric name")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise InvalidAgentPlanReplanningRequestError("metric values must be non-negative integers.")
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
    raise InvalidAgentPlanReplanningRequestError("value is not deterministically serializable.")


def _forbidden_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS + _DYNAMIC_KEY_PARTS)


def _safe_text(value: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_REPLANNING_STRING_LENGTH:
        raise InvalidAgentPlanReplanningRequestError("text is invalid.")
    lowered = value.replace("-", "_").lower()
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        raise InvalidAgentPlanReplanningRequestError("text contains sensitive content.")
    return " ".join(value.split())


def _safe_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 128:
        raise InvalidAgentPlanReplanningRequestError(f"{field_name} contains unsupported characters.")
    if not all(character.isalnum() or character in "_.-" for character in value):
        raise InvalidAgentPlanReplanningRequestError(f"{field_name} contains unsupported characters.")
    return value


def _identifier_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise InvalidAgentPlanReplanningRequestError(f"{field_name} must be a sequence.")
    return tuple(dict.fromkeys(_safe_identifier(value, field_name) for value in tuple(values)))


def _bounded_int(value: int, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise InvalidAgentPlanReplanningRequestError(f"{field_name} is outside the allowed range.")
    return value


def _validate_signature(value: str, field_name: str, *, allow_empty: bool = False) -> None:
    if allow_empty and value == "":
        return
    if not isinstance(value, str) or len(value) != 64 or any(character not in _SIGNATURE_HEX for character in value):
        raise InvalidAgentPlanReplanningRequestError(f"{field_name} must be a SHA-256 hex digest.")


def _status(value: AgentPlanReplanningStatus | str) -> AgentPlanReplanningStatus:
    try:
        return value if isinstance(value, AgentPlanReplanningStatus) else AgentPlanReplanningStatus(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentPlanReplanningRequestError("invalid replanning status.") from error


def _decision_type(value: AgentPlanReplanningDecisionType | str) -> AgentPlanReplanningDecisionType:
    try:
        return value if isinstance(value, AgentPlanReplanningDecisionType) else AgentPlanReplanningDecisionType(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentPlanReplanningRequestError("invalid replanning decision.") from error


def _action_type(value: AgentPlanReplanningActionType | str) -> AgentPlanReplanningActionType:
    try:
        return value if isinstance(value, AgentPlanReplanningActionType) else AgentPlanReplanningActionType(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentPlanReplanningRequestError("invalid replanning action.") from error

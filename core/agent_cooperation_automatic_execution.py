"""Controlled execution of deterministically generated cooperation plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType

from core.agent_context import AgentContextBuilder
from core.agent_cooperation_automatic_planner import (
    AgentCooperationAutomaticPlanner,
    AgentCooperationPlanningPolicy,
    AgentCooperationPlanningRequest,
    AgentCooperationPlanningResult,
    AgentCooperationPlanningStatus,
    agent_cooperation_planning_request_signature,
)
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
from core.agent_delegation import AgentDelegationService
from core.agent_delegation_chain import AgentDelegationChainService
from core.agent_delegation_coordinator import AgentDelegationCoordinator
from core.agent_executor import AgentExecutor
from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolver
from core.multi_agent import MultiAgentCoordinator, MultiAgentResolver
from core.skill_system import SkillSystem


MAX_AUTO_EXECUTION_TASKS = 32
MAX_AUTO_EXECUTION_AGENTS = 16
MAX_AUTO_EXECUTION_DEPENDENCIES = 128
MAX_AUTO_EXECUTION_DEPTH = 16
MAX_AUTO_EXECUTION_OUTPUT_ITEMS = 512
MAX_AUTO_EXECUTION_LOGICAL_TIME = 10_000
MAX_AUTO_EXECUTION_EVENTS = 256
MAX_AUTO_EXECUTION_METADATA_ITEMS = 32
MAX_AUTO_EXECUTION_SEQUENCE_ITEMS = 128
MAX_AUTO_EXECUTION_VALUE_DEPTH = 8
MAX_AUTO_EXECUTION_TOTAL_ITEMS = 1_024
MAX_AUTO_EXECUTION_STRING_LENGTH = 2_000

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SIGNATURE_PATTERN = re.compile(r"^[0-9a-f]{64}$")
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


class AgentCooperationAutomaticExecutionError(RuntimeError):
    """Base error for controlled automatic cooperation execution."""


class InvalidAgentCooperationAutomaticExecutionRequestError(
    AgentCooperationAutomaticExecutionError
):
    """Raised when an automatic execution request or policy is malformed."""


class AgentCooperationAutomaticExecutionStatus(str, Enum):
    """Structured lifecycle and terminal statuses."""

    DISABLED = "DISABLED"
    INVALID_REQUEST = "INVALID_REQUEST"
    PLANNING_STARTED = "PLANNING_STARTED"
    PLANNING_FAILED = "PLANNING_FAILED"
    PLANNING_AMBIGUOUS = "PLANNING_AMBIGUOUS"
    NO_VALID_PLAN = "NO_VALID_PLAN"
    PLAN_VALIDATION_FAILED = "PLAN_VALIDATION_FAILED"
    DRY_RUN_COMPLETED = "DRY_RUN_COMPLETED"
    EXECUTION_STARTED = "EXECUTION_STARTED"
    EXECUTION_SUCCEEDED = "EXECUTION_SUCCEEDED"
    EXECUTION_PARTIAL = "EXECUTION_PARTIAL"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    LIMIT_REACHED = "LIMIT_REACHED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentCooperationAutomaticExecutionDecisionType(str, Enum):
    """Explainable decisions made by the controlled composition layer."""

    PLANNING_ALLOWED = "PLANNING_ALLOWED"
    PLANNING_BLOCKED = "PLANNING_BLOCKED"
    PLAN_ACCEPTED = "PLAN_ACCEPTED"
    PLAN_REJECTED = "PLAN_REJECTED"
    EXECUTION_ALLOWED = "EXECUTION_ALLOWED"
    EXECUTION_SKIPPED = "EXECUTION_SKIPPED"
    EXECUTION_COMPLETED = "EXECUTION_COMPLETED"


@dataclass(frozen=True, slots=True)
class AgentCooperationAutomaticExecutionPolicy:
    """Explicit opt-in policies and conservative outer limits."""

    enabled: bool = False
    planning_policy: AgentCooperationPlanningPolicy | None = None
    execution_policy: AgentCooperationPlanPolicy | None = None
    execute_generated_plan: bool = False
    dry_run: bool = False
    max_tasks: int = 16
    max_agents: int = 8
    max_dependencies: int = 48
    max_plan_depth: int = 8
    max_output_items: int = 256
    max_total_executions: int = 16
    max_logical_time: int = 64

    def __post_init__(self) -> None:
        for name in ("enabled", "execute_generated_plan", "dry_run"):
            if type(getattr(self, name)) is not bool:
                raise InvalidAgentCooperationAutomaticExecutionRequestError(
                    f"{name} must be a bool."
                )
        if self.planning_policy is not None and not isinstance(
            self.planning_policy, AgentCooperationPlanningPolicy
        ):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                "planning_policy must be AgentCooperationPlanningPolicy or None."
            )
        if self.execution_policy is not None and not isinstance(
            self.execution_policy, AgentCooperationPlanPolicy
        ):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                "execution_policy must be AgentCooperationPlanPolicy or None."
            )
        limits = (
            ("max_tasks", MAX_AUTO_EXECUTION_TASKS),
            ("max_agents", MAX_AUTO_EXECUTION_AGENTS),
            ("max_dependencies", MAX_AUTO_EXECUTION_DEPENDENCIES),
            ("max_plan_depth", MAX_AUTO_EXECUTION_DEPTH),
            ("max_output_items", MAX_AUTO_EXECUTION_OUTPUT_ITEMS),
            ("max_total_executions", MAX_AUTO_EXECUTION_TASKS),
            ("max_logical_time", MAX_AUTO_EXECUTION_LOGICAL_TIME),
        )
        for name, maximum in limits:
            object.__setattr__(self, name, _bounded_int(getattr(self, name), name, maximum))


@dataclass(frozen=True, slots=True)
class AgentCooperationAutomaticExecutionRequest:
    """Request to plan once and optionally execute the generated plan once."""

    planning_request: AgentCooperationPlanningRequest
    policy: AgentCooperationAutomaticExecutionPolicy = field(
        default_factory=AgentCooperationAutomaticExecutionPolicy
    )
    execution_id: str | None = None
    correlation_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.planning_request, AgentCooperationPlanningRequest):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                "planning_request must be AgentCooperationPlanningRequest."
            )
        if not isinstance(self.policy, AgentCooperationAutomaticExecutionPolicy):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                "policy must be AgentCooperationAutomaticExecutionPolicy."
            )
        for name in ("execution_id", "correlation_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_safe_mapping(self.metadata, "metadata")),
        )

    @property
    def effective_execution_id(self) -> str:
        return (
            self.execution_id
            or self.planning_request.execution_id
            or f"auto.{self.planning_request.objective_id}"
        )


@dataclass(frozen=True, slots=True)
class AgentCooperationAutomaticExecutionDecision:
    """One safe explanation for a planning, validation, or execution choice."""

    decision: AgentCooperationAutomaticExecutionDecisionType
    reason_code: str
    policy_name: str
    limit_name: str | None = None
    outcome: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", _decision_type(self.decision))
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        object.__setattr__(self, "policy_name", _identifier(self.policy_name, "policy_name"))
        if self.limit_name is not None:
            object.__setattr__(self, "limit_name", _identifier(self.limit_name, "limit_name"))
        if self.outcome is not None:
            object.__setattr__(self, "outcome", _identifier(self.outcome, "outcome"))


@dataclass(frozen=True, slots=True)
class AgentCooperationAutomaticExecutionEvent:
    """Bounded event without inputs, outputs, prompts, or secrets."""

    name: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "event name"))
        object.__setattr__(self, "status", _identifier(self.status, "event status"))
        object.__setattr__(
            self,
            "details",
            MappingProxyType(_safe_mapping(self.details, "event details")),
        )


@dataclass(frozen=True, slots=True)
class AgentCooperationAutomaticExecutionResult:
    """Aggregate immutable result for one controlled automatic execution."""

    execution_id: str
    status: AgentCooperationAutomaticExecutionStatus
    decision: AgentCooperationAutomaticExecutionDecision
    signature: str
    planning_result: AgentCooperationPlanningResult | None = None
    generated_plan: AgentCooperationPlan | None = None
    cooperation_result: AgentCooperationPlanResult | None = None
    plan_signature: str | None = None
    decisions: tuple[AgentCooperationAutomaticExecutionDecision, ...] = ()
    safe_summary: Mapping[str, object] = field(default_factory=dict)
    events: tuple[AgentCooperationAutomaticExecutionEvent, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    error_code: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_id", _identifier(self.execution_id, "execution_id"))
        object.__setattr__(self, "status", _status(self.status))
        if not isinstance(self.decision, AgentCooperationAutomaticExecutionDecision):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                "decision must be AgentCooperationAutomaticExecutionDecision."
            )
        _validate_signature(self.signature, "signature", allow_empty=True)
        if self.planning_result is not None and not isinstance(
            self.planning_result, AgentCooperationPlanningResult
        ):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                "planning_result must be AgentCooperationPlanningResult or None."
            )
        if self.generated_plan is not None and not isinstance(
            self.generated_plan, AgentCooperationPlan
        ):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                "generated_plan must be AgentCooperationPlan or None."
            )
        if self.cooperation_result is not None and not isinstance(
            self.cooperation_result, AgentCooperationPlanResult
        ):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                "cooperation_result must be AgentCooperationPlanResult or None."
            )
        if self.plan_signature is not None:
            _validate_signature(self.plan_signature, "plan_signature")
        decisions = tuple(self.decisions)
        if not decisions:
            decisions = (self.decision,)
        if not all(
            isinstance(item, AgentCooperationAutomaticExecutionDecision)
            for item in decisions
        ):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                "decisions must contain AgentCooperationAutomaticExecutionDecision values."
            )
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self,
            "safe_summary",
            MappingProxyType(_safe_mapping(self.safe_summary, "safe_summary")),
        )
        events = tuple(self.events)[-MAX_AUTO_EXECUTION_EVENTS:]
        if not all(isinstance(item, AgentCooperationAutomaticExecutionEvent) for item in events):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                "events must contain AgentCooperationAutomaticExecutionEvent values."
            )
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "metrics", MappingProxyType(_metric_mapping(self.metrics)))
        if self.error_code is not None:
            object.__setattr__(self, "error_code", _identifier(self.error_code, "error_code"))
        if self.reason is not None:
            object.__setattr__(self, "reason", _safe_message(self.reason))

    @property
    def request_signature(self) -> str:
        return self.signature


class AgentCooperationAutomaticExecutionService:
    """Plan once, validate explicitly, and optionally execute once."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        agent_resolver: AgentResolver,
        agent_context_builder: AgentContextBuilder,
        agent_executor: AgentExecutor,
        skill_system: SkillSystem,
        agent_delegation_service: AgentDelegationService,
        agent_delegation_chain_service: AgentDelegationChainService,
        agent_delegation_coordinator: AgentDelegationCoordinator,
        multi_agent_resolver: MultiAgentResolver,
        multi_agent_coordinator: MultiAgentCoordinator,
        agent_cooperation_planner: AgentCooperationPlanner,
        agent_cooperation_automatic_planner: AgentCooperationAutomaticPlanner,
    ) -> None:
        dependencies = (
            (agent_registry, AgentRegistry, "agent_registry"),
            (agent_resolver, AgentResolver, "agent_resolver"),
            (agent_context_builder, AgentContextBuilder, "agent_context_builder"),
            (agent_executor, AgentExecutor, "agent_executor"),
            (skill_system, SkillSystem, "skill_system"),
            (agent_delegation_service, AgentDelegationService, "agent_delegation_service"),
            (
                agent_delegation_chain_service,
                AgentDelegationChainService,
                "agent_delegation_chain_service",
            ),
            (
                agent_delegation_coordinator,
                AgentDelegationCoordinator,
                "agent_delegation_coordinator",
            ),
            (multi_agent_resolver, MultiAgentResolver, "multi_agent_resolver"),
            (multi_agent_coordinator, MultiAgentCoordinator, "multi_agent_coordinator"),
            (agent_cooperation_planner, AgentCooperationPlanner, "agent_cooperation_planner"),
            (
                agent_cooperation_automatic_planner,
                AgentCooperationAutomaticPlanner,
                "agent_cooperation_automatic_planner",
            ),
        )
        for value, expected, name in dependencies:
            if not isinstance(value, expected):
                raise AgentCooperationAutomaticExecutionError(
                    f"{name} must be {expected.__name__}."
                )
        self._agent_registry = agent_registry
        self._agent_resolver = agent_resolver
        self._agent_context_builder = agent_context_builder
        self._agent_executor = agent_executor
        self._skill_system = skill_system
        self._agent_delegation_service = agent_delegation_service
        self._agent_delegation_chain_service = agent_delegation_chain_service
        self._agent_delegation_coordinator = agent_delegation_coordinator
        self._multi_agent_resolver = multi_agent_resolver
        self._multi_agent_coordinator = multi_agent_coordinator
        self._agent_cooperation_planner = agent_cooperation_planner
        self._agent_cooperation_automatic_planner = agent_cooperation_automatic_planner

    def execute(
        self,
        request: AgentCooperationAutomaticExecutionRequest,
    ) -> AgentCooperationAutomaticExecutionResult:
        """Execute the guarded planning-to-execution flow."""

        events: list[AgentCooperationAutomaticExecutionEvent] = []
        metrics = _base_metrics()
        metrics["requests"] = 1
        _event(events, "cooperation_auto_execution_requested", "requested")
        _event(events, "cooperation_auto_execution_validation_started", "started")
        if not isinstance(request, AgentCooperationAutomaticExecutionRequest):
            metrics["validation_failures"] = 1
            _event(events, "cooperation_auto_execution_validation_failed", "failed")
            return _result(
                execution_id="auto.invalid",
                status=AgentCooperationAutomaticExecutionStatus.INVALID_REQUEST,
                decision=_decision("PLANNING_BLOCKED", "INVALID_REQUEST", "request"),
                signature="",
                events=events,
                metrics=metrics,
                error_code="INVALID_REQUEST",
                reason="request must be AgentCooperationAutomaticExecutionRequest.",
            )

        signature = agent_cooperation_automatic_execution_request_signature(request)
        execution_id = request.effective_execution_id
        policy = request.policy
        if not policy.enabled:
            _event(events, "cooperation_auto_execution_skipped", "disabled")
            _event(events, "cooperation_auto_execution_completed", "disabled")
            return _result(
                execution_id=execution_id,
                status=AgentCooperationAutomaticExecutionStatus.DISABLED,
                decision=_decision("PLANNING_BLOCKED", "POLICY_DISABLED", "automatic_execution"),
                signature=signature,
                events=events,
                metrics=metrics,
                error_code="DISABLED",
                reason="automatic cooperation execution is disabled.",
            )

        validation = _validate_request_policy(request)
        if validation is not None:
            code, message = validation
            metrics["validation_failures"] = 1
            _event(
                events,
                "cooperation_auto_execution_validation_failed",
                "failed",
                error_code=code,
            )
            return _result(
                execution_id=execution_id,
                status=AgentCooperationAutomaticExecutionStatus.INVALID_REQUEST,
                decision=_decision("PLANNING_BLOCKED", code, "automatic_execution"),
                signature=signature,
                events=events,
                metrics=metrics,
                error_code=code,
                reason=message,
            )

        planning_policy = policy.planning_policy
        execution_policy = policy.execution_policy
        assert planning_policy is not None
        assert execution_policy is not None
        _event(events, "cooperation_auto_execution_validation_succeeded", "succeeded")
        planning_request = replace(request.planning_request, policy=planning_policy)
        expected_planning_signature = agent_cooperation_planning_request_signature(
            planning_request
        )
        metrics["planning_started"] = 1
        _event(events, "cooperation_auto_planning_started", "started")
        try:
            planning_result = self._agent_cooperation_automatic_planner.plan(planning_request)
        except (RuntimeError, TypeError, ValueError) as error:
            metrics["planning_failed"] = 1
            _event(events, "cooperation_auto_planning_failed", "failed")
            return _result(
                execution_id=execution_id,
                status=AgentCooperationAutomaticExecutionStatus.INTERNAL_ERROR,
                decision=_decision("PLAN_REJECTED", "PLANNING_EXCEPTION", "planning"),
                signature=signature,
                events=events,
                metrics=metrics,
                error_code="PLANNING_EXCEPTION",
                reason=str(error),
            )

        if not isinstance(planning_result, AgentCooperationPlanningResult):
            metrics["planning_failed"] = 1
            _event(events, "cooperation_auto_planning_failed", "failed")
            return _result(
                execution_id=execution_id,
                status=AgentCooperationAutomaticExecutionStatus.PLANNING_FAILED,
                decision=_decision("PLAN_REJECTED", "INVALID_PLANNING_RESULT", "planning"),
                signature=signature,
                events=events,
                metrics=metrics,
                error_code="INVALID_PLANNING_RESULT",
                reason="automatic planner returned an invalid result.",
            )

        if planning_result.status is not AgentCooperationPlanningStatus.SUCCESS:
            return self._planning_failure(
                execution_id,
                signature,
                planning_result,
                events,
                metrics,
            )

        metrics["planning_succeeded"] = 1
        _event(events, "cooperation_auto_planning_succeeded", "succeeded")
        _event(events, "cooperation_generated_plan_validation_started", "started")
        plan_validation = self._validate_generated_plan(
            planning_result,
            expected_planning_signature,
            policy,
            execution_policy,
        )
        if plan_validation is not None:
            status, code, message, limit_name = plan_validation
            metrics["plans_rejected"] = 1
            if status is AgentCooperationAutomaticExecutionStatus.LIMIT_REACHED:
                metrics["limits_reached"] = 1
                _event(
                    events,
                    "cooperation_auto_execution_limit_reached",
                    "failed",
                    limit_name=limit_name or code,
                )
            _event(
                events,
                "cooperation_generated_plan_validation_failed",
                "failed",
                error_code=code,
            )
            return _result(
                execution_id=execution_id,
                status=status,
                decision=_decision(
                    "PLAN_REJECTED",
                    code,
                    "automatic_execution",
                    limit_name=limit_name,
                ),
                signature=signature,
                planning_result=planning_result,
                generated_plan=planning_result.plan,
                plan_signature=planning_result.plan_signature,
                events=events,
                metrics=metrics,
                error_code=code,
                reason=message,
            )

        plan = planning_result.plan
        assert plan is not None
        plan_signature = planning_result.plan_signature
        assert plan_signature is not None
        metrics["plans_generated"] = 1
        metrics["tasks_planned"] = len(plan.tasks)
        metrics["agents_selected"] = len(planning_result.selected_agent_ids)
        metrics["skills_required"] = len(
            {skill_id for task in plan.tasks for skill_id in task.required_skill_ids}
        )
        _event(events, "cooperation_generated_plan_validation_succeeded", "succeeded")

        if policy.dry_run:
            metrics["dry_runs"] = 1
            _event(events, "cooperation_auto_execution_skipped", "dry_run")
            _event(events, "cooperation_auto_execution_completed", "dry_run")
            return _result(
                execution_id=execution_id,
                status=AgentCooperationAutomaticExecutionStatus.DRY_RUN_COMPLETED,
                decision=_decision("EXECUTION_SKIPPED", "DRY_RUN", "automatic_execution"),
                signature=signature,
                planning_result=planning_result,
                generated_plan=plan,
                plan_signature=plan_signature,
                events=events,
                metrics=metrics,
            )

        metrics["executions_started"] = 1
        _event(events, "cooperation_auto_execution_started", "started")
        plan_request = AgentCooperationPlanRequest(
            plan=plan,
            policy=execution_policy,
            execution_id=execution_id,
            correlation_id=request.correlation_id or planning_request.correlation_id,
            metadata=request.metadata,
        )
        try:
            cooperation_result = self._agent_cooperation_planner.execute(plan_request)
        except (RuntimeError, TypeError, ValueError) as error:
            metrics["executions_failed"] = 1
            _event(events, "cooperation_auto_execution_failed", "failed")
            _event(events, "cooperation_auto_execution_completed", "failed")
            return _result(
                execution_id=execution_id,
                status=AgentCooperationAutomaticExecutionStatus.EXECUTION_FAILED,
                decision=_decision("EXECUTION_COMPLETED", "EXECUTION_EXCEPTION", "execution"),
                signature=signature,
                planning_result=planning_result,
                generated_plan=plan,
                plan_signature=plan_signature,
                events=events,
                metrics=metrics,
                error_code="EXECUTION_EXCEPTION",
                reason=str(error),
            )

        if not isinstance(cooperation_result, AgentCooperationPlanResult):
            metrics["executions_failed"] = 1
            _event(events, "cooperation_auto_execution_failed", "failed")
            _event(events, "cooperation_auto_execution_completed", "failed")
            return _result(
                execution_id=execution_id,
                status=AgentCooperationAutomaticExecutionStatus.EXECUTION_FAILED,
                decision=_decision("EXECUTION_COMPLETED", "INVALID_EXECUTION_RESULT", "execution"),
                signature=signature,
                planning_result=planning_result,
                generated_plan=plan,
                plan_signature=plan_signature,
                events=events,
                metrics=metrics,
                error_code="INVALID_EXECUTION_RESULT",
                reason="cooperation planner returned an invalid result.",
            )

        if cooperation_result.plan_signature != plan_signature:
            metrics["executions_failed"] = 1
            _event(events, "cooperation_auto_execution_failed", "failed")
            _event(events, "cooperation_auto_execution_completed", "failed")
            return _result(
                execution_id=execution_id,
                status=AgentCooperationAutomaticExecutionStatus.EXECUTION_FAILED,
                decision=_decision("EXECUTION_COMPLETED", "EXECUTED_PLAN_MISMATCH", "execution"),
                signature=signature,
                planning_result=planning_result,
                generated_plan=plan,
                cooperation_result=cooperation_result,
                plan_signature=plan_signature,
                events=events,
                metrics=metrics,
                error_code="EXECUTED_PLAN_MISMATCH",
                reason="execution result does not match the generated plan.",
            )

        safe_cooperation_result = _sanitize_cooperation_result(cooperation_result)
        _merge_execution_metrics(metrics, safe_cooperation_result)
        if cooperation_result.status is AgentCooperationPlanStatus.SUCCESS:
            status = AgentCooperationAutomaticExecutionStatus.EXECUTION_SUCCEEDED
            metrics["executions_succeeded"] = 1
            event_name = "cooperation_auto_execution_succeeded"
            outcome = "SUCCESS"
        elif cooperation_result.status is AgentCooperationPlanStatus.PARTIAL_SUCCESS:
            status = AgentCooperationAutomaticExecutionStatus.EXECUTION_PARTIAL
            metrics["executions_partial"] = 1
            event_name = "cooperation_auto_execution_partial"
            outcome = "PARTIAL"
        elif cooperation_result.status is AgentCooperationPlanStatus.LIMIT_REACHED:
            status = AgentCooperationAutomaticExecutionStatus.LIMIT_REACHED
            metrics["executions_failed"] = 1
            metrics["limits_reached"] = 1
            event_name = "cooperation_auto_execution_limit_reached"
            outcome = "FAILED"
        else:
            status = AgentCooperationAutomaticExecutionStatus.EXECUTION_FAILED
            metrics["executions_failed"] = 1
            event_name = "cooperation_auto_execution_failed"
            outcome = "FAILED"
        _event(events, event_name, outcome.lower())
        _event(events, "cooperation_auto_execution_completed", outcome.lower())
        return _result(
            execution_id=execution_id,
            status=status,
            decision=_decision(
                "EXECUTION_COMPLETED",
                cooperation_result.error_code or outcome,
                "execution",
                outcome=outcome,
            ),
            signature=signature,
            planning_result=planning_result,
            generated_plan=plan,
            cooperation_result=safe_cooperation_result,
            plan_signature=plan_signature,
            events=events,
            metrics=metrics,
            error_code=(
                None
                if status
                in (
                    AgentCooperationAutomaticExecutionStatus.EXECUTION_SUCCEEDED,
                    AgentCooperationAutomaticExecutionStatus.EXECUTION_PARTIAL,
                )
                else cooperation_result.error_code or "EXECUTION_FAILED"
            ),
            reason=cooperation_result.safe_message,
        )

    def _planning_failure(
        self,
        execution_id: str,
        signature: str,
        planning_result: AgentCooperationPlanningResult,
        events: list[AgentCooperationAutomaticExecutionEvent],
        metrics: dict[str, int],
    ) -> AgentCooperationAutomaticExecutionResult:
        metrics["planning_failed"] = 1
        status = AgentCooperationAutomaticExecutionStatus.PLANNING_FAILED
        if planning_result.status is AgentCooperationPlanningStatus.AMBIGUOUS:
            status = AgentCooperationAutomaticExecutionStatus.PLANNING_AMBIGUOUS
        elif planning_result.status is AgentCooperationPlanningStatus.LIMIT_REACHED:
            status = AgentCooperationAutomaticExecutionStatus.LIMIT_REACHED
            metrics["limits_reached"] = 1
            _event(events, "cooperation_auto_execution_limit_reached", "failed")
        _event(events, "cooperation_auto_planning_failed", "failed")
        return _result(
            execution_id=execution_id,
            status=status,
            decision=_decision(
                "PLAN_REJECTED",
                planning_result.error_code or planning_result.status.value,
                "planning",
            ),
            signature=signature,
            planning_result=planning_result,
            events=events,
            metrics=metrics,
            error_code=planning_result.error_code or planning_result.status.value,
            reason=planning_result.safe_message,
        )

    def _validate_generated_plan(
        self,
        planning_result: AgentCooperationPlanningResult,
        expected_planning_signature: str,
        policy: AgentCooperationAutomaticExecutionPolicy,
        execution_policy: AgentCooperationPlanPolicy,
    ) -> tuple[
        AgentCooperationAutomaticExecutionStatus,
        str,
        str,
        str | None,
    ] | None:
        plan = planning_result.plan
        if planning_result.planning_request_signature != expected_planning_signature:
            return (
                AgentCooperationAutomaticExecutionStatus.NO_VALID_PLAN,
                "PLANNING_SIGNATURE_MISMATCH",
                "planning result does not match the submitted request.",
                None,
            )
        if plan is None or not isinstance(plan, AgentCooperationPlan) or not plan.tasks:
            return (
                AgentCooperationAutomaticExecutionStatus.NO_VALID_PLAN,
                "NO_VALID_PLAN",
                "automatic planner did not return a non-empty plan.",
                None,
            )
        if planning_result.plan_signature is None:
            return (
                AgentCooperationAutomaticExecutionStatus.NO_VALID_PLAN,
                "MISSING_PLAN_SIGNATURE",
                "generated plan has no signature.",
                None,
            )
        actual_signature = agent_cooperation_plan_signature(plan)
        if actual_signature != planning_result.plan_signature:
            return (
                AgentCooperationAutomaticExecutionStatus.NO_VALID_PLAN,
                "PLAN_SIGNATURE_MISMATCH",
                "generated plan signature is inconsistent.",
                None,
            )
        if planning_result.created_task_ids != tuple(task.task_id for task in plan.tasks):
            return (
                AgentCooperationAutomaticExecutionStatus.NO_VALID_PLAN,
                "PLAN_TASKS_MISMATCH",
                "generated plan tasks are inconsistent with the planning result.",
                None,
            )
        if planning_result.created_dependencies != plan.dependencies:
            return (
                AgentCooperationAutomaticExecutionStatus.NO_VALID_PLAN,
                "PLAN_DEPENDENCIES_MISMATCH",
                "generated plan dependencies are inconsistent with the planning result.",
                None,
            )
        checks = (
            (len(plan.tasks), policy.max_tasks, "MAX_TASKS"),
            (len(planning_result.selected_agent_ids), policy.max_agents, "MAX_AGENTS"),
            (len(plan.dependencies), policy.max_dependencies, "MAX_DEPENDENCIES"),
            (_plan_depth(plan), policy.max_plan_depth, "MAX_PLAN_DEPTH"),
            (
                sum(task.logical_timeout_limit for task in plan.tasks),
                policy.max_logical_time,
                "MAX_LOGICAL_TIME",
            ),
        )
        for actual, maximum, name in checks:
            if actual > maximum:
                return (
                    AgentCooperationAutomaticExecutionStatus.LIMIT_REACHED,
                    name,
                    "generated plan exceeds an automatic execution limit.",
                    name,
                )
        if any(
            not self._agent_registry.contains(agent_id)
            for agent_id in planning_result.selected_agent_ids
        ):
            return (
                AgentCooperationAutomaticExecutionStatus.PLAN_VALIDATION_FAILED,
                "AGENT_NOT_FOUND",
                "generated plan references an unregistered agent.",
                None,
            )
        validation = self._agent_cooperation_planner._validate(plan, execution_policy)
        if validation is not None:
            plan_status, code, message = validation
            status = (
                AgentCooperationAutomaticExecutionStatus.LIMIT_REACHED
                if plan_status is AgentCooperationPlanStatus.LIMIT_REACHED
                else AgentCooperationAutomaticExecutionStatus.PLAN_VALIDATION_FAILED
            )
            return status, code, message, code if status is AgentCooperationAutomaticExecutionStatus.LIMIT_REACHED else None
        return None


def agent_cooperation_automatic_execution_request_signature(
    request: AgentCooperationAutomaticExecutionRequest,
) -> str:
    """Return the stable SHA-256 signature of one automatic execution request."""

    if not isinstance(request, AgentCooperationAutomaticExecutionRequest):
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            "request must be AgentCooperationAutomaticExecutionRequest."
        )
    return _signature(_jsonable_dataclass(request))


def _validate_request_policy(
    request: AgentCooperationAutomaticExecutionRequest,
) -> tuple[str, str] | None:
    policy = request.policy
    planning_policy = policy.planning_policy
    execution_policy = policy.execution_policy
    if planning_policy is None:
        return "MISSING_PLANNING_POLICY", "planning_policy is required."
    if execution_policy is None:
        return "MISSING_EXECUTION_POLICY", "execution_policy is required."
    if not planning_policy.enabled:
        return "PLANNING_POLICY_DISABLED", "planning_policy must be enabled."
    if not policy.dry_run and not policy.execute_generated_plan:
        return "EXECUTION_NOT_AUTHORIZED", "execute_generated_plan must be enabled."
    if not policy.dry_run and not execution_policy.enabled:
        return "EXECUTION_POLICY_DISABLED", "execution_policy must be enabled."
    policy_checks = (
        (planning_policy.max_tasks, policy.max_tasks, "PLANNING_MAX_TASKS"),
        (planning_policy.max_agents, policy.max_agents, "PLANNING_MAX_AGENTS"),
        (
            planning_policy.max_dependencies,
            policy.max_dependencies,
            "PLANNING_MAX_DEPENDENCIES",
        ),
        (planning_policy.max_plan_depth, policy.max_plan_depth, "PLANNING_MAX_DEPTH"),
        (execution_policy.max_tasks, policy.max_tasks, "EXECUTION_MAX_TASKS"),
        (
            execution_policy.max_dependencies,
            policy.max_dependencies,
            "EXECUTION_MAX_DEPENDENCIES",
        ),
        (execution_policy.max_depth, policy.max_plan_depth, "EXECUTION_MAX_DEPTH"),
        (
            execution_policy.max_total_executions,
            policy.max_total_executions,
            "EXECUTION_MAX_EXECUTIONS",
        ),
        (
            execution_policy.max_output_items,
            policy.max_output_items,
            "EXECUTION_MAX_OUTPUT_ITEMS",
        ),
    )
    for actual, maximum, code in policy_checks:
        if actual > maximum:
            return code, "nested policy exceeds an automatic execution limit."
    return None


def _plan_depth(plan: AgentCooperationPlan) -> int:
    prerequisites: dict[str, list[str]] = {task.task_id: [] for task in plan.tasks}
    for dependency in plan.dependencies:
        prerequisites[dependency.dependent_task_id].append(
            dependency.prerequisite_task_id
        )
    remaining = set(prerequisites)
    depths: dict[str, int] = {}
    while remaining:
        ready = sorted(
            task_id
            for task_id in remaining
            if all(item in depths for item in prerequisites[task_id])
        )
        if not ready:
            return MAX_AUTO_EXECUTION_DEPTH + 1
        for task_id in ready:
            depths[task_id] = 1 + max(
                (depths[item] for item in prerequisites[task_id]),
                default=0,
            )
            remaining.remove(task_id)
    return max(depths.values(), default=0)


def _merge_execution_metrics(
    metrics: dict[str, int],
    result: AgentCooperationPlanResult,
) -> None:
    metrics["tasks_executed"] = len(result.execution_order)
    metrics["tasks_failed"] = sum(
        item.status
        not in (AgentCooperationTaskStatus.SUCCESS, AgentCooperationTaskStatus.SKIPPED)
        for item in result.task_results
    )
    metrics["tasks_skipped"] = sum(
        item.status is AgentCooperationTaskStatus.SKIPPED
        for item in result.task_results
    )
    metrics["agents_executed"] = len(
        {
            agent_id
            for item in result.task_results
            if item.task_id in result.execution_order
            for agent_id in item.agent_ids
        }
    )
    metrics["dependencies_processed"] = result.metrics.get(
        "cooperation_dependencies_evaluated",
        0,
    )


def _sanitize_cooperation_result(
    result: AgentCooperationPlanResult,
) -> AgentCooperationPlanResult:
    safe_task_results = tuple(
        replace(item, execution_result=None)
        for item in result.task_results
    )
    return replace(result, task_results=safe_task_results)


def _result(
    *,
    execution_id: str,
    status: AgentCooperationAutomaticExecutionStatus,
    decision: AgentCooperationAutomaticExecutionDecision,
    signature: str,
    planning_result: AgentCooperationPlanningResult | None = None,
    generated_plan: AgentCooperationPlan | None = None,
    cooperation_result: AgentCooperationPlanResult | None = None,
    plan_signature: str | None = None,
    events: Sequence[AgentCooperationAutomaticExecutionEvent] = (),
    metrics: Mapping[str, int] | None = None,
    error_code: str | None = None,
    reason: str | None = None,
) -> AgentCooperationAutomaticExecutionResult:
    values = _base_metrics() if metrics is None else dict(metrics)
    summary = {
        "status": status.value,
        "planned": generated_plan is not None,
        "executed": cooperation_result is not None,
        "task_count": len(generated_plan.tasks) if generated_plan is not None else 0,
        "selected_agent_count": (
            len(planning_result.selected_agent_ids) if planning_result is not None else 0
        ),
        "successful_task_count": (
            sum(
                item.status is AgentCooperationTaskStatus.SUCCESS
                for item in cooperation_result.task_results
            )
            if cooperation_result is not None
            else 0
        ),
    }
    stage_decisions: list[AgentCooperationAutomaticExecutionDecision] = []
    if planning_result is not None:
        stage_decisions.append(
            _decision("PLANNING_ALLOWED", "PLANNING_POLICY_ENABLED", "planning")
        )
    accepted_statuses = (
        AgentCooperationAutomaticExecutionStatus.DRY_RUN_COMPLETED,
        AgentCooperationAutomaticExecutionStatus.EXECUTION_SUCCEEDED,
        AgentCooperationAutomaticExecutionStatus.EXECUTION_PARTIAL,
        AgentCooperationAutomaticExecutionStatus.EXECUTION_FAILED,
    )
    if generated_plan is not None and status in accepted_statuses:
        stage_decisions.append(
            _decision("PLAN_ACCEPTED", "PLAN_VALIDATED", "automatic_execution")
        )
    if cooperation_result is not None:
        stage_decisions.append(
            _decision("EXECUTION_ALLOWED", "EXECUTION_POLICY_ENABLED", "execution")
        )
    if decision not in stage_decisions:
        stage_decisions.append(decision)
    return AgentCooperationAutomaticExecutionResult(
        execution_id=execution_id,
        status=status,
        decision=decision,
        signature=signature,
        planning_result=planning_result,
        generated_plan=generated_plan,
        cooperation_result=cooperation_result,
        plan_signature=plan_signature,
        decisions=tuple(stage_decisions),
        safe_summary=summary,
        events=tuple(events),
        metrics=values,
        error_code=error_code,
        reason=reason,
    )


def _decision(
    decision: AgentCooperationAutomaticExecutionDecisionType | str,
    reason_code: str,
    policy_name: str,
    *,
    limit_name: str | None = None,
    outcome: str | None = None,
) -> AgentCooperationAutomaticExecutionDecision:
    return AgentCooperationAutomaticExecutionDecision(
        decision=decision,
        reason_code=reason_code,
        policy_name=policy_name,
        limit_name=limit_name,
        outcome=outcome,
    )


def _event(
    events: list[AgentCooperationAutomaticExecutionEvent],
    name: str,
    status: str,
    **details: object,
) -> None:
    if len(events) < MAX_AUTO_EXECUTION_EVENTS:
        events.append(AgentCooperationAutomaticExecutionEvent(name, status, details))


def _base_metrics() -> dict[str, int]:
    return {
        "requests": 0,
        "validation_failures": 0,
        "planning_started": 0,
        "planning_succeeded": 0,
        "planning_failed": 0,
        "plans_generated": 0,
        "plans_rejected": 0,
        "dry_runs": 0,
        "executions_started": 0,
        "executions_succeeded": 0,
        "executions_partial": 0,
        "executions_failed": 0,
        "limits_reached": 0,
        "tasks_planned": 0,
        "tasks_executed": 0,
        "tasks_failed": 0,
        "tasks_skipped": 0,
        "agents_selected": 0,
        "agents_executed": 0,
        "skills_required": 0,
        "dependencies_processed": 0,
    }


def _safe_mapping(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            f"{field_name} must be a mapping."
        )
    counter = {"items": 0}
    return _safe_mapping_inner(value, field_name, 0, counter)


def _safe_mapping_inner(
    value: Mapping[str, object],
    field_name: str,
    depth: int,
    counter: dict[str, int],
) -> dict[str, object]:
    if depth > MAX_AUTO_EXECUTION_VALUE_DEPTH:
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            f"{field_name} exceeds maximum depth."
        )
    if len(value) > MAX_AUTO_EXECUTION_METADATA_ITEMS:
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            f"{field_name} has too many items."
        )
    result: dict[str, object] = {}
    for raw_key in sorted(value, key=lambda item: str(item)):
        key = _key(raw_key, field_name)
        if _is_sensitive_key(key):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                f"{field_name} contains a sensitive key."
            )
        if _is_dynamic_code_key(key):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                f"{field_name} contains a dynamic code key."
            )
        result[key] = _safe_value(value[raw_key], field_name, depth + 1, counter)
    return result


def _safe_value(
    value: object,
    field_name: str,
    depth: int,
    counter: dict[str, int],
) -> object:
    if depth > MAX_AUTO_EXECUTION_VALUE_DEPTH:
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            f"{field_name} exceeds maximum depth."
        )
    counter["items"] += 1
    if counter["items"] > MAX_AUTO_EXECUTION_TOTAL_ITEMS:
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            f"{field_name} exceeds total item limit."
        )
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                f"{field_name} floats must be finite."
            )
        return value
    if isinstance(value, str):
        if len(value) > MAX_AUTO_EXECUTION_STRING_LENGTH:
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                f"{field_name} string exceeds length limit."
            )
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_mapping_inner(value, field_name, depth, counter))
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_AUTO_EXECUTION_SEQUENCE_ITEMS:
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                f"{field_name} sequence exceeds item limit."
            )
        return tuple(
            _safe_value(item, field_name, depth + 1, counter) for item in value
        )
    raise InvalidAgentCooperationAutomaticExecutionRequestError(
        f"{field_name} contains an unsupported object."
    )


def _jsonable_dataclass(value: object) -> object:
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is None:
        return _jsonable(value)
    return {
        name: _jsonable_dataclass(getattr(value, name))
        for name in sorted(fields)
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    return _jsonable_dataclass(value)


def _signature(value: object) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identifier(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not _IDENTIFIER_PATTERN.fullmatch(value)
    ):
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            f"{field_name} contains unsupported characters."
        )
    return value


def _key(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            f"{field_name} keys must be strings."
        )
    return _identifier(value, f"{field_name} key")


def _bounded_int(value: int, field_name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            f"{field_name} is outside the allowed range."
        )
    return value


def _metric_mapping(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            "metrics must be a mapping."
        )
    result: dict[str, int] = {}
    for key, item in value.items():
        normalized = _identifier(key, "metric name")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise InvalidAgentCooperationAutomaticExecutionRequestError(
                "metric values must be non-negative integers."
            )
        result[normalized] = item
    return result


def _safe_message(value: str) -> str:
    if not isinstance(value, str):
        return "automatic cooperation execution failed."
    lowered = value.lower()
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return "automatic cooperation execution failed."
    return " ".join(value.split())[:240] or "automatic cooperation execution failed."


def _is_sensitive_key(value: str) -> bool:
    lowered = value.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _is_dynamic_code_key(value: str) -> bool:
    lowered = value.lower()
    return any(part in lowered for part in _DYNAMIC_CODE_KEY_PARTS)


def _validate_signature(
    value: str,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if allow_empty and value == "":
        return
    if not isinstance(value, str) or _SIGNATURE_PATTERN.fullmatch(value) is None:
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            f"{field_name} must be a SHA-256 hex digest."
        )


def _status(
    value: AgentCooperationAutomaticExecutionStatus | str,
) -> AgentCooperationAutomaticExecutionStatus:
    try:
        return (
            value
            if isinstance(value, AgentCooperationAutomaticExecutionStatus)
            else AgentCooperationAutomaticExecutionStatus(value)
        )
    except (TypeError, ValueError) as error:
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            "invalid automatic execution status."
        ) from error


def _decision_type(
    value: AgentCooperationAutomaticExecutionDecisionType | str,
) -> AgentCooperationAutomaticExecutionDecisionType:
    try:
        return (
            value
            if isinstance(value, AgentCooperationAutomaticExecutionDecisionType)
            else AgentCooperationAutomaticExecutionDecisionType(value)
        )
    except (TypeError, ValueError) as error:
        raise InvalidAgentCooperationAutomaticExecutionRequestError(
            "invalid automatic execution decision."
        ) from error

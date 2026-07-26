"""Safe deterministic planning of declarative multi-agent cooperation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType

from core.agent_context import AgentContextBuilder
from core.agent_cooperation_plan import (
    AgentCooperationDependency,
    AgentCooperationExecutionType,
    AgentCooperationOutputBinding,
    AgentCooperationPlan,
    AgentCooperationPlanError,
    AgentCooperationPlanPolicy,
    AgentCooperationPlanner,
    AgentCooperationTask,
    agent_cooperation_plan_signature,
)
from core.agent_delegation import AgentDelegationPolicy, AgentDelegationRequest, AgentDelegationService
from core.agent_delegation_chain import (
    AgentDelegationChainPolicy,
    AgentDelegationChainRequest,
    AgentDelegationChainService,
    AgentDelegationChainStep,
)
from core.agent_delegation_coordinator import (
    AgentDelegationCoordinationChain,
    AgentDelegationCoordinationPlan,
    AgentDelegationCoordinationPolicy,
    AgentDelegationCoordinationRequest,
    AgentDelegationCoordinator,
)
from core.agent_executor import AgentExecutor
from core.agent_registry import AgentDefinition, AgentRegistry, AgentType, validate_agent_id
from core.agent_resolver import AgentResolutionRequest, AgentResolutionStatus, AgentResolver
from core.multi_agent import MultiAgentCoordinator, MultiAgentExecutionPolicy, MultiAgentExecutionRequest
from core.skill_registry import SkillDefinition, validate_skill_id
from core.skill_resolver import SkillResolutionRequest, SkillResolutionStatus
from core.skill_system import SkillSystem


MAX_PLANNING_AGENTS = 16
MAX_PLANNING_TASKS = 64
MAX_PLANNING_DEPENDENCIES = 256
MAX_PLANNING_DEPTH = 32
MAX_PLANNING_REQUIRED_SKILLS = 32
MAX_PLANNING_REQUIREMENTS = 32
MAX_PLANNING_CANDIDATES = 64
MAX_PLANNING_STATES = 65_536
MAX_PLANNING_METADATA_ITEMS = 32
MAX_PLANNING_SEQUENCE_ITEMS = 128
MAX_PLANNING_VALUE_DEPTH = 8
MAX_PLANNING_TOTAL_ITEMS = 1_024
MAX_PLANNING_STRING_LENGTH = 2_000
MAX_PLANNING_EVENTS = 1_024

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
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


class AgentCooperationPlanningError(RuntimeError):
    """Base error for deterministic automatic cooperation planning."""


class InvalidAgentCooperationPlanningRequestError(AgentCooperationPlanningError):
    """Raised when a planning request or policy is malformed."""


class AgentCooperationPlanningStatus(str, Enum):
    """Terminal statuses for automatic cooperation planning."""

    SUCCESS = "SUCCESS"
    DISABLED = "DISABLED"
    INVALID_REQUEST = "INVALID_REQUEST"
    NO_MATCHING_AGENTS = "NO_MATCHING_AGENTS"
    MISSING_CAPABILITY = "MISSING_CAPABILITY"
    MISSING_PERMISSION = "MISSING_PERMISSION"
    MISSING_SKILL = "MISSING_SKILL"
    AMBIGUOUS = "AMBIGUOUS"
    LIMIT_REACHED = "LIMIT_REACHED"
    PLAN_VALIDATION_FAILED = "PLAN_VALIDATION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentCooperationObjectiveType(str, Enum):
    """Closed structured objective categories without text classification."""

    ANALYSIS = "ANALYSIS"
    IMPLEMENTATION = "IMPLEMENTATION"
    VALIDATION = "VALIDATION"
    REVIEW = "REVIEW"
    DOCUMENTATION = "DOCUMENTATION"
    RESEARCH = "RESEARCH"
    ORCHESTRATION = "ORCHESTRATION"
    CUSTOM = "CUSTOM"


class AgentCooperationPlanningDecisionStatus(str, Enum):
    """Disposition of one evaluated candidate."""

    SELECTED = "SELECTED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class AgentCooperationPlanningReasonCode(str, Enum):
    """Structured, explainable planning reasons."""

    REQUIRED_AGENT = "REQUIRED_AGENT"
    REQUIRED_TYPE_MATCH = "REQUIRED_TYPE_MATCH"
    REQUIRED_CAPABILITY_MATCH = "REQUIRED_CAPABILITY_MATCH"
    REQUIRED_PERMISSION_MATCH = "REQUIRED_PERMISSION_MATCH"
    REQUIRED_SKILL_AVAILABLE = "REQUIRED_SKILL_AVAILABLE"
    PREFERRED_AGENT = "PREFERRED_AGENT"
    PREFERRED_TYPE_MATCH = "PREFERRED_TYPE_MATCH"
    PREFERRED_CAPABILITY_MATCH = "PREFERRED_CAPABILITY_MATCH"
    MINIMAL_COVERAGE_SET = "MINIMAL_COVERAGE_SET"
    EXCLUDED_AGENT = "EXCLUDED_AGENT"
    DISABLED_AGENT = "DISABLED_AGENT"
    MISSING_CAPABILITY = "MISSING_CAPABILITY"
    MISSING_PERMISSION = "MISSING_PERMISSION"
    MISSING_SKILL = "MISSING_SKILL"
    AMBIGUOUS_SELECTION = "AMBIGUOUS_SELECTION"
    LIMIT_REACHED = "LIMIT_REACHED"


@dataclass(frozen=True, slots=True)
class AgentCooperationPlanningPolicy:
    """Immutable opt-in policy and hard planning bounds."""

    enabled: bool = False
    max_agents: int = 8
    max_tasks: int = 16
    max_dependencies: int = 48
    max_plan_depth: int = 8
    max_required_skills: int = 16
    max_candidates: int = 32
    require_unique_agent_per_task: bool = True
    allow_agent_reuse: bool = True
    require_all_capabilities_covered: bool = True
    require_all_skills_available: bool = True
    fail_on_ambiguous_agent: bool = True
    fail_on_missing_optional_agent: bool = False
    allow_multi_agent_tasks: bool = True
    allow_delegation_tasks: bool = True
    allow_skill_requirements: bool = True
    deterministic_ordering: bool = True

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "require_unique_agent_per_task",
            "allow_agent_reuse",
            "require_all_capabilities_covered",
            "require_all_skills_available",
            "fail_on_ambiguous_agent",
            "fail_on_missing_optional_agent",
            "allow_multi_agent_tasks",
            "allow_delegation_tasks",
            "allow_skill_requirements",
            "deterministic_ordering",
        ):
            if type(getattr(self, name)) is not bool:
                raise InvalidAgentCooperationPlanningRequestError(f"{name} must be a bool.")
        object.__setattr__(self, "max_agents", _bounded_int(self.max_agents, "max_agents", MAX_PLANNING_AGENTS))
        object.__setattr__(self, "max_tasks", _bounded_int(self.max_tasks, "max_tasks", MAX_PLANNING_TASKS))
        object.__setattr__(
            self,
            "max_dependencies",
            _bounded_int(self.max_dependencies, "max_dependencies", MAX_PLANNING_DEPENDENCIES),
        )
        object.__setattr__(
            self,
            "max_plan_depth",
            _bounded_int(self.max_plan_depth, "max_plan_depth", MAX_PLANNING_DEPTH),
        )
        object.__setattr__(
            self,
            "max_required_skills",
            _bounded_int(
                self.max_required_skills,
                "max_required_skills",
                MAX_PLANNING_REQUIRED_SKILLS,
            ),
        )
        object.__setattr__(
            self,
            "max_candidates",
            _bounded_int(self.max_candidates, "max_candidates", MAX_PLANNING_CANDIDATES),
        )
        if not self.deterministic_ordering:
            raise InvalidAgentCooperationPlanningRequestError("deterministic_ordering must be True.")


@dataclass(frozen=True, slots=True)
class AgentCooperationPlanningTaskRequirement:
    """Structured requirements for one generated cooperation task."""

    task_id: str
    objective_id: str | None = None
    execution_type: AgentCooperationExecutionType | str | None = None
    source_agent_id: str | None = None
    required_capability_ids: tuple[str, ...] = ()
    preferred_capability_ids: tuple[str, ...] = ()
    required_agent_types: tuple[AgentType | str, ...] = ()
    preferred_agent_types: tuple[AgentType | str, ...] = ()
    required_agent_ids: tuple[str, ...] = ()
    preferred_agent_ids: tuple[str, ...] = ()
    excluded_agent_ids: tuple[str, ...] = ()
    required_skill_ids: tuple[str, ...] = ()
    optional_skill_ids: tuple[str, ...] = ()
    required_permission_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    output_bindings: tuple[AgentCooperationOutputBinding, ...] = ()
    inherit_objective_requirements: bool = True
    order: int = 0
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        if self.objective_id is not None:
            object.__setattr__(self, "objective_id", _identifier(self.objective_id, "objective_id"))
        if self.execution_type is not None:
            object.__setattr__(self, "execution_type", _execution_type(self.execution_type))
        if self.source_agent_id is not None:
            object.__setattr__(self, "source_agent_id", validate_agent_id(self.source_agent_id))
        _normalize_requirement_fields(self)
        object.__setattr__(self, "depends_on", _identifier_tuple(self.depends_on, "depends_on"))
        bindings = tuple(self.output_bindings)
        if not all(isinstance(item, AgentCooperationOutputBinding) for item in bindings):
            raise InvalidAgentCooperationPlanningRequestError(
                "output_bindings must contain AgentCooperationOutputBinding values."
            )
        object.__setattr__(self, "output_bindings", bindings)
        if type(self.inherit_objective_requirements) is not bool:
            raise InvalidAgentCooperationPlanningRequestError("inherit_objective_requirements must be a bool.")
        for name in ("order", "priority"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or abs(value) > 1_000_000:
                raise InvalidAgentCooperationPlanningRequestError(f"{name} is outside the allowed range.")


@dataclass(frozen=True, slots=True)
class AgentCooperationPlanningRequest:
    """Structured objective that may be transformed into a validated plan."""

    objective_id: str
    objective_type: AgentCooperationObjectiveType | str
    required_capability_ids: tuple[str, ...] = ()
    preferred_capability_ids: tuple[str, ...] = ()
    required_agent_types: tuple[AgentType | str, ...] = ()
    preferred_agent_types: tuple[AgentType | str, ...] = ()
    required_agent_ids: tuple[str, ...] = ()
    preferred_agent_ids: tuple[str, ...] = ()
    excluded_agent_ids: tuple[str, ...] = ()
    required_skill_ids: tuple[str, ...] = ()
    optional_skill_ids: tuple[str, ...] = ()
    required_permission_ids: tuple[str, ...] = ()
    structured_input: Mapping[str, object] = field(default_factory=dict)
    shared_context: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    execution_id: str | None = None
    correlation_id: str | None = None
    execution_type: AgentCooperationExecutionType | str | None = None
    source_agent_id: str | None = None
    task_requirements: tuple[AgentCooperationPlanningTaskRequirement, ...] = ()
    policy: AgentCooperationPlanningPolicy = field(default_factory=AgentCooperationPlanningPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_id", _identifier(self.objective_id, "objective_id"))
        object.__setattr__(self, "objective_type", _objective_type(self.objective_type))
        _normalize_requirement_fields(self)
        object.__setattr__(
            self,
            "structured_input",
            MappingProxyType(_safe_mapping(self.structured_input, "structured_input")),
        )
        object.__setattr__(
            self,
            "shared_context",
            MappingProxyType(_safe_mapping(self.shared_context, "shared_context")),
        )
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))
        for name in ("execution_id", "correlation_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        if self.execution_type is not None:
            object.__setattr__(self, "execution_type", _execution_type(self.execution_type))
        if self.source_agent_id is not None:
            object.__setattr__(self, "source_agent_id", validate_agent_id(self.source_agent_id))
        requirements = tuple(self.task_requirements)
        if not all(isinstance(item, AgentCooperationPlanningTaskRequirement) for item in requirements):
            raise InvalidAgentCooperationPlanningRequestError(
                "task_requirements must contain AgentCooperationPlanningTaskRequirement values."
            )
        if len({item.task_id for item in requirements}) != len(requirements):
            raise InvalidAgentCooperationPlanningRequestError("task requirement ids must be unique.")
        object.__setattr__(self, "task_requirements", requirements)
        if not isinstance(self.policy, AgentCooperationPlanningPolicy):
            raise InvalidAgentCooperationPlanningRequestError(
                "policy must be AgentCooperationPlanningPolicy."
            )


@dataclass(frozen=True, slots=True)
class AgentCooperationPlanningDecision:
    """Explainable disposition for one agent candidate and task."""

    task_id: str
    agent_id: str
    status: AgentCooperationPlanningDecisionStatus
    covered_capability_ids: tuple[str, ...] = ()
    covered_skill_ids: tuple[str, ...] = ()
    reasons: tuple[AgentCooperationPlanningReasonCode, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        object.__setattr__(self, "status", _decision_status(self.status))
        object.__setattr__(
            self,
            "covered_capability_ids",
            _identifier_tuple(self.covered_capability_ids, "covered_capability_ids"),
        )
        object.__setattr__(
            self,
            "covered_skill_ids",
            _skill_id_tuple(self.covered_skill_ids, "covered_skill_ids"),
        )
        reasons = tuple(_reason_code(item) for item in self.reasons)
        object.__setattr__(self, "reasons", tuple(dict.fromkeys(reasons)))


@dataclass(frozen=True, slots=True)
class AgentCooperationPlanningEvent:
    """Safe structured planning event."""

    name: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "event name"))
        object.__setattr__(self, "status", _identifier(self.status, "event status"))
        object.__setattr__(self, "details", MappingProxyType(_safe_mapping(self.details, "event details")))


@dataclass(frozen=True, slots=True)
class AgentCooperationPlanningResult:
    """Immutable result of deterministic plan construction."""

    status: AgentCooperationPlanningStatus
    planning_request_signature: str
    plan: AgentCooperationPlan | None = None
    plan_signature: str | None = None
    decisions: tuple[AgentCooperationPlanningDecision, ...] = ()
    evaluated_agent_ids: tuple[str, ...] = ()
    accepted_agent_ids: tuple[str, ...] = ()
    rejected_agent_ids: tuple[str, ...] = ()
    selected_agent_ids: tuple[str, ...] = ()
    covered_capability_ids: tuple[str, ...] = ()
    missing_capability_ids: tuple[str, ...] = ()
    available_skill_ids: tuple[str, ...] = ()
    missing_skill_ids: tuple[str, ...] = ()
    created_task_ids: tuple[str, ...] = ()
    created_dependencies: tuple[AgentCooperationDependency, ...] = ()
    events: tuple[AgentCooperationPlanningEvent, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    error_code: str | None = None
    safe_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _planning_status(self.status))
        _validate_signature(self.planning_request_signature, "planning_request_signature", allow_empty=True)
        if self.plan is not None and not isinstance(self.plan, AgentCooperationPlan):
            raise InvalidAgentCooperationPlanningRequestError("plan must be AgentCooperationPlan or None.")
        if self.plan_signature is not None:
            _validate_signature(self.plan_signature, "plan_signature")
        object.__setattr__(self, "decisions", tuple(self.decisions))
        if not all(isinstance(item, AgentCooperationPlanningDecision) for item in self.decisions):
            raise InvalidAgentCooperationPlanningRequestError(
                "decisions must contain AgentCooperationPlanningDecision values."
            )
        for name in (
            "evaluated_agent_ids",
            "accepted_agent_ids",
            "rejected_agent_ids",
            "selected_agent_ids",
        ):
            object.__setattr__(self, name, _agent_id_tuple(getattr(self, name), name))
        for name in ("covered_capability_ids", "missing_capability_ids", "created_task_ids"):
            object.__setattr__(self, name, _identifier_tuple(getattr(self, name), name))
        for name in ("available_skill_ids", "missing_skill_ids"):
            object.__setattr__(self, name, _skill_id_tuple(getattr(self, name), name))
        dependencies = tuple(self.created_dependencies)
        if not all(isinstance(item, AgentCooperationDependency) for item in dependencies):
            raise InvalidAgentCooperationPlanningRequestError(
                "created_dependencies must contain AgentCooperationDependency values."
            )
        object.__setattr__(self, "created_dependencies", dependencies)
        object.__setattr__(self, "events", tuple(self.events)[-MAX_PLANNING_EVENTS:])
        object.__setattr__(self, "metrics", MappingProxyType(_metric_mapping(self.metrics)))
        if self.safe_message is not None:
            object.__setattr__(self, "safe_message", _safe_message(self.safe_message))


@dataclass(frozen=True, slots=True)
class _TaskCriteria:
    task_id: str
    objective_id: str
    execution_type: AgentCooperationExecutionType | None
    source_agent_id: str | None
    required_capability_ids: tuple[str, ...]
    preferred_capability_ids: tuple[str, ...]
    required_agent_types: tuple[AgentType, ...]
    preferred_agent_types: tuple[AgentType, ...]
    required_agent_ids: tuple[str, ...]
    preferred_agent_ids: tuple[str, ...]
    excluded_agent_ids: tuple[str, ...]
    required_skill_ids: tuple[str, ...]
    optional_skill_ids: tuple[str, ...]
    required_permission_ids: tuple[str, ...]
    depends_on: tuple[str, ...]
    output_bindings: tuple[AgentCooperationOutputBinding, ...]
    order: int
    priority: int


@dataclass(frozen=True, slots=True)
class _Selection:
    status: AgentCooperationPlanningStatus
    selected: tuple[AgentDefinition, ...] = ()
    decisions: tuple[AgentCooperationPlanningDecision, ...] = ()
    covered_capabilities: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    available_skills: tuple[str, ...] = ()
    missing_skills: tuple[str, ...] = ()
    error_code: str | None = None
    safe_message: str | None = None


class AgentCooperationAutomaticPlanner:
    """Build validated AgentCooperationPlan values without executing anything."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        agent_resolver: AgentResolver,
        agent_context_builder: AgentContextBuilder,
        agent_executor: AgentExecutor,
        skill_system: SkillSystem,
        agent_cooperation_planner: AgentCooperationPlanner,
        agent_delegation_service: AgentDelegationService,
        agent_delegation_chain_service: AgentDelegationChainService,
        agent_delegation_coordinator: AgentDelegationCoordinator,
        multi_agent_coordinator: MultiAgentCoordinator,
    ) -> None:
        dependencies = (
            (agent_registry, AgentRegistry, "agent_registry"),
            (agent_resolver, AgentResolver, "agent_resolver"),
            (agent_context_builder, AgentContextBuilder, "agent_context_builder"),
            (agent_executor, AgentExecutor, "agent_executor"),
            (skill_system, SkillSystem, "skill_system"),
            (agent_cooperation_planner, AgentCooperationPlanner, "agent_cooperation_planner"),
            (agent_delegation_service, AgentDelegationService, "agent_delegation_service"),
            (agent_delegation_chain_service, AgentDelegationChainService, "agent_delegation_chain_service"),
            (agent_delegation_coordinator, AgentDelegationCoordinator, "agent_delegation_coordinator"),
            (multi_agent_coordinator, MultiAgentCoordinator, "multi_agent_coordinator"),
        )
        for value, expected, name in dependencies:
            if not isinstance(value, expected):
                raise AgentCooperationPlanningError(f"{name} must be {expected.__name__}.")
        self._agent_registry = agent_registry
        self._agent_resolver = agent_resolver
        self._agent_context_builder = agent_context_builder
        self._agent_executor = agent_executor
        self._skill_system = skill_system
        self._agent_cooperation_planner = agent_cooperation_planner
        self._agent_delegation_service = agent_delegation_service
        self._agent_delegation_chain_service = agent_delegation_chain_service
        self._agent_delegation_coordinator = agent_delegation_coordinator
        self._multi_agent_coordinator = multi_agent_coordinator

    def plan(self, request: AgentCooperationPlanningRequest) -> AgentCooperationPlanningResult:
        """Return one fully declared plan and never execute it."""

        events: list[AgentCooperationPlanningEvent] = []
        metrics = _base_metrics()
        if not isinstance(request, AgentCooperationPlanningRequest):
            metrics["cooperation_planning_failed"] = 1
            return _result(
                AgentCooperationPlanningStatus.INVALID_REQUEST,
                "",
                events=events,
                metrics=metrics,
                error_code="INVALID_REQUEST",
                safe_message="request must be AgentCooperationPlanningRequest.",
            )
        signature = agent_cooperation_planning_request_signature(request)
        policy = request.policy
        metrics["cooperation_planning_requests"] = 1
        _event(events, "agent_cooperation_planning_requested", "requested", objective_id=request.objective_id)
        _event(events, "agent_cooperation_planning_validation_started", "started", objective_id=request.objective_id)
        if not policy.enabled:
            metrics["cooperation_planning_failed"] = 1
            _event(events, "agent_cooperation_planning_validation_failed", "disabled")
            return _result(
                AgentCooperationPlanningStatus.DISABLED,
                signature,
                events=events,
                metrics=metrics,
                error_code="DISABLED",
                safe_message="automatic cooperation planning is disabled.",
            )
        validation = _validate_request_limits(request)
        if validation is not None:
            status, code, message = validation
            metrics["cooperation_planning_failed"] = 1
            if status is AgentCooperationPlanningStatus.LIMIT_REACHED:
                metrics["cooperation_planning_limits_reached"] = 1
                _event(events, "agent_cooperation_planning_limit_reached", "failed", error_code=code)
            else:
                _event(events, "agent_cooperation_planning_validation_failed", "failed", error_code=code)
            return _result(
                status,
                signature,
                events=events,
                metrics=metrics,
                error_code=code,
                safe_message=message,
            )

        criteria = _task_criteria(request)
        tasks: list[AgentCooperationTask] = []
        dependencies: list[AgentCooperationDependency] = []
        all_decisions: list[AgentCooperationPlanningDecision] = []
        selected_by_task: dict[str, tuple[AgentDefinition, ...]] = {}
        selected_globally: dict[str, AgentDefinition] = {}
        covered_capabilities: set[str] = set()
        missing_capabilities: set[str] = set()
        available_skills: set[str] = set()
        missing_skills: set[str] = set()

        for item in sorted(criteria, key=lambda value: (value.order, -value.priority, value.task_id)):
            selection = self._select(item, policy, tuple(selected_globally), events, metrics)
            all_decisions.extend(selection.decisions)
            covered_capabilities.update(selection.covered_capabilities)
            missing_capabilities.update(selection.missing_capabilities)
            available_skills.update(selection.available_skills)
            missing_skills.update(selection.missing_skills)
            if selection.status is not AgentCooperationPlanningStatus.SUCCESS:
                metrics["cooperation_planning_failed"] = 1
                if selection.status is AgentCooperationPlanningStatus.AMBIGUOUS:
                    metrics["cooperation_planning_ambiguous"] = 1
                    _event(events, "agent_cooperation_planning_ambiguous", "ambiguous", task_id=item.task_id)
                elif selection.status is AgentCooperationPlanningStatus.LIMIT_REACHED:
                    metrics["cooperation_planning_limits_reached"] = 1
                    _event(events, "agent_cooperation_planning_limit_reached", "failed", task_id=item.task_id)
                else:
                    _event(events, "agent_cooperation_requirements_missing", "failed", task_id=item.task_id)
                _event(events, "agent_cooperation_planning_failed", "failed", task_id=item.task_id)
                return _result(
                    selection.status,
                    signature,
                    decisions=tuple(all_decisions),
                    covered_capability_ids=tuple(sorted(covered_capabilities)),
                    missing_capability_ids=tuple(sorted(missing_capabilities)),
                    available_skill_ids=tuple(sorted(available_skills)),
                    missing_skill_ids=tuple(sorted(missing_skills)),
                    events=events,
                    metrics=metrics,
                    error_code=selection.error_code,
                    safe_message=selection.safe_message,
                )
            selected_by_task[item.task_id] = selection.selected
            for agent in selection.selected:
                selected_globally[agent.agent_id] = agent
            if len(selected_globally) > policy.max_agents:
                metrics["cooperation_planning_failed"] = 1
                metrics["cooperation_planning_limits_reached"] = 1
                _event(events, "agent_cooperation_planning_limit_reached", "failed", task_id=item.task_id)
                return _result(
                    AgentCooperationPlanningStatus.LIMIT_REACHED,
                    signature,
                    decisions=tuple(all_decisions),
                    events=events,
                    metrics=metrics,
                    error_code="MAX_AGENTS",
                    safe_message="selected agents exceed max_agents.",
                )
            _event(events, "agent_cooperation_requirements_covered", "succeeded", task_id=item.task_id)
            try:
                task = _build_task(item, selection.selected, request, policy)
            except (AgentCooperationPlanningError, AgentCooperationPlanError, ValueError, TypeError) as error:
                metrics["cooperation_planning_failed"] = 1
                _event(events, "agent_cooperation_plan_validation_failed", "failed", task_id=item.task_id)
                return _result(
                    AgentCooperationPlanningStatus.PLAN_VALIDATION_FAILED,
                    signature,
                    decisions=tuple(all_decisions),
                    events=events,
                    metrics=metrics,
                    error_code="TASK_BUILD_FAILED",
                    safe_message=str(error),
                )
            tasks.append(task)
            dependencies.extend(
                AgentCooperationDependency(source_task_id, item.task_id)
                for source_task_id in item.depends_on
            )

        _event(events, "agent_cooperation_plan_build_started", "started", tasks=len(tasks))
        try:
            plan = AgentCooperationPlan(
                plan_id=request.objective_id,
                tasks=tuple(tasks),
                dependencies=tuple(dependencies),
                metadata={
                    "objective_id": request.objective_id,
                    "objective_type": request.objective_type.value,
                    "planning_request_signature": signature,
                },
                policy=AgentCooperationPlanPolicy(
                    enabled=False,
                    max_tasks=policy.max_tasks,
                    max_dependencies=policy.max_dependencies,
                    max_depth=policy.max_plan_depth,
                    max_total_executions=policy.max_tasks,
                    max_output_items=256,
                    max_propagated_items=128,
                    propagate_dependency_outputs=any(task.dependency_output_bindings for task in tasks),
                ),
            )
            plan_signature = agent_cooperation_plan_signature(plan)
        except (AgentCooperationPlanError, ValueError, TypeError) as error:
            metrics["cooperation_planning_failed"] = 1
            _event(events, "agent_cooperation_plan_validation_failed", "failed")
            return _result(
                AgentCooperationPlanningStatus.PLAN_VALIDATION_FAILED,
                signature,
                decisions=tuple(all_decisions),
                events=events,
                metrics=metrics,
                error_code="PLAN_VALIDATION_FAILED",
                safe_message=str(error),
            )

        metrics["cooperation_planning_succeeded"] = 1
        metrics["cooperation_agents_selected"] = len(selected_globally)
        metrics["cooperation_tasks_created"] = len(tasks)
        metrics["cooperation_dependencies_created"] = len(dependencies)
        metrics["cooperation_required_capabilities_covered"] = len(covered_capabilities)
        metrics["cooperation_required_skills_available"] = len(available_skills)
        _event(events, "agent_cooperation_plan_built", "succeeded", tasks=len(tasks))
        _event(events, "agent_cooperation_planning_completed", "succeeded", objective_id=request.objective_id)
        return _result(
            AgentCooperationPlanningStatus.SUCCESS,
            signature,
            plan=plan,
            plan_signature=plan_signature,
            decisions=tuple(all_decisions),
            selected_agent_ids=tuple(sorted(selected_globally)),
            covered_capability_ids=tuple(sorted(covered_capabilities)),
            available_skill_ids=tuple(sorted(available_skills)),
            created_task_ids=tuple(task.task_id for task in tasks),
            created_dependencies=tuple(dependencies),
            events=events,
            metrics=metrics,
        )

    def _select(
        self,
        criteria: _TaskCriteria,
        policy: AgentCooperationPlanningPolicy,
        already_selected_ids: tuple[str, ...],
        events: list[AgentCooperationPlanningEvent],
        metrics: dict[str, int],
    ) -> _Selection:
        available_skills, missing_skills, skill_definitions = _resolve_skills(
            self._skill_system,
            criteria,
            policy,
        )
        if missing_skills:
            missing = missing_skills
            metrics["cooperation_required_skills_missing"] += len(missing)
            return _Selection(
                AgentCooperationPlanningStatus.MISSING_SKILL,
                available_skills=available_skills,
                missing_skills=missing,
                error_code="MISSING_SKILL",
                safe_message="one or more required skills are unavailable.",
            )
        try:
            agents = tuple(sorted(self._agent_registry.list_agents(enabled_only=False), key=lambda item: item.agent_id))
        except (RuntimeError, TypeError, ValueError):
            return _Selection(
                AgentCooperationPlanningStatus.NO_MATCHING_AGENTS,
                error_code="REGISTRY_UNAVAILABLE",
                safe_message="agent registry is unavailable.",
            )
        if len(agents) > policy.max_candidates:
            return _Selection(
                AgentCooperationPlanningStatus.LIMIT_REACHED,
                available_skills=available_skills,
                error_code="MAX_CANDIDATES",
                safe_message="candidate limit reached.",
            )
        known_ids = {agent.agent_id for agent in agents}
        if criteria.source_agent_id is not None:
            source = next(
                (agent for agent in agents if agent.agent_id == criteria.source_agent_id),
                None,
            )
            if source is None or not source.enabled:
                return _Selection(
                    AgentCooperationPlanningStatus.NO_MATCHING_AGENTS,
                    available_skills=available_skills,
                    error_code="SOURCE_AGENT_MISSING",
                    safe_message="delegation source agent is unavailable.",
                )
        missing_required_agents = tuple(
            agent_id for agent_id in criteria.required_agent_ids if agent_id not in known_ids
        )
        missing_preferred_agents = tuple(
            agent_id for agent_id in criteria.preferred_agent_ids if agent_id not in known_ids
        )
        if missing_required_agents:
            return _Selection(
                AgentCooperationPlanningStatus.NO_MATCHING_AGENTS,
                available_skills=available_skills,
                error_code="REQUIRED_AGENT_MISSING",
                safe_message="a required agent is not registered.",
            )
        if missing_preferred_agents and policy.fail_on_missing_optional_agent:
            return _Selection(
                AgentCooperationPlanningStatus.NO_MATCHING_AGENTS,
                available_skills=available_skills,
                error_code="PREFERRED_AGENT_MISSING",
                safe_message="a preferred agent is not registered.",
            )

        accepted: list[AgentDefinition] = []
        decisions: list[AgentCooperationPlanningDecision] = []
        for agent in agents:
            metrics["cooperation_candidates_evaluated"] += 1
            _event(events, "agent_cooperation_candidate_evaluated", "evaluated", task_id=criteria.task_id, agent_id=agent.agent_id)
            rejection = _candidate_rejection(agent, criteria, policy, already_selected_ids)
            if rejection is not None:
                metrics["cooperation_candidates_rejected"] += 1
                decisions.append(
                    AgentCooperationPlanningDecision(
                        criteria.task_id,
                        agent.agent_id,
                        AgentCooperationPlanningDecisionStatus.REJECTED,
                        reasons=(rejection,),
                    )
                )
                _event(events, "agent_cooperation_candidate_rejected", "rejected", task_id=criteria.task_id, agent_id=agent.agent_id)
                continue
            resolution = self._agent_resolver.resolve(
                AgentResolutionRequest(
                    required_agent_ids=(agent.agent_id,),
                    required_agent_types=criteria.required_agent_types,
                    required_permission_ids=criteria.required_permission_ids,
                    enabled_only=True,
                    require_unique_top_score=False,
                    metadata={"route": "cooperation_automatic_planner", "task_id": criteria.task_id},
                )
            )
            if resolution.status is not AgentResolutionStatus.RESOLVED:
                metrics["cooperation_candidates_rejected"] += 1
                decisions.append(
                    AgentCooperationPlanningDecision(
                        criteria.task_id,
                        agent.agent_id,
                        AgentCooperationPlanningDecisionStatus.REJECTED,
                        reasons=(AgentCooperationPlanningReasonCode.MISSING_PERMISSION,),
                    )
                )
                _event(events, "agent_cooperation_candidate_rejected", "rejected", task_id=criteria.task_id, agent_id=agent.agent_id)
                continue
            accepted.append(agent)
            metrics["cooperation_candidates_accepted"] += 1
            decisions.append(
                _candidate_decision(criteria, agent, available_skills, skill_definitions)
            )
            _event(events, "agent_cooperation_candidate_accepted", "accepted", task_id=criteria.task_id, agent_id=agent.agent_id)

        if not accepted:
            status = (
                AgentCooperationPlanningStatus.MISSING_PERMISSION
                if criteria.required_permission_ids
                else AgentCooperationPlanningStatus.NO_MATCHING_AGENTS
            )
            return _Selection(
                status,
                decisions=tuple(decisions),
                available_skills=available_skills,
                error_code=status.value,
                safe_message="no compatible agent candidates are available.",
            )

        coverage = _minimal_coverage_set(
            criteria,
            tuple(accepted),
            available_skills,
            skill_definitions,
            policy,
        )
        if coverage[0] is not AgentCooperationPlanningStatus.SUCCESS:
            missing_caps = coverage[2]
            missing_skill_ids = coverage[3]
            metrics["cooperation_required_capabilities_missing"] += len(missing_caps)
            metrics["cooperation_required_skills_missing"] += len(missing_skill_ids)
            return _Selection(
                coverage[0],
                decisions=tuple(decisions),
                covered_capabilities=tuple(
                    sorted(set(criteria.required_capability_ids) - set(missing_caps))
                ),
                missing_capabilities=missing_caps,
                available_skills=available_skills,
                missing_skills=missing_skill_ids,
                error_code=coverage[0].value,
                safe_message=coverage[4],
            )
        selected = coverage[1]
        selected_ids = {agent.agent_id for agent in selected}
        selected_decisions = tuple(
            AgentCooperationPlanningDecision(
                item.task_id,
                item.agent_id,
                AgentCooperationPlanningDecisionStatus.SELECTED
                if item.agent_id in selected_ids
                else item.status,
                covered_capability_ids=item.covered_capability_ids,
                covered_skill_ids=item.covered_skill_ids,
                reasons=item.reasons
                + (
                    (AgentCooperationPlanningReasonCode.MINIMAL_COVERAGE_SET,)
                    if item.agent_id in selected_ids
                    else ()
                ),
            )
            for item in decisions
        )
        return _Selection(
            AgentCooperationPlanningStatus.SUCCESS,
            selected=selected,
            decisions=selected_decisions,
            covered_capabilities=criteria.required_capability_ids,
            available_skills=available_skills,
        )


def agent_cooperation_planning_request_signature(request: AgentCooperationPlanningRequest) -> str:
    """Return a stable SHA-256 signature for a planning request."""

    if not isinstance(request, AgentCooperationPlanningRequest):
        raise InvalidAgentCooperationPlanningRequestError(
            "request must be AgentCooperationPlanningRequest."
        )
    return _signature(_jsonable_dataclass(request))


def _task_criteria(request: AgentCooperationPlanningRequest) -> tuple[_TaskCriteria, ...]:
    requirements = request.task_requirements or (
        AgentCooperationPlanningTaskRequirement(
            task_id=request.objective_id,
            objective_id=request.objective_id,
            execution_type=request.execution_type,
            source_agent_id=request.source_agent_id,
        ),
    )
    result: list[_TaskCriteria] = []
    for item in requirements:
        inherit = item.inherit_objective_requirements
        result.append(
            _TaskCriteria(
                task_id=item.task_id,
                objective_id=item.objective_id or request.objective_id,
                execution_type=item.execution_type if item.execution_type is not None else request.execution_type,
                source_agent_id=item.source_agent_id or request.source_agent_id,
                required_capability_ids=_merge_ids(
                    request.required_capability_ids if inherit else (),
                    item.required_capability_ids,
                ),
                preferred_capability_ids=_merge_ids(
                    request.preferred_capability_ids if inherit else (),
                    item.preferred_capability_ids,
                ),
                required_agent_types=_merge_agent_types(
                    request.required_agent_types if inherit else (),
                    item.required_agent_types,
                ),
                preferred_agent_types=_merge_agent_types(
                    request.preferred_agent_types if inherit else (),
                    item.preferred_agent_types,
                ),
                required_agent_ids=_merge_ids(
                    request.required_agent_ids if inherit else (),
                    item.required_agent_ids,
                ),
                preferred_agent_ids=_merge_ids(
                    request.preferred_agent_ids if inherit else (),
                    item.preferred_agent_ids,
                ),
                excluded_agent_ids=_merge_ids(
                    request.excluded_agent_ids if inherit else (),
                    item.excluded_agent_ids,
                ),
                required_skill_ids=_merge_ids(
                    request.required_skill_ids if inherit else (),
                    item.required_skill_ids,
                ),
                optional_skill_ids=_merge_ids(
                    request.optional_skill_ids if inherit else (),
                    item.optional_skill_ids,
                ),
                required_permission_ids=_merge_ids(
                    request.required_permission_ids if inherit else (),
                    item.required_permission_ids,
                ),
                depends_on=item.depends_on,
                output_bindings=item.output_bindings,
                order=item.order,
                priority=item.priority,
            )
        )
    return tuple(result)


def _minimal_coverage_set(
    criteria: _TaskCriteria,
    candidates: tuple[AgentDefinition, ...],
    available_skills: tuple[str, ...],
    skill_definitions: Mapping[str, SkillDefinition],
    policy: AgentCooperationPlanningPolicy,
) -> tuple[
    AgentCooperationPlanningStatus,
    tuple[AgentDefinition, ...],
    tuple[str, ...],
    tuple[str, ...],
    str | None,
]:
    required_capabilities = (
        criteria.required_capability_ids if policy.require_all_capabilities_covered else ()
    )
    required_skills = (
        criteria.required_skill_ids if policy.require_all_skills_available else ()
    )
    capability_bits = {value: index for index, value in enumerate(required_capabilities)}
    skill_bits = {
        value: index + len(capability_bits)
        for index, value in enumerate(required_skills)
    }
    objective_bit = len(capability_bits) + len(skill_bits)
    full_mask = (1 << (objective_bit + 1)) - 1
    by_id = {agent.agent_id: agent for agent in candidates}
    required = tuple(by_id[agent_id] for agent_id in criteria.required_agent_ids if agent_id in by_id)
    if len(required) != len(criteria.required_agent_ids):
        return (
            AgentCooperationPlanningStatus.NO_MATCHING_AGENTS,
            (),
            (),
            (),
            "a required agent is not compatible.",
        )
    initial_mask = 0
    for agent in required:
        initial_mask |= _coverage_mask(
            agent,
            capability_bits,
            skill_bits,
            available_skills,
            skill_definitions,
        )
        initial_mask |= 1 << objective_bit
    initial_ids = tuple(sorted(agent.agent_id for agent in required))
    states: dict[int, tuple[tuple[str, ...], ...]] = {initial_mask: (initial_ids,)}
    optional = tuple(agent for agent in candidates if agent.agent_id not in initial_ids)
    for agent in optional:
        updates: dict[int, list[tuple[str, ...]]] = {}
        agent_mask = _coverage_mask(
            agent,
            capability_bits,
            skill_bits,
            available_skills,
            skill_definitions,
        ) | (1 << objective_bit)
        for mask, selections in tuple(states.items()):
            for selection in selections:
                if len(selection) >= MAX_PLANNING_AGENTS:
                    continue
                new_selection = tuple(sorted(selection + (agent.agent_id,)))
                new_mask = mask | agent_mask
                existing = list(states.get(new_mask, ())) + updates.get(new_mask, [])
                updates[new_mask] = _best_equivalent_sets(
                    existing + [new_selection],
                    by_id,
                    criteria,
                    skill_definitions,
                )
        for mask, selections in updates.items():
            existing = list(states.get(mask, ()))
            states[mask] = tuple(
                _best_equivalent_sets(
                    existing + selections,
                    by_id,
                    criteria,
                    skill_definitions,
                )
            )
        if len(states) > MAX_PLANNING_STATES:
            return (
                AgentCooperationPlanningStatus.LIMIT_REACHED,
                (),
                (),
                (),
                "planning state limit reached.",
            )
    solutions = states.get(full_mask, ())
    if not solutions:
        covered_mask = max(states, key=lambda mask: (mask.bit_count(), -len(states[mask][0])))
        missing_capabilities = tuple(
            value for value, bit in capability_bits.items() if not covered_mask & (1 << bit)
        )
        missing_skills = tuple(
            value for value, bit in skill_bits.items() if not covered_mask & (1 << bit)
        )
        if missing_capabilities and policy.require_all_capabilities_covered:
            return (
                AgentCooperationPlanningStatus.MISSING_CAPABILITY,
                (),
                missing_capabilities,
                missing_skills,
                "required capabilities are not fully covered.",
            )
        if missing_skills and policy.require_all_skills_available:
            return (
                AgentCooperationPlanningStatus.MISSING_SKILL,
                (),
                missing_capabilities,
                missing_skills,
                "required skills are not authorized by the selected agents.",
            )
        return (
            AgentCooperationPlanningStatus.NO_MATCHING_AGENTS,
            (),
            missing_capabilities,
            missing_skills,
            "no sufficient agent set exists.",
        )
    ranked = _best_equivalent_sets(
        list(solutions),
        by_id,
        criteria,
        skill_definitions,
    )
    if len(ranked) > 1 and policy.fail_on_ambiguous_agent:
        return (
            AgentCooperationPlanningStatus.AMBIGUOUS,
            (),
            (),
            (),
            "multiple equivalent minimal agent sets exist.",
        )
    selected_ids = min(ranked)
    selected = tuple(by_id[agent_id] for agent_id in selected_ids)
    if len(selected) > policy.max_agents:
        return (
            AgentCooperationPlanningStatus.LIMIT_REACHED,
            (),
            (),
            (),
            "selected agents exceed max_agents.",
        )
    return AgentCooperationPlanningStatus.SUCCESS, selected, (), (), None


def _best_equivalent_sets(
    selections: list[tuple[str, ...]],
    by_id: Mapping[str, AgentDefinition],
    criteria: _TaskCriteria,
    skill_definitions: Mapping[str, SkillDefinition],
) -> list[tuple[str, ...]]:
    unique = tuple(dict.fromkeys(selections))
    if not unique:
        return []
    ranks = {
        selection: _selection_rank(
            selection,
            by_id,
            criteria,
            skill_definitions,
        )
        for selection in unique
    }
    best_rank = min(ranks.values())
    return sorted(selection for selection in unique if ranks[selection] == best_rank)[:2]


def _selection_rank(
    selection: tuple[str, ...],
    by_id: Mapping[str, AgentDefinition],
    criteria: _TaskCriteria,
    skill_definitions: Mapping[str, SkillDefinition],
) -> tuple[int, int, int, int, int, int]:
    agents = tuple(by_id[agent_id] for agent_id in selection)
    preferred_agents = sum(agent.agent_id in criteria.preferred_agent_ids for agent in agents)
    preferred_types = sum(agent.agent_type in criteria.preferred_agent_types for agent in agents)
    preferred_capabilities = sum(
        len(set(agent.capabilities.capabilities).intersection(criteria.preferred_capability_ids))
        for agent in agents
    )
    optional_skills = sum(
        _skill_authorized(skill_definitions[skill_id], agent)
        for agent in agents
        for skill_id in criteria.optional_skill_ids
        if skill_id in skill_definitions
    )
    priority = sum(_agent_priority(agent) for agent in agents)
    return (
        len(selection),
        -preferred_agents,
        -preferred_types,
        -preferred_capabilities,
        -optional_skills,
        -priority,
    )


def _coverage_mask(
    agent: AgentDefinition,
    capability_bits: Mapping[str, int],
    skill_bits: Mapping[str, int],
    available_skills: tuple[str, ...],
    skill_definitions: Mapping[str, SkillDefinition],
) -> int:
    mask = 0
    for capability, bit in capability_bits.items():
        if capability in agent.capabilities.capabilities:
            mask |= 1 << bit
    for skill_id, bit in skill_bits.items():
        if (
            skill_id in available_skills
            and _skill_authorized(skill_definitions[skill_id], agent)
        ):
            mask |= 1 << bit
    return mask


def _resolve_skills(
    skill_system: SkillSystem,
    criteria: _TaskCriteria,
    policy: AgentCooperationPlanningPolicy,
) -> tuple[tuple[str, ...], tuple[str, ...], Mapping[str, SkillDefinition]]:
    requested = _merge_ids(criteria.required_skill_ids, criteria.optional_skill_ids)
    if requested and not policy.allow_skill_requirements:
        return (), criteria.required_skill_ids, MappingProxyType({})
    available: list[str] = []
    missing: list[str] = []
    definitions: dict[str, SkillDefinition] = {}
    for skill_id in requested:
        result = skill_system.skill_resolver.resolve(
            SkillResolutionRequest(
                required_skill_ids=(skill_id,),
                enabled_only=True,
                require_unique_top_score=False,
                metadata={"route": "cooperation_automatic_planner"},
            )
        )
        if result.status is SkillResolutionStatus.RESOLVED:
            available.append(skill_id)
            if result.selected_skill is not None:
                definitions[skill_id] = result.selected_skill
        elif skill_id in criteria.required_skill_ids:
            missing.append(skill_id)
    if missing and not policy.require_all_skills_available:
        missing = []
    return tuple(available), tuple(missing), MappingProxyType(definitions)


def _candidate_rejection(
    agent: AgentDefinition,
    criteria: _TaskCriteria,
    policy: AgentCooperationPlanningPolicy,
    already_selected_ids: tuple[str, ...],
) -> AgentCooperationPlanningReasonCode | None:
    if not agent.enabled:
        return AgentCooperationPlanningReasonCode.DISABLED_AGENT
    if agent.agent_id in criteria.excluded_agent_ids:
        return AgentCooperationPlanningReasonCode.EXCLUDED_AGENT
    if (
        criteria.source_agent_id is not None
        and agent.agent_id == criteria.source_agent_id
        and criteria.execution_type in (
            AgentCooperationExecutionType.DELEGATION,
            AgentCooperationExecutionType.DELEGATION_CHAIN,
            AgentCooperationExecutionType.COORDINATED_CHAINS,
        )
    ):
        return AgentCooperationPlanningReasonCode.EXCLUDED_AGENT
    if not policy.allow_agent_reuse and agent.agent_id in already_selected_ids:
        return AgentCooperationPlanningReasonCode.EXCLUDED_AGENT
    if criteria.required_agent_types and agent.agent_type not in criteria.required_agent_types:
        return AgentCooperationPlanningReasonCode.MISSING_CAPABILITY
    if any(not bool(getattr(agent.permissions, item)) for item in criteria.required_permission_ids):
        return AgentCooperationPlanningReasonCode.MISSING_PERMISSION
    return None


def _candidate_decision(
    criteria: _TaskCriteria,
    agent: AgentDefinition,
    available_skills: tuple[str, ...],
    skill_definitions: Mapping[str, SkillDefinition],
) -> AgentCooperationPlanningDecision:
    covered_capabilities = tuple(
        value for value in criteria.required_capability_ids if value in agent.capabilities.capabilities
    )
    covered_skills = tuple(
        skill_id
        for skill_id in criteria.required_skill_ids
        if skill_id in available_skills
        and _skill_authorized(skill_definitions[skill_id], agent)
    )
    reasons: list[AgentCooperationPlanningReasonCode] = []
    if agent.agent_id in criteria.required_agent_ids:
        reasons.append(AgentCooperationPlanningReasonCode.REQUIRED_AGENT)
    if criteria.required_agent_types and agent.agent_type in criteria.required_agent_types:
        reasons.append(AgentCooperationPlanningReasonCode.REQUIRED_TYPE_MATCH)
    if covered_capabilities:
        reasons.append(AgentCooperationPlanningReasonCode.REQUIRED_CAPABILITY_MATCH)
    if criteria.required_permission_ids:
        reasons.append(AgentCooperationPlanningReasonCode.REQUIRED_PERMISSION_MATCH)
    if covered_skills:
        reasons.append(AgentCooperationPlanningReasonCode.REQUIRED_SKILL_AVAILABLE)
    if agent.agent_id in criteria.preferred_agent_ids:
        reasons.append(AgentCooperationPlanningReasonCode.PREFERRED_AGENT)
    if agent.agent_type in criteria.preferred_agent_types:
        reasons.append(AgentCooperationPlanningReasonCode.PREFERRED_TYPE_MATCH)
    if set(agent.capabilities.capabilities).intersection(criteria.preferred_capability_ids):
        reasons.append(AgentCooperationPlanningReasonCode.PREFERRED_CAPABILITY_MATCH)
    return AgentCooperationPlanningDecision(
        criteria.task_id,
        agent.agent_id,
        AgentCooperationPlanningDecisionStatus.ACCEPTED,
        covered_capability_ids=covered_capabilities,
        covered_skill_ids=covered_skills,
        reasons=tuple(reasons),
    )


def _skill_authorized(skill: SkillDefinition, agent: AgentDefinition) -> bool:
    denied = _metadata_ids(agent.metadata.get("denied_skill_ids"))
    required = _metadata_ids(agent.metadata.get("required_skill_ids"))
    allowed = _metadata_ids(agent.metadata.get("allowed_skill_ids"))
    if skill.skill_id in denied:
        return False
    if required and skill.skill_id not in required:
        return False
    if allowed and skill.skill_id not in allowed:
        return False
    if skill.allowed_agent_types and agent.agent_type not in skill.allowed_agent_types:
        return False
    if any(item not in agent.capabilities.capabilities for item in skill.required_capability_ids):
        return False
    if any(not bool(getattr(agent.permissions, item)) for item in skill.required_permission_ids):
        return False
    return True


def _metadata_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    return tuple(validate_skill_id(part.strip()) for part in value.split(",") if part.strip())


def _build_task(
    criteria: _TaskCriteria,
    selected: tuple[AgentDefinition, ...],
    request: AgentCooperationPlanningRequest,
    policy: AgentCooperationPlanningPolicy,
) -> AgentCooperationTask:
    execution_type = criteria.execution_type
    if execution_type is None:
        execution_type = (
            AgentCooperationExecutionType.SINGLE_AGENT
            if len(selected) == 1
            else AgentCooperationExecutionType.MULTI_AGENT
        )
    if execution_type is AgentCooperationExecutionType.MULTI_AGENT and not policy.allow_multi_agent_tasks:
        raise AgentCooperationPlanningError("multi-agent tasks are disabled by policy.")
    if execution_type in (
        AgentCooperationExecutionType.DELEGATION,
        AgentCooperationExecutionType.DELEGATION_CHAIN,
        AgentCooperationExecutionType.COORDINATED_CHAINS,
    ) and not policy.allow_delegation_tasks:
        raise AgentCooperationPlanningError("delegation tasks are disabled by policy.")
    common = {
        "task_id": criteria.task_id,
        "objective_id": criteria.objective_id,
        "execution_type": execution_type,
        "required_agent_types": criteria.required_agent_types,
        "required_capability_ids": criteria.required_capability_ids,
        "required_permission_ids": criteria.required_permission_ids,
        "preferred_agent_ids": criteria.preferred_agent_ids,
        "excluded_agent_ids": criteria.excluded_agent_ids,
        "structured_input": request.structured_input,
        "shared_context": request.shared_context,
        "metadata": request.metadata,
        "required_skill_ids": criteria.required_skill_ids,
        "dependency_output_bindings": criteria.output_bindings,
        "order": criteria.order,
        "priority": criteria.priority,
    }
    selected_ids = tuple(agent.agent_id for agent in selected)
    if execution_type is AgentCooperationExecutionType.SINGLE_AGENT:
        if len(selected) != 1:
            raise AgentCooperationPlanningError("SINGLE_AGENT requires exactly one selected agent.")
        return AgentCooperationTask(agent_id=selected[0].agent_id, **common)
    if execution_type is AgentCooperationExecutionType.MULTI_AGENT:
        if len(selected) < 2:
            raise AgentCooperationPlanningError("MULTI_AGENT requires at least two selected agents.")
        return AgentCooperationTask(
            multi_agent_request=MultiAgentExecutionRequest(
                required_agent_ids=selected_ids,
                payload=request.structured_input,
                shared_context=request.shared_context,
                task_id=criteria.task_id,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                policy=MultiAgentExecutionPolicy(
                    min_agents=len(selected),
                    max_agents=len(selected),
                ),
            ),
            **common,
        )
    source = criteria.source_agent_id
    if source is None:
        raise AgentCooperationPlanningError("delegation execution requires source_agent_id.")
    if execution_type is AgentCooperationExecutionType.DELEGATION:
        if len(selected) != 1:
            raise AgentCooperationPlanningError("DELEGATION requires exactly one selected target.")
        return AgentCooperationTask(
            delegation_request=AgentDelegationRequest(
                origin_agent_id=source,
                target_agent_id=selected[0].agent_id,
                structured_input=request.structured_input,
                shared_context=request.shared_context,
                execution_id=request.execution_id,
                policy=AgentDelegationPolicy(
                    enabled=True,
                    allowed_target_agent_ids=(selected[0].agent_id,),
                    propagate_structured_input=True,
                    propagate_shared_context=True,
                ),
            ),
            **common,
        )
    if execution_type is AgentCooperationExecutionType.DELEGATION_CHAIN:
        steps: list[AgentDelegationChainStep] = []
        current_source = source
        for agent in selected:
            steps.append(
                AgentDelegationChainStep(
                    source_agent_id=current_source,
                    target_agent_id=agent.agent_id,
                    execution_required_capability_ids=tuple(
                        value for value in criteria.required_capability_ids
                        if value in agent.capabilities.capabilities
                    ),
                    execution_required_permission_ids=criteria.required_permission_ids,
                )
            )
            current_source = agent.agent_id
        return AgentCooperationTask(
            delegation_chain_request=AgentDelegationChainRequest(
                steps=tuple(steps),
                policy=AgentDelegationChainPolicy(
                    enabled=True,
                    max_steps=max(1, len(steps)),
                    max_depth=max(1, len(steps)),
                    max_total_delegations=max(1, len(steps)),
                ),
                initial_input=request.structured_input,
                shared_context=request.shared_context,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
            ),
            **common,
        )
    chains = tuple(
        AgentDelegationCoordinationChain(
            chain_id=f"chain.{criteria.task_id}.{index}",
            chain_request=AgentDelegationChainRequest(
                steps=(
                    AgentDelegationChainStep(
                        source_agent_id=source,
                        target_agent_id=agent.agent_id,
                        execution_required_capability_ids=tuple(
                            value for value in criteria.required_capability_ids
                            if value in agent.capabilities.capabilities
                        ),
                        execution_required_permission_ids=criteria.required_permission_ids,
                    ),
                ),
                policy=AgentDelegationChainPolicy(
                    enabled=True,
                    max_steps=1,
                    max_depth=1,
                    max_total_delegations=1,
                ),
            ),
        )
        for index, agent in enumerate(selected)
    )
    return AgentCooperationTask(
        coordinated_chains_request=AgentDelegationCoordinationRequest(
            source_agent_id=source,
            plan=AgentDelegationCoordinationPlan(
                plan_id=f"coord.{criteria.task_id}",
                chains=chains,
            ),
            policy=AgentDelegationCoordinationPolicy(
                enabled=True,
                max_chains=len(chains),
                max_total_steps=len(chains),
            ),
            structured_input=request.structured_input,
            shared_context=request.shared_context,
            execution_id=request.execution_id,
        ),
        **common,
    )


def _validate_request_limits(
    request: AgentCooperationPlanningRequest,
) -> tuple[AgentCooperationPlanningStatus, str, str] | None:
    policy = request.policy
    criteria = _task_criteria(request)
    if len(criteria) > policy.max_tasks:
        return AgentCooperationPlanningStatus.LIMIT_REACHED, "MAX_TASKS", "maximum tasks reached."
    dependency_count = sum(len(item.depends_on) for item in criteria)
    if dependency_count > policy.max_dependencies:
        return (
            AgentCooperationPlanningStatus.LIMIT_REACHED,
            "MAX_DEPENDENCIES",
            "maximum dependencies reached.",
        )
    task_ids = {item.task_id for item in criteria}
    if any(source not in task_ids for item in criteria for source in item.depends_on):
        return (
            AgentCooperationPlanningStatus.INVALID_REQUEST,
            "INVALID_DEPENDENCY",
            "a dependency references an unknown task.",
        )
    required_skills = {skill for item in criteria for skill in item.required_skill_ids}
    if len(required_skills) > policy.max_required_skills:
        return (
            AgentCooperationPlanningStatus.LIMIT_REACHED,
            "MAX_REQUIRED_SKILLS",
            "maximum required skills reached.",
        )
    if len({capability for item in criteria for capability in item.required_capability_ids}) > MAX_PLANNING_REQUIREMENTS:
        return (
            AgentCooperationPlanningStatus.LIMIT_REACHED,
            "MAX_REQUIREMENTS",
            "maximum capability requirements reached.",
        )
    try:
        depth = _criteria_depth(criteria)
    except InvalidAgentCooperationPlanningRequestError:
        return (
            AgentCooperationPlanningStatus.INVALID_REQUEST,
            "CYCLE_DETECTED",
            "task requirements contain a cycle.",
        )
    if depth > policy.max_plan_depth:
        return AgentCooperationPlanningStatus.LIMIT_REACHED, "MAX_PLAN_DEPTH", "maximum plan depth reached."
    return None


def _criteria_depth(criteria: tuple[_TaskCriteria, ...]) -> int:
    dependencies = {item.task_id: item.depends_on for item in criteria}
    visiting: set[str] = set()
    visited: set[str] = set()
    depths: dict[str, int] = {}

    def visit(task_id: str) -> int:
        if task_id in visiting:
            raise InvalidAgentCooperationPlanningRequestError("task requirements contain a cycle.")
        if task_id in visited:
            return depths[task_id]
        visiting.add(task_id)
        depth = 1 + max((visit(source) for source in dependencies[task_id]), default=0)
        visiting.remove(task_id)
        visited.add(task_id)
        depths[task_id] = depth
        return depth

    return max((visit(task_id) for task_id in dependencies), default=0)


def _normalize_requirement_fields(value: object) -> None:
    for name in ("required_capability_ids", "preferred_capability_ids"):
        object.__setattr__(value, name, _identifier_tuple(getattr(value, name), name))
    for name in ("required_agent_types", "preferred_agent_types"):
        object.__setattr__(value, name, _agent_type_tuple(getattr(value, name), name))
    for name in ("required_agent_ids", "preferred_agent_ids", "excluded_agent_ids"):
        object.__setattr__(value, name, _agent_id_tuple(getattr(value, name), name))
    for name in ("required_skill_ids", "optional_skill_ids"):
        object.__setattr__(value, name, _skill_id_tuple(getattr(value, name), name))
    object.__setattr__(
        value,
        "required_permission_ids",
        _permission_tuple(getattr(value, "required_permission_ids"), "required_permission_ids"),
    )
    if set(value.required_agent_ids).intersection(value.excluded_agent_ids):
        raise InvalidAgentCooperationPlanningRequestError(
            "required and excluded agent ids cannot overlap."
        )


def _result(
    status: AgentCooperationPlanningStatus,
    request_signature: str,
    *,
    plan: AgentCooperationPlan | None = None,
    plan_signature: str | None = None,
    decisions: tuple[AgentCooperationPlanningDecision, ...] = (),
    selected_agent_ids: tuple[str, ...] = (),
    covered_capability_ids: tuple[str, ...] = (),
    missing_capability_ids: tuple[str, ...] = (),
    available_skill_ids: tuple[str, ...] = (),
    missing_skill_ids: tuple[str, ...] = (),
    created_task_ids: tuple[str, ...] = (),
    created_dependencies: tuple[AgentCooperationDependency, ...] = (),
    events: Sequence[AgentCooperationPlanningEvent] = (),
    metrics: Mapping[str, int] | None = None,
    error_code: str | None = None,
    safe_message: str | None = None,
) -> AgentCooperationPlanningResult:
    evaluated = tuple(sorted({item.agent_id for item in decisions}))
    accepted = tuple(
        sorted({
            item.agent_id
            for item in decisions
            if item.status in (
                AgentCooperationPlanningDecisionStatus.ACCEPTED,
                AgentCooperationPlanningDecisionStatus.SELECTED,
            )
        })
    )
    rejected = tuple(
        sorted({
            item.agent_id
            for item in decisions
            if item.status is AgentCooperationPlanningDecisionStatus.REJECTED
        })
    )
    return AgentCooperationPlanningResult(
        status=status,
        planning_request_signature=request_signature,
        plan=plan,
        plan_signature=plan_signature,
        decisions=decisions,
        evaluated_agent_ids=evaluated,
        accepted_agent_ids=accepted,
        rejected_agent_ids=rejected,
        selected_agent_ids=selected_agent_ids,
        covered_capability_ids=covered_capability_ids,
        missing_capability_ids=missing_capability_ids,
        available_skill_ids=available_skill_ids,
        missing_skill_ids=missing_skill_ids,
        created_task_ids=created_task_ids,
        created_dependencies=created_dependencies,
        events=tuple(events),
        metrics=_base_metrics() if metrics is None else metrics,
        error_code=error_code,
        safe_message=safe_message,
    )


def _event(
    events: list[AgentCooperationPlanningEvent],
    name: str,
    status: str,
    **details: object,
) -> None:
    if len(events) < MAX_PLANNING_EVENTS:
        events.append(AgentCooperationPlanningEvent(name, status, details))


def _base_metrics() -> dict[str, int]:
    return {
        "cooperation_planning_requests": 0,
        "cooperation_planning_succeeded": 0,
        "cooperation_planning_failed": 0,
        "cooperation_planning_ambiguous": 0,
        "cooperation_planning_limits_reached": 0,
        "cooperation_candidates_evaluated": 0,
        "cooperation_candidates_accepted": 0,
        "cooperation_candidates_rejected": 0,
        "cooperation_agents_selected": 0,
        "cooperation_tasks_created": 0,
        "cooperation_dependencies_created": 0,
        "cooperation_required_capabilities_covered": 0,
        "cooperation_required_capabilities_missing": 0,
        "cooperation_required_skills_available": 0,
        "cooperation_required_skills_missing": 0,
    }


def _safe_mapping(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentCooperationPlanningRequestError(f"{field_name} must be a mapping.")
    counter = {"items": 0}
    return _safe_mapping_inner(value, field_name, 0, counter)


def _safe_mapping_inner(
    value: Mapping[str, object],
    field_name: str,
    depth: int,
    counter: dict[str, int],
) -> dict[str, object]:
    if depth > MAX_PLANNING_VALUE_DEPTH:
        raise InvalidAgentCooperationPlanningRequestError(f"{field_name} exceeds maximum depth.")
    if len(value) > MAX_PLANNING_METADATA_ITEMS:
        raise InvalidAgentCooperationPlanningRequestError(f"{field_name} has too many items.")
    result: dict[str, object] = {}
    for raw_key in sorted(value, key=lambda item: str(item)):
        key = _key(raw_key, field_name)
        if _is_sensitive_key(key):
            raise InvalidAgentCooperationPlanningRequestError(
                f"{field_name} contains a sensitive key."
            )
        if _is_dynamic_code_key(key):
            raise InvalidAgentCooperationPlanningRequestError(
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
    if depth > MAX_PLANNING_VALUE_DEPTH:
        raise InvalidAgentCooperationPlanningRequestError(f"{field_name} exceeds maximum depth.")
    counter["items"] += 1
    if counter["items"] > MAX_PLANNING_TOTAL_ITEMS:
        raise InvalidAgentCooperationPlanningRequestError(f"{field_name} exceeds total item limit.")
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentCooperationPlanningRequestError(f"{field_name} floats must be finite.")
        return value
    if isinstance(value, str):
        if len(value) > MAX_PLANNING_STRING_LENGTH:
            raise InvalidAgentCooperationPlanningRequestError(f"{field_name} string exceeds length limit.")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_mapping_inner(value, field_name, depth, counter))
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_PLANNING_SEQUENCE_ITEMS:
            raise InvalidAgentCooperationPlanningRequestError(f"{field_name} sequence exceeds item limit.")
        return tuple(_safe_value(item, field_name, depth + 1, counter) for item in value)
    raise InvalidAgentCooperationPlanningRequestError(f"{field_name} contains an unsupported object.")


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
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _merge_ids(first: Iterable[str], second: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(tuple(first) + tuple(second)))


def _merge_agent_types(
    first: Iterable[AgentType],
    second: Iterable[AgentType],
) -> tuple[AgentType, ...]:
    return tuple(dict.fromkeys(tuple(first) + tuple(second)))


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise InvalidAgentCooperationPlanningRequestError(
            f"{field_name} contains unsupported characters."
        )
    return value


def _key(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidAgentCooperationPlanningRequestError(f"{field_name} keys must be strings.")
    return _identifier(value, f"{field_name} key")


def _identifier_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentCooperationPlanningRequestError(f"{field_name} must be an iterable.")
    result = tuple(dict.fromkeys(_identifier(value, field_name) for value in values))
    if len(result) > MAX_PLANNING_SEQUENCE_ITEMS:
        raise InvalidAgentCooperationPlanningRequestError(f"{field_name} exceeds item limit.")
    return result


def _agent_id_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentCooperationPlanningRequestError(f"{field_name} must be an iterable.")
    return tuple(dict.fromkeys(validate_agent_id(value) for value in values))


def _skill_id_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentCooperationPlanningRequestError(f"{field_name} must be an iterable.")
    return tuple(dict.fromkeys(validate_skill_id(value) for value in values))


def _agent_type_tuple(
    values: Iterable[AgentType | str],
    field_name: str,
) -> tuple[AgentType, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentCooperationPlanningRequestError(f"{field_name} must be an iterable.")
    result: list[AgentType] = []
    for value in values:
        try:
            normalized = value if isinstance(value, AgentType) else AgentType(value)
        except (TypeError, ValueError) as error:
            raise InvalidAgentCooperationPlanningRequestError(
                f"{field_name} contains an invalid agent type."
            ) from error
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def _permission_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    allowed = {
        "can_read_project",
        "can_write_files",
        "can_execute_tools",
        "can_modify_memory",
        "can_use_network",
        "requires_confirmation",
    }
    result = _identifier_tuple(values, field_name)
    if any(value not in allowed for value in result):
        raise InvalidAgentCooperationPlanningRequestError(
            f"{field_name} contains an invalid permission."
        )
    return result


def _bounded_int(value: int, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise InvalidAgentCooperationPlanningRequestError(
            f"{field_name} is outside the allowed range."
        )
    return value


def _metric_mapping(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise InvalidAgentCooperationPlanningRequestError("metrics must be a mapping.")
    result: dict[str, int] = {}
    for key, item in value.items():
        normalized = _identifier(key, "metric name")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise InvalidAgentCooperationPlanningRequestError(
                "metric values must be non-negative integers."
            )
        result[normalized] = item
    return result


def _agent_priority(agent: AgentDefinition) -> int:
    value = agent.metadata.get("planning_priority", 0)
    return value if type(value) is int and -1_000_000 <= value <= 1_000_000 else 0


def _safe_message(value: str) -> str:
    if not isinstance(value, str):
        return "planning failed."
    lowered = value.lower()
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return "planning failed."
    return " ".join(value.split())[:240] or "planning failed."


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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InvalidAgentCooperationPlanningRequestError(
            f"{field_name} must be a lowercase SHA-256 signature."
        )


def _objective_type(
    value: AgentCooperationObjectiveType | str,
) -> AgentCooperationObjectiveType:
    if isinstance(value, AgentCooperationObjectiveType):
        return value
    try:
        return AgentCooperationObjectiveType(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentCooperationPlanningRequestError("invalid objective_type.") from error


def _execution_type(
    value: AgentCooperationExecutionType | str,
) -> AgentCooperationExecutionType:
    if isinstance(value, AgentCooperationExecutionType):
        return value
    try:
        return AgentCooperationExecutionType(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentCooperationPlanningRequestError("invalid execution_type.") from error


def _planning_status(
    value: AgentCooperationPlanningStatus | str,
) -> AgentCooperationPlanningStatus:
    if isinstance(value, AgentCooperationPlanningStatus):
        return value
    try:
        return AgentCooperationPlanningStatus(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentCooperationPlanningRequestError("invalid planning status.") from error


def _decision_status(
    value: AgentCooperationPlanningDecisionStatus | str,
) -> AgentCooperationPlanningDecisionStatus:
    if isinstance(value, AgentCooperationPlanningDecisionStatus):
        return value
    try:
        return AgentCooperationPlanningDecisionStatus(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentCooperationPlanningRequestError("invalid decision status.") from error


def _reason_code(
    value: AgentCooperationPlanningReasonCode | str,
) -> AgentCooperationPlanningReasonCode:
    if isinstance(value, AgentCooperationPlanningReasonCode):
        return value
    try:
        return AgentCooperationPlanningReasonCode(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentCooperationPlanningRequestError("invalid planning reason.") from error

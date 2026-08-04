"""Declarative, bounded cooperation plans for specialized Atlas agents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType

from core.agent_context import AgentContextBuilder
from core.agent_delegation import (
    AgentDelegationRequest,
    AgentDelegationResult,
    AgentDelegationService,
    AgentDelegationStatus,
)
from core.agent_delegation_chain import (
    AgentDelegationChainRequest,
    AgentDelegationChainResult,
    AgentDelegationChainService,
    AgentDelegationChainStatus,
)
from core.agent_delegation_coordinator import (
    AgentDelegationCoordinationRequest,
    AgentDelegationCoordinationResult,
    AgentDelegationCoordinationStatus,
    AgentDelegationCoordinator,
)
from core.agent_executor import (
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentExecutionStatus,
    AgentExecutor,
)
from core.agent_registry import AgentDefinition, AgentRegistry, AgentType, validate_agent_id
from core.agent_resolver import (
    AgentResolutionRequest,
    AgentResolutionStatus,
    AgentResolver,
)
from core.multi_agent import (
    MultiAgentCoordinator,
    MultiAgentExecutionRequest,
    MultiAgentExecutionResult,
    MultiAgentExecutionStatus,
)
from core.skill_registry import SkillDefinition, SkillNotFoundError, validate_skill_id
from core.skill_executor import SkillExecutionRequest, SkillExecutionResult, SkillExecutionStatus
from core.skill_resolver import SkillResolutionRequest, SkillResolutionStatus
from core.skill_system import SkillSystem


MAX_COOPERATION_TASKS = 100
MAX_COOPERATION_DEPENDENCIES = 400
MAX_COOPERATION_DEPTH = 32
MAX_COOPERATION_EXECUTIONS = 100
MAX_COOPERATION_OUTPUT_ITEMS = 512
MAX_COOPERATION_PROPAGATED_ITEMS = 256
MAX_COOPERATION_METADATA_ITEMS = 32
MAX_COOPERATION_SEQUENCE_ITEMS = 128
MAX_COOPERATION_TOTAL_ITEMS = 1_024
MAX_COOPERATION_VALUE_DEPTH = 8
MAX_COOPERATION_STRING_LENGTH = 2_000
MAX_COOPERATION_EVENTS = 1_024
MAX_LOGICAL_TIMEOUT = 1_000_000

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+){1,2}$")
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


class AgentCooperationPlanError(RuntimeError):
    """Base error for declarative cooperation plans."""


class InvalidAgentCooperationPlanError(AgentCooperationPlanError):
    """Raised when a cooperation plan or request is malformed."""


class AgentCooperationDependencyError(AgentCooperationPlanError):
    """Raised when declared dependencies are invalid."""


class AgentCooperationCycleError(AgentCooperationDependencyError):
    """Raised when a cooperation plan contains a cycle."""


class AgentCooperationExecutionError(AgentCooperationPlanError):
    """Raised when a task cannot be dispatched safely."""


class AgentCooperationExecutionType(str, Enum):
    """Existing execution boundary selected by one task."""

    SINGLE_AGENT = "SINGLE_AGENT"
    DELEGATION = "DELEGATION"
    DELEGATION_CHAIN = "DELEGATION_CHAIN"
    MULTI_AGENT = "MULTI_AGENT"
    COORDINATED_CHAINS = "COORDINATED_CHAINS"


class AgentCooperationFailureMode(str, Enum):
    """Deterministic plan failure behavior."""

    STOP_ON_FIRST_FAILURE = "STOP_ON_FIRST_FAILURE"
    CONTINUE_INDEPENDENT_TASKS = "CONTINUE_INDEPENDENT_TASKS"
    REQUIRE_ALL_SUCCESS = "REQUIRE_ALL_SUCCESS"
    REQUIRE_MINIMUM_SUCCESS = "REQUIRE_MINIMUM_SUCCESS"


class AgentCooperationPlanStatus(str, Enum):
    """Terminal statuses for a cooperation plan."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INVALID_PLAN = "INVALID_PLAN"
    LIMIT_REACHED = "LIMIT_REACHED"
    CYCLE_DETECTED = "CYCLE_DETECTED"
    MINIMUM_SUCCESS_NOT_REACHED = "MINIMUM_SUCCESS_NOT_REACHED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentCooperationTaskStatus(str, Enum):
    """Terminal statuses for one declared cooperation task."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"
    AGENT_RESOLUTION_FAILED = "AGENT_RESOLUTION_FAILED"
    AGENT_EXECUTION_FAILED = "AGENT_EXECUTION_FAILED"
    SKILL_AUTHORIZATION_FAILED = "SKILL_AUTHORIZATION_FAILED"
    SKILL_EXECUTION_FAILED = "SKILL_EXECUTION_FAILED"
    LIMIT_REACHED = "LIMIT_REACHED"
    OUTPUT_BINDING_FAILED = "OUTPUT_BINDING_FAILED"


@dataclass(frozen=True, slots=True)
class AgentCooperationPlanPolicy:
    """Opt-in limits and failure rules for one plan execution."""

    enabled: bool = False
    max_tasks: int = 20
    max_dependencies: int = 80
    max_depth: int = 10
    max_total_executions: int = 20
    max_output_items: int = 256
    max_propagated_items: int = 128
    failure_mode: AgentCooperationFailureMode | str = AgentCooperationFailureMode.STOP_ON_FIRST_FAILURE
    require_all_success: bool = False
    minimum_successful_tasks: int = 1
    allow_partial_success: bool = False
    allow_skipped_tasks: bool = False
    propagate_dependency_outputs: bool = False
    fail_on_missing_dependency_output: bool = True
    stop_on_blocked_task: bool = True

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "require_all_success",
            "allow_partial_success",
            "allow_skipped_tasks",
            "propagate_dependency_outputs",
            "fail_on_missing_dependency_output",
            "stop_on_blocked_task",
        ):
            if type(getattr(self, name)) is not bool:
                raise InvalidAgentCooperationPlanError(f"{name} must be a bool.")
        object.__setattr__(self, "max_tasks", _bounded_int(self.max_tasks, "max_tasks", MAX_COOPERATION_TASKS))
        object.__setattr__(
            self,
            "max_dependencies",
            _bounded_int(self.max_dependencies, "max_dependencies", MAX_COOPERATION_DEPENDENCIES),
        )
        object.__setattr__(self, "max_depth", _bounded_int(self.max_depth, "max_depth", MAX_COOPERATION_DEPTH))
        object.__setattr__(
            self,
            "max_total_executions",
            _bounded_int(self.max_total_executions, "max_total_executions", MAX_COOPERATION_EXECUTIONS),
        )
        object.__setattr__(
            self,
            "max_output_items",
            _bounded_int(self.max_output_items, "max_output_items", MAX_COOPERATION_OUTPUT_ITEMS),
        )
        object.__setattr__(
            self,
            "max_propagated_items",
            _bounded_int(
                self.max_propagated_items,
                "max_propagated_items",
                MAX_COOPERATION_PROPAGATED_ITEMS,
            ),
        )
        object.__setattr__(self, "failure_mode", _failure_mode(self.failure_mode))
        object.__setattr__(
            self,
            "minimum_successful_tasks",
            _bounded_int(self.minimum_successful_tasks, "minimum_successful_tasks", MAX_COOPERATION_TASKS),
        )
        if self.require_all_success and self.failure_mode is AgentCooperationFailureMode.REQUIRE_MINIMUM_SUCCESS:
            raise InvalidAgentCooperationPlanError("require_all_success conflicts with REQUIRE_MINIMUM_SUCCESS.")


@dataclass(frozen=True, slots=True)
class AgentCooperationDependency:
    """Explicit directed dependency from one task to another."""

    prerequisite_task_id: str
    dependent_task_id: str
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "prerequisite_task_id", _identifier(self.prerequisite_task_id, "prerequisite_task_id"))
        object.__setattr__(self, "dependent_task_id", _identifier(self.dependent_task_id, "dependent_task_id"))
        if type(self.required) is not bool:
            raise AgentCooperationDependencyError("required must be a bool.")
        if self.prerequisite_task_id == self.dependent_task_id:
            raise AgentCooperationDependencyError("self dependencies are not allowed.")


@dataclass(frozen=True, slots=True)
class AgentCooperationOutputBinding:
    """Safe mapping from a dependency result into task input or context."""

    source_task_id: str
    source_path: tuple[str, ...] = ("result",)
    target_path: tuple[str, ...] = ("input", "dependency_result")
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_task_id", _identifier(self.source_task_id, "source_task_id"))
        object.__setattr__(self, "source_path", _path(self.source_path, "source_path"))
        object.__setattr__(self, "target_path", _path(self.target_path, "target_path"))
        if self.source_path[0] != "result":
            raise InvalidAgentCooperationPlanError("source_path must start with result.")
        if self.target_path[0] not in ("input", "context") or len(self.target_path) < 2:
            raise InvalidAgentCooperationPlanError("target_path must start with input or context and name a destination.")
        if type(self.required) is not bool:
            raise InvalidAgentCooperationPlanError("binding required must be a bool.")


@dataclass(frozen=True, slots=True)
class AgentCooperationTask:
    """One immutable task in a fully declared cooperation plan."""

    task_id: str
    objective_id: str
    execution_type: AgentCooperationExecutionType | str = AgentCooperationExecutionType.SINGLE_AGENT
    agent_id: str | None = None
    required_agent_types: tuple[AgentType | str, ...] = ()
    required_capability_ids: tuple[str, ...] = ()
    required_permission_ids: tuple[str, ...] = ()
    preferred_agent_ids: tuple[str, ...] = ()
    excluded_agent_ids: tuple[str, ...] = ()
    enabled_only: bool = True
    structured_input: Mapping[str, object] = field(default_factory=dict)
    shared_context: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    required_skill_ids: tuple[str, ...] = ()
    dependency_output_bindings: tuple[AgentCooperationOutputBinding, ...] = ()
    logical_timeout_limit: int = 1
    continue_on_failure: bool = False
    order: int = 0
    priority: int = 0
    delegation_request: AgentDelegationRequest | None = None
    delegation_chain_request: AgentDelegationChainRequest | None = None
    multi_agent_request: MultiAgentExecutionRequest | None = None
    coordinated_chains_request: AgentDelegationCoordinationRequest | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(self, "objective_id", _identifier(self.objective_id, "objective_id"))
        object.__setattr__(self, "execution_type", _execution_type(self.execution_type))
        if self.agent_id is not None:
            object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        object.__setattr__(self, "required_agent_types", _agent_type_tuple(self.required_agent_types, "required_agent_types"))
        object.__setattr__(
            self,
            "required_capability_ids",
            _identifier_tuple(self.required_capability_ids, "required_capability_ids"),
        )
        object.__setattr__(
            self,
            "required_permission_ids",
            _permission_tuple(self.required_permission_ids, "required_permission_ids"),
        )
        object.__setattr__(self, "preferred_agent_ids", _agent_id_tuple(self.preferred_agent_ids, "preferred_agent_ids"))
        object.__setattr__(self, "excluded_agent_ids", _agent_id_tuple(self.excluded_agent_ids, "excluded_agent_ids"))
        if type(self.enabled_only) is not bool or type(self.continue_on_failure) is not bool:
            raise InvalidAgentCooperationPlanError("enabled_only and continue_on_failure must be bool.")
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
        object.__setattr__(self, "required_skill_ids", _skill_id_tuple(self.required_skill_ids))
        bindings = tuple(self.dependency_output_bindings)
        if not all(isinstance(binding, AgentCooperationOutputBinding) for binding in bindings):
            raise InvalidAgentCooperationPlanError(
                "dependency_output_bindings must contain AgentCooperationOutputBinding values."
            )
        if len({binding.target_path for binding in bindings}) != len(bindings):
            raise InvalidAgentCooperationPlanError("binding target paths must be unique.")
        object.__setattr__(self, "dependency_output_bindings", bindings)
        object.__setattr__(
            self,
            "logical_timeout_limit",
            _bounded_int(self.logical_timeout_limit, "logical_timeout_limit", MAX_LOGICAL_TIMEOUT),
        )
        for name in ("order", "priority"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or abs(value) > MAX_LOGICAL_TIMEOUT:
                raise InvalidAgentCooperationPlanError(f"{name} is outside the allowed range.")
        self._validate_execution_xor()

    def _validate_execution_xor(self) -> None:
        requests = {
            AgentCooperationExecutionType.DELEGATION: self.delegation_request,
            AgentCooperationExecutionType.DELEGATION_CHAIN: self.delegation_chain_request,
            AgentCooperationExecutionType.MULTI_AGENT: self.multi_agent_request,
            AgentCooperationExecutionType.COORDINATED_CHAINS: self.coordinated_chains_request,
        }
        supplied = tuple(key for key, value in requests.items() if value is not None)
        if self.execution_type is AgentCooperationExecutionType.SINGLE_AGENT:
            if supplied:
                raise InvalidAgentCooperationPlanError("SINGLE_AGENT cannot contain another execution request.")
            if self.agent_id is None and not (
                self.required_agent_types or self.required_capability_ids or self.required_permission_ids
            ):
                raise InvalidAgentCooperationPlanError(
                    "SINGLE_AGENT requires agent_id or declarative selection criteria."
                )
            return
        if supplied != (self.execution_type,):
            raise InvalidAgentCooperationPlanError(
                "task must contain exactly the request matching its execution_type."
            )
        if self.agent_id is not None:
            raise InvalidAgentCooperationPlanError("non-single tasks cannot also declare agent_id.")


@dataclass(frozen=True, slots=True)
class AgentCooperationPlan:
    """Complete predeclared DAG of cooperative tasks."""

    plan_id: str
    tasks: Sequence[AgentCooperationTask]
    dependencies: Sequence[AgentCooperationDependency] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "1.0"
    execution_strategy: str = "SEQUENTIAL"
    policy: AgentCooperationPlanPolicy = field(default_factory=AgentCooperationPlanPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        if self.execution_strategy != "SEQUENTIAL":
            raise InvalidAgentCooperationPlanError("execution_strategy must be SEQUENTIAL.")
        if not isinstance(self.policy, AgentCooperationPlanPolicy):
            raise InvalidAgentCooperationPlanError("policy must be AgentCooperationPlanPolicy.")
        if isinstance(self.tasks, (str, bytes)) or not isinstance(self.tasks, Sequence):
            raise InvalidAgentCooperationPlanError("tasks must be a sequence.")
        tasks = tuple(self.tasks)
        if not tasks:
            raise InvalidAgentCooperationPlanError("plan tasks cannot be empty.")
        if len(tasks) > MAX_COOPERATION_TASKS:
            raise InvalidAgentCooperationPlanError("tasks exceeds the absolute limit.")
        if not all(isinstance(task, AgentCooperationTask) for task in tasks):
            raise InvalidAgentCooperationPlanError("tasks must contain AgentCooperationTask values.")
        task_ids = tuple(task.task_id for task in tasks)
        if len(set(task_ids)) != len(task_ids):
            raise InvalidAgentCooperationPlanError("task_id values must be unique.")
        object.__setattr__(self, "tasks", tasks)
        if isinstance(self.dependencies, (str, bytes)) or not isinstance(self.dependencies, Sequence):
            raise AgentCooperationDependencyError("dependencies must be a sequence.")
        dependencies = tuple(self.dependencies)
        if len(dependencies) > MAX_COOPERATION_DEPENDENCIES:
            raise AgentCooperationDependencyError("dependencies exceeds the absolute limit.")
        if not all(isinstance(item, AgentCooperationDependency) for item in dependencies):
            raise AgentCooperationDependencyError(
                "dependencies must contain AgentCooperationDependency values."
            )
        dependency_keys = tuple(
            (item.prerequisite_task_id, item.dependent_task_id) for item in dependencies
        )
        if len(set(dependency_keys)) != len(dependency_keys):
            raise AgentCooperationDependencyError("duplicate dependencies are not allowed.")
        unknown = {
            item_id
            for dependency in dependencies
            for item_id in (dependency.prerequisite_task_id, dependency.dependent_task_id)
            if item_id not in task_ids
        }
        if unknown:
            raise AgentCooperationDependencyError("dependencies reference unknown tasks.")
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))
        _topological_order(self)
        dependency_pairs = set(dependency_keys)
        for task in tasks:
            for binding in task.dependency_output_bindings:
                if (binding.source_task_id, task.task_id) not in dependency_pairs:
                    raise AgentCooperationDependencyError(
                        "binding source must be an explicit dependency of its task."
                    )


@dataclass(frozen=True, slots=True)
class AgentCooperationPlanRequest:
    """Request to execute one complete, already-declared plan."""

    plan: AgentCooperationPlan
    policy: AgentCooperationPlanPolicy | None = None
    execution_id: str | None = None
    correlation_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, AgentCooperationPlan):
            raise InvalidAgentCooperationPlanError("plan must be AgentCooperationPlan.")
        if self.policy is not None and not isinstance(self.policy, AgentCooperationPlanPolicy):
            raise InvalidAgentCooperationPlanError("policy must be AgentCooperationPlanPolicy or None.")
        for name in ("execution_id", "correlation_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))

    @property
    def effective_policy(self) -> AgentCooperationPlanPolicy:
        return self.policy if self.policy is not None else self.plan.policy


@dataclass(frozen=True, slots=True)
class AgentCooperationPlanEvent:
    """Safe event emitted during validation and sequential execution."""

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
class AgentCooperationTaskResult:
    """Structured result for one plan task."""

    task_id: str
    status: AgentCooperationTaskStatus
    execution_type: AgentCooperationExecutionType
    agent_ids: tuple[str, ...] = ()
    output: Mapping[str, object] | None = None
    execution_result: (
        AgentExecutionResult
        | AgentDelegationResult
        | AgentDelegationChainResult
        | MultiAgentExecutionResult
        | AgentDelegationCoordinationResult
        | SkillExecutionResult
        | None
    ) = None
    error_code: str | None = None
    safe_message: str | None = None
    position: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(self, "status", _task_status(self.status))
        object.__setattr__(self, "execution_type", _execution_type(self.execution_type))
        object.__setattr__(self, "agent_ids", _agent_id_tuple(self.agent_ids, "agent_ids"))
        if self.output is not None:
            output, _ = _safe_output(self.output)
            object.__setattr__(self, "output", MappingProxyType(output))
        allowed_results = (
            AgentExecutionResult,
            AgentDelegationResult,
            AgentDelegationChainResult,
            MultiAgentExecutionResult,
            AgentDelegationCoordinationResult,
            SkillExecutionResult,
        )
        if self.execution_result is not None and not isinstance(self.execution_result, allowed_results):
            raise InvalidAgentCooperationPlanError("execution_result has an unsupported type.")
        if self.safe_message is not None:
            object.__setattr__(self, "safe_message", _safe_message(self.safe_message))
        if isinstance(self.position, bool) or not isinstance(self.position, int) or self.position < 0:
            raise InvalidAgentCooperationPlanError("position must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class AgentCooperationPlanResult:
    """Immutable terminal result for a cooperation plan."""

    status: AgentCooperationPlanStatus
    plan_id: str | None
    plan_signature: str
    request_signature: str
    task_results: tuple[AgentCooperationTaskResult, ...] = ()
    execution_order: tuple[str, ...] = ()
    outputs: Mapping[str, object] = field(default_factory=dict)
    events: tuple[AgentCooperationPlanEvent, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    error_code: str | None = None
    safe_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _plan_status(self.status))
        if self.plan_id is not None:
            object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(self, "task_results", tuple(self.task_results))
        object.__setattr__(self, "execution_order", _identifier_tuple(self.execution_order, "execution_order"))
        object.__setattr__(self, "outputs", MappingProxyType(_safe_mapping(self.outputs, "outputs")))
        object.__setattr__(self, "events", tuple(self.events)[-MAX_COOPERATION_EVENTS:])
        object.__setattr__(self, "metrics", MappingProxyType(_metric_mapping(self.metrics)))
        if self.safe_message is not None:
            object.__setattr__(self, "safe_message", _safe_message(self.safe_message))


class AgentCooperationPlanner:
    """Validate and execute complete declarative plans through existing services."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        agent_resolver: AgentResolver,
        agent_context_builder: AgentContextBuilder,
        agent_executor: AgentExecutor,
        agent_delegation_service: AgentDelegationService,
        agent_delegation_chain_service: AgentDelegationChainService,
        agent_delegation_coordinator: AgentDelegationCoordinator,
        multi_agent_coordinator: MultiAgentCoordinator,
        skill_system: SkillSystem,
    ) -> None:
        dependencies = (
            (agent_registry, AgentRegistry, "agent_registry"),
            (agent_resolver, AgentResolver, "agent_resolver"),
            (agent_context_builder, AgentContextBuilder, "agent_context_builder"),
            (agent_executor, AgentExecutor, "agent_executor"),
            (agent_delegation_service, AgentDelegationService, "agent_delegation_service"),
            (agent_delegation_chain_service, AgentDelegationChainService, "agent_delegation_chain_service"),
            (agent_delegation_coordinator, AgentDelegationCoordinator, "agent_delegation_coordinator"),
            (multi_agent_coordinator, MultiAgentCoordinator, "multi_agent_coordinator"),
            (skill_system, SkillSystem, "skill_system"),
        )
        for value, expected, name in dependencies:
            if not isinstance(value, expected):
                raise AgentCooperationPlanError(f"{name} must be {expected.__name__}.")
        self._agent_registry = agent_registry
        self._agent_resolver = agent_resolver
        self._agent_context_builder = agent_context_builder
        self._agent_executor = agent_executor
        self._agent_delegation_service = agent_delegation_service
        self._agent_delegation_chain_service = agent_delegation_chain_service
        self._agent_delegation_coordinator = agent_delegation_coordinator
        self._multi_agent_coordinator = multi_agent_coordinator
        self._skill_system = skill_system

    def execute(self, request: AgentCooperationPlanRequest) -> AgentCooperationPlanResult:
        """Validate and execute a predeclared plan sequentially."""

        metrics = _base_metrics()
        events: list[AgentCooperationPlanEvent] = []
        if not isinstance(request, AgentCooperationPlanRequest):
            metrics["cooperation_plans_invalid"] = 1
            return _result(
                AgentCooperationPlanStatus.INVALID_PLAN,
                metrics=metrics,
                events=events,
                error_code="INVALID_PLAN",
                safe_message="request must be AgentCooperationPlanRequest.",
            )
        plan = request.plan
        policy = request.effective_policy
        plan_signature = agent_cooperation_plan_signature(plan)
        request_signature = agent_cooperation_plan_request_signature(request)
        metrics["cooperation_plans_requested"] = 1
        _event(events, "agent_cooperation_plan_requested", "requested", plan_id=plan.plan_id)
        _event(events, "agent_cooperation_plan_validation_started", "started", plan_id=plan.plan_id)
        validation = self._validate(plan, policy)
        if validation is not None:
            status, code, message = validation
            metrics["cooperation_plans_invalid" if status is AgentCooperationPlanStatus.INVALID_PLAN else "cooperation_limits_reached"] = 1
            metrics["cooperation_plans_failed"] = 1
            _event(
                events,
                "agent_cooperation_limit_reached"
                if status is AgentCooperationPlanStatus.LIMIT_REACHED
                else "agent_cooperation_plan_validation_failed",
                "failed",
                plan_id=plan.plan_id,
                error_code=code,
            )
            return _result(
                status,
                plan=plan,
                plan_signature=plan_signature,
                request_signature=request_signature,
                metrics=metrics,
                events=events,
                error_code=code,
                safe_message=message,
            )
        if not policy.enabled:
            _event(events, "agent_cooperation_plan_validation_failed", "blocked", plan_id=plan.plan_id)
            metrics["cooperation_plans_failed"] = 1
            return _result(
                AgentCooperationPlanStatus.BLOCKED,
                plan=plan,
                plan_signature=plan_signature,
                request_signature=request_signature,
                metrics=metrics,
                events=events,
                error_code="DISABLED",
                safe_message="cooperation plan policy is disabled.",
            )

        _event(events, "agent_cooperation_plan_validation_succeeded", "succeeded", plan_id=plan.plan_id)
        _event(events, "agent_cooperation_plan_started", "started", plan_id=plan.plan_id)
        metrics["cooperation_plans_started"] = 1
        metrics["cooperation_tasks_total"] = len(plan.tasks)
        ordered = _topological_order(plan)
        dependencies = _dependencies_by_task(plan)
        results: dict[str, AgentCooperationTaskResult] = {}
        outputs: dict[str, object] = {}
        execution_order: list[str] = []
        executions = 0
        propagated_items = 0
        stopped = False

        for position, task in enumerate(ordered):
            dependency_results = tuple(results[item.prerequisite_task_id] for item in dependencies[task.task_id])
            metrics["cooperation_dependencies_evaluated"] += len(dependency_results)
            blocking = _dependency_blocking_status(dependency_results)
            if stopped:
                task_result = _non_execution_result(task, AgentCooperationTaskStatus.SKIPPED, position, "PLAN_STOPPED")
                metrics["cooperation_tasks_skipped"] += 1
                _event(events, "agent_cooperation_task_skipped", "skipped", task_id=task.task_id)
            elif blocking is not None:
                task_status = (
                    AgentCooperationTaskStatus.DEPENDENCY_BLOCKED
                    if blocking is AgentCooperationTaskStatus.BLOCKED
                    else AgentCooperationTaskStatus.DEPENDENCY_FAILED
                )
                task_result = _non_execution_result(task, task_status, position, task_status.value)
                metrics["cooperation_dependency_failures"] += 1
                metrics["cooperation_tasks_blocked"] += 1
                _event(events, "agent_cooperation_dependency_failed", "failed", task_id=task.task_id)
                _event(events, "agent_cooperation_task_blocked", "blocked", task_id=task.task_id)
                if policy.stop_on_blocked_task:
                    stopped = True
            else:
                metrics["cooperation_dependencies_satisfied"] += len(dependency_results)
                for dependency in dependencies[task.task_id]:
                    _event(
                        events,
                        "agent_cooperation_dependency_satisfied",
                        "succeeded",
                        task_id=task.task_id,
                        source_task_id=dependency.prerequisite_task_id,
                    )
                if executions >= policy.max_total_executions:
                    task_result = _non_execution_result(
                        task,
                        AgentCooperationTaskStatus.LIMIT_REACHED,
                        position,
                        "MAX_TOTAL_EXECUTIONS",
                    )
                    metrics["cooperation_limits_reached"] += 1
                    metrics["cooperation_tasks_failed"] += 1
                    stopped = True
                    _event(events, "agent_cooperation_limit_reached", "failed", task_id=task.task_id)
                else:
                    _event(events, "agent_cooperation_task_ready", "ready", task_id=task.task_id)
                    try:
                        structured_input, shared_context, propagated = self._bind_inputs(
                            task,
                            outputs,
                            policy,
                        )
                        propagated_items += propagated
                        if propagated_items > policy.max_propagated_items:
                            raise AgentCooperationExecutionError("maximum propagated items reached.")
                    except AgentCooperationExecutionError as error:
                        task_result = _non_execution_result(
                            task,
                            AgentCooperationTaskStatus.OUTPUT_BINDING_FAILED,
                            position,
                            "OUTPUT_BINDING_FAILED",
                            str(error),
                        )
                        metrics["cooperation_output_binding_failures"] += 1
                        metrics["cooperation_tasks_failed"] += 1
                        _event(events, "agent_cooperation_output_binding_failed", "failed", task_id=task.task_id)
                    else:
                        if propagated:
                            metrics["cooperation_outputs_propagated"] += propagated
                            _event(
                                events,
                                "agent_cooperation_output_propagated",
                                "succeeded",
                                task_id=task.task_id,
                                items=propagated,
                            )
                        _event(events, "agent_cooperation_task_started", "started", task_id=task.task_id)
                        metrics["cooperation_tasks_started"] += 1
                        execution_order.append(task.task_id)
                        executions += 1
                        task_result = self._execute_task(
                            task,
                            structured_input,
                            shared_context,
                            request,
                            position,
                        )
                        if task_result.status is AgentCooperationTaskStatus.SUCCESS:
                            metrics["cooperation_tasks_succeeded"] += 1
                            _event(events, "agent_cooperation_task_succeeded", "succeeded", task_id=task.task_id)
                            if task_result.output is not None:
                                outputs[task.task_id] = task_result.output
                                if _count_items(outputs) > policy.max_output_items:
                                    task_result = replace(
                                        task_result,
                                        status=AgentCooperationTaskStatus.LIMIT_REACHED,
                                        output=None,
                                        error_code="MAX_OUTPUT_ITEMS",
                                        safe_message="maximum output items reached.",
                                    )
                                    outputs.pop(task.task_id, None)
                                    metrics["cooperation_tasks_succeeded"] -= 1
                                    metrics["cooperation_tasks_failed"] += 1
                                    metrics["cooperation_limits_reached"] += 1
                                    _event(events, "agent_cooperation_limit_reached", "failed", task_id=task.task_id)
                        else:
                            metrics["cooperation_tasks_failed"] += 1
                            _event(events, "agent_cooperation_task_failed", "failed", task_id=task.task_id)

            results[task.task_id] = task_result
            if task_result.status is not AgentCooperationTaskStatus.SUCCESS:
                if _stop_after_task_failure(task, policy):
                    stopped = True

        final_status = _final_status(tuple(results.values()), policy)
        if final_status is AgentCooperationPlanStatus.SUCCESS:
            metrics["cooperation_plans_succeeded"] = 1
            _event(events, "agent_cooperation_plan_completed", "succeeded", plan_id=plan.plan_id)
        elif final_status is AgentCooperationPlanStatus.PARTIAL_SUCCESS:
            metrics["cooperation_plans_partial"] = 1
            _event(events, "agent_cooperation_plan_partial", "partial", plan_id=plan.plan_id)
        else:
            metrics["cooperation_plans_failed"] = 1
            _event(events, "agent_cooperation_plan_failed", "failed", plan_id=plan.plan_id)
        return _result(
            final_status,
            plan=plan,
            plan_signature=plan_signature,
            request_signature=request_signature,
            task_results=tuple(results[task.task_id] for task in ordered),
            execution_order=tuple(execution_order),
            outputs=outputs,
            events=events,
            metrics=metrics,
            error_code=None if final_status in (AgentCooperationPlanStatus.SUCCESS, AgentCooperationPlanStatus.PARTIAL_SUCCESS) else final_status.value,
        )

    def _validate(
        self,
        plan: AgentCooperationPlan,
        policy: AgentCooperationPlanPolicy,
    ) -> tuple[AgentCooperationPlanStatus, str, str] | None:
        if len(plan.tasks) > policy.max_tasks:
            return AgentCooperationPlanStatus.LIMIT_REACHED, "MAX_TASKS", "maximum tasks reached."
        if len(plan.dependencies) > policy.max_dependencies:
            return AgentCooperationPlanStatus.LIMIT_REACHED, "MAX_DEPENDENCIES", "maximum dependencies reached."
        try:
            depth = _plan_depth(plan)
        except AgentCooperationCycleError:
            return AgentCooperationPlanStatus.CYCLE_DETECTED, "CYCLE_DETECTED", "plan contains a cycle."
        if depth > policy.max_depth:
            return AgentCooperationPlanStatus.LIMIT_REACHED, "MAX_DEPTH", "maximum dependency depth reached."
        if policy.minimum_successful_tasks > len(plan.tasks):
            return (
                AgentCooperationPlanStatus.INVALID_PLAN,
                "INVALID_MINIMUM_SUCCESS",
                "minimum_successful_tasks exceeds task count.",
            )
        for task in plan.tasks:
            if len(task.required_skill_ids) > 1:
                return (
                    AgentCooperationPlanStatus.INVALID_PLAN,
                    "MULTIPLE_SKILLS_UNSUPPORTED",
                    "one task can execute at most one required skill.",
                )
            if task.continue_on_failure and policy.failure_mode is not AgentCooperationFailureMode.CONTINUE_INDEPENDENT_TASKS:
                return (
                    AgentCooperationPlanStatus.INVALID_PLAN,
                    "INVALID_CONTINUE_ON_FAILURE",
                    "continue_on_failure requires CONTINUE_INDEPENDENT_TASKS.",
                )
            if task.agent_id is not None:
                if not self._agent_registry.contains(task.agent_id):
                    return AgentCooperationPlanStatus.INVALID_PLAN, "AGENT_NOT_FOUND", "task agent is not registered."
                agent = self._agent_registry.get(task.agent_id)
                mismatch = _agent_mismatch(agent, task)
                if mismatch is not None:
                    return AgentCooperationPlanStatus.INVALID_PLAN, mismatch, "task agent is incompatible."
            for skill_id in task.required_skill_ids:
                try:
                    skill = self._skill_system.skill_registry.get(skill_id)
                except SkillNotFoundError:
                    return AgentCooperationPlanStatus.INVALID_PLAN, "SKILL_NOT_FOUND", "required skill is not registered."
                if not skill.enabled:
                    return AgentCooperationPlanStatus.INVALID_PLAN, "SKILL_DISABLED", "required skill is disabled."
        return None

    def _bind_inputs(
        self,
        task: AgentCooperationTask,
        outputs: Mapping[str, object],
        policy: AgentCooperationPlanPolicy,
    ) -> tuple[Mapping[str, object], Mapping[str, object], int]:
        input_data = _mutable_copy(task.structured_input)
        context_data = _mutable_copy(task.shared_context)
        if task.dependency_output_bindings and not policy.propagate_dependency_outputs:
            raise AgentCooperationExecutionError("dependency output propagation is disabled.")
        propagated = 0
        for binding in task.dependency_output_bindings:
            try:
                value = _read_binding_source(outputs, binding)
            except AgentCooperationExecutionError:
                if binding.required and policy.fail_on_missing_dependency_output:
                    raise
                continue
            target = input_data if binding.target_path[0] == "input" else context_data
            _write_binding_target(target, binding.target_path[1:], value)
            propagated += _count_items(value)
        return (
            MappingProxyType(_safe_mapping(input_data, "structured_input")),
            MappingProxyType(_safe_mapping(context_data, "shared_context")),
            propagated,
        )

    def _execute_task(
        self,
        task: AgentCooperationTask,
        structured_input: Mapping[str, object],
        shared_context: Mapping[str, object],
        request: AgentCooperationPlanRequest,
        position: int,
    ) -> AgentCooperationTaskResult:
        try:
            if task.execution_type is AgentCooperationExecutionType.SINGLE_AGENT:
                return self._execute_single(task, structured_input, shared_context, request, position)
            if task.execution_type is AgentCooperationExecutionType.DELEGATION:
                service_request = task.delegation_request
                assert service_request is not None
                result = self._agent_delegation_service.delegate(
                    replace(
                        service_request,
                        structured_input=_merge_no_collision(service_request.structured_input, structured_input),
                        shared_context=_merge_no_collision(service_request.shared_context, shared_context),
                    )
                )
                return _delegation_task_result(task, result, position)
            if task.execution_type is AgentCooperationExecutionType.DELEGATION_CHAIN:
                service_request = task.delegation_chain_request
                assert service_request is not None
                result = self._agent_delegation_chain_service.execute(
                    replace(
                        service_request,
                        initial_input=_merge_no_collision(service_request.initial_input, structured_input),
                        shared_context=_merge_no_collision(service_request.shared_context, shared_context),
                    )
                )
                return _chain_task_result(task, result, position)
            if task.execution_type is AgentCooperationExecutionType.MULTI_AGENT:
                service_request = task.multi_agent_request
                assert service_request is not None
                result = self._multi_agent_coordinator.execute(
                    replace(
                        service_request,
                        payload=_merge_no_collision(service_request.payload, structured_input),
                        shared_context=_merge_no_collision(service_request.shared_context, shared_context),
                    )
                )
                return _multi_task_result(task, result, position)
            service_request = task.coordinated_chains_request
            assert service_request is not None
            result = self._agent_delegation_coordinator.coordinate(
                replace(
                    service_request,
                    structured_input=_merge_no_collision(service_request.structured_input, structured_input),
                    shared_context=_merge_no_collision(service_request.shared_context, shared_context),
                )
            )
            return _coordinated_task_result(task, result, position)
        except (RuntimeError, TypeError, ValueError) as error:
            return AgentCooperationTaskResult(
                task_id=task.task_id,
                status=AgentCooperationTaskStatus.AGENT_EXECUTION_FAILED,
                execution_type=task.execution_type,
                error_code=type(error).__name__,
                safe_message=str(error),
                position=position,
            )

    def _execute_single(
        self,
        task: AgentCooperationTask,
        structured_input: Mapping[str, object],
        shared_context: Mapping[str, object],
        request: AgentCooperationPlanRequest,
        position: int,
    ) -> AgentCooperationTaskResult:
        resolution_request = AgentResolutionRequest(
            required_agent_ids=() if task.agent_id is None else (task.agent_id,),
            required_agent_types=task.required_agent_types,
            required_capability_ids=task.required_capability_ids,
            required_permission_ids=task.required_permission_ids,
            preferred_agent_ids=task.preferred_agent_ids,
            excluded_agent_ids=task.excluded_agent_ids,
            enabled_only=task.enabled_only,
            require_unique_top_score=True,
            metadata={"route": "agent_cooperation_plan", "task_id": task.task_id},
        )
        resolution = self._agent_resolver.resolve(resolution_request)
        if resolution.status is not AgentResolutionStatus.RESOLVED or resolution.selected_agent is None:
            return AgentCooperationTaskResult(
                task.task_id,
                AgentCooperationTaskStatus.AGENT_RESOLUTION_FAILED,
                task.execution_type,
                error_code=resolution.error_code or resolution.status.value,
                safe_message=resolution.error_message,
                position=position,
            )
        agent = resolution.selected_agent
        if not self._skills_authorized(agent, task.required_skill_ids):
            return AgentCooperationTaskResult(
                task.task_id,
                AgentCooperationTaskStatus.SKILL_AUTHORIZATION_FAILED,
                task.execution_type,
                agent_ids=(agent.agent_id,),
                error_code="SKILL_AUTHORIZATION_FAILED",
                safe_message="selected agent is not authorized for all required skills.",
                position=position,
            )
        if task.required_skill_ids:
            return self._execute_skill(task, structured_input, agent, position)
        execution = self._agent_executor.execute(
            AgentExecutionRequest(
                resolution_request=AgentResolutionRequest(
                    required_agent_ids=(agent.agent_id,),
                    enabled_only=task.enabled_only,
                    require_unique_top_score=False,
                    metadata={"route": "agent_cooperation_plan", "task_id": task.task_id},
                ),
                task_id=task.task_id,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                structured_input=structured_input,
                shared_context=shared_context,
                metadata=task.metadata,
                required_capability_ids=task.required_capability_ids,
                required_permission_ids=task.required_permission_ids,
            )
        )
        output, _ = _safe_output(execution.output or {})
        return AgentCooperationTaskResult(
            task.task_id,
            AgentCooperationTaskStatus.SUCCESS
            if execution.status is AgentExecutionStatus.COMPLETED
            else AgentCooperationTaskStatus.AGENT_EXECUTION_FAILED,
            task.execution_type,
            agent_ids=(agent.agent_id,),
            output=output if execution.status is AgentExecutionStatus.COMPLETED else None,
            execution_result=execution,
            error_code=execution.error_code,
            safe_message=execution.safe_message,
            position=position,
        )

    def _execute_skill(
        self,
        task: AgentCooperationTask,
        structured_input: Mapping[str, object],
        agent: AgentDefinition,
        position: int,
    ) -> AgentCooperationTaskResult:
        skill_id = task.required_skill_ids[0]
        resolution = self._skill_system.skill_resolver.resolve(
            SkillResolutionRequest(required_skill_ids=(skill_id,))
        )
        if resolution.status is not SkillResolutionStatus.RESOLVED or resolution.selected_skill is None:
            return AgentCooperationTaskResult(
                task.task_id,
                AgentCooperationTaskStatus.SKILL_EXECUTION_FAILED,
                task.execution_type,
                agent_ids=(agent.agent_id,),
                error_code=resolution.error_code or resolution.status.value,
                safe_message="required skill could not be resolved.",
                position=position,
            )
        execution = self._skill_system.skill_executor.execute(
            SkillExecutionRequest(
                skill=resolution.selected_skill,
                inputs=structured_input,
                agent=agent,
                metadata={"route": "agent_cooperation_plan", "task_id": task.task_id},
            )
        )
        output, _ = _safe_output(execution.output or {})
        if execution.status is SkillExecutionStatus.COMPLETED:
            status = AgentCooperationTaskStatus.SUCCESS
        elif execution.status is SkillExecutionStatus.SKILL_NOT_AUTHORIZED:
            status = AgentCooperationTaskStatus.SKILL_AUTHORIZATION_FAILED
        else:
            status = AgentCooperationTaskStatus.SKILL_EXECUTION_FAILED
        return AgentCooperationTaskResult(
            task.task_id,
            status,
            task.execution_type,
            agent_ids=(agent.agent_id,),
            output=output if execution.status is SkillExecutionStatus.COMPLETED else None,
            execution_result=execution,
            error_code=execution.error_code,
            safe_message=execution.safe_message,
            position=position,
        )

    def _skills_authorized(self, agent: AgentDefinition, skill_ids: tuple[str, ...]) -> bool:
        for skill_id in skill_ids:
            skill = self._skill_system.skill_registry.get(skill_id)
            if not _skill_authorized(skill, agent):
                return False
        return True


def agent_cooperation_plan_signature(plan: AgentCooperationPlan) -> str:
    """Return a stable SHA-256 signature for a complete plan."""

    if not isinstance(plan, AgentCooperationPlan):
        raise InvalidAgentCooperationPlanError("plan must be AgentCooperationPlan.")
    return _signature(_plan_payload(plan))


def agent_cooperation_plan_request_signature(request: AgentCooperationPlanRequest) -> str:
    """Return a stable SHA-256 signature for a plan request."""

    if not isinstance(request, AgentCooperationPlanRequest):
        raise InvalidAgentCooperationPlanError("request must be AgentCooperationPlanRequest.")
    return _signature(
        {
            "plan_signature": agent_cooperation_plan_signature(request.plan),
            "policy": _policy_payload(request.effective_policy),
            "execution_id": request.execution_id,
            "correlation_id": request.correlation_id,
            "metadata": _jsonable(request.metadata),
        }
    )


def _topological_order(plan: AgentCooperationPlan) -> tuple[AgentCooperationTask, ...]:
    tasks = {task.task_id: task for task in plan.tasks}
    incoming = {task_id: 0 for task_id in tasks}
    outgoing = {task_id: [] for task_id in tasks}
    for dependency in plan.dependencies:
        incoming[dependency.dependent_task_id] += 1
        outgoing[dependency.prerequisite_task_id].append(dependency.dependent_task_id)
    ready = [tasks[task_id] for task_id, count in incoming.items() if count == 0]
    ordered: list[AgentCooperationTask] = []
    while ready:
        ready.sort(key=_task_sort_key)
        task = ready.pop(0)
        ordered.append(task)
        for dependent_id in sorted(outgoing[task.task_id]):
            incoming[dependent_id] -= 1
            if incoming[dependent_id] == 0:
                ready.append(tasks[dependent_id])
    if len(ordered) != len(tasks):
        raise AgentCooperationCycleError("plan contains a dependency cycle.")
    return tuple(ordered)


def _plan_depth(plan: AgentCooperationPlan) -> int:
    ordered = _topological_order(plan)
    prerequisites = _dependencies_by_task(plan)
    depths: dict[str, int] = {}
    for task in ordered:
        previous = prerequisites[task.task_id]
        depths[task.task_id] = 1 + max(
            (depths[item.prerequisite_task_id] for item in previous),
            default=0,
        )
    return max(depths.values(), default=0)


def _dependencies_by_task(
    plan: AgentCooperationPlan,
) -> dict[str, tuple[AgentCooperationDependency, ...]]:
    result: dict[str, list[AgentCooperationDependency]] = {task.task_id: [] for task in plan.tasks}
    for dependency in plan.dependencies:
        result[dependency.dependent_task_id].append(dependency)
    return {
        task_id: tuple(sorted(items, key=lambda item: item.prerequisite_task_id))
        for task_id, items in result.items()
    }


def _task_sort_key(task: AgentCooperationTask) -> tuple[int, int, str]:
    return task.order, -task.priority, task.task_id


def _dependency_blocking_status(
    results: tuple[AgentCooperationTaskResult, ...],
) -> AgentCooperationTaskStatus | None:
    if any(result.status in (AgentCooperationTaskStatus.BLOCKED, AgentCooperationTaskStatus.DEPENDENCY_BLOCKED) for result in results):
        return AgentCooperationTaskStatus.BLOCKED
    if any(result.status is not AgentCooperationTaskStatus.SUCCESS for result in results):
        return AgentCooperationTaskStatus.FAILED
    return None


def _stop_after_task_failure(
    task: AgentCooperationTask,
    policy: AgentCooperationPlanPolicy,
) -> bool:
    if task.continue_on_failure and policy.failure_mode is AgentCooperationFailureMode.CONTINUE_INDEPENDENT_TASKS:
        return False
    return policy.failure_mode in (
        AgentCooperationFailureMode.STOP_ON_FIRST_FAILURE,
        AgentCooperationFailureMode.REQUIRE_ALL_SUCCESS,
    )


def _final_status(
    results: tuple[AgentCooperationTaskResult, ...],
    policy: AgentCooperationPlanPolicy,
) -> AgentCooperationPlanStatus:
    succeeded = sum(result.status is AgentCooperationTaskStatus.SUCCESS for result in results)
    failed = len(results) - succeeded
    skipped = sum(result.status is AgentCooperationTaskStatus.SKIPPED for result in results)
    if succeeded == len(results):
        return AgentCooperationPlanStatus.SUCCESS
    if policy.failure_mode is AgentCooperationFailureMode.REQUIRE_MINIMUM_SUCCESS:
        if succeeded < policy.minimum_successful_tasks:
            return AgentCooperationPlanStatus.MINIMUM_SUCCESS_NOT_REACHED
        return AgentCooperationPlanStatus.PARTIAL_SUCCESS if policy.allow_partial_success else AgentCooperationPlanStatus.FAILED
    if policy.require_all_success or policy.failure_mode is AgentCooperationFailureMode.REQUIRE_ALL_SUCCESS:
        return AgentCooperationPlanStatus.FAILED
    if skipped and not policy.allow_skipped_tasks:
        return AgentCooperationPlanStatus.FAILED
    if succeeded and failed and policy.allow_partial_success:
        return AgentCooperationPlanStatus.PARTIAL_SUCCESS
    return AgentCooperationPlanStatus.FAILED


def _delegation_task_result(
    task: AgentCooperationTask,
    result: AgentDelegationResult,
    position: int,
) -> AgentCooperationTaskResult:
    output, _ = _safe_output(result.safe_output or {})
    return AgentCooperationTaskResult(
        task.task_id,
        AgentCooperationTaskStatus.SUCCESS
        if result.status is AgentDelegationStatus.SUCCESS
        else AgentCooperationTaskStatus.AGENT_EXECUTION_FAILED,
        task.execution_type,
        agent_ids=() if result.target_agent_id is None else (result.target_agent_id,),
        output=output if result.status is AgentDelegationStatus.SUCCESS else None,
        execution_result=result,
        error_code=result.error_code,
        safe_message=result.error_message,
        position=position,
    )


def _chain_task_result(
    task: AgentCooperationTask,
    result: AgentDelegationChainResult,
    position: int,
) -> AgentCooperationTaskResult:
    output, _ = _safe_output(result.final_output or {})
    agent_ids = tuple(
        item.resolved_target_agent_id
        for item in result.step_results
        if item.resolved_target_agent_id is not None
    )
    return AgentCooperationTaskResult(
        task.task_id,
        AgentCooperationTaskStatus.SUCCESS
        if result.status is AgentDelegationChainStatus.SUCCESS
        else AgentCooperationTaskStatus.AGENT_EXECUTION_FAILED,
        task.execution_type,
        agent_ids=agent_ids,
        output=output if result.status is AgentDelegationChainStatus.SUCCESS else None,
        execution_result=result,
        error_code=result.error_code,
        safe_message=result.error_message,
        position=position,
    )


def _multi_task_result(
    task: AgentCooperationTask,
    result: MultiAgentExecutionResult,
    position: int,
) -> AgentCooperationTaskResult:
    success = result.status in (
        MultiAgentExecutionStatus.SUCCESS,
        MultiAgentExecutionStatus.PARTIAL_SUCCESS,
    )
    output, _ = _safe_output(result.output or {})
    agent_ids = (
        result.team_resolution_result.selected_agent_ids
        if result.team_resolution_result is not None
        else ()
    )
    return AgentCooperationTaskResult(
        task.task_id,
        AgentCooperationTaskStatus.SUCCESS if success else AgentCooperationTaskStatus.AGENT_EXECUTION_FAILED,
        task.execution_type,
        agent_ids=agent_ids,
        output=output if success else None,
        execution_result=result,
        error_code=result.error_code,
        safe_message=result.safe_message,
        position=position,
    )


def _coordinated_task_result(
    task: AgentCooperationTask,
    result: AgentDelegationCoordinationResult,
    position: int,
) -> AgentCooperationTaskResult:
    success = result.status in (
        AgentDelegationCoordinationStatus.SUCCESS,
        AgentDelegationCoordinationStatus.PARTIAL_SUCCESS,
    )
    output, _ = _safe_output(result.aggregated_outputs)
    return AgentCooperationTaskResult(
        task.task_id,
        AgentCooperationTaskStatus.SUCCESS if success else AgentCooperationTaskStatus.AGENT_EXECUTION_FAILED,
        task.execution_type,
        output=output if success else None,
        execution_result=result,
        error_code=result.error_code,
        safe_message=result.error_message,
        position=position,
    )


def _non_execution_result(
    task: AgentCooperationTask,
    status: AgentCooperationTaskStatus,
    position: int,
    error_code: str,
    safe_message: str | None = None,
) -> AgentCooperationTaskResult:
    return AgentCooperationTaskResult(
        task.task_id,
        status,
        task.execution_type,
        error_code=error_code,
        safe_message=safe_message,
        position=position,
    )


def _read_binding_source(
    outputs: Mapping[str, object],
    binding: AgentCooperationOutputBinding,
) -> object:
    if binding.source_task_id not in outputs:
        raise AgentCooperationExecutionError("binding source output is unavailable.")
    value: object = outputs[binding.source_task_id]
    for segment in binding.source_path[1:]:
        if not isinstance(value, Mapping) or segment not in value:
            raise AgentCooperationExecutionError("binding source path is unavailable.")
        value = value[segment]
    safe, _ = _safe_value(value, "binding value", depth=0, counter={"items": 0}, drop_sensitive=True)
    return safe


def _write_binding_target(target: dict[str, object], path: tuple[str, ...], value: object) -> None:
    current = target
    for segment in path[:-1]:
        if segment in current:
            existing = current[segment]
            if not isinstance(existing, dict):
                if isinstance(existing, Mapping):
                    existing = _mutable_copy(existing)
                    current[segment] = existing
                else:
                    raise AgentCooperationExecutionError("binding target collides with an existing value.")
            current = existing
        else:
            nested: dict[str, object] = {}
            current[segment] = nested
            current = nested
    if path[-1] in current:
        raise AgentCooperationExecutionError("binding target already exists.")
    current[path[-1]] = value


def _merge_no_collision(
    existing: Mapping[str, object] | None,
    addition: Mapping[str, object],
) -> Mapping[str, object]:
    merged = {} if existing is None else _mutable_copy(existing)
    for key, value in addition.items():
        if key in merged and merged[key] != value:
            raise AgentCooperationExecutionError("task input collides with service request input.")
        merged[key] = value
    return MappingProxyType(_safe_mapping(merged, "merged input"))


def _agent_mismatch(agent: AgentDefinition, task: AgentCooperationTask) -> str | None:
    if task.enabled_only and not agent.enabled:
        return "AGENT_DISABLED"
    if task.required_agent_types and agent.agent_type not in task.required_agent_types:
        return "AGENT_TYPE_INCOMPATIBLE"
    if any(item not in agent.capabilities.capabilities for item in task.required_capability_ids):
        return "AGENT_CAPABILITY_INCOMPATIBLE"
    if any(not bool(getattr(agent.permissions, item)) for item in task.required_permission_ids):
        return "AGENT_PERMISSION_INCOMPATIBLE"
    return None


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


def _plan_payload(plan: AgentCooperationPlan) -> Mapping[str, object]:
    return {
        "plan_id": plan.plan_id,
        "schema_version": plan.schema_version,
        "execution_strategy": plan.execution_strategy,
        "tasks": [
            {
                "task_id": task.task_id,
                "objective_id": task.objective_id,
                "execution_type": task.execution_type.value,
                "agent_id": task.agent_id,
                "required_agent_types": sorted(item.value for item in task.required_agent_types),
                "required_capability_ids": sorted(task.required_capability_ids),
                "required_permission_ids": sorted(task.required_permission_ids),
                "preferred_agent_ids": sorted(task.preferred_agent_ids),
                "excluded_agent_ids": sorted(task.excluded_agent_ids),
                "enabled_only": task.enabled_only,
                "structured_input": _jsonable(task.structured_input),
                "shared_context": _jsonable(task.shared_context),
                "metadata": _jsonable(task.metadata),
                "required_skill_ids": sorted(task.required_skill_ids),
                "bindings": [
                    {
                        "source_task_id": item.source_task_id,
                        "source_path": list(item.source_path),
                        "target_path": list(item.target_path),
                        "required": item.required,
                    }
                    for item in task.dependency_output_bindings
                ],
                "logical_timeout_limit": task.logical_timeout_limit,
                "continue_on_failure": task.continue_on_failure,
                "order": task.order,
                "priority": task.priority,
                "service_request_signature": _service_request_signature(task),
            }
            for task in sorted(plan.tasks, key=lambda item: item.task_id)
        ],
        "dependencies": [
            {
                "prerequisite_task_id": item.prerequisite_task_id,
                "dependent_task_id": item.dependent_task_id,
                "required": item.required,
            }
            for item in sorted(
                plan.dependencies,
                key=lambda item: (item.prerequisite_task_id, item.dependent_task_id),
            )
        ],
        "metadata": _jsonable(plan.metadata),
        "policy": _policy_payload(plan.policy),
    }


def _service_request_signature(task: AgentCooperationTask) -> str | None:
    request = (
        task.delegation_request
        or task.delegation_chain_request
        or task.multi_agent_request
        or task.coordinated_chains_request
    )
    if request is None:
        return None
    return _signature(_jsonable_dataclass(request))


def _jsonable_dataclass(value: object) -> object:
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is None:
        return _jsonable(value)
    return {
        name: _jsonable_dataclass(getattr(value, name))
        for name in sorted(fields)
    }


def _policy_payload(policy: AgentCooperationPlanPolicy) -> Mapping[str, object]:
    return {
        name: getattr(policy, name).value
        if isinstance(getattr(policy, name), Enum)
        else getattr(policy, name)
        for name in sorted(policy.__dataclass_fields__)
    }


def _result(
    status: AgentCooperationPlanStatus,
    *,
    plan: AgentCooperationPlan | None = None,
    plan_signature: str = "",
    request_signature: str = "",
    task_results: tuple[AgentCooperationTaskResult, ...] = (),
    execution_order: tuple[str, ...] = (),
    outputs: Mapping[str, object] | None = None,
    events: Sequence[AgentCooperationPlanEvent] = (),
    metrics: Mapping[str, int] | None = None,
    error_code: str | None = None,
    safe_message: str | None = None,
) -> AgentCooperationPlanResult:
    return AgentCooperationPlanResult(
        status=status,
        plan_id=None if plan is None else plan.plan_id,
        plan_signature=plan_signature,
        request_signature=request_signature,
        task_results=task_results,
        execution_order=execution_order,
        outputs={} if outputs is None else outputs,
        events=tuple(events),
        metrics=_base_metrics() if metrics is None else metrics,
        error_code=error_code,
        safe_message=safe_message,
    )


def _event(
    events: list[AgentCooperationPlanEvent],
    name: str,
    status: str,
    **details: object,
) -> None:
    if len(events) < MAX_COOPERATION_EVENTS:
        events.append(AgentCooperationPlanEvent(name, status, details))


def _base_metrics() -> dict[str, int]:
    return {
        "cooperation_plans_requested": 0,
        "cooperation_plans_started": 0,
        "cooperation_plans_succeeded": 0,
        "cooperation_plans_partial": 0,
        "cooperation_plans_failed": 0,
        "cooperation_plans_invalid": 0,
        "cooperation_tasks_total": 0,
        "cooperation_tasks_started": 0,
        "cooperation_tasks_succeeded": 0,
        "cooperation_tasks_failed": 0,
        "cooperation_tasks_blocked": 0,
        "cooperation_tasks_skipped": 0,
        "cooperation_dependencies_evaluated": 0,
        "cooperation_dependencies_satisfied": 0,
        "cooperation_dependency_failures": 0,
        "cooperation_outputs_propagated": 0,
        "cooperation_output_binding_failures": 0,
        "cooperation_cycles_detected": 0,
        "cooperation_limits_reached": 0,
    }


def _safe_mapping(
    value: Mapping[str, object],
    field_name: str,
    *,
    drop_sensitive: bool = False,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentCooperationPlanError(f"{field_name} must be a mapping.")
    if len(value) > MAX_COOPERATION_METADATA_ITEMS:
        raise InvalidAgentCooperationPlanError(f"{field_name} has too many items.")
    counter = {"items": 0}
    result: dict[str, object] = {}
    for raw_key in sorted(value, key=lambda item: str(item)):
        key = _key(raw_key, field_name)
        if _is_sensitive_key(key):
            if drop_sensitive:
                continue
            raise InvalidAgentCooperationPlanError(f"{field_name} contains a sensitive key.")
        safe, _ = _safe_value(
            value[raw_key],
            field_name,
            depth=0,
            counter=counter,
            drop_sensitive=drop_sensitive,
        )
        result[key] = safe
    return result


def _safe_output(value: Mapping[str, object]) -> tuple[dict[str, object], int]:
    before = _count_items(value)
    safe = _safe_mapping(value, "output", drop_sensitive=True)
    return safe, max(0, before - _count_items(safe))


def _safe_value(
    value: object,
    field_name: str,
    *,
    depth: int,
    counter: dict[str, int],
    drop_sensitive: bool,
) -> tuple[object, int]:
    if depth > MAX_COOPERATION_VALUE_DEPTH:
        raise InvalidAgentCooperationPlanError(f"{field_name} exceeds maximum depth.")
    counter["items"] += 1
    if counter["items"] > MAX_COOPERATION_TOTAL_ITEMS:
        raise InvalidAgentCooperationPlanError(f"{field_name} exceeds total item limit.")
    if value is None or type(value) in (bool, int):
        return value, 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentCooperationPlanError(f"{field_name} floats must be finite.")
        return value, 0
    if isinstance(value, str):
        if len(value) > MAX_COOPERATION_STRING_LENGTH:
            raise InvalidAgentCooperationPlanError(f"{field_name} string exceeds length limit.")
        return value, 0
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        removed = 0
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = _key(raw_key, field_name)
            if _is_sensitive_key(key):
                if drop_sensitive:
                    removed += 1
                    continue
                raise InvalidAgentCooperationPlanError(f"{field_name} contains a sensitive key.")
            item, count = _safe_value(
                value[raw_key],
                field_name,
                depth=depth + 1,
                counter=counter,
                drop_sensitive=drop_sensitive,
            )
            result[key] = item
            removed += count
        return MappingProxyType(result), removed
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_COOPERATION_SEQUENCE_ITEMS:
            raise InvalidAgentCooperationPlanError(f"{field_name} sequence exceeds item limit.")
        safe_items = []
        removed = 0
        for item in value:
            safe, count = _safe_value(
                item,
                field_name,
                depth=depth + 1,
                counter=counter,
                drop_sensitive=drop_sensitive,
            )
            safe_items.append(safe)
            removed += count
        return tuple(safe_items), removed
    raise InvalidAgentCooperationPlanError(f"{field_name} contains an unsupported object.")


def _mutable_copy(value: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[key] = _mutable_copy(item)
        elif isinstance(item, tuple):
            result[key] = tuple(_mutable_sequence_item(part) for part in item)
        else:
            result[key] = item
    return result


def _mutable_sequence_item(value: object) -> object:
    if isinstance(value, Mapping):
        return _mutable_copy(value)
    if isinstance(value, tuple):
        return tuple(_mutable_sequence_item(item) for item in value)
    return value


def _count_items(value: object) -> int:
    if isinstance(value, Mapping):
        return len(value) + sum(_count_items(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return len(value) + sum(_count_items(item) for item in value)
    return 1


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        return _jsonable_dataclass(value)
    raise InvalidAgentCooperationPlanError("value is not deterministically serializable.")


def _signature(value: object) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidAgentCooperationPlanError(f"{field_name} must be a string.")
    if value != value.strip() or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise InvalidAgentCooperationPlanError(f"{field_name} contains unsupported characters.")
    return value


def _key(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidAgentCooperationPlanError(f"{field_name} keys must be strings.")
    return _identifier(value, f"{field_name} key")


def _path(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    result = _identifier_tuple(values, field_name)
    if not result or len(result) > MAX_COOPERATION_VALUE_DEPTH:
        raise InvalidAgentCooperationPlanError(f"{field_name} has invalid depth.")
    if any(_is_sensitive_key(segment) for segment in result):
        raise InvalidAgentCooperationPlanError(f"{field_name} contains a sensitive segment.")
    return result


def _identifier_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentCooperationPlanError(f"{field_name} must be an iterable of strings.")
    result = tuple(dict.fromkeys(_identifier(value, field_name) for value in values))
    if len(result) > MAX_COOPERATION_SEQUENCE_ITEMS:
        raise InvalidAgentCooperationPlanError(f"{field_name} exceeds item limit.")
    return result


def _agent_id_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentCooperationPlanError(f"{field_name} must be an iterable of agent ids.")
    return tuple(dict.fromkeys(validate_agent_id(value) for value in values))


def _skill_id_tuple(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentCooperationPlanError("required_skill_ids must be an iterable.")
    return tuple(dict.fromkeys(validate_skill_id(value) for value in values))


def _agent_type_tuple(
    values: Iterable[AgentType | str],
    field_name: str,
) -> tuple[AgentType, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentCooperationPlanError(f"{field_name} must be an iterable.")
    result: list[AgentType] = []
    for value in values:
        try:
            normalized = value if isinstance(value, AgentType) else AgentType(value)
        except (TypeError, ValueError) as error:
            raise InvalidAgentCooperationPlanError(f"{field_name} contains an invalid agent type.") from error
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
        raise InvalidAgentCooperationPlanError(f"{field_name} contains an invalid permission.")
    return result


def _version(value: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not _VERSION_PATTERN.fullmatch(value):
        raise InvalidAgentCooperationPlanError("schema_version is invalid.")
    return value


def _bounded_int(value: int, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise InvalidAgentCooperationPlanError(f"{field_name} is outside the allowed range.")
    return value


def _metric_mapping(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise InvalidAgentCooperationPlanError("metrics must be a mapping.")
    result: dict[str, int] = {}
    for key, item in value.items():
        normalized = _identifier(key, "metric name")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise InvalidAgentCooperationPlanError("metric values must be non-negative integers.")
        result[normalized] = item
    return result


def _safe_message(value: str) -> str:
    if not isinstance(value, str):
        return "operation failed."
    lowered = value.lower()
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return "operation failed."
    return " ".join(value.split())[:240] or "operation failed."


def _is_sensitive_key(value: str) -> bool:
    lowered = value.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _execution_type(value: AgentCooperationExecutionType | str) -> AgentCooperationExecutionType:
    if isinstance(value, AgentCooperationExecutionType):
        return value
    try:
        return AgentCooperationExecutionType(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentCooperationPlanError("invalid execution_type.") from error


def _failure_mode(value: AgentCooperationFailureMode | str) -> AgentCooperationFailureMode:
    if isinstance(value, AgentCooperationFailureMode):
        return value
    try:
        return AgentCooperationFailureMode(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentCooperationPlanError("invalid failure_mode.") from error


def _task_status(value: AgentCooperationTaskStatus | str) -> AgentCooperationTaskStatus:
    if isinstance(value, AgentCooperationTaskStatus):
        return value
    try:
        return AgentCooperationTaskStatus(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentCooperationPlanError("invalid task status.") from error


def _plan_status(value: AgentCooperationPlanStatus | str) -> AgentCooperationPlanStatus:
    if isinstance(value, AgentCooperationPlanStatus):
        return value
    try:
        return AgentCooperationPlanStatus(value)
    except (TypeError, ValueError) as error:
        raise InvalidAgentCooperationPlanError("invalid plan status.") from error

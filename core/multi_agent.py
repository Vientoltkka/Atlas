"""Deterministic sequential coordination for specialized Atlas agents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType

from core.agent_executor import AgentExecutionRequest, AgentExecutionResult, AgentExecutionStatus, AgentExecutor
from core.agent_registry import AgentDefinition, AgentRegistry, AgentType, validate_agent_id
from core.agent_resolver import AgentResolutionRequest


MAX_MULTI_AGENT_IDS = 32
MAX_MULTI_AGENT_STEPS = 16
MAX_MULTI_AGENT_DEPTH = 5
MAX_MULTI_AGENT_NODES = 256
MAX_MULTI_AGENT_STRING_LENGTH = 1_000
MAX_MULTI_AGENT_METADATA_ITEMS = 16
_PERMISSION_IDS = frozenset(
    {
        "can_read_project",
        "can_write_files",
        "can_execute_tools",
        "can_modify_memory",
        "can_use_network",
        "requires_confirmation",
    }
)
_SENSITIVE_KEY_PARTS = (
    "token",
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


class MultiAgentExecutionError(RuntimeError):
    """Base error for deterministic multi-agent coordination."""


class InvalidMultiAgentExecutionRequestError(MultiAgentExecutionError):
    """Raised when a multi-agent request is malformed."""


class MultiAgentFailurePolicy(str, Enum):
    """Bounded failure behavior for sequential multi-agent execution."""

    STOP_ON_FIRST_FAILURE = "STOP_ON_FIRST_FAILURE"
    CONTINUE_ON_FAILURE = "CONTINUE_ON_FAILURE"


class MultiAgentExecutionStatus(str, Enum):
    """Terminal statuses for coordinated multi-agent execution."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    INVALID_REQUEST = "INVALID_REQUEST"
    NO_MATCHING_TEAM = "NO_MATCHING_TEAM"
    TEAM_RESOLUTION_AMBIGUOUS = "TEAM_RESOLUTION_AMBIGUOUS"
    AGENT_EXECUTION_FAILED = "AGENT_EXECUTION_FAILED"
    AGGREGATION_FAILED = "AGGREGATION_FAILED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class MultiAgentAggregationStatus(str, Enum):
    """Terminal statuses for deterministic multi-agent aggregation."""

    AGGREGATED = "AGGREGATED"
    AGGREGATION_FAILED = "AGGREGATION_FAILED"


class MultiAgentTeamResolutionStatus(str, Enum):
    """Terminal statuses for deterministic team resolution."""

    RESOLVED = "RESOLVED"
    INVALID_REQUEST = "INVALID_REQUEST"
    NO_MATCHING_TEAM = "NO_MATCHING_TEAM"
    AMBIGUOUS = "AMBIGUOUS"
    REGISTRY_UNAVAILABLE = "REGISTRY_UNAVAILABLE"


class MultiAgentRejectionCode(str, Enum):
    """Safe reasons for excluding an agent from a team."""

    DISABLED = "DISABLED"
    EXCLUDED = "EXCLUDED"
    REQUIRED_AGENT_ID_MISSING = "REQUIRED_AGENT_ID_MISSING"
    REQUIRED_AGENT_TYPE_MISSING = "REQUIRED_AGENT_TYPE_MISSING"
    REQUIRED_CAPABILITY_MISSING = "REQUIRED_CAPABILITY_MISSING"
    REQUIRED_PERMISSION_MISSING = "REQUIRED_PERMISSION_MISSING"
    MAX_AGENTS_LIMIT = "MAX_AGENTS_LIMIT"


@dataclass(frozen=True, slots=True)
class MultiAgentExecutionPolicy:
    """Immutable policy for bounded sequential multi-agent execution."""

    min_agents: int = 2
    max_agents: int = 4
    enabled_only: bool = True
    failure_policy: MultiAgentFailurePolicy | str = MultiAgentFailurePolicy.STOP_ON_FIRST_FAILURE
    require_unique_team: bool = False
    share_previous_outputs: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_agents", _positive_int(self.min_agents, "min_agents"))
        object.__setattr__(self, "max_agents", _positive_int(self.max_agents, "max_agents"))
        if self.min_agents > self.max_agents:
            raise InvalidMultiAgentExecutionRequestError("min_agents cannot exceed max_agents.")
        if self.max_agents > MAX_MULTI_AGENT_STEPS:
            raise InvalidMultiAgentExecutionRequestError("max_agents exceeds the step limit.")
        if not isinstance(self.enabled_only, bool):
            raise InvalidMultiAgentExecutionRequestError("enabled_only must be a bool.")
        object.__setattr__(self, "failure_policy", _failure_policy(self.failure_policy))
        if not isinstance(self.require_unique_team, bool):
            raise InvalidMultiAgentExecutionRequestError("require_unique_team must be a bool.")
        if not isinstance(self.share_previous_outputs, bool):
            raise InvalidMultiAgentExecutionRequestError("share_previous_outputs must be a bool.")


@dataclass(frozen=True, slots=True)
class MultiAgentExecutionRequest:
    """Immutable request for selecting and executing a bounded agent team."""

    required_agent_ids: tuple[str, ...] = ()
    required_agent_types: tuple[AgentType | str, ...] = ()
    required_capability_ids: tuple[str, ...] = ()
    required_permission_ids: tuple[str, ...] = ()
    excluded_agent_ids: tuple[str, ...] = ()
    payload: Mapping[str, object] | None = None
    shared_context: Mapping[str, object] | None = None
    user_input: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    task_id: str | None = None
    execution_id: str | None = None
    correlation_id: str | None = None
    session_id: str | None = None
    policy: MultiAgentExecutionPolicy = field(default_factory=MultiAgentExecutionPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_agent_ids", _agent_id_tuple(self.required_agent_ids, "required_agent_ids"))
        object.__setattr__(self, "required_agent_types", _agent_type_tuple(self.required_agent_types, "required_agent_types"))
        object.__setattr__(
            self,
            "required_capability_ids",
            _sorted_identifier_tuple(self.required_capability_ids, "required_capability_ids"),
        )
        object.__setattr__(
            self,
            "required_permission_ids",
            _permission_tuple(self.required_permission_ids, "required_permission_ids"),
        )
        object.__setattr__(self, "excluded_agent_ids", _sorted_agent_id_tuple(self.excluded_agent_ids, "excluded_agent_ids"))
        object.__setattr__(self, "payload", _optional_safe_mapping(self.payload, "payload"))
        object.__setattr__(self, "shared_context", _optional_safe_mapping(self.shared_context, "shared_context"))
        if self.user_input is not None and not isinstance(self.user_input, str):
            raise InvalidMultiAgentExecutionRequestError("user_input must be a string or None.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))
        for field_name in ("task_id", "execution_id", "correlation_id", "session_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _identifier(value, field_name))
        if not isinstance(self.policy, MultiAgentExecutionPolicy):
            raise InvalidMultiAgentExecutionRequestError("policy must be MultiAgentExecutionPolicy.")
        if not _has_team_criteria(self):
            raise InvalidMultiAgentExecutionRequestError("multi-agent execution requires declarative team criteria.")


@dataclass(frozen=True, slots=True)
class MultiAgentTeamRejection:
    """Safe rejection reason for one registered or requested agent."""

    agent_id: str
    reason_code: MultiAgentRejectionCode
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        object.__setattr__(self, "reason_code", _rejection_code(self.reason_code))
        object.__setattr__(self, "message", _safe_message(self.message))


@dataclass(frozen=True, slots=True)
class MultiAgentTeamCandidate:
    """Accepted team candidate with stable score and explanation."""

    agent: AgentDefinition
    score: int
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.agent, AgentDefinition):
            raise InvalidMultiAgentExecutionRequestError("candidate agent must be AgentDefinition.")
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise InvalidMultiAgentExecutionRequestError("candidate score must be an int.")
        object.__setattr__(self, "reasons", _safe_string_tuple(self.reasons, "reasons"))


@dataclass(frozen=True, slots=True)
class MultiAgentTeamResolutionResult:
    """Immutable result of deterministic team resolution."""

    status: MultiAgentTeamResolutionStatus
    selected_agents: tuple[AgentDefinition, ...] = ()
    candidates: tuple[MultiAgentTeamCandidate, ...] = ()
    rejections: tuple[MultiAgentTeamRejection, ...] = ()
    criteria: Mapping[str, object] = field(default_factory=dict)
    request_signature: str = ""
    result_signature: str = ""
    error_code: str | None = None
    safe_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _team_status(self.status))
        object.__setattr__(self, "selected_agents", tuple(self.selected_agents))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "rejections", tuple(self.rejections))
        object.__setattr__(self, "criteria", MappingProxyType(_safe_mapping(self.criteria, "criteria")))
        if self.safe_message is not None:
            object.__setattr__(self, "safe_message", _safe_message(self.safe_message))

    @property
    def selected_agent_ids(self) -> tuple[str, ...]:
        """Return selected agent ids in execution order."""

        return tuple(agent.agent_id for agent in self.selected_agents)


@dataclass(frozen=True, slots=True)
class MultiAgentExecutionStep:
    """One explicit sequential step in a multi-agent plan."""

    step_id: str
    agent_id: str
    input_binding: tuple[str, ...] = ("request_payload",)
    required_capability_ids: tuple[str, ...] = ()
    required_permission_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    failure_policy: MultiAgentFailurePolicy | str = MultiAgentFailurePolicy.STOP_ON_FIRST_FAILURE
    output_binding: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        object.__setattr__(self, "input_binding", _safe_string_tuple(self.input_binding, "input_binding"))
        object.__setattr__(
            self,
            "required_capability_ids",
            _sorted_identifier_tuple(self.required_capability_ids, "required_capability_ids"),
        )
        object.__setattr__(self, "required_permission_ids", _permission_tuple(self.required_permission_ids, "required_permission_ids"))
        object.__setattr__(self, "depends_on", _safe_string_tuple(self.depends_on, "depends_on"))
        object.__setattr__(self, "failure_policy", _failure_policy(self.failure_policy))
        if self.output_binding is not None:
            object.__setattr__(self, "output_binding", _identifier(self.output_binding, "output_binding"))


@dataclass(frozen=True, slots=True)
class MultiAgentExecutionPlan:
    """Immutable sequential plan for selected agents."""

    steps: tuple[MultiAgentExecutionStep, ...]
    request_signature: str
    plan_signature: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            raise InvalidMultiAgentExecutionRequestError("plan requires at least one step.")
        if len(self.steps) > MAX_MULTI_AGENT_STEPS:
            raise InvalidMultiAgentExecutionRequestError("plan exceeds the step limit.")
        step_ids = tuple(step.step_id for step in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise InvalidMultiAgentExecutionRequestError("plan step ids must be unique.")
        agent_ids = tuple(step.agent_id for step in self.steps)
        if len(set(agent_ids)) != len(agent_ids):
            raise InvalidMultiAgentExecutionRequestError("plan agent ids must be unique.")
        seen: set[str] = set()
        for step in self.steps:
            if any(dependency not in seen for dependency in step.depends_on):
                raise InvalidMultiAgentExecutionRequestError("step dependencies must refer to previous steps only.")
            seen.add(step.step_id)
        if not self.plan_signature:
            object.__setattr__(self, "plan_signature", _signature(_plan_payload(self.steps, self.request_signature)))


@dataclass(frozen=True, slots=True)
class MultiAgentStepResult:
    """Result of executing one planned agent step."""

    step_id: str
    agent_id: str
    status: AgentExecutionStatus
    execution_result: AgentExecutionResult | None = None
    output: Mapping[str, object] | None = None
    error_code: str | None = None
    safe_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        if not isinstance(self.status, AgentExecutionStatus):
            object.__setattr__(self, "status", AgentExecutionStatus(self.status))
        if self.execution_result is not None and not isinstance(self.execution_result, AgentExecutionResult):
            raise InvalidMultiAgentExecutionRequestError("execution_result must be AgentExecutionResult or None.")
        object.__setattr__(self, "output", None if self.output is None else MappingProxyType(_safe_mapping(self.output, "output")))
        if self.safe_message is not None:
            object.__setattr__(self, "safe_message", _safe_message(self.safe_message))


@dataclass(frozen=True, slots=True)
class MultiAgentAggregationResult:
    """Deterministic structured aggregation of step results."""

    status: MultiAgentAggregationStatus
    output: Mapping[str, object]
    aggregation_signature: str
    error_code: str | None = None
    safe_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _aggregation_status(self.status))
        object.__setattr__(self, "output", MappingProxyType(_safe_mapping(self.output, "output")))
        if self.safe_message is not None:
            object.__setattr__(self, "safe_message", _safe_message(self.safe_message))


@dataclass(frozen=True, slots=True)
class MultiAgentExecutionEvent:
    """Safe event emitted during multi-agent coordination."""

    name: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "event name"))
        if not isinstance(self.status, str) or not self.status.strip():
            raise InvalidMultiAgentExecutionRequestError("event status must be a non-empty string.")
        object.__setattr__(self, "status", self.status.strip())
        object.__setattr__(self, "details", MappingProxyType(_safe_metadata(self.details)))


@dataclass(frozen=True, slots=True)
class MultiAgentExecutionResult:
    """Terminal result of sequential multi-agent coordination."""

    status: MultiAgentExecutionStatus
    request_signature: str
    team_resolution_result: MultiAgentTeamResolutionResult | None = None
    plan: MultiAgentExecutionPlan | None = None
    step_results: tuple[MultiAgentStepResult, ...] = ()
    aggregation_result: MultiAgentAggregationResult | None = None
    output: Mapping[str, object] | None = None
    events: tuple[MultiAgentExecutionEvent, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    error_code: str | None = None
    safe_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _execution_status(self.status))
        object.__setattr__(self, "step_results", tuple(self.step_results))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "metrics", MappingProxyType(_safe_int_mapping(self.metrics, "metrics")))
        object.__setattr__(self, "output", None if self.output is None else MappingProxyType(_safe_mapping(self.output, "output")))
        if self.safe_message is not None:
            object.__setattr__(self, "safe_message", _safe_message(self.safe_message))

    @property
    def completed(self) -> bool:
        """Return whether all selected agents completed."""

        return self.status is MultiAgentExecutionStatus.SUCCESS


class MultiAgentResolver:
    """Resolve deterministic bounded teams from AgentRegistry."""

    def __init__(self, agent_registry: AgentRegistry) -> None:
        if not isinstance(agent_registry, AgentRegistry):
            raise MultiAgentExecutionError("MultiAgentResolver requires AgentRegistry.")
        self._agent_registry = agent_registry

    def resolve(self, request: MultiAgentExecutionRequest) -> MultiAgentTeamResolutionResult:
        """Resolve a team without executing handlers."""

        if not isinstance(request, MultiAgentExecutionRequest):
            return _team_result(
                MultiAgentTeamResolutionStatus.INVALID_REQUEST,
                "",
                error_code="INVALID_REQUEST",
                safe_message="request must be MultiAgentExecutionRequest.",
            )
        signature = multi_agent_execution_request_signature(request)
        try:
            agents = self._agent_registry.list_agents(enabled_only=False)
        except (RuntimeError, ValueError, TypeError):
            return _team_result(
                MultiAgentTeamResolutionStatus.REGISTRY_UNAVAILABLE,
                signature,
                error_code="REGISTRY_UNAVAILABLE",
                safe_message="agent registry is unavailable.",
            )

        by_id = {agent.agent_id: agent for agent in agents}
        rejections: list[MultiAgentTeamRejection] = []
        candidates: list[MultiAgentTeamCandidate] = []
        requested_missing = tuple(agent_id for agent_id in request.required_agent_ids if agent_id not in by_id)
        for agent_id in requested_missing:
            rejections.append(
                MultiAgentTeamRejection(agent_id, MultiAgentRejectionCode.REQUIRED_AGENT_ID_MISSING, "required agent is not registered")
            )
        if requested_missing:
            return _team_result(
                MultiAgentTeamResolutionStatus.NO_MATCHING_TEAM,
                signature,
                rejections=tuple(rejections),
                criteria=_criteria(request),
                error_code="REQUIRED_AGENT_ID_MISSING",
                safe_message="one or more required agents are not registered.",
            )

        pool = tuple(by_id[agent_id] for agent_id in request.required_agent_ids) if request.required_agent_ids else agents
        for agent in pool:
            rejection = _team_rejection(agent, request)
            if rejection is not None:
                rejections.append(rejection)
                continue
            candidates.append(_team_candidate(agent, request))

        ordered = tuple(sorted(candidates, key=lambda candidate: _team_sort_key(candidate, request)))
        selected = ordered[: request.policy.max_agents]
        if len(ordered) > request.policy.max_agents:
            for candidate in ordered[request.policy.max_agents :]:
                rejections.append(
                    MultiAgentTeamRejection(
                        candidate.agent.agent_id,
                        MultiAgentRejectionCode.MAX_AGENTS_LIMIT,
                        "agent was not selected because max_agents was reached",
                    )
                )
        if len(selected) < request.policy.min_agents:
            return _team_result(
                MultiAgentTeamResolutionStatus.NO_MATCHING_TEAM,
                signature,
                candidates=ordered,
                rejections=tuple(rejections),
                criteria=_criteria(request),
                error_code="MIN_AGENTS_NOT_REACHED",
                safe_message="not enough compatible agents were selected.",
            )
        if request.policy.require_unique_team and len(ordered) > request.policy.max_agents:
            return _team_result(
                MultiAgentTeamResolutionStatus.AMBIGUOUS,
                signature,
                candidates=ordered,
                rejections=tuple(rejections),
                criteria=_criteria(request),
                error_code="TEAM_RESOLUTION_AMBIGUOUS",
                safe_message="more compatible agents exist than max_agents allows.",
            )
        return _team_result(
            MultiAgentTeamResolutionStatus.RESOLVED,
            signature,
            selected_agents=tuple(candidate.agent for candidate in selected),
            candidates=ordered,
            rejections=tuple(rejections),
            criteria=_criteria(request),
        )


class MultiAgentCoordinator:
    """Coordinate a deterministic sequential team through AgentExecutor."""

    def __init__(self, team_resolver: MultiAgentResolver, agent_executor: AgentExecutor) -> None:
        if not isinstance(team_resolver, MultiAgentResolver):
            raise MultiAgentExecutionError("MultiAgentCoordinator requires MultiAgentResolver.")
        if not isinstance(agent_executor, AgentExecutor):
            raise MultiAgentExecutionError("MultiAgentCoordinator requires AgentExecutor.")
        self._team_resolver = team_resolver
        self._agent_executor = agent_executor

    def execute(self, request: MultiAgentExecutionRequest) -> MultiAgentExecutionResult:
        """Resolve, plan, execute sequentially, and aggregate a team."""

        if not isinstance(request, MultiAgentExecutionRequest):
            return _execution_result(
                MultiAgentExecutionStatus.INVALID_REQUEST,
                "",
                error_code="INVALID_REQUEST",
                safe_message="request must be MultiAgentExecutionRequest.",
            )
        signature = multi_agent_execution_request_signature(request)
        events: list[MultiAgentExecutionEvent] = []
        _event(events, "multi_agent_execution_requested", "started")
        _event(events, "multi_agent_team_resolution_started", "started")
        team = self._team_resolver.resolve(request)
        if team.status is not MultiAgentTeamResolutionStatus.RESOLVED:
            _event(events, "multi_agent_team_resolution_failed", "failed", {"team_status": team.status.value})
            return _execution_result(
                _execution_status_for_team(team.status),
                signature,
                team_resolution_result=team,
                events=events,
                metrics=_metrics(requested=1, failed=1, resolution_failures=1),
                error_code=team.error_code,
                safe_message=team.safe_message,
            )
        _event(events, "multi_agent_team_resolved", "finished", {"selected_agents": len(team.selected_agents)})

        try:
            plan = _build_plan(request, team)
        except (RuntimeError, ValueError, TypeError) as error:
            _event(events, "multi_agent_execution_failed", "failed", {"reason": "plan_failed"})
            return _execution_result(
                MultiAgentExecutionStatus.INVALID_REQUEST,
                signature,
                team_resolution_result=team,
                events=events,
                metrics=_metrics(requested=1, failed=1),
                error_code="INVALID_PLAN",
                safe_message=str(error),
            )
        _event(events, "multi_agent_plan_created", "finished", {"steps": len(plan.steps)})

        step_results: list[MultiAgentStepResult] = []
        previous_outputs: dict[str, object] = {}
        stopped = False
        for step in plan.steps:
            _event(events, "multi_agent_step_started", "started", {"step_id": step.step_id, "agent_id": step.agent_id})
            structured_input = _step_structured_input(request, previous_outputs)
            execution_request = AgentExecutionRequest(
                resolution_request=AgentResolutionRequest(
                    required_agent_ids=(step.agent_id,),
                    enabled_only=False,
                    require_unique_top_score=False,
                    metadata={"route": "multi_agent", "step_id": step.step_id},
                ),
                task_id=request.task_id,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                session_id=request.session_id,
                user_input=request.user_input,
                structured_input=structured_input,
                shared_context=request.shared_context,
                metadata=request.metadata,
                required_capability_ids=step.required_capability_ids,
                required_permission_ids=step.required_permission_ids,
            )
            execution_result = self._agent_executor.execute(execution_request)
            if execution_result.status is AgentExecutionStatus.COMPLETED:
                output = {} if execution_result.output is None else dict(execution_result.output)
                step_results.append(
                    MultiAgentStepResult(
                        step.step_id,
                        step.agent_id,
                        execution_result.status,
                        execution_result=execution_result,
                        output=output,
                    )
                )
                previous_outputs[step.output_binding or step.agent_id] = output
                _event(events, "multi_agent_step_succeeded", "finished", {"step_id": step.step_id, "agent_id": step.agent_id})
                continue

            step_results.append(
                MultiAgentStepResult(
                    step.step_id,
                    step.agent_id,
                    execution_result.status,
                    execution_result=execution_result,
                    error_code=execution_result.error_code,
                    safe_message=execution_result.safe_message,
                )
            )
            _event(events, "multi_agent_step_failed", "failed", {"step_id": step.step_id, "agent_id": step.agent_id})
            if request.policy.failure_policy is MultiAgentFailurePolicy.STOP_ON_FIRST_FAILURE:
                stopped = True
                break

        _event(events, "multi_agent_aggregation_started", "started")
        try:
            aggregation = _aggregate(team, step_results, stopped)
        except (RuntimeError, ValueError, TypeError) as error:
            _event(events, "multi_agent_aggregation_failed", "failed")
            return _execution_result(
                MultiAgentExecutionStatus.AGGREGATION_FAILED,
                signature,
                team_resolution_result=team,
                plan=plan,
                step_results=tuple(step_results),
                events=events,
                metrics=_metrics(requested=1, failed=1, steps_started=len(step_results), steps_failed=_failed_steps(step_results), aggregations_failed=1),
                error_code="AGGREGATION_FAILED",
                safe_message=str(error),
            )
        _event(events, "multi_agent_aggregation_succeeded", "finished")

        status = _terminal_status(step_results, stopped, request.policy)
        _event(
            events,
            "multi_agent_execution_completed" if status in (MultiAgentExecutionStatus.SUCCESS, MultiAgentExecutionStatus.PARTIAL_SUCCESS) else "multi_agent_execution_failed",
            "finished" if status in (MultiAgentExecutionStatus.SUCCESS, MultiAgentExecutionStatus.PARTIAL_SUCCESS) else "failed",
            {"status": status.value},
        )
        return _execution_result(
            status,
            signature,
            team_resolution_result=team,
            plan=plan,
            step_results=tuple(step_results),
            aggregation_result=aggregation,
            output=aggregation.output,
            events=events,
            metrics=_metrics(
                requested=1,
                succeeded=1 if status is MultiAgentExecutionStatus.SUCCESS else 0,
                partial=1 if status is MultiAgentExecutionStatus.PARTIAL_SUCCESS else 0,
                failed=1 if status not in (MultiAgentExecutionStatus.SUCCESS, MultiAgentExecutionStatus.PARTIAL_SUCCESS) else 0,
                teams_resolved=1,
                steps_started=len(step_results),
                steps_succeeded=_succeeded_steps(step_results),
                steps_failed=_failed_steps(step_results),
                aggregations_succeeded=1,
            ),
            error_code=None if status in (MultiAgentExecutionStatus.SUCCESS, MultiAgentExecutionStatus.PARTIAL_SUCCESS) else "AGENT_EXECUTION_FAILED",
            safe_message=None if status is MultiAgentExecutionStatus.SUCCESS else "one or more agent steps failed.",
        )


def multi_agent_execution_request_signature(request: MultiAgentExecutionRequest) -> str:
    """Return a stable SHA-256 signature for a multi-agent execution request."""

    if not isinstance(request, MultiAgentExecutionRequest):
        raise InvalidMultiAgentExecutionRequestError("request must be MultiAgentExecutionRequest.")
    return _signature(
        {
            "required_agent_ids": request.required_agent_ids,
            "required_agent_types": tuple(agent_type.value for agent_type in request.required_agent_types),
            "required_capability_ids": request.required_capability_ids,
            "required_permission_ids": request.required_permission_ids,
            "excluded_agent_ids": request.excluded_agent_ids,
            "payload": _jsonable(request.payload),
            "shared_context": _jsonable(request.shared_context),
            "user_input": request.user_input,
            "metadata": _jsonable(request.metadata),
            "task_id": request.task_id,
            "execution_id": request.execution_id,
            "correlation_id": request.correlation_id,
            "session_id": request.session_id,
            "policy": {
                "min_agents": request.policy.min_agents,
                "max_agents": request.policy.max_agents,
                "enabled_only": request.policy.enabled_only,
                "failure_policy": request.policy.failure_policy.value,
                "require_unique_team": request.policy.require_unique_team,
                "share_previous_outputs": request.policy.share_previous_outputs,
            },
        }
    )


def _has_team_criteria(request: MultiAgentExecutionRequest) -> bool:
    return bool(request.required_agent_ids or request.required_agent_types or request.required_capability_ids)


def _team_rejection(agent: AgentDefinition, request: MultiAgentExecutionRequest) -> MultiAgentTeamRejection | None:
    if request.policy.enabled_only and not agent.enabled:
        return MultiAgentTeamRejection(agent.agent_id, MultiAgentRejectionCode.DISABLED, "agent is disabled")
    if agent.agent_id in request.excluded_agent_ids:
        return MultiAgentTeamRejection(agent.agent_id, MultiAgentRejectionCode.EXCLUDED, "agent is excluded")
    if request.required_agent_types and agent.agent_type not in request.required_agent_types:
        return MultiAgentTeamRejection(agent.agent_id, MultiAgentRejectionCode.REQUIRED_AGENT_TYPE_MISSING, "required type is missing")
    if any(capability not in agent.capabilities.capabilities for capability in request.required_capability_ids):
        return MultiAgentTeamRejection(agent.agent_id, MultiAgentRejectionCode.REQUIRED_CAPABILITY_MISSING, "required capability is missing")
    if any(not bool(getattr(agent.permissions, permission)) for permission in request.required_permission_ids):
        return MultiAgentTeamRejection(agent.agent_id, MultiAgentRejectionCode.REQUIRED_PERMISSION_MISSING, "required permission is missing")
    return None


def _team_candidate(agent: AgentDefinition, request: MultiAgentExecutionRequest) -> MultiAgentTeamCandidate:
    score = 1
    reasons = ["enabled_agent"] if agent.enabled else []
    for capability in request.required_capability_ids:
        if capability in agent.capabilities.capabilities:
            score += 30
            reasons.append(f"required_capability:{capability}")
    for permission in request.required_permission_ids:
        if bool(getattr(agent.permissions, permission)):
            score += 10
            reasons.append(f"required_permission:{permission}")
    if agent.agent_type in request.required_agent_types:
        score += 20
        reasons.append(f"required_type:{agent.agent_type.value}")
    if agent.agent_id in request.required_agent_ids:
        score += 100
        reasons.append("required_agent_id")
    return MultiAgentTeamCandidate(agent, score, tuple(reasons))


def _team_sort_key(candidate: MultiAgentTeamCandidate, request: MultiAgentExecutionRequest) -> tuple[object, ...]:
    required_index = (
        request.required_agent_ids.index(candidate.agent.agent_id)
        if candidate.agent.agent_id in request.required_agent_ids
        else MAX_MULTI_AGENT_IDS + 1
    )
    return (required_index, -candidate.score, candidate.agent.agent_type.value, candidate.agent.agent_id)


def _build_plan(request: MultiAgentExecutionRequest, team: MultiAgentTeamResolutionResult) -> MultiAgentExecutionPlan:
    steps = tuple(
        MultiAgentExecutionStep(
            step_id=f"step_{index:03d}",
            agent_id=agent.agent_id,
            input_binding=("request_payload", "shared_context")
            if not request.policy.share_previous_outputs
            else ("request_payload", "shared_context", "previous_outputs"),
            required_capability_ids=request.required_capability_ids,
            required_permission_ids=request.required_permission_ids,
            depends_on=() if index == 1 else (f"step_{index - 1:03d}",),
            failure_policy=request.policy.failure_policy,
            output_binding=agent.agent_id,
        )
        for index, agent in enumerate(team.selected_agents, start=1)
    )
    return MultiAgentExecutionPlan(steps, request.request_signature if hasattr(request, "request_signature") else multi_agent_execution_request_signature(request))


def _step_structured_input(request: MultiAgentExecutionRequest, previous_outputs: Mapping[str, object]) -> Mapping[str, object] | None:
    base: dict[str, object] = {}
    if request.payload is not None:
        base.update(dict(request.payload))
    if request.policy.share_previous_outputs and previous_outputs:
        base["previous_outputs"] = dict(previous_outputs)
    return MappingProxyType(_safe_mapping(base, "structured_input")) if base else None


def _aggregate(
    team: MultiAgentTeamResolutionResult,
    step_results: list[MultiAgentStepResult],
    stopped: bool,
) -> MultiAgentAggregationResult:
    completed = tuple(result.agent_id for result in step_results if result.status is AgentExecutionStatus.COMPLETED)
    failed = tuple(result.agent_id for result in step_results if result.status is not AgentExecutionStatus.COMPLETED)
    outputs = {
        result.agent_id: ({} if result.output is None else dict(result.output))
        for result in step_results
        if result.status is AgentExecutionStatus.COMPLETED
    }
    output = {
        "team": team.selected_agent_ids,
        "completed_agents": completed,
        "failed_agents": failed,
        "outputs": outputs,
        "summary": {
            "status": "stopped" if stopped else "completed",
            "completed_count": len(completed),
            "failed_count": len(failed),
        },
    }
    return MultiAgentAggregationResult(
        MultiAgentAggregationStatus.AGGREGATED,
        output,
        _signature(_jsonable(output)),
    )


def _terminal_status(
    step_results: list[MultiAgentStepResult],
    stopped: bool,
    policy: MultiAgentExecutionPolicy,
) -> MultiAgentExecutionStatus:
    if not step_results:
        return MultiAgentExecutionStatus.FAILED
    failed = _failed_steps(step_results)
    succeeded = _succeeded_steps(step_results)
    if failed == 0:
        return MultiAgentExecutionStatus.SUCCESS
    if policy.failure_policy is MultiAgentFailurePolicy.CONTINUE_ON_FAILURE and succeeded > 0:
        return MultiAgentExecutionStatus.PARTIAL_SUCCESS
    if stopped and succeeded > 0:
        return MultiAgentExecutionStatus.AGENT_EXECUTION_FAILED
    return MultiAgentExecutionStatus.FAILED


def _execution_status_for_team(status: MultiAgentTeamResolutionStatus) -> MultiAgentExecutionStatus:
    if status is MultiAgentTeamResolutionStatus.INVALID_REQUEST:
        return MultiAgentExecutionStatus.INVALID_REQUEST
    if status is MultiAgentTeamResolutionStatus.AMBIGUOUS:
        return MultiAgentExecutionStatus.TEAM_RESOLUTION_AMBIGUOUS
    if status is MultiAgentTeamResolutionStatus.REGISTRY_UNAVAILABLE:
        return MultiAgentExecutionStatus.SERVICE_UNAVAILABLE
    return MultiAgentExecutionStatus.NO_MATCHING_TEAM


def _succeeded_steps(results: Iterable[MultiAgentStepResult]) -> int:
    return sum(1 for result in results if result.status is AgentExecutionStatus.COMPLETED)


def _failed_steps(results: Iterable[MultiAgentStepResult]) -> int:
    return sum(1 for result in results if result.status is not AgentExecutionStatus.COMPLETED)


def _team_result(
    status: MultiAgentTeamResolutionStatus,
    request_signature: str,
    *,
    selected_agents: tuple[AgentDefinition, ...] = (),
    candidates: tuple[MultiAgentTeamCandidate, ...] = (),
    rejections: tuple[MultiAgentTeamRejection, ...] = (),
    criteria: Mapping[str, object] | None = None,
    error_code: str | None = None,
    safe_message: str | None = None,
) -> MultiAgentTeamResolutionResult:
    payload = {
        "status": status.value,
        "selected_agent_ids": tuple(agent.agent_id for agent in selected_agents),
        "candidate_agent_ids": tuple(candidate.agent.agent_id for candidate in candidates),
        "rejections": tuple({"agent_id": item.agent_id, "reason": item.reason_code.value} for item in rejections),
        "request_signature": request_signature,
    }
    return MultiAgentTeamResolutionResult(
        status,
        selected_agents=selected_agents,
        candidates=candidates,
        rejections=rejections,
        criteria={} if criteria is None else criteria,
        request_signature=request_signature,
        result_signature=_signature(payload),
        error_code=error_code,
        safe_message=safe_message,
    )


def _execution_result(
    status: MultiAgentExecutionStatus,
    request_signature: str,
    *,
    team_resolution_result: MultiAgentTeamResolutionResult | None = None,
    plan: MultiAgentExecutionPlan | None = None,
    step_results: tuple[MultiAgentStepResult, ...] = (),
    aggregation_result: MultiAgentAggregationResult | None = None,
    output: Mapping[str, object] | None = None,
    events: list[MultiAgentExecutionEvent] | tuple[MultiAgentExecutionEvent, ...] = (),
    metrics: Mapping[str, int] | None = None,
    error_code: str | None = None,
    safe_message: str | None = None,
) -> MultiAgentExecutionResult:
    return MultiAgentExecutionResult(
        status,
        request_signature,
        team_resolution_result=team_resolution_result,
        plan=plan,
        step_results=step_results,
        aggregation_result=aggregation_result,
        output=output,
        events=tuple(events),
        metrics={} if metrics is None else metrics,
        error_code=error_code,
        safe_message=safe_message,
    )


def _event(
    events: list[MultiAgentExecutionEvent],
    name: str,
    status: str,
    details: Mapping[str, object] | None = None,
) -> None:
    events.append(MultiAgentExecutionEvent(name, status, {} if details is None else details))


def _metrics(
    *,
    requested: int = 0,
    succeeded: int = 0,
    partial: int = 0,
    failed: int = 0,
    teams_resolved: int = 0,
    resolution_failures: int = 0,
    steps_started: int = 0,
    steps_succeeded: int = 0,
    steps_failed: int = 0,
    aggregations_succeeded: int = 0,
    aggregations_failed: int = 0,
) -> Mapping[str, int]:
    return {
        "multi_agent_executions_requested": requested,
        "multi_agent_executions_succeeded": succeeded,
        "multi_agent_executions_partial": partial,
        "multi_agent_executions_failed": failed,
        "multi_agent_teams_resolved": teams_resolved,
        "multi_agent_team_resolution_failures": resolution_failures,
        "multi_agent_steps_started": steps_started,
        "multi_agent_steps_succeeded": steps_succeeded,
        "multi_agent_steps_failed": steps_failed,
        "multi_agent_aggregations_succeeded": aggregations_succeeded,
        "multi_agent_aggregations_failed": aggregations_failed,
    }


def _criteria(request: MultiAgentExecutionRequest) -> Mapping[str, object]:
    return {
        "required_agent_ids": request.required_agent_ids,
        "required_agent_types": tuple(agent_type.value for agent_type in request.required_agent_types),
        "required_capability_ids": request.required_capability_ids,
        "required_permission_ids": request.required_permission_ids,
        "excluded_agent_ids": request.excluded_agent_ids,
        "min_agents": request.policy.min_agents,
        "max_agents": request.policy.max_agents,
        "enabled_only": request.policy.enabled_only,
    }


def _failure_policy(value: MultiAgentFailurePolicy | str) -> MultiAgentFailurePolicy:
    if isinstance(value, MultiAgentFailurePolicy):
        return value
    if isinstance(value, str):
        return MultiAgentFailurePolicy(value)
    raise InvalidMultiAgentExecutionRequestError("failure_policy is invalid.")


def _team_status(value: MultiAgentTeamResolutionStatus | str) -> MultiAgentTeamResolutionStatus:
    if isinstance(value, MultiAgentTeamResolutionStatus):
        return value
    if isinstance(value, str):
        return MultiAgentTeamResolutionStatus(value)
    raise InvalidMultiAgentExecutionRequestError("team status is invalid.")


def _execution_status(value: MultiAgentExecutionStatus | str) -> MultiAgentExecutionStatus:
    if isinstance(value, MultiAgentExecutionStatus):
        return value
    if isinstance(value, str):
        return MultiAgentExecutionStatus(value)
    raise InvalidMultiAgentExecutionRequestError("execution status is invalid.")


def _aggregation_status(value: MultiAgentAggregationStatus | str) -> MultiAgentAggregationStatus:
    if isinstance(value, MultiAgentAggregationStatus):
        return value
    if isinstance(value, str):
        return MultiAgentAggregationStatus(value)
    raise InvalidMultiAgentExecutionRequestError("aggregation status is invalid.")


def _rejection_code(value: MultiAgentRejectionCode | str) -> MultiAgentRejectionCode:
    if isinstance(value, MultiAgentRejectionCode):
        return value
    if isinstance(value, str):
        return MultiAgentRejectionCode(value)
    raise InvalidMultiAgentExecutionRequestError("rejection code is invalid.")


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} must be an integer.")
    if value <= 0 or value > MAX_MULTI_AGENT_STEPS:
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} is outside the allowed range.")
    return value


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if normalized != value:
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} cannot contain surrounding whitespace.")
    if len(normalized) > MAX_MULTI_AGENT_STRING_LENGTH:
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} is too long.")
    if _is_sensitive_key(normalized):
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} cannot contain sensitive content.")
    return normalized


def _safe_string_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} must be an iterable.")
    normalized = tuple(dict.fromkeys(_identifier(value, field_name) for value in values))
    if len(normalized) > MAX_MULTI_AGENT_IDS:
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} has too many items.")
    return normalized


def _identifier_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    return _safe_string_tuple(values, field_name)


def _sorted_identifier_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    return tuple(sorted(_identifier_tuple(values, field_name)))


def _agent_id_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} must be an iterable.")
    try:
        normalized = tuple(dict.fromkeys(validate_agent_id(value) for value in values))
    except Exception as error:
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} contains an invalid agent id.") from error
    if len(normalized) > MAX_MULTI_AGENT_IDS:
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} has too many items.")
    return normalized


def _sorted_agent_id_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    return tuple(sorted(_agent_id_tuple(values, field_name)))


def _agent_type_tuple(values: Iterable[AgentType | str], field_name: str) -> tuple[AgentType, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} must be an iterable.")
    normalized: list[AgentType] = []
    for value in values:
        if isinstance(value, AgentType):
            agent_type = value
        elif isinstance(value, str):
            agent_type = AgentType(value.strip().lower())
        else:
            raise InvalidMultiAgentExecutionRequestError(f"{field_name} contains an invalid value.")
        if agent_type not in normalized:
            normalized.append(agent_type)
    if len(normalized) > MAX_MULTI_AGENT_IDS:
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} has too many items.")
    return tuple(normalized)


def _permission_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    normalized = _sorted_identifier_tuple(values, field_name)
    invalid = tuple(permission for permission in normalized if permission not in _PERMISSION_IDS)
    if invalid:
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} contains an unsupported permission.")
    return normalized


def _optional_safe_mapping(value: Mapping[str, object] | None, field_name: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return MappingProxyType(_safe_mapping(value, field_name))


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidMultiAgentExecutionRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_MULTI_AGENT_METADATA_ITEMS:
        raise InvalidMultiAgentExecutionRequestError("metadata has too many items.")
    return _safe_mapping(metadata, "metadata")


def _safe_int_mapping(mapping: Mapping[str, int], field_name: str) -> dict[str, int]:
    if not isinstance(mapping, Mapping):
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} must be a mapping.")
    safe: dict[str, int] = {}
    for key, value in mapping.items():
        normalized = _identifier(key, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidMultiAgentExecutionRequestError(f"{field_name} values must be non-negative integers.")
        safe[normalized] = value
    return safe


def _safe_mapping(mapping: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(mapping, Mapping):
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} must be a mapping.")
    safe = _copy_safe_value(mapping, depth=0, counter={"nodes": 0}, field_name=field_name)
    if not isinstance(safe, Mapping):
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} must be a mapping.")
    return dict(safe)


def _copy_safe_value(value: object, *, depth: int, counter: dict[str, int], field_name: str) -> object:
    if depth > MAX_MULTI_AGENT_DEPTH:
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} is too deep.")
    counter["nodes"] += 1
    if counter["nodes"] > MAX_MULTI_AGENT_NODES:
        raise InvalidMultiAgentExecutionRequestError(f"{field_name} is too large.")
    if value is None or type(value) in (bool, int, str):
        if isinstance(value, str) and len(value) > MAX_MULTI_AGENT_STRING_LENGTH:
            raise InvalidMultiAgentExecutionRequestError(f"{field_name} strings are too long.")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidMultiAgentExecutionRequestError(f"{field_name} floats must be finite.")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
            key = _identifier(raw_key, field_name)
            if _is_sensitive_key(key):
                raise InvalidMultiAgentExecutionRequestError(f"{field_name} cannot contain sensitive keys.")
            copied[key] = _copy_safe_value(raw_value, depth=depth + 1, counter=counter, field_name=field_name)
        return MappingProxyType(copied)
    if isinstance(value, (tuple, list)):
        return tuple(_copy_safe_value(item, depth=depth + 1, counter=counter, field_name=field_name) for item in value)
    raise InvalidMultiAgentExecutionRequestError(f"{field_name} contains an unsupported value.")


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if value is not None and type(value) not in (bool, int, float, str):
        raise InvalidMultiAgentExecutionRequestError("unsupported signature value.")
    return value


def _plan_payload(steps: tuple[MultiAgentExecutionStep, ...], request_signature: str) -> Mapping[str, object]:
    return {
        "request_signature": request_signature,
        "steps": tuple(
            {
                "step_id": step.step_id,
                "agent_id": step.agent_id,
                "input_binding": step.input_binding,
                "required_capability_ids": step.required_capability_ids,
                "required_permission_ids": step.required_permission_ids,
                "depends_on": step.depends_on,
                "failure_policy": step.failure_policy.value,
                "output_binding": step.output_binding,
            }
            for step in steps
        ),
    }


def _signature(value: object) -> str:
    encoded = json.dumps(_jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_message(value: str) -> str:
    message = " ".join(str(value).split())
    for key in _SENSITIVE_KEY_PARTS:
        message = message.replace(key, "[redacted]")
    return message[:300]


def _is_sensitive_key(key: str) -> bool:
    normalized = str(key).replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)

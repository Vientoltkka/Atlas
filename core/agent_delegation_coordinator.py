"""Deterministic coordination of multiple declared agent delegation chains."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import inspect
import json
import math
import types
from types import MappingProxyType

from core.agent_context import AgentContextBuilder
from core.agent_delegation import AgentDelegationService
from core.agent_delegation_chain import (
    AgentDelegationChainRequest,
    AgentDelegationChainResult,
    AgentDelegationChainService,
    AgentDelegationChainStatus,
    agent_delegation_chain_request_signature,
)
from core.agent_executor import AgentExecutor
from core.agent_registry import AgentRegistry, validate_agent_id
from core.agent_resolver import AgentResolver


MAX_COORDINATION_CHAINS = 20
MAX_COORDINATION_TOTAL_STEPS = 100
MAX_COORDINATION_OUTPUT_ITEMS = 128
MAX_COORDINATION_METADATA_ITEMS = 16
MAX_COORDINATION_MAPPING_ITEMS = 32
MAX_COORDINATION_SEQUENCE_ITEMS = 32
MAX_COORDINATION_TOTAL_ITEMS = 512
MAX_COORDINATION_STRING_LENGTH = 1_000
MAX_COORDINATION_EVENTS = 128
PREVIOUS_CHAIN_OUTPUTS_KEY = "previous_chain_outputs"
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


class AgentDelegationCoordinationError(RuntimeError):
    """Base error for delegation coordination."""


class InvalidAgentDelegationCoordinationRequestError(AgentDelegationCoordinationError):
    """Raised when a coordination request is malformed."""


class AgentDelegationCoordinationFailureMode(str, Enum):
    """Supported coordination failure policies."""

    STOP_ON_FIRST_FAILURE = "STOP_ON_FIRST_FAILURE"
    CONTINUE_ON_FAILURE = "CONTINUE_ON_FAILURE"
    REQUIRE_ALL_SUCCESS = "REQUIRE_ALL_SUCCESS"
    REQUIRE_MINIMUM_SUCCESS = "REQUIRE_MINIMUM_SUCCESS"


class AgentDelegationCoordinationStatus(str, Enum):
    """Structured statuses for a delegation coordination run."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INVALID_REQUEST = "INVALID_REQUEST"
    LIMIT_REACHED = "LIMIT_REACHED"
    NO_CHAINS = "NO_CHAINS"
    MINIMUM_SUCCESS_NOT_REACHED = "MINIMUM_SUCCESS_NOT_REACHED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class AgentDelegationCoordinationPolicy:
    """Immutable policy for coordinating declared chains."""

    enabled: bool = False
    failure_mode: AgentDelegationCoordinationFailureMode | str = (
        AgentDelegationCoordinationFailureMode.STOP_ON_FIRST_FAILURE
    )
    min_successful_chains: int = 1
    max_chains: int = 5
    max_total_steps: int = 20
    propagate_chain_outputs: bool = False
    include_partial_results: bool = True
    stop_after_failure: bool = True
    max_output_items: int = MAX_COORDINATION_OUTPUT_ITEMS

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise InvalidAgentDelegationCoordinationRequestError("enabled must be a bool.")
        for field_name in ("propagate_chain_outputs", "include_partial_results", "stop_after_failure"):
            if type(getattr(self, field_name)) is not bool:
                raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} must be a bool.")
        object.__setattr__(self, "failure_mode", _failure_mode(self.failure_mode))
        object.__setattr__(self, "max_chains", _bounded_int(self.max_chains, "max_chains", MAX_COORDINATION_CHAINS))
        object.__setattr__(
            self,
            "max_total_steps",
            _bounded_int(self.max_total_steps, "max_total_steps", MAX_COORDINATION_TOTAL_STEPS),
        )
        object.__setattr__(
            self,
            "max_output_items",
            _bounded_int(self.max_output_items, "max_output_items", MAX_COORDINATION_OUTPUT_ITEMS),
        )
        if isinstance(self.min_successful_chains, bool) or not isinstance(self.min_successful_chains, int):
            raise InvalidAgentDelegationCoordinationRequestError("min_successful_chains must be an integer.")
        if self.min_successful_chains <= 0 or self.min_successful_chains > self.max_chains:
            raise InvalidAgentDelegationCoordinationRequestError("min_successful_chains is outside the allowed range.")


@dataclass(frozen=True, slots=True)
class AgentDelegationCoordinationChain:
    """One named chain entry in a coordination plan."""

    chain_id: str
    chain_request: AgentDelegationChainRequest

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_id", _identifier(self.chain_id, "chain_id"))
        if not isinstance(self.chain_request, AgentDelegationChainRequest):
            raise InvalidAgentDelegationCoordinationRequestError("chain_request must be AgentDelegationChainRequest.")


@dataclass(frozen=True, slots=True)
class AgentDelegationCoordinationPlan:
    """Declarative plan containing named delegation chains."""

    plan_id: str
    chains: Sequence[AgentDelegationCoordinationChain]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        if isinstance(self.chains, (str, bytes)) or not isinstance(self.chains, Sequence):
            raise InvalidAgentDelegationCoordinationRequestError("chains must be a sequence.")
        chains = tuple(self.chains)
        if not chains:
            raise InvalidAgentDelegationCoordinationRequestError("chains cannot be empty.")
        if len(chains) > MAX_COORDINATION_CHAINS:
            raise InvalidAgentDelegationCoordinationRequestError("chains exceeds the absolute limit.")
        if not all(isinstance(chain, AgentDelegationCoordinationChain) for chain in chains):
            raise InvalidAgentDelegationCoordinationRequestError(
                "chains must contain AgentDelegationCoordinationChain values."
            )
        chain_ids = tuple(chain.chain_id for chain in chains)
        if len(set(chain_ids)) != len(chain_ids):
            raise InvalidAgentDelegationCoordinationRequestError("chain_id values must be unique.")
        object.__setattr__(self, "chains", chains)
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))


@dataclass(frozen=True, slots=True)
class AgentDelegationCoordinationRequest:
    """Request for coordinating one declarative multi-chain plan."""

    source_agent_id: str
    plan: AgentDelegationCoordinationPlan
    policy: AgentDelegationCoordinationPolicy = field(default_factory=AgentDelegationCoordinationPolicy)
    structured_input: Mapping[str, object] | None = None
    shared_context: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    execution_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_agent_id", validate_agent_id(self.source_agent_id))
        if not isinstance(self.plan, AgentDelegationCoordinationPlan):
            raise InvalidAgentDelegationCoordinationRequestError("plan must be AgentDelegationCoordinationPlan.")
        if not isinstance(self.policy, AgentDelegationCoordinationPolicy):
            raise InvalidAgentDelegationCoordinationRequestError("policy must be AgentDelegationCoordinationPolicy.")
        object.__setattr__(self, "structured_input", _optional_safe_mapping(self.structured_input, "structured_input"))
        object.__setattr__(self, "shared_context", _optional_safe_mapping(self.shared_context, "shared_context"))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))
        if self.execution_id is not None:
            object.__setattr__(self, "execution_id", _identifier(self.execution_id, "execution_id"))


@dataclass(frozen=True, slots=True)
class AgentDelegationCoordinationEvent:
    """Safe event emitted by the coordination layer."""

    name: str
    status: AgentDelegationCoordinationStatus
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "event name"))
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(self, "details", MappingProxyType(_safe_mapping(self.details, "event details")))


@dataclass(frozen=True, slots=True)
class AgentDelegationCoordinationChainResult:
    """Structured result for one coordinated chain."""

    chain_id: str
    status: AgentDelegationCoordinationStatus
    chain_result: AgentDelegationChainResult | None = None
    output: Mapping[str, object] | None = None
    error_code: str | None = None
    position: int = 0
    started: bool = False
    completed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_id", _identifier(self.chain_id, "chain_id"))
        object.__setattr__(self, "status", _status(self.status))
        if self.output is not None:
            object.__setattr__(self, "output", MappingProxyType(_safe_mapping(self.output, "output")))
        if isinstance(self.position, bool) or not isinstance(self.position, int) or self.position < 0:
            raise InvalidAgentDelegationCoordinationRequestError("position must be a non-negative integer.")
        if type(self.started) is not bool or type(self.completed) is not bool:
            raise InvalidAgentDelegationCoordinationRequestError("started and completed must be bool.")


@dataclass(frozen=True, slots=True)
class AgentDelegationCoordinationResult:
    """Immutable result for a coordination run."""

    status: AgentDelegationCoordinationStatus
    plan_id: str | None
    signature: str
    chain_results: tuple[AgentDelegationCoordinationChainResult, ...] = ()
    successful_chain_ids: tuple[str, ...] = ()
    failed_chain_ids: tuple[str, ...] = ()
    skipped_chain_ids: tuple[str, ...] = ()
    aggregated_outputs: Mapping[str, object] = field(default_factory=dict)
    summary: Mapping[str, object] = field(default_factory=dict)
    events: tuple[AgentDelegationCoordinationEvent, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        if self.plan_id is not None:
            object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(self, "chain_results", tuple(self.chain_results))
        if not all(isinstance(result, AgentDelegationCoordinationChainResult) for result in self.chain_results):
            raise InvalidAgentDelegationCoordinationRequestError(
                "chain_results must contain AgentDelegationCoordinationChainResult values."
            )
        object.__setattr__(self, "successful_chain_ids", _identifier_tuple(self.successful_chain_ids, "successful_chain_ids"))
        object.__setattr__(self, "failed_chain_ids", _identifier_tuple(self.failed_chain_ids, "failed_chain_ids"))
        object.__setattr__(self, "skipped_chain_ids", _identifier_tuple(self.skipped_chain_ids, "skipped_chain_ids"))
        object.__setattr__(self, "aggregated_outputs", MappingProxyType(_safe_mapping(self.aggregated_outputs, "aggregated_outputs")))
        object.__setattr__(self, "summary", MappingProxyType(_safe_mapping(self.summary, "summary")))
        object.__setattr__(self, "events", tuple(self.events))
        if not all(isinstance(event, AgentDelegationCoordinationEvent) for event in self.events):
            raise InvalidAgentDelegationCoordinationRequestError(
                "events must contain AgentDelegationCoordinationEvent values."
            )
        object.__setattr__(self, "metrics", MappingProxyType(_metric_mapping(self.metrics)))
        if self.error_message is not None:
            object.__setattr__(self, "error_message", _safe_message(self.error_message))


class AgentDelegationCoordinator:
    """Coordinate multiple declared chains through AgentDelegationChainService."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        agent_resolver: AgentResolver,
        agent_context_builder: AgentContextBuilder,
        agent_executor: AgentExecutor,
        agent_delegation_service: AgentDelegationService,
        agent_delegation_chain_service: AgentDelegationChainService,
    ) -> None:
        if not isinstance(agent_registry, AgentRegistry):
            raise AgentDelegationCoordinationError("agent_registry must be AgentRegistry.")
        if not isinstance(agent_resolver, AgentResolver):
            raise AgentDelegationCoordinationError("agent_resolver must be AgentResolver.")
        if not isinstance(agent_context_builder, AgentContextBuilder):
            raise AgentDelegationCoordinationError("agent_context_builder must be AgentContextBuilder.")
        if not isinstance(agent_executor, AgentExecutor):
            raise AgentDelegationCoordinationError("agent_executor must be AgentExecutor.")
        if not isinstance(agent_delegation_service, AgentDelegationService):
            raise AgentDelegationCoordinationError("agent_delegation_service must be AgentDelegationService.")
        if not isinstance(agent_delegation_chain_service, AgentDelegationChainService):
            raise AgentDelegationCoordinationError(
                "agent_delegation_chain_service must be AgentDelegationChainService."
            )
        self._agent_registry = agent_registry
        self._agent_resolver = agent_resolver
        self._agent_context_builder = agent_context_builder
        self._agent_executor = agent_executor
        self._agent_delegation_service = agent_delegation_service
        self._agent_delegation_chain_service = agent_delegation_chain_service

    def coordinate(
        self,
        request: AgentDelegationCoordinationRequest,
    ) -> AgentDelegationCoordinationResult:
        """Coordinate declared chains sequentially and aggregate safe outputs."""

        events: list[AgentDelegationCoordinationEvent] = []
        metrics = _base_metrics()
        try:
            if not isinstance(request, AgentDelegationCoordinationRequest):
                raise InvalidAgentDelegationCoordinationRequestError(
                    "request must be AgentDelegationCoordinationRequest."
                )
            signature = agent_delegation_coordination_request_signature(request)
        except (AgentDelegationCoordinationError, TypeError, ValueError) as error:
            metrics["delegation_coordinations_failed"] = 1
            return _result(
                AgentDelegationCoordinationStatus.INVALID_REQUEST,
                signature="",
                metrics=metrics,
                events=events,
                error_code="INVALID_REQUEST",
                error_message=str(error),
            )

        metrics["delegation_coordinations_requested"] = 1
        events.append(_event("agent_delegation_coordination_requested", AgentDelegationCoordinationStatus.SUCCESS, request=request))
        events.append(_event("agent_delegation_coordination_validation_started", AgentDelegationCoordinationStatus.SUCCESS, request=request))
        blocked = self._validate_request(request, signature, events, metrics)
        if blocked is not None:
            return blocked
        events.append(_event("agent_delegation_coordination_validation_succeeded", AgentDelegationCoordinationStatus.SUCCESS, request=request))
        events.append(_event("agent_delegation_coordination_started", AgentDelegationCoordinationStatus.SUCCESS, request=request))
        metrics["delegation_coordinations_started"] = 1

        chain_results: list[AgentDelegationCoordinationChainResult] = []
        successful: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        aggregated: dict[str, object] = {}
        previous_outputs: dict[str, object] = {}

        for position, chain in enumerate(request.plan.chains):
            if _should_skip_remaining(request.policy, failed):
                skipped.append(chain.chain_id)
                metrics["delegation_coordination_chains_skipped"] += 1
                chain_results.append(
                    AgentDelegationCoordinationChainResult(
                        chain_id=chain.chain_id,
                        status=AgentDelegationCoordinationStatus.BLOCKED,
                        position=position,
                        started=False,
                        completed=False,
                        error_code="SKIPPED",
                    )
                )
                events.append(_event("agent_delegation_coordination_chain_skipped", AgentDelegationCoordinationStatus.BLOCKED, request=request, chain_id=chain.chain_id))
                continue

            events.append(_event("agent_delegation_coordination_chain_started", AgentDelegationCoordinationStatus.SUCCESS, request=request, chain_id=chain.chain_id))
            metrics["delegation_coordination_chains_started"] += 1
            try:
                chain_request = _chain_request_for_execution(request, chain.chain_request, previous_outputs)
            except InvalidAgentDelegationCoordinationRequestError as error:
                failed.append(chain.chain_id)
                chain_results.append(
                    AgentDelegationCoordinationChainResult(
                        chain_id=chain.chain_id,
                        status=AgentDelegationCoordinationStatus.FAILED,
                        position=position,
                        started=True,
                        completed=True,
                        error_code="INVALID_CHAIN_CONTEXT",
                    )
                )
                metrics["delegation_coordination_chains_failed"] += 1
                events.append(_event("agent_delegation_coordination_chain_failed", AgentDelegationCoordinationStatus.FAILED, request=request, chain_id=chain.chain_id))
                if request.policy.stop_after_failure:
                    continue
                continue

            result = self._agent_delegation_chain_service.execute(chain_request)
            output = _safe_mapping(result.final_output or {}, "chain_output")
            metrics["delegation_coordination_steps_executed"] += result.completed_steps
            chain_status = _status_from_chain(result.status)
            chain_results.append(
                AgentDelegationCoordinationChainResult(
                    chain_id=chain.chain_id,
                    status=chain_status,
                    chain_result=result,
                    output=output if (result.status is AgentDelegationChainStatus.SUCCESS or request.policy.include_partial_results) else None,
                    error_code=result.error_code,
                    position=position,
                    started=True,
                    completed=True,
                )
            )
            if result.status is AgentDelegationChainStatus.SUCCESS:
                successful.append(chain.chain_id)
                previous_outputs[chain.chain_id] = output
                aggregated[chain.chain_id] = output
                if _count_items(aggregated) > request.policy.max_output_items:
                    metrics["delegation_coordination_limits_reached"] = 1
                    metrics["delegation_coordinations_failed"] = 1
                    events.append(
                        _event(
                            "agent_delegation_coordination_limit_reached",
                            AgentDelegationCoordinationStatus.LIMIT_REACHED,
                            request=request,
                        )
                    )
                    return _result(
                        AgentDelegationCoordinationStatus.LIMIT_REACHED,
                        signature=signature,
                        request=request,
                        chain_results=tuple(chain_results),
                        successful_chain_ids=tuple(successful),
                        failed_chain_ids=tuple(failed),
                        skipped_chain_ids=tuple(skipped),
                        aggregated_outputs=aggregated,
                        summary=_summary(request, AgentDelegationCoordinationStatus.LIMIT_REACHED, successful, failed, skipped, chain_results),
                        metrics=metrics,
                        events=events,
                        error_code="LIMIT_REACHED",
                        error_message="maximum output items reached.",
                    )
                metrics["delegation_coordination_chains_succeeded"] += 1
                events.append(_event("agent_delegation_coordination_chain_succeeded", AgentDelegationCoordinationStatus.SUCCESS, request=request, chain_id=chain.chain_id))
            else:
                failed.append(chain.chain_id)
                if request.policy.include_partial_results and output:
                    previous_outputs[chain.chain_id] = output
                    aggregated[chain.chain_id] = output
                metrics["delegation_coordination_chains_failed"] += 1
                events.append(_event("agent_delegation_coordination_chain_failed", AgentDelegationCoordinationStatus.FAILED, request=request, chain_id=chain.chain_id))

        events.append(_event("agent_delegation_coordination_aggregation_started", AgentDelegationCoordinationStatus.SUCCESS, request=request))
        final_status = _final_status(request.policy, len(request.plan.chains), len(successful), len(failed), len(skipped))
        if final_status is AgentDelegationCoordinationStatus.SUCCESS:
            metrics["delegation_coordinations_succeeded"] = 1
            events.append(_event("agent_delegation_coordination_completed", final_status, request=request))
        elif final_status is AgentDelegationCoordinationStatus.PARTIAL_SUCCESS:
            metrics["delegation_coordinations_partial"] = 1
            events.append(_event("agent_delegation_coordination_partial", final_status, request=request))
        else:
            metrics["delegation_coordinations_failed"] = 1
            if final_status is AgentDelegationCoordinationStatus.MINIMUM_SUCCESS_NOT_REACHED:
                metrics["delegation_coordination_minimum_not_reached"] = 1
            events.append(_event("agent_delegation_coordination_failed", final_status, request=request))
        summary = _summary(request, final_status, successful, failed, skipped, chain_results)
        return _result(
            final_status,
            signature=signature,
            request=request,
            chain_results=tuple(chain_results),
            successful_chain_ids=tuple(successful),
            failed_chain_ids=tuple(failed),
            skipped_chain_ids=tuple(skipped),
            aggregated_outputs=aggregated,
            summary=summary,
            metrics=metrics,
            events=events,
            error_code=None if final_status in (AgentDelegationCoordinationStatus.SUCCESS, AgentDelegationCoordinationStatus.PARTIAL_SUCCESS) else final_status.value,
        )

    def _validate_request(
        self,
        request: AgentDelegationCoordinationRequest,
        signature: str,
        events: list[AgentDelegationCoordinationEvent],
        metrics: dict[str, int],
    ) -> AgentDelegationCoordinationResult | None:
        policy = request.policy
        if not policy.enabled:
            metrics["delegation_coordinations_blocked"] = 1
            events.append(_event("agent_delegation_coordination_blocked", AgentDelegationCoordinationStatus.BLOCKED, request=request))
            return _result(
                AgentDelegationCoordinationStatus.BLOCKED,
                signature=signature,
                request=request,
                metrics=metrics,
                events=events,
                error_code="DISABLED",
                error_message="coordination policy is disabled.",
            )
        chains = request.plan.chains
        if not chains:
            return _blocked(request, signature, events, metrics, AgentDelegationCoordinationStatus.NO_CHAINS, "plan has no chains.")
        if len(chains) > policy.max_chains:
            return _limit(request, signature, events, metrics, "maximum chains reached.")
        total_steps = sum(len(chain.chain_request.steps) for chain in chains)
        if total_steps > policy.max_total_steps:
            return _limit(request, signature, events, metrics, "maximum total steps reached.")
        return None


def agent_delegation_coordination_request_signature(
    request: AgentDelegationCoordinationRequest,
) -> str:
    """Return a deterministic SHA-256 signature for a coordination request."""

    if not isinstance(request, AgentDelegationCoordinationRequest):
        raise InvalidAgentDelegationCoordinationRequestError("request must be AgentDelegationCoordinationRequest.")
    payload = {
        "source_agent_id": request.source_agent_id,
        "plan": {
            "plan_id": request.plan.plan_id,
            "chains": tuple(
                {
                    "chain_id": chain.chain_id,
                    "chain_request_signature": agent_delegation_chain_request_signature(chain.chain_request),
                }
                for chain in request.plan.chains
            ),
            "metadata": _jsonable(request.plan.metadata),
        },
        "policy": _policy_payload(request.policy),
        "structured_input": _jsonable(request.structured_input),
        "shared_context": _jsonable(request.shared_context),
        "metadata": _jsonable(request.metadata),
        "execution_id": request.execution_id,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _chain_request_for_execution(
    request: AgentDelegationCoordinationRequest,
    chain_request: AgentDelegationChainRequest,
    previous_outputs: Mapping[str, object],
) -> AgentDelegationChainRequest:
    initial_input = _merge_mapping(request.structured_input, chain_request.initial_input, "initial_input")
    shared_context = _merge_mapping(request.shared_context, chain_request.shared_context, "shared_context")
    if request.policy.propagate_chain_outputs and previous_outputs:
        context = dict(shared_context or {})
        if PREVIOUS_CHAIN_OUTPUTS_KEY in context:
            raise InvalidAgentDelegationCoordinationRequestError("previous_chain_outputs is reserved.")
        context[PREVIOUS_CHAIN_OUTPUTS_KEY] = _safe_mapping(previous_outputs, PREVIOUS_CHAIN_OUTPUTS_KEY)
        shared_context = MappingProxyType(_safe_mapping(context, "shared_context"))
    return AgentDelegationChainRequest(
        steps=chain_request.steps,
        policy=chain_request.policy,
        initial_input=initial_input,
        shared_context=shared_context,
        metadata=_merge_mapping(request.metadata, chain_request.metadata, "metadata") or {},
        execution_id=chain_request.execution_id,
        correlation_id=chain_request.correlation_id,
    )


def _should_skip_remaining(policy: AgentDelegationCoordinationPolicy, failed: Sequence[str]) -> bool:
    return bool(failed) and (
        policy.failure_mode is AgentDelegationCoordinationFailureMode.STOP_ON_FIRST_FAILURE
        or policy.stop_after_failure
        or policy.failure_mode is AgentDelegationCoordinationFailureMode.REQUIRE_ALL_SUCCESS
    )


def _final_status(
    policy: AgentDelegationCoordinationPolicy,
    total: int,
    successful: int,
    failed: int,
    skipped: int,
) -> AgentDelegationCoordinationStatus:
    if policy.failure_mode is AgentDelegationCoordinationFailureMode.REQUIRE_MINIMUM_SUCCESS:
        if successful < policy.min_successful_chains:
            return AgentDelegationCoordinationStatus.MINIMUM_SUCCESS_NOT_REACHED
        return AgentDelegationCoordinationStatus.SUCCESS if failed == 0 and skipped == 0 else AgentDelegationCoordinationStatus.PARTIAL_SUCCESS
    if policy.failure_mode is AgentDelegationCoordinationFailureMode.REQUIRE_ALL_SUCCESS:
        return AgentDelegationCoordinationStatus.SUCCESS if successful == total else AgentDelegationCoordinationStatus.FAILED
    if successful == total:
        return AgentDelegationCoordinationStatus.SUCCESS
    if policy.failure_mode is AgentDelegationCoordinationFailureMode.CONTINUE_ON_FAILURE and successful > 0:
        return AgentDelegationCoordinationStatus.PARTIAL_SUCCESS
    return AgentDelegationCoordinationStatus.FAILED


def _status_from_chain(status: AgentDelegationChainStatus) -> AgentDelegationCoordinationStatus:
    if status is AgentDelegationChainStatus.SUCCESS:
        return AgentDelegationCoordinationStatus.SUCCESS
    if status is AgentDelegationChainStatus.PARTIAL_SUCCESS:
        return AgentDelegationCoordinationStatus.PARTIAL_SUCCESS
    return AgentDelegationCoordinationStatus.FAILED


def _summary(
    request: AgentDelegationCoordinationRequest,
    status: AgentDelegationCoordinationStatus,
    successful: Sequence[str],
    failed: Sequence[str],
    skipped: Sequence[str],
    chain_results: Sequence[AgentDelegationCoordinationChainResult],
) -> Mapping[str, object]:
    return MappingProxyType(
        _safe_mapping(
            {
                "plan_id": request.plan.plan_id,
                "status": status.value,
                "total_chains": len(request.plan.chains),
                "chains_executed": len(chain_results) - len(skipped),
                "chains_succeeded": len(successful),
                "chains_failed": len(failed),
                "chains_skipped": len(skipped),
                "steps_executed": sum(result.chain_result.completed_steps for result in chain_results if result.chain_result),
            },
            "summary",
        )
    )


def _blocked(
    request: AgentDelegationCoordinationRequest,
    signature: str,
    events: list[AgentDelegationCoordinationEvent],
    metrics: dict[str, int],
    status: AgentDelegationCoordinationStatus,
    message: str,
) -> AgentDelegationCoordinationResult:
    metrics["delegation_coordinations_blocked"] = 1
    events.append(_event("agent_delegation_coordination_blocked", status, request=request))
    return _result(status, signature=signature, request=request, metrics=metrics, events=events, error_code=status.value, error_message=message)


def _limit(
    request: AgentDelegationCoordinationRequest,
    signature: str,
    events: list[AgentDelegationCoordinationEvent],
    metrics: dict[str, int],
    message: str,
) -> AgentDelegationCoordinationResult:
    metrics["delegation_coordination_limits_reached"] = 1
    events.append(_event("agent_delegation_coordination_limit_reached", AgentDelegationCoordinationStatus.LIMIT_REACHED, request=request))
    return _blocked(request, signature, events, metrics, AgentDelegationCoordinationStatus.LIMIT_REACHED, message)


def _result(
    status: AgentDelegationCoordinationStatus,
    *,
    signature: str,
    request: AgentDelegationCoordinationRequest | None = None,
    chain_results: tuple[AgentDelegationCoordinationChainResult, ...] = (),
    successful_chain_ids: tuple[str, ...] = (),
    failed_chain_ids: tuple[str, ...] = (),
    skipped_chain_ids: tuple[str, ...] = (),
    aggregated_outputs: Mapping[str, object] | None = None,
    summary: Mapping[str, object] | None = None,
    events: list[AgentDelegationCoordinationEvent] | tuple[AgentDelegationCoordinationEvent, ...] = (),
    metrics: Mapping[str, int] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> AgentDelegationCoordinationResult:
    if len(events) > MAX_COORDINATION_EVENTS:
        events = tuple(events)[-MAX_COORDINATION_EVENTS:]
    return AgentDelegationCoordinationResult(
        status=status,
        plan_id=request.plan.plan_id if request is not None else None,
        signature=signature,
        chain_results=chain_results,
        successful_chain_ids=successful_chain_ids,
        failed_chain_ids=failed_chain_ids,
        skipped_chain_ids=skipped_chain_ids,
        aggregated_outputs=aggregated_outputs or {},
        summary=summary or {},
        events=tuple(events),
        metrics=metrics or _base_metrics(),
        error_code=error_code,
        error_message=error_message,
    )


def _event(
    name: str,
    status: AgentDelegationCoordinationStatus,
    *,
    request: AgentDelegationCoordinationRequest | None = None,
    chain_id: str | None = None,
) -> AgentDelegationCoordinationEvent:
    details: dict[str, object] = {}
    if request is not None:
        details["plan_id"] = request.plan.plan_id
        details["execution_id"] = request.execution_id or ""
    if chain_id is not None:
        details["chain_id"] = chain_id
    return AgentDelegationCoordinationEvent(name=name, status=status, details=details)


def _base_metrics() -> dict[str, int]:
    return {
        "delegation_coordinations_requested": 0,
        "delegation_coordinations_started": 0,
        "delegation_coordinations_succeeded": 0,
        "delegation_coordinations_partial": 0,
        "delegation_coordinations_failed": 0,
        "delegation_coordinations_blocked": 0,
        "delegation_coordination_chains_started": 0,
        "delegation_coordination_chains_succeeded": 0,
        "delegation_coordination_chains_failed": 0,
        "delegation_coordination_chains_skipped": 0,
        "delegation_coordination_steps_executed": 0,
        "delegation_coordination_limits_reached": 0,
        "delegation_coordination_minimum_not_reached": 0,
    }


def _policy_payload(policy: AgentDelegationCoordinationPolicy) -> Mapping[str, object]:
    return {
        "enabled": policy.enabled,
        "failure_mode": policy.failure_mode.value,
        "min_successful_chains": policy.min_successful_chains,
        "max_chains": policy.max_chains,
        "max_total_steps": policy.max_total_steps,
        "propagate_chain_outputs": policy.propagate_chain_outputs,
        "include_partial_results": policy.include_partial_results,
        "stop_after_failure": policy.stop_after_failure,
        "max_output_items": policy.max_output_items,
    }


def _merge_mapping(
    first: Mapping[str, object] | None,
    second: Mapping[str, object] | None,
    field_name: str,
) -> Mapping[str, object] | None:
    if first is None and second is None:
        return None
    merged: dict[str, object] = {}
    if first is not None:
        merged.update(_safe_mapping(first, field_name))
    if second is not None:
        overlap = set(merged).intersection(second)
        if overlap:
            raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} cannot overwrite existing keys.")
        merged.update(_safe_mapping(second, field_name))
    return MappingProxyType(_safe_mapping(merged, field_name))


def _count_items(value: object) -> int:
    if isinstance(value, Mapping):
        return len(value) + sum(_count_items(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) + sum(_count_items(child) for child in value)
    return 1


def _bounded_int(value: int, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} must be an integer.")
    if value <= 0 or value > maximum:
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} is outside the allowed range.")
    return value


def _identifier_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} must be an iterable of strings.")
    return tuple(dict.fromkeys(_identifier(value, field_name) for value in values))


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} must be a string.")
    if not value or value.strip() != value:
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} cannot be empty or padded.")
    if "/" in value or "\\" in value or ".." in value:
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} cannot be path-like.")
    if any(ord(character) < 32 for character in value):
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} cannot contain control characters.")
    if not all(character.isalnum() or character in "_.-" for character in value):
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} contains unsupported characters.")
    if len(value) > 128:
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} exceeds the length limit.")
    if _is_sensitive_key(value):
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} cannot contain sensitive content.")
    return value


def _optional_safe_mapping(value: Mapping[str, object] | None, field_name: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return MappingProxyType(_safe_mapping(value, field_name))


def _safe_mapping(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} must be a mapping.")
    if len(value) > MAX_COORDINATION_METADATA_ITEMS and field_name == "metadata":
        raise InvalidAgentDelegationCoordinationRequestError("metadata has too many items.")
    return _safe_mapping_inner(value, field_name, depth=0, counter={"nodes": 0, "chars": 0})


def _safe_mapping_inner(
    value: Mapping[str, object],
    field_name: str,
    *,
    depth: int,
    counter: dict[str, int],
) -> dict[str, object]:
    if depth > 5:
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} is too deep.")
    if len(value) > MAX_COORDINATION_MAPPING_ITEMS:
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} has too many items.")
    result: dict[str, object] = {}
    for raw_key in sorted(value):
        if isinstance(raw_key, str) and _is_sensitive_key(raw_key):
            continue
        key = _key(raw_key, field_name)
        if _is_sensitive_key(key):
            continue
        result[key] = _safe_value(value[raw_key], field_name, depth=depth + 1, counter=counter)
    return result


def _safe_sequence(value: Sequence[object], field_name: str, *, depth: int, counter: dict[str, int]) -> tuple[object, ...]:
    if len(value) > MAX_COORDINATION_SEQUENCE_ITEMS:
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} has too many items.")
    return tuple(_safe_value(item, field_name, depth=depth + 1, counter=counter) for item in value)


def _safe_value(value: object, field_name: str, *, depth: int, counter: dict[str, int]) -> object:
    counter["nodes"] += 1
    if counter["nodes"] > MAX_COORDINATION_TOTAL_ITEMS:
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} is too large.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} floats must be finite.")
        return value
    if isinstance(value, str):
        if len(value) > MAX_COORDINATION_STRING_LENGTH:
            raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} strings are too long.")
        counter["chars"] += len(value)
        if counter["chars"] > MAX_COORDINATION_STRING_LENGTH * MAX_COORDINATION_MAPPING_ITEMS:
            raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} string budget exceeded.")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_mapping_inner(value, field_name, depth=depth, counter=counter))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _safe_sequence(value, field_name, depth=depth, counter=counter)
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value) or isinstance(value, types.ModuleType):
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} cannot contain executable objects.")
    raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} contains an unsupported value.")


def _key(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidAgentDelegationCoordinationRequestError(f"{field_name} keys must be strings.")
    return _identifier(value, f"{field_name} key")


def _metric_mapping(metrics: Mapping[str, int]) -> dict[str, int]:
    safe: dict[str, int] = {}
    for key, value in metrics.items():
        name = _identifier(str(key), "metric name")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidAgentDelegationCoordinationRequestError("metric values must be non-negative integers.")
        safe[name] = value
    return safe


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidAgentDelegationCoordinationRequestError("non-finite floats are not allowed.")
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value) or isinstance(value, types.ModuleType):
        raise InvalidAgentDelegationCoordinationRequestError("executable objects are not allowed.")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise InvalidAgentDelegationCoordinationRequestError("unsupported signature value.")
    return value


def _safe_message(value: str | None) -> str:
    text = " ".join(str(value or "").split())
    lowered = text.lower()
    if "[redacted]" in lowered or any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        text = "[redacted]"
    return text[:240]


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _failure_mode(
    value: AgentDelegationCoordinationFailureMode | str,
) -> AgentDelegationCoordinationFailureMode:
    if isinstance(value, AgentDelegationCoordinationFailureMode):
        return value
    if isinstance(value, str):
        try:
            return AgentDelegationCoordinationFailureMode(value.upper())
        except ValueError as error:
            raise InvalidAgentDelegationCoordinationRequestError("failure_mode is invalid.") from error
    raise InvalidAgentDelegationCoordinationRequestError("failure_mode must be AgentDelegationCoordinationFailureMode.")


def _status(value: AgentDelegationCoordinationStatus | str) -> AgentDelegationCoordinationStatus:
    if isinstance(value, AgentDelegationCoordinationStatus):
        return value
    if isinstance(value, str):
        return AgentDelegationCoordinationStatus(value)
    raise InvalidAgentDelegationCoordinationRequestError("status must be AgentDelegationCoordinationStatus.")

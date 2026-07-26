from __future__ import annotations

from collections.abc import Mapping
import math

import pytest

from bootstrap.agent_delegation_coordinator import build_core_agent_delegation_coordinator
from bootstrap.agent_system import build_core_agent_system
from core.agent_context import AgentContext
from core.agent_delegation import AgentDelegationPolicy, AgentDelegationRequest
from core.agent_delegation_chain import AgentDelegationChainPolicy, AgentDelegationChainRequest, AgentDelegationChainStep
from core.agent_delegation_coordinator import (
    PREVIOUS_CHAIN_OUTPUTS_KEY,
    AgentDelegationCoordinationChain,
    AgentDelegationCoordinationFailureMode,
    AgentDelegationCoordinationPlan,
    AgentDelegationCoordinationPolicy,
    AgentDelegationCoordinationRequest,
    AgentDelegationCoordinationStatus,
    AgentDelegationCoordinator,
    InvalidAgentDelegationCoordinationRequestError,
    agent_delegation_coordination_request_signature,
)
from core.agent_registry import AgentCapabilities, AgentContextPolicy, AgentDefinition, AgentPermissions, AgentRegistry, AgentType


class CoordinationHandler:
    calls: list[tuple[str, Mapping[str, object], Mapping[str, object]]] = []

    def __init__(self, agent_id: str, *, fail: bool = False) -> None:
        self._agent_id = agent_id
        self.fail = fail

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        CoordinationHandler.calls.append((self._agent_id, context.structured_input, context.shared_context))
        if self.fail:
            raise RuntimeError("handler failed with authorization token raw-secret")
        return {
            "agent_id": self._agent_id,
            "input": context.structured_input,
            "shared": context.shared_context,
            "token": "hidden",
        }


def _definition(agent_id: str) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=AgentType.GENERAL,
        name=agent_id,
        description="Coordination test agent.",
        capabilities=AgentCapabilities(capabilities=("agent.inspect",)),
        permissions=AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_shared_context=True),
    )


def _system(*, fail: tuple[str, ...] = ()):
    result = build_core_agent_system()
    assert result.system is not None
    system = result.system
    for agent_id in ("agent.a", "agent.b", "agent.c", "agent.d"):
        system.agent_registry.register(_definition(agent_id))
    for agent_id in ("agent.b", "agent.d"):
        system.agent_handler_registry.register(CoordinationHandler(agent_id, fail=agent_id in fail))
    return system


def _chain_policy(**overrides: object) -> AgentDelegationChainPolicy:
    values = {"enabled": True, "max_steps": 3, "max_depth": 3, "max_total_delegations": 3}
    values.update(overrides)
    return AgentDelegationChainPolicy(**values)


def _chain(source: str, target: str, *, chain_id: str, shared_context: Mapping[str, object] | None = None):
    return AgentDelegationCoordinationChain(
        chain_id=chain_id,
        chain_request=AgentDelegationChainRequest(
            steps=(
                AgentDelegationChainStep(
                    source_agent_id=source,
                    target_agent_id=target,
                    execution_required_capability_ids=("agent.inspect",),
                    structured_input={"source": source},
                ),
            ),
            policy=_chain_policy(propagate_shared_context=True),
            shared_context=shared_context,
            execution_id=f"{chain_id}.exec",
            correlation_id="coord-1",
        ),
    )


def _policy(**overrides: object) -> AgentDelegationCoordinationPolicy:
    values = {"enabled": True, "max_chains": 5, "max_total_steps": 10}
    values.update(overrides)
    return AgentDelegationCoordinationPolicy(**values)


def _request(*chains: AgentDelegationCoordinationChain, **overrides: object) -> AgentDelegationCoordinationRequest:
    values = {
        "source_agent_id": "agent.a",
        "plan": AgentDelegationCoordinationPlan(
            plan_id="plan.main",
            chains=chains or (_chain("agent.a", "agent.b", chain_id="chain.a"), _chain("agent.c", "agent.d", chain_id="chain.b")),
            metadata={"owner": "test"},
        ),
        "policy": _policy(),
        "structured_input": {"root": "ok"},
        "shared_context": {"trace": "coord", "token": "hidden"},
        "metadata": {"source": "test"},
        "execution_id": "coord-1",
    }
    values.update(overrides)
    return AgentDelegationCoordinationRequest(**values)


def setup_function() -> None:
    CoordinationHandler.calls = []


def test_service_construction_and_agent_system_shared_instances() -> None:
    system = _system()
    service = build_core_agent_delegation_coordinator(
        agent_registry=system.agent_registry,
        agent_resolver=system.agent_resolver,
        agent_context_builder=system.agent_context_builder,
        agent_executor=system.agent_executor,
        agent_delegation_service=system.agent_delegation_service,
        agent_delegation_chain_service=system.agent_delegation_chain_service,
    )

    assert isinstance(service, AgentDelegationCoordinator)
    assert isinstance(system.agent_delegation_coordinator, AgentDelegationCoordinator)
    assert service._agent_executor is system.agent_executor
    assert system.agent_delegation_coordinator._agent_executor is system.agent_executor
    assert system.agent_delegation_coordinator._agent_delegation_chain_service is system.agent_delegation_chain_service


def test_two_chain_plan_executes_in_declared_order_and_succeeds() -> None:
    result = _system().agent_delegation_coordinator.coordinate(_request())

    assert result.status is AgentDelegationCoordinationStatus.SUCCESS
    assert result.successful_chain_ids == ("chain.a", "chain.b")
    assert [call[0] for call in CoordinationHandler.calls] == ["agent.b", "agent.d"]
    assert tuple(result.aggregated_outputs) == ("chain.a", "chain.b")
    assert result.summary["total_chains"] == 2


def test_continue_on_failure_returns_partial_success() -> None:
    result = _system(fail=("agent.b",)).agent_delegation_coordinator.coordinate(
        _request(
            policy=_policy(
                failure_mode=AgentDelegationCoordinationFailureMode.CONTINUE_ON_FAILURE,
                stop_after_failure=False,
            )
        )
    )

    assert result.status is AgentDelegationCoordinationStatus.PARTIAL_SUCCESS
    assert result.failed_chain_ids == ("chain.a",)
    assert result.successful_chain_ids == ("chain.b",)


def test_stop_on_first_failure_skips_later_chains() -> None:
    result = _system(fail=("agent.b",)).agent_delegation_coordinator.coordinate(_request())

    assert result.status is AgentDelegationCoordinationStatus.FAILED
    assert result.failed_chain_ids == ("chain.a",)
    assert result.skipped_chain_ids == ("chain.b",)
    assert [call[0] for call in CoordinationHandler.calls] == ["agent.b"]


def test_require_all_success_and_minimum_success_policies() -> None:
    require_all = _system(fail=("agent.b",)).agent_delegation_coordinator.coordinate(
        _request(policy=_policy(failure_mode=AgentDelegationCoordinationFailureMode.REQUIRE_ALL_SUCCESS))
    )
    minimum_ok = _system(fail=("agent.b",)).agent_delegation_coordinator.coordinate(
        _request(
            policy=_policy(
                failure_mode=AgentDelegationCoordinationFailureMode.REQUIRE_MINIMUM_SUCCESS,
                min_successful_chains=1,
                stop_after_failure=False,
            )
        )
    )
    minimum_failed = _system(fail=("agent.b", "agent.d")).agent_delegation_coordinator.coordinate(
        _request(
            policy=_policy(
                failure_mode=AgentDelegationCoordinationFailureMode.REQUIRE_MINIMUM_SUCCESS,
                min_successful_chains=1,
                stop_after_failure=False,
            )
        )
    )

    assert require_all.status is AgentDelegationCoordinationStatus.FAILED
    assert minimum_ok.status is AgentDelegationCoordinationStatus.PARTIAL_SUCCESS
    assert minimum_failed.status is AgentDelegationCoordinationStatus.MINIMUM_SUCCESS_NOT_REACHED


def test_empty_plan_duplicate_chain_id_and_invalid_plan_id_are_rejected() -> None:
    with pytest.raises(InvalidAgentDelegationCoordinationRequestError):
        AgentDelegationCoordinationPlan(plan_id="plan.empty", chains=())
    with pytest.raises(InvalidAgentDelegationCoordinationRequestError):
        AgentDelegationCoordinationPlan(
            plan_id="plan.dup",
            chains=(_chain("agent.a", "agent.b", chain_id="chain.a"), _chain("agent.c", "agent.d", chain_id="chain.a")),
        )
    with pytest.raises(InvalidAgentDelegationCoordinationRequestError):
        AgentDelegationCoordinationPlan(plan_id="../bad", chains=(_chain("agent.a", "agent.b", chain_id="chain.a"),))


def test_limits_are_enforced() -> None:
    too_many_chains = _system().agent_delegation_coordinator.coordinate(_request(policy=_policy(max_chains=1)))
    too_many_steps = _system().agent_delegation_coordinator.coordinate(_request(policy=_policy(max_total_steps=1)))
    too_many_outputs = _system().agent_delegation_coordinator.coordinate(_request(policy=_policy(max_output_items=1)))

    assert too_many_chains.status is AgentDelegationCoordinationStatus.LIMIT_REACHED
    assert too_many_steps.status is AgentDelegationCoordinationStatus.LIMIT_REACHED
    assert too_many_outputs.status is AgentDelegationCoordinationStatus.LIMIT_REACHED


def test_propagation_disabled_by_default_and_enabled_explicitly() -> None:
    default = _system().agent_delegation_coordinator.coordinate(_request())
    enabled = _system().agent_delegation_coordinator.coordinate(
        _request(policy=_policy(propagate_chain_outputs=True))
    )

    assert default.status is AgentDelegationCoordinationStatus.SUCCESS
    assert PREVIOUS_CHAIN_OUTPUTS_KEY not in CoordinationHandler.calls[1][2]
    assert enabled.status is AgentDelegationCoordinationStatus.SUCCESS
    assert PREVIOUS_CHAIN_OUTPUTS_KEY in CoordinationHandler.calls[-1][2]


def test_chain_output_propagation_collision_is_structured_failure() -> None:
    result = _system().agent_delegation_coordinator.coordinate(
        _request(
            _chain("agent.a", "agent.b", chain_id="chain.a"),
            _chain("agent.c", "agent.d", chain_id="chain.b", shared_context={PREVIOUS_CHAIN_OUTPUTS_KEY: {}}),
            policy=_policy(propagate_chain_outputs=True),
        )
    )

    assert result.status is AgentDelegationCoordinationStatus.FAILED
    assert result.failed_chain_ids == ("chain.b",)


def test_sensitive_keys_are_sanitized_and_unsafe_values_rejected() -> None:
    request = _request(shared_context={"trace": "ok", "api_key": "raw-secret"})

    assert request.shared_context == {"trace": "ok"}
    with pytest.raises(InvalidAgentDelegationCoordinationRequestError):
        _request(structured_input={"callback": lambda: None})
    with pytest.raises(InvalidAgentDelegationCoordinationRequestError):
        _request(structured_input={"module": math})
    with pytest.raises(InvalidAgentDelegationCoordinationRequestError):
        _request(structured_input={"bad": float("nan")})


def test_signature_is_deterministic() -> None:
    first = _request()
    second = _request()
    changed = _request(
        _chain("agent.c", "agent.d", chain_id="chain.b"),
        _chain("agent.a", "agent.b", chain_id="chain.a"),
    )

    assert agent_delegation_coordination_request_signature(first) == agent_delegation_coordination_request_signature(second)
    assert agent_delegation_coordination_request_signature(first) != agent_delegation_coordination_request_signature(changed)


def test_events_and_metrics_are_reported() -> None:
    result = _system().agent_delegation_coordinator.coordinate(_request())
    names = {event.name for event in result.events}

    assert "agent_delegation_coordination_requested" in names
    assert "agent_delegation_coordination_validation_succeeded" in names
    assert "agent_delegation_coordination_completed" in names
    assert result.metrics["delegation_coordinations_requested"] == 1
    assert result.metrics["delegation_coordination_chains_started"] == 2
    assert result.metrics["delegation_coordination_chains_succeeded"] == 2
    assert result.metrics["delegation_coordination_steps_executed"] == 2


def test_policy_disabled_blocks_without_execution() -> None:
    result = _system().agent_delegation_coordinator.coordinate(
        _request(policy=AgentDelegationCoordinationPolicy())
    )

    assert result.status is AgentDelegationCoordinationStatus.BLOCKED
    assert result.error_code == "DISABLED"
    assert CoordinationHandler.calls == []


def test_phase_12_1_and_12_2_compatibility() -> None:
    system = _system()
    direct = system.agent_delegation_service.delegate(
        AgentDelegationRequest(
            origin_agent_id="agent.a",
            target_agent_id="agent.b",
            required_capability_ids=("agent.inspect",),
            policy=AgentDelegationPolicy(enabled=True),
        )
    )
    chain = system.agent_delegation_chain_service.execute(
        AgentDelegationChainRequest(
            steps=(
                AgentDelegationChainStep(
                    source_agent_id="agent.a",
                    target_agent_id="agent.b",
                    execution_required_capability_ids=("agent.inspect",),
                ),
            ),
            policy=_chain_policy(),
        )
    )

    assert direct.success is True
    assert chain.status.name == "SUCCESS"

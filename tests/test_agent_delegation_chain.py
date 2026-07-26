from __future__ import annotations

from collections.abc import Mapping

import pytest

from bootstrap.agent_delegation_chain import build_core_agent_delegation_chain_service
from bootstrap.agent_system import build_core_agent_system
from core.agent_context import AgentContext, AgentContextBuilder
from core.agent_delegation import AgentDelegationPolicy, AgentDelegationRequest, AgentDelegationService
from core.agent_delegation_chain import (
    PREVIOUS_OUTPUT_KEY,
    AgentDelegationChainPolicy,
    AgentDelegationChainRequest,
    AgentDelegationChainService,
    AgentDelegationChainStatus,
    AgentDelegationChainStep,
    InvalidAgentDelegationChainRequestError,
    agent_delegation_chain_request_signature,
)
from core.agent_executor import AgentExecutor
from core.agent_registry import AgentCapabilities, AgentContextPolicy, AgentDefinition, AgentPermissions, AgentRegistry, AgentType
from core.agent_resolver import AgentResolver


class ChainHandler:
    calls: list[tuple[str, Mapping[str, object], Mapping[str, object]]] = []

    def __init__(self, agent_id: str, *, fail: bool = False) -> None:
        self._agent_id = agent_id
        self.fail = fail

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        ChainHandler.calls.append((self._agent_id, context.structured_input, context.shared_context))
        if self.fail:
            raise RuntimeError("handler failed with authorization token raw-secret")
        return {
            "agent_id": self._agent_id,
            "input": context.structured_input,
            "shared": context.shared_context,
            "token": "hidden",
        }


def _definition(
    agent_id: str,
    *,
    agent_type: AgentType = AgentType.GENERAL,
    capabilities: tuple[str, ...] = ("agent.inspect",),
    enabled: bool = True,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=agent_type,
        name=agent_id,
        description="Chain test agent.",
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_shared_context=True, allow_user_input=False),
        enabled=enabled,
    )


def _system(
    definitions: tuple[AgentDefinition, ...] | None = None,
    handlers: tuple[ChainHandler, ...] | None = None,
):
    result = build_core_agent_system()
    assert result.system is not None
    system = result.system
    for definition in definitions or (
        _definition("agent.a"),
        _definition("agent.b"),
        _definition("agent.c"),
    ):
        system.agent_registry.register(definition)
    for handler in handlers or (ChainHandler("agent.b"), ChainHandler("agent.c")):
        system.agent_handler_registry.register(handler)
    return system


def _policy(**overrides: object) -> AgentDelegationChainPolicy:
    values = {"enabled": True, "max_steps": 5, "max_depth": 5, "max_total_delegations": 5}
    values.update(overrides)
    return AgentDelegationChainPolicy(**values)


def _step(source: str, target: str | None = None, **overrides: object) -> AgentDelegationChainStep:
    values = {
        "source_agent_id": source,
        "target_agent_id": target,
        "execution_required_capability_ids": ("agent.inspect",),
        "structured_input": {"step": source},
    }
    values.update(overrides)
    return AgentDelegationChainStep(**values)


def _request(*steps: AgentDelegationChainStep, **overrides: object) -> AgentDelegationChainRequest:
    values = {
        "steps": steps or (_step("agent.a", "agent.b"), _step("agent.b", "agent.c")),
        "policy": _policy(),
        "initial_input": {"root": "ok"},
        "shared_context": {"trace": "chain", "token": "hidden"},
        "metadata": {"source": "test"},
        "execution_id": "chain-1",
        "correlation_id": "corr-1",
    }
    values.update(overrides)
    return AgentDelegationChainRequest(**values)


def setup_function() -> None:
    ChainHandler.calls = []


def test_valid_chain_a_to_b_to_c_with_real_handlers() -> None:
    result = _system().agent_delegation_chain_service.execute(_request())

    assert result.status is AgentDelegationChainStatus.SUCCESS
    assert result.completed_steps == 2
    assert [step.resolved_target_agent_id for step in result.step_results] == ["agent.b", "agent.c"]
    assert [call[0] for call in ChainHandler.calls] == ["agent.b", "agent.c"]
    assert result.final_output is not None
    assert result.final_output["agent_id"] == "agent.c"


def test_single_step_is_compatible_with_phase_12_1_delegation() -> None:
    result = _system().agent_delegation_chain_service.execute(_request(_step("agent.a", "agent.b")))

    assert result.status is AgentDelegationChainStatus.SUCCESS
    assert result.step_results[0].delegation_result is not None
    assert result.step_results[0].delegation_result.success is True


def test_policy_disabled_and_empty_steps_are_rejected() -> None:
    disabled = _system().agent_delegation_chain_service.execute(_request(policy=AgentDelegationChainPolicy()))

    assert disabled.status is AgentDelegationChainStatus.DISABLED
    with pytest.raises(InvalidAgentDelegationChainRequestError):
        AgentDelegationChainRequest(steps=(), policy=_policy())


def test_limits_are_enforced() -> None:
    system = _system()
    steps = (_step("agent.a", "agent.b"), _step("agent.b", "agent.c"))

    assert system.agent_delegation_chain_service.execute(_request(*steps, policy=_policy(max_steps=1))).status is AgentDelegationChainStatus.MAX_STEPS_REACHED
    assert system.agent_delegation_chain_service.execute(_request(*steps, policy=_policy(max_depth=1))).status is AgentDelegationChainStatus.MAX_DEPTH_REACHED
    assert system.agent_delegation_chain_service.execute(_request(*steps, policy=_policy(max_total_delegations=1))).status is AgentDelegationChainStatus.MAX_TOTAL_DELEGATIONS_REACHED


def test_missing_source_and_explicit_missing_target_are_structured() -> None:
    missing_source = _system().agent_delegation_chain_service.execute(_request(_step("agent.missing", "agent.b")))
    missing_target = _system().agent_delegation_chain_service.execute(_request(_step("agent.a", "agent.missing")))

    assert missing_source.status is AgentDelegationChainStatus.SOURCE_AGENT_NOT_FOUND
    assert missing_target.status is AgentDelegationChainStatus.TARGET_AGENT_NOT_FOUND


def test_automatic_resolution_success_no_match_and_ambiguous() -> None:
    no_match = _system().agent_delegation_chain_service.execute(
        _request(
            AgentDelegationChainStep(
                source_agent_id="agent.a",
                required_capability_ids=("agent.special",),
                execution_required_capability_ids=("agent.special",),
            )
        )
    )
    assert no_match.status is AgentDelegationChainStatus.TARGET_RESOLUTION_FAILED

    system = _system(
        definitions=(
            _definition("agent.a"),
            _definition("agent.b", capabilities=("agent.special",)),
            _definition("agent.c", capabilities=("agent.other",)),
        ),
        handlers=(ChainHandler("agent.b"),),
    )
    ok = system.agent_delegation_chain_service.execute(
        _request(
            AgentDelegationChainStep(
                source_agent_id="agent.a",
                required_capability_ids=("agent.special",),
                execution_required_capability_ids=("agent.special",),
            )
        )
    )
    ambiguous = _system().agent_delegation_chain_service.execute(
        _request(
            AgentDelegationChainStep(
                source_agent_id="agent.a",
                required_capability_ids=("agent.inspect",),
                execution_required_capability_ids=("agent.inspect",),
            )
        )
    )

    assert ok.status is AgentDelegationChainStatus.SUCCESS
    assert ok.step_results[0].resolved_target_agent_id == "agent.b"
    assert ambiguous.status is AgentDelegationChainStatus.TARGET_RESOLUTION_AMBIGUOUS


def test_cycles_a_to_a_a_to_b_to_a_and_a_to_b_to_c_to_a_are_blocked() -> None:
    system = _system(handlers=(ChainHandler("agent.a"), ChainHandler("agent.b"), ChainHandler("agent.c")))

    direct = system.agent_delegation_chain_service.execute(_request(_step("agent.a", "agent.a")))
    two = system.agent_delegation_chain_service.execute(
        _request(_step("agent.a", "agent.b"), _step("agent.b", "agent.a"))
    )
    three = system.agent_delegation_chain_service.execute(
        _request(_step("agent.a", "agent.b"), _step("agent.b", "agent.c"), _step("agent.c", "agent.a"))
    )

    assert direct.status is AgentDelegationChainStatus.CYCLE_DETECTED
    assert two.status is AgentDelegationChainStatus.CYCLE_DETECTED
    assert three.status is AgentDelegationChainStatus.CYCLE_DETECTED


def test_repeated_target_blocked_and_allowed_without_cycle() -> None:
    blocked = _system(
        definitions=(_definition("agent.a"), _definition("agent.b"), _definition("agent.c")),
        handlers=(ChainHandler("agent.b"),),
    ).agent_delegation_chain_service.execute(
        _request(_step("agent.a", "agent.b"), _step("agent.c", "agent.b"))
    )
    allowed = _system(
        definitions=(_definition("agent.a"), _definition("agent.b"), _definition("agent.c")),
        handlers=(ChainHandler("agent.b"),),
    ).agent_delegation_chain_service.execute(
        _request(
            _step("agent.a", "agent.b"),
            _step("agent.c", "agent.b"),
            policy=_policy(allow_repeated_agents=True),
        )
    )

    assert blocked.status is AgentDelegationChainStatus.REPEATED_AGENT_DENIED
    assert allowed.status is AgentDelegationChainStatus.SUCCESS


def test_stop_on_failure_true_and_false_partial_success() -> None:
    failing = _system(
        handlers=(ChainHandler("agent.b", fail=True), ChainHandler("agent.c")),
    ).agent_delegation_chain_service.execute(_request())
    continuing = _system(
        handlers=(ChainHandler("agent.b", fail=True), ChainHandler("agent.c")),
    ).agent_delegation_chain_service.execute(_request(policy=_policy(stop_on_failure=False)))

    assert failing.status is AgentDelegationChainStatus.FAILED
    assert len(failing.step_results) == 1
    assert continuing.status is AgentDelegationChainStatus.PARTIAL_SUCCESS
    assert continuing.completed_steps == 1
    assert [call[0] for call in ChainHandler.calls][-2:] == ["agent.b", "agent.c"]


def test_previous_output_propagation_default_enabled_and_reserved_key() -> None:
    default = _system().agent_delegation_chain_service.execute(_request())
    propagated = _system().agent_delegation_chain_service.execute(
        _request(policy=_policy(propagate_previous_output=True))
    )

    assert PREVIOUS_OUTPUT_KEY not in ChainHandler.calls[1][1]
    assert default.status is AgentDelegationChainStatus.SUCCESS
    assert propagated.status is AgentDelegationChainStatus.SUCCESS
    assert PREVIOUS_OUTPUT_KEY in ChainHandler.calls[-1][1]
    reserved = _system().agent_delegation_chain_service.execute(
        _request(
            _step("agent.a", "agent.b"),
            _step("agent.b", "agent.c", structured_input={PREVIOUS_OUTPUT_KEY: {}}),
            policy=_policy(propagate_previous_output=True),
        )
    )
    assert reserved.status is AgentDelegationChainStatus.FAILED
    assert reserved.step_results[-1].status is AgentDelegationChainStatus.INVALID_REQUEST


def test_shared_context_propagation_sanitizes_sensitive_keys() -> None:
    blocked = _system().agent_delegation_chain_service.execute(_request())
    allowed = _system().agent_delegation_chain_service.execute(_request(policy=_policy(propagate_shared_context=True)))

    assert blocked.status is AgentDelegationChainStatus.SUCCESS
    assert ChainHandler.calls[0][2] == {}
    assert allowed.status is AgentDelegationChainStatus.SUCCESS
    assert ChainHandler.calls[-1][2] == {"trace": "chain"}


def test_unsafe_values_are_rejected() -> None:
    with pytest.raises(InvalidAgentDelegationChainRequestError):
        _step("agent.a", "agent.b", structured_input={"fn": lambda: None})
    with pytest.raises(InvalidAgentDelegationChainRequestError):
        _request(_step("agent.a", "agent.b"), initial_input={"bad": float("inf")})


def test_handler_error_is_sanitized() -> None:
    result = _system(handlers=(ChainHandler("agent.b", fail=True),)).agent_delegation_chain_service.execute(
        _request(_step("agent.a", "agent.b"))
    )

    assert result.status is AgentDelegationChainStatus.FAILED
    assert result.error_message == "[redacted]"
    assert "raw-secret" not in repr(result)


def test_signature_is_deterministic_and_order_sensitive() -> None:
    first = _request(_step("agent.a", "agent.b"), _step("agent.b", "agent.c"))
    second = _request(_step("agent.a", "agent.b"), _step("agent.b", "agent.c"))
    reordered = _request(_step("agent.b", "agent.c"), _step("agent.a", "agent.b"))

    assert agent_delegation_chain_request_signature(first) == agent_delegation_chain_request_signature(second)
    assert agent_delegation_chain_request_signature(first) != agent_delegation_chain_request_signature(reordered)


def test_agent_system_uses_shared_instances_and_does_not_duplicate_executor() -> None:
    system = _system()
    service = build_core_agent_delegation_chain_service(
        agent_registry=system.agent_registry,
        agent_resolver=system.agent_resolver,
        agent_context_builder=system.agent_context_builder,
        agent_executor=system.agent_executor,
        agent_delegation_service=system.agent_delegation_service,
    )

    assert isinstance(system.agent_delegation_chain_service, AgentDelegationChainService)
    assert service._agent_executor is system.agent_executor
    assert system.agent_delegation_chain_service._agent_executor is system.agent_executor
    assert system.agent_delegation_chain_service._agent_delegation_service is system.agent_delegation_service


def test_events_and_metrics_are_safe_and_complete() -> None:
    result = _system().agent_delegation_chain_service.execute(_request())
    names = {event.name for event in result.events}

    assert "agent_delegation_chain_requested" in names
    assert "agent_delegation_chain_validation_succeeded" in names
    assert "agent_delegation_chain_completed" in names
    assert result.metrics["delegation_chains_requested"] == 1
    assert result.metrics["delegation_chains_started"] == 1
    assert result.metrics["delegation_chains_succeeded"] == 1
    assert result.metrics["delegation_chain_steps_succeeded"] == 2


def test_phase_12_1_delegation_tests_remain_compatible() -> None:
    system = _system()

    direct = system.agent_delegation_service.delegate(
        AgentDelegationRequest(
            origin_agent_id="agent.a",
            target_agent_id="agent.b",
            required_capability_ids=("agent.inspect",),
            policy=AgentDelegationPolicy(enabled=True),
        )
    )

    assert direct.success is True

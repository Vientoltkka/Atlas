from __future__ import annotations

import pytest

from core.agent_registry import AgentCapabilities, AgentDefinition, AgentPermissions, AgentType, AgentRegistry
from core.multi_agent import (
    InvalidMultiAgentExecutionRequestError,
    MultiAgentExecutionPolicy,
    MultiAgentExecutionRequest,
    MultiAgentResolver,
    MultiAgentTeamResolutionStatus,
    multi_agent_execution_request_signature,
)


def _agent(
    agent_id: str,
    *,
    agent_type: AgentType = AgentType.GENERAL,
    enabled: bool = True,
    capabilities: tuple[str, ...] = ("agent.inspect",),
    permissions: AgentPermissions | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=agent_type,
        name=agent_id,
        description="Resolver test agent.",
        enabled=enabled,
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=permissions or AgentPermissions(requires_confirmation=False),
    )


def _resolver(*agents: AgentDefinition) -> MultiAgentResolver:
    return MultiAgentResolver(AgentRegistry(agents))


def test_selects_two_agents_by_required_agent_ids_in_requested_order() -> None:
    result = _resolver(_agent("agent.b"), _agent("agent.a")).resolve(
        MultiAgentExecutionRequest(
            required_agent_ids=("agent.a", "agent.b"),
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        )
    )

    assert result.status is MultiAgentTeamResolutionStatus.RESOLVED
    assert result.selected_agent_ids == ("agent.a", "agent.b")


def test_selects_by_capabilities_with_deterministic_order_and_max_agents() -> None:
    result = _resolver(_agent("agent.c"), _agent("agent.a"), _agent("agent.b")).resolve(
        MultiAgentExecutionRequest(
            required_capability_ids=("agent.inspect",),
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        )
    )

    assert result.status is MultiAgentTeamResolutionStatus.RESOLVED
    assert result.selected_agent_ids == ("agent.a", "agent.b")
    assert [rejection.agent_id for rejection in result.rejections] == ["agent.c"]


def test_filters_disabled_excluded_capability_permission_and_type_mismatches() -> None:
    result = _resolver(
        _agent("agent.disabled", enabled=False),
        _agent("agent.excluded"),
        _agent("agent.no_capability", capabilities=("other.capability",)),
        _agent("agent.no_permission", permissions=AgentPermissions(can_execute_tools=False, requires_confirmation=False)),
        _agent("agent.wrong_type", agent_type=AgentType.MEMORY),
        _agent(
            "agent.ok",
            agent_type=AgentType.CODING,
            permissions=AgentPermissions(can_execute_tools=True, requires_confirmation=False),
        ),
    ).resolve(
        MultiAgentExecutionRequest(
            required_agent_types=(AgentType.CODING,),
            required_capability_ids=("agent.inspect",),
            required_permission_ids=("can_execute_tools",),
            excluded_agent_ids=("agent.excluded",),
            policy=MultiAgentExecutionPolicy(min_agents=1, max_agents=2),
        )
    )

    assert result.status is MultiAgentTeamResolutionStatus.RESOLVED
    assert result.selected_agent_ids == ("agent.ok",)
    assert {rejection.agent_id for rejection in result.rejections} == {
        "agent.disabled",
        "agent.excluded",
        "agent.no_capability",
        "agent.no_permission",
        "agent.wrong_type",
    }


def test_required_agent_id_missing_and_min_agents_not_reached_fail_closed() -> None:
    missing = _resolver(_agent("agent.a")).resolve(
        MultiAgentExecutionRequest(
            required_agent_ids=("agent.a", "agent.missing"),
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        )
    )
    too_few = _resolver(_agent("agent.a")).resolve(
        MultiAgentExecutionRequest(
            required_capability_ids=("agent.inspect",),
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=3),
        )
    )

    assert missing.status is MultiAgentTeamResolutionStatus.NO_MATCHING_TEAM
    assert missing.error_code == "REQUIRED_AGENT_ID_MISSING"
    assert too_few.status is MultiAgentTeamResolutionStatus.NO_MATCHING_TEAM
    assert too_few.error_code == "MIN_AGENTS_NOT_REACHED"


def test_duplicate_ids_are_canonicalized_and_invalid_requests_are_rejected() -> None:
    request = MultiAgentExecutionRequest(
        required_agent_ids=("agent.a", "agent.a", "agent.b"),
        policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
    )

    assert request.required_agent_ids == ("agent.a", "agent.b")
    with pytest.raises(InvalidMultiAgentExecutionRequestError):
        MultiAgentExecutionRequest(required_agent_ids=("__class__",))
    with pytest.raises(InvalidMultiAgentExecutionRequestError):
        MultiAgentExecutionRequest(required_capability_ids=("agent.inspect",), payload={"api_key": "hidden"})
    with pytest.raises(InvalidMultiAgentExecutionRequestError):
        MultiAgentExecutionPolicy(min_agents=3, max_agents=2)


def test_require_unique_team_fails_when_max_agents_truncates_candidates() -> None:
    result = _resolver(_agent("agent.a"), _agent("agent.b"), _agent("agent.c")).resolve(
        MultiAgentExecutionRequest(
            required_capability_ids=("agent.inspect",),
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2, require_unique_team=True),
        )
    )

    assert result.status is MultiAgentTeamResolutionStatus.AMBIGUOUS
    assert result.error_code == "TEAM_RESOLUTION_AMBIGUOUS"


def test_signature_is_deterministic_for_normalized_lists() -> None:
    first = MultiAgentExecutionRequest(
        required_capability_ids=("b.capability", "a.capability"),
        excluded_agent_ids=("agent.z", "agent.y"),
        policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=3),
    )
    second = MultiAgentExecutionRequest(
        required_capability_ids=("a.capability", "b.capability"),
        excluded_agent_ids=("agent.y", "agent.z"),
        policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=3),
    )

    assert multi_agent_execution_request_signature(first) == multi_agent_execution_request_signature(second)

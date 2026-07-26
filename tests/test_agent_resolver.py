from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from bootstrap.agent_resolver import build_core_agent_resolver
from core.agent_registry import (
    AgentCapabilities,
    AgentDefinition,
    AgentLimits,
    AgentPermissions,
    AgentRegistry,
    AgentSecurityPolicy,
    AgentType,
)
from core.agent_resolver import (
    AgentResolutionCandidate,
    AgentResolutionRequest,
    AgentResolutionStatus,
    AgentResolver,
    InvalidAgentResolutionRequestError,
    agent_resolution_request_signature,
)


def _agent(
    agent_id: str,
    *,
    agent_type: AgentType = AgentType.PROJECT_ANALYSIS,
    capabilities: tuple[str, ...] = ("project.inspect",),
    enabled: bool = True,
    can_execute_tools: bool = True,
    can_write_files: bool = False,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=agent_type,
        name=agent_id,
        description="Declarative test agent.",
        permissions=AgentPermissions(
            can_read_project=True,
            can_execute_tools=can_execute_tools,
            can_write_files=can_write_files,
        ),
        limits=AgentLimits(max_steps=2, max_tool_calls=1 if can_execute_tools else 0),
        capabilities=AgentCapabilities(capabilities=capabilities, tags=("safe",)),
        security_policy=AgentSecurityPolicy(
            allow_file_write=can_write_files,
            allowed_tools=("read_file",),
        ),
        enabled=enabled,
    )


def _resolver(*agents: AgentDefinition) -> AgentResolver:
    return AgentResolver(AgentRegistry(agents))


def test_resolves_single_compatible_agent() -> None:
    agent = _agent("agent.project")

    result = _resolver(agent).resolve(
        AgentResolutionRequest(required_capability_ids=("project.inspect",))
    )

    assert result.status is AgentResolutionStatus.RESOLVED
    assert result.selected_agent is agent
    assert result.selected_agent_id == "agent.project"
    assert result.scanned_agents == 1
    assert result.matched_agents == 1


def test_filters_by_required_capability() -> None:
    matching = _agent("agent.match", capabilities=("code.edit",))
    other = _agent("agent.other", capabilities=("project.inspect",))

    result = _resolver(other, matching).resolve(
        AgentResolutionRequest(required_capability_ids=("code.edit",))
    )

    assert result.selected_agent is matching
    assert [item.agent_id for item in result.rejections] == ["agent.other"]


def test_filters_by_required_agent_type() -> None:
    project = _agent("agent.project", agent_type=AgentType.PROJECT_ANALYSIS)
    coding = _agent("agent.coding", agent_type=AgentType.CODING)

    result = _resolver(project, coding).resolve(
        AgentResolutionRequest(required_agent_types=(AgentType.CODING,))
    )

    assert result.selected_agent is coding


def test_filters_by_required_permission() -> None:
    allowed = _agent("agent.tool", can_execute_tools=True)
    blocked = _agent("agent.readonly", can_execute_tools=False)

    result = _resolver(blocked, allowed).resolve(
        AgentResolutionRequest(required_permission_ids=("can_execute_tools",))
    )

    assert result.selected_agent is allowed
    assert result.rejections[0].agent_id == "agent.readonly"


def test_preferences_add_score_without_becoming_requirements() -> None:
    preferred = _agent("agent.preferred", capabilities=("project.inspect", "code.edit"))
    fallback = _agent("agent.fallback", capabilities=("project.inspect",))

    result = _resolver(fallback, preferred).resolve(
        AgentResolutionRequest(
            required_capability_ids=("project.inspect",),
            preferred_capability_ids=("code.edit",),
        )
    )

    assert result.selected_agent is preferred
    assert {candidate.agent.agent_id for candidate in result.candidates} == {"agent.fallback", "agent.preferred"}


def test_filters_by_required_agent_id() -> None:
    required = _agent("agent.required")
    other = _agent("agent.other")

    result = _resolver(other, required).resolve(
        AgentResolutionRequest(required_agent_ids=("agent.required",))
    )

    assert result.status is AgentResolutionStatus.RESOLVED
    assert result.selected_agent is required
    assert [item.agent_id for item in result.rejections] == ["agent.other"]


def test_excludes_agents() -> None:
    excluded = _agent("agent.excluded", capabilities=("project.inspect", "code.edit"))
    selected = _agent("agent.selected", capabilities=("project.inspect",))

    result = _resolver(excluded, selected).resolve(
        AgentResolutionRequest(
            required_capability_ids=("project.inspect",),
            preferred_capability_ids=("code.edit",),
            excluded_agent_ids=("agent.excluded",),
        )
    )

    assert result.selected_agent is selected
    assert result.rejections[0].agent_id == "agent.excluded"


def test_enabled_only_filters_disabled_agents() -> None:
    disabled = _agent("agent.disabled", capabilities=("project.inspect", "code.edit"), enabled=False)
    enabled = _agent("agent.enabled", capabilities=("project.inspect",))

    enabled_only = _resolver(disabled, enabled).resolve(
        AgentResolutionRequest(required_capability_ids=("project.inspect",))
    )
    include_disabled = _resolver(disabled, enabled).resolve(
        AgentResolutionRequest(
            required_capability_ids=("project.inspect",),
            preferred_capability_ids=("code.edit",),
            enabled_only=False,
            require_unique_top_score=False,
        )
    )

    assert enabled_only.selected_agent is enabled
    assert include_disabled.selected_agent is disabled


def test_minimum_score_can_reject_top_candidate() -> None:
    result = _resolver(_agent("agent.low")).resolve(
        AgentResolutionRequest(minimum_score=500)
    )

    assert result.status is AgentResolutionStatus.BELOW_MINIMUM_SCORE
    assert result.selected_agent is None


def test_ambiguity_when_unique_top_score_required() -> None:
    result = _resolver(_agent("agent.a"), _agent("agent.b")).resolve(
        AgentResolutionRequest(required_capability_ids=("project.inspect",))
    )

    assert result.status is AgentResolutionStatus.AMBIGUOUS
    assert result.selected_agent is None


def test_deterministic_tiebreak_when_unique_top_score_not_required() -> None:
    result = _resolver(_agent("agent.b"), _agent("agent.a")).resolve(
        AgentResolutionRequest(
            required_capability_ids=("project.inspect",),
            require_unique_top_score=False,
        )
    )

    assert result.status is AgentResolutionStatus.RESOLVED
    assert result.selected_agent_id == "agent.a"


def test_stable_order_uses_score_then_tiebreakers() -> None:
    best = _agent("agent.best", capabilities=("project.inspect", "code.edit"))
    middle = _agent("agent.middle", capabilities=("project.inspect",))
    lower = _agent("agent.lower", capabilities=("other.capability",))

    result = _resolver(lower, middle, best).resolve(
        AgentResolutionRequest(
            preferred_capability_ids=("code.edit",),
            require_unique_top_score=False,
        )
    )

    assert [candidate.agent.agent_id for candidate in result.candidates] == [
        "agent.best",
        "agent.lower",
        "agent.middle",
    ]


def test_reason_scores_sum_to_candidate_score() -> None:
    result = _resolver(_agent("agent.project")).resolve(
        AgentResolutionRequest(
            required_capability_ids=("project.inspect",),
            preferred_agent_ids=("agent.project",),
            preferred_permission_ids=("can_execute_tools",),
        )
    )

    assert result.candidates
    assert all(sum(reason.score for reason in candidate.reasons) == candidate.score for candidate in result.candidates)


def test_empty_registry_returns_no_agents() -> None:
    result = AgentResolver(AgentRegistry()).resolve(AgentResolutionRequest())

    assert result.status is AgentResolutionStatus.NO_AGENTS
    assert result.scanned_agents == 0


def test_no_matching_agents() -> None:
    result = _resolver(_agent("agent.project")).resolve(
        AgentResolutionRequest(required_capability_ids=("missing.capability",))
    )

    assert result.status is AgentResolutionStatus.NO_MATCHING_AGENTS
    assert result.rejections[0].agent_id == "agent.project"


def test_invalid_request_result_and_constructor_validation() -> None:
    result = _resolver(_agent("agent.project")).resolve(object())  # type: ignore[arg-type]

    assert result.status is AgentResolutionStatus.INVALID_REQUEST
    with pytest.raises(InvalidAgentResolutionRequestError):
        AgentResolutionRequest(required_capability_ids=("bad value",))


def test_size_limits_are_enforced() -> None:
    too_many = tuple(f"cap.{index}" for index in range(33))

    with pytest.raises(InvalidAgentResolutionRequestError):
        AgentResolutionRequest(required_capability_ids=too_many)
    with pytest.raises(InvalidAgentResolutionRequestError):
        AgentResolutionRequest(maximum_candidates_considered=65)


def test_invalid_metadata_is_rejected() -> None:
    with pytest.raises(InvalidAgentResolutionRequestError):
        AgentResolutionRequest(metadata={"api_token": "hidden"})
    with pytest.raises(InvalidAgentResolutionRequestError):
        AgentResolutionRequest(metadata={"value": float("nan")})
    with pytest.raises(InvalidAgentResolutionRequestError):
        AgentResolutionRequest(metadata={"callback": lambda: None})


def test_request_and_candidate_are_immutable_and_defensively_copied() -> None:
    metadata = {"trace": ["a", "b"]}
    request = AgentResolutionRequest(metadata=metadata)
    candidate = AgentResolutionCandidate(_agent("agent.project"), 0, ())

    metadata["trace"].append("c")

    assert request.metadata["trace"] == ("a", "b")
    assert isinstance(request.metadata, MappingProxyType)
    with pytest.raises(FrozenInstanceError):
        candidate.score = 10  # type: ignore[misc]


def test_signature_is_deterministic_and_changes_with_structure() -> None:
    first = AgentResolutionRequest(
        required_capability_ids=("b.capability", "a.capability"),
        metadata={"trace": ("x", "y")},
    )
    same = AgentResolutionRequest(
        required_capability_ids=("a.capability", "b.capability"),
        metadata={"trace": ("x", "y")},
    )
    different = AgentResolutionRequest(
        required_capability_ids=("a.capability",),
        metadata={"trace": ("x", "y")},
    )

    assert agent_resolution_request_signature(first) == agent_resolution_request_signature(same)
    assert agent_resolution_request_signature(first) != agent_resolution_request_signature(different)


def test_safe_errors_do_not_expose_secret_values() -> None:
    result = _resolver(_agent("agent.project")).resolve(
        AgentResolutionRequest(minimum_score=500, metadata={"trace": "safe"})
    )

    assert "secret" not in str(result.error_message).lower()
    assert "token" not in str(result.error_message).lower()


def test_bootstrap_builds_resolver_from_agent_registry() -> None:
    registry = AgentRegistry((_agent("agent.project"),))
    resolver = build_core_agent_resolver(registry)

    result = resolver.resolve(AgentResolutionRequest(required_capability_ids=("project.inspect",)))

    assert isinstance(resolver, AgentResolver)
    assert result.selected_agent_id == "agent.project"


def test_agent_registry_phase_10_1_compatibility() -> None:
    registry = AgentRegistry((_agent("agent.project"),))

    assert registry.find_by_capability("project.inspect")
    assert AgentResolver(registry).resolve(AgentResolutionRequest()).status is AgentResolutionStatus.RESOLVED

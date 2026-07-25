from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import sys
from types import MappingProxyType

import pytest

from bootstrap.agent_context import build_core_agent_context_builder
from core.agent_context import (
    AgentContextBuilder,
    AgentContextRequest,
    AgentContextStatus,
    InvalidAgentContextRequestError,
    agent_context_request_signature,
)
from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentLimits,
    AgentMemoryPolicy,
    AgentPermissions,
    AgentRegistry,
    AgentSecurityPolicy,
    AgentType,
)
from core.agent_resolver import AgentResolutionRequest, AgentResolver


def _agent(
    agent_id: str = "agent.context",
    *,
    context_policy: AgentContextPolicy | None = None,
    memory_policy: AgentMemoryPolicy | None = None,
    permissions: AgentPermissions | None = None,
    security_policy: AgentSecurityPolicy | None = None,
) -> AgentDefinition:
    active_permissions = permissions or AgentPermissions(
        can_read_project=True,
        can_execute_tools=True,
        can_modify_memory=True,
    )
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=AgentType.PROJECT_ANALYSIS,
        name="Context agent",
        description="Declarative context test agent.",
        permissions=active_permissions,
        limits=AgentLimits(max_steps=2, max_tool_calls=1 if active_permissions.can_execute_tools else 0),
        capabilities=AgentCapabilities(capabilities=("project.inspect",)),
        context_policy=context_policy or AgentContextPolicy(),
        memory_policy=memory_policy or AgentMemoryPolicy(),
        security_policy=security_policy or AgentSecurityPolicy(),
    )


def _builder() -> AgentContextBuilder:
    return AgentContextBuilder()


def test_minimal_valid_context_is_built() -> None:
    result = _builder().build(AgentContextRequest(agent=_agent(), task_id="task.1"))

    assert result.status is AgentContextStatus.BUILT
    assert result.context is not None
    assert result.context.agent_id == "agent.context"
    assert result.context.task_id == "task.1"
    assert result.context.context_signature


def test_context_is_immutable() -> None:
    result = _builder().build(AgentContextRequest(agent=_agent(), structured_input={"a": 1}))
    assert result.context is not None

    with pytest.raises(FrozenInstanceError):
        result.context.agent_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.context.structured_input["a"] = 2  # type: ignore[index]


def test_defensive_copies_prevent_external_mutation() -> None:
    source = {"items": ["a", "b"]}
    result = _builder().build(AgentContextRequest(agent=_agent(), structured_input=source))
    source["items"].append("c")

    assert result.context is not None
    assert result.context.structured_input["items"] == ("a", "b")


def test_same_input_produces_same_signature() -> None:
    request = AgentContextRequest(agent=_agent(), structured_input={"b": 2, "a": 1})

    first = _builder().build(request)
    second = _builder().build(request)

    assert first.context is not None
    assert second.context is not None
    assert first.context.context_signature == second.context.context_signature
    assert first.request_signature == second.request_signature


def test_real_input_change_changes_signature() -> None:
    first = _builder().build(AgentContextRequest(agent=_agent(), structured_input={"a": 1}))
    second = _builder().build(AgentContextRequest(agent=_agent(), structured_input={"a": 2}))

    assert first.context is not None
    assert second.context is not None
    assert first.context.context_signature != second.context.context_signature


def test_policy_allows_user_input_and_conversation() -> None:
    agent = _agent(
        context_policy=AgentContextPolicy(
            allow_user_input=True,
            include_conversation_context=True,
        )
    )
    result = _builder().build(
        AgentContextRequest(
            agent=agent,
            user_input="analiza",
            conversation_context=({"role": "user", "text": "hola"},),
        )
    )

    assert result.context is not None
    assert result.context.user_input == "analiza"
    assert result.context.conversation_context == (MappingProxyType({"role": "user", "text": "hola"}),)


def test_policy_denies_user_input_section() -> None:
    result = _builder().build(AgentContextRequest(agent=_agent(), user_input="no entra"))

    assert result.context is not None
    assert result.context.user_input is None
    assert "user_input:policy_denied" in result.omitted_sections


def test_memory_denied() -> None:
    result = _builder().build(AgentContextRequest(agent=_agent(), memory_context={"note": "x"}))

    assert result.context is not None
    assert dict(result.context.memory_context) == {}
    assert "memory_context:policy_denied" in result.omitted_sections


def test_conversation_denied() -> None:
    result = _builder().build(
        AgentContextRequest(agent=_agent(), conversation_context=({"text": "x"},))
    )

    assert result.context is not None
    assert result.context.conversation_context == ()
    assert "conversation_context:policy_denied" in result.omitted_sections


def test_tool_results_denied() -> None:
    result = _builder().build(AgentContextRequest(agent=_agent(), tool_results={"read_file": {"ok": True}}))

    assert result.context is not None
    assert dict(result.context.tool_results) == {}
    assert "tool_results:policy_denied" in result.omitted_sections


def test_workflow_results_denied() -> None:
    result = _builder().build(AgentContextRequest(agent=_agent(), workflow_results={"workflow": {"ok": True}}))

    assert result.context is not None
    assert dict(result.context.workflow_results) == {}
    assert "workflow_results:policy_denied" in result.omitted_sections


def test_sensitive_nested_keys_are_removed() -> None:
    result = _builder().build(
        AgentContextRequest(
            agent=_agent(),
            structured_input={"safe": {"password": "hidden", "value": "ok"}},
        )
    )

    assert result.context is not None
    assert "password" not in result.context.structured_input["safe"]
    assert result.sanitized_fields_count == 1


def test_sensitive_keys_with_case_and_hyphen_are_removed() -> None:
    result = _builder().build(
        AgentContextRequest(
            agent=_agent(),
            structured_input={"Authorization": "Bearer x", "private-key": "k", "safe": "ok"},
        )
    )

    assert result.context is not None
    assert dict(result.context.structured_input) == {"safe": "ok"}
    assert result.sanitized_fields_count == 2


@pytest.mark.parametrize("value", (lambda: None, object, sys))
def test_rejects_functions_classes_and_modules(value: object) -> None:
    result = _builder().build(AgentContextRequest(agent=_agent(), structured_input={"bad": value}))

    assert result.status is AgentContextStatus.INVALID_REQUEST


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_rejects_nan_and_infinity(value: float) -> None:
    result = _builder().build(AgentContextRequest(agent=_agent(), structured_input={"bad": value}))

    assert result.status is AgentContextStatus.INVALID_REQUEST
    assert math.isfinite(value) is False


def test_depth_limit() -> None:
    agent = _agent(context_policy=AgentContextPolicy(max_context_depth=1))
    result = _builder().build(
        AgentContextRequest(agent=agent, structured_input={"a": {"b": {"c": 1}}})
    )

    assert result.status is AgentContextStatus.LIMIT_EXCEEDED


def test_string_limit() -> None:
    agent = _agent(context_policy=AgentContextPolicy(max_string_length=3))
    result = _builder().build(AgentContextRequest(agent=agent, structured_input={"text": "abcd"}))

    assert result.status is AgentContextStatus.LIMIT_EXCEEDED


def test_sequence_limit() -> None:
    agent = _agent(context_policy=AgentContextPolicy(max_sequence_items=2))
    result = _builder().build(AgentContextRequest(agent=agent, structured_input={"items": [1, 2, 3]}))

    assert result.status is AgentContextStatus.LIMIT_EXCEEDED


def test_mapping_limit() -> None:
    agent = _agent(context_policy=AgentContextPolicy(max_mapping_items=1))
    result = _builder().build(AgentContextRequest(agent=agent, structured_input={"a": 1, "b": 2}))

    assert result.status is AgentContextStatus.LIMIT_EXCEEDED


def test_total_context_limit() -> None:
    agent = _agent(context_policy=AgentContextPolicy(max_total_items=2))
    result = _builder().build(AgentContextRequest(agent=agent, structured_input={"a": [1, 2, 3]}))

    assert result.status is AgentContextStatus.LIMIT_EXCEEDED


def test_context_from_other_agent_is_not_mixed() -> None:
    result = _builder().build(
        AgentContextRequest(agent=_agent("agent.one"), shared_context={"agent_id": "agent.two", "value": "x"})
    )

    assert result.context is not None
    assert dict(result.context.shared_context) == {}


def test_errors_do_not_expose_secrets() -> None:
    result = _builder().build(
        AgentContextRequest(agent=_agent(), structured_input={"safe": "x" * 5000, "api_key": "SECRET"})
    )

    assert result.status is AgentContextStatus.LIMIT_EXCEEDED
    assert "SECRET" not in str(result.safe_message)
    assert "api_key" not in str(result.safe_message)


def test_compatibility_with_real_agent_definition_and_resolver() -> None:
    agent = _agent(
        context_policy=AgentContextPolicy(allow_user_input=True),
        memory_policy=AgentMemoryPolicy(can_read_memory=True, memory_scopes=("project",), max_memory_items=1),
    )
    registry = AgentRegistry((agent,))
    resolved = AgentResolver(registry).resolve(AgentResolutionRequest(required_capability_ids=("project.inspect",)))
    assert resolved.selected_agent is not None

    result = _builder().build(AgentContextRequest(agent=resolved.selected_agent, user_input="ok"))

    assert result.status is AgentContextStatus.BUILT
    assert result.context is not None
    assert result.context.agent_id == agent.agent_id


def test_bootstrap_builds_context_builder() -> None:
    builder = build_core_agent_context_builder()

    assert isinstance(builder, AgentContextBuilder)


def test_request_signature_is_deterministic() -> None:
    request = AgentContextRequest(agent=_agent(), structured_input={"b": 2, "a": 1})

    assert agent_context_request_signature(request) == agent_context_request_signature(request)

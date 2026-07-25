from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agents.registry import AgentRegistry as RuntimeAgentRegistry
from core.agent_registry import (
    AgentAlreadyRegisteredError,
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentLimits,
    AgentMemoryPolicy,
    AgentNotFoundError,
    AgentPermissions,
    AgentRegistry,
    AgentSecurityPolicy,
    AgentType,
    InvalidAgentDefinitionError,
)


def _definition(
    agent_id: str = "agent.project",
    *,
    agent_type: AgentType = AgentType.PROJECT_ANALYSIS,
    capability: str = "project.inspect",
    enabled: bool = True,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=agent_type,
        name=f"Agent {agent_id}",
        description="Specialized declarative test agent.",
        permissions=AgentPermissions(
            can_read_project=True,
            can_execute_tools=True,
            can_modify_memory=True,
        ),
        limits=AgentLimits(max_steps=3, max_tool_calls=2, max_context_items=8, max_memory_items=1),
        capabilities=AgentCapabilities(
            capabilities=(capability, "shared.capability"),
            tools=("read_file",),
            tags=("safe", "test"),
        ),
        context_policy=AgentContextPolicy(
            include_project_context=True,
            include_conversation_context=False,
            max_context_items=8,
            allowed_context_keys=("project.tree",),
        ),
        memory_policy=AgentMemoryPolicy(
            can_read_memory=True,
            can_write_memory=False,
            memory_scopes=("project",),
            max_memory_items=1,
        ),
        security_policy=AgentSecurityPolicy(
            allow_network=False,
            allow_file_write=False,
            allowed_tools=("read_file",),
            blocked_tools=("write_file",),
        ),
        enabled=enabled,
        metadata={"owner": "atlas", "version": 1},
    )


def test_agent_definition_is_immutable_and_normalized() -> None:
    definition = _definition()

    assert definition.id == "agent.project"
    assert definition.agent_type is AgentType.PROJECT_ANALYSIS
    assert definition.capabilities.capabilities == ("project.inspect", "shared.capability")
    assert definition.metadata["owner"] == "atlas"

    with pytest.raises(FrozenInstanceError):
        definition.name = "Other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        definition.metadata["owner"] = "other"  # type: ignore[index]


def test_agent_type_accepts_string_values() -> None:
    definition = AgentDefinition(
        agent_id="agent.coding",
        agent_type="coding",
        name="Coding",
        description="Coding agent.",
    )

    assert definition.agent_type is AgentType.CODING


@pytest.mark.parametrize(
    "agent_id",
    ("", " agent.project ", "../agent", "agent/project", "agent$", "__class__"),
)
def test_agent_id_validation_rejects_unsafe_values(agent_id: str) -> None:
    with pytest.raises(InvalidAgentDefinitionError):
        _definition(agent_id)


def test_policy_validation_rejects_inconsistent_permissions() -> None:
    with pytest.raises(InvalidAgentDefinitionError):
        AgentDefinition(
            agent_id="agent.write",
            agent_type=AgentType.CODING,
            name="Writer",
            description="Invalid writer.",
            permissions=AgentPermissions(can_write_files=False),
            security_policy=AgentSecurityPolicy(allow_file_write=True),
        )
    with pytest.raises(InvalidAgentDefinitionError):
        AgentMemoryPolicy(can_read_memory=False, can_write_memory=True)
    with pytest.raises(InvalidAgentDefinitionError):
        AgentDefinition(
            agent_id="agent.tool",
            agent_type=AgentType.EXECUTION,
            name="Tool",
            description="Invalid tool limit.",
            permissions=AgentPermissions(can_execute_tools=False),
            limits=AgentLimits(max_tool_calls=1),
        )


def test_capability_and_security_validation() -> None:
    assert AgentCapabilities(capabilities=("a.b", "a.b")).capabilities == ("a.b",)
    with pytest.raises(InvalidAgentDefinitionError):
        AgentCapabilities(capabilities=("bad value",))
    with pytest.raises(InvalidAgentDefinitionError):
        AgentSecurityPolicy(allowed_tools=("read_file",), blocked_tools=("read_file",))
    with pytest.raises(InvalidAgentDefinitionError):
        AgentLimits(max_steps=0)


def test_registry_register_get_contains_list_and_iteration_order() -> None:
    first = _definition("agent.project", capability="project.inspect")
    second = _definition("agent.coding", agent_type=AgentType.CODING, capability="code.edit")
    registry = AgentRegistry((first,))

    registry.register(second)

    assert len(registry) == 2
    assert registry.contains("agent.project") is True
    assert registry.get("agent.project") is first
    assert registry.list_agents() == (first, second)
    assert tuple(registry) == (first, second)


def test_registry_rejects_duplicate_ids_unless_replace_is_explicit() -> None:
    first = _definition("agent.project", capability="project.inspect")
    replacement = _definition("agent.project", capability="project.analyze")
    registry = AgentRegistry((first,))

    with pytest.raises(AgentAlreadyRegisteredError):
        registry.register(replacement)

    assert registry.register(replacement, replace=True) is replacement
    assert registry.get("agent.project") is replacement


def test_registry_get_missing_raises_structured_error() -> None:
    with pytest.raises(AgentNotFoundError):
        AgentRegistry().get("agent.missing")


def test_find_by_capability_filters_enabled_agents_by_default() -> None:
    enabled = _definition("agent.enabled", capability="project.inspect", enabled=True)
    disabled = _definition("agent.disabled", capability="project.inspect", enabled=False)
    registry = AgentRegistry((enabled, disabled))

    assert registry.find_by_capability("project.inspect") == (enabled,)
    assert registry.find_by_capability("project.inspect", enabled_only=False) == (enabled, disabled)


def test_find_by_type_filters_enabled_agents_by_default() -> None:
    project = _definition("agent.project", agent_type=AgentType.PROJECT_ANALYSIS, enabled=True)
    coding = _definition("agent.coding", agent_type=AgentType.CODING, capability="code.edit", enabled=False)
    registry = AgentRegistry((project, coding))

    assert registry.find_by_type(AgentType.PROJECT_ANALYSIS) == (project,)
    assert registry.find_by_type("coding") == ()
    assert registry.find_by_type("coding", enabled_only=False) == (coding,)


def test_list_agents_can_filter_enabled_and_clear_registry() -> None:
    enabled = _definition("agent.enabled", enabled=True)
    disabled = _definition("agent.disabled", capability="other.capability", enabled=False)
    registry = AgentRegistry((enabled, disabled))

    assert registry.list_agents(enabled_only=True) == (enabled,)
    registry.clear()
    assert registry.list_agents() == ()


def test_invalid_registry_inputs_are_rejected() -> None:
    registry = AgentRegistry()
    with pytest.raises(InvalidAgentDefinitionError):
        registry.register(object())  # type: ignore[arg-type]
    with pytest.raises(InvalidAgentDefinitionError):
        registry.find_by_capability("bad capability")


def test_runtime_agent_registry_compatibility_is_preserved() -> None:
    registry = RuntimeAgentRegistry()

    assert registry.list() == []
    assert registry.get("missing") is None

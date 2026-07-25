"""Factory for the deterministic specialized-agent resolver."""

from __future__ import annotations

from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolver


def build_core_agent_resolver(
    agent_registry: AgentRegistry,
) -> AgentResolver:
    """Build an AgentResolver from an explicitly injected AgentRegistry."""

    return AgentResolver(agent_registry)

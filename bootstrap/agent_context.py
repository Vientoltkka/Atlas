"""Factory for safe specialized-agent context construction."""

from __future__ import annotations

from core.agent_context import AgentContextBuilder


def build_core_agent_context_builder() -> AgentContextBuilder:
    """Build an AgentContextBuilder with no hidden global dependencies."""

    return AgentContextBuilder()

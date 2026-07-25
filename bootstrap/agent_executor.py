"""Factory for controlled specialized-agent execution."""

from __future__ import annotations

from core.agent_context import AgentContextBuilder
from core.agent_executor import AgentExecutor, AgentHandlerRegistry
from core.agent_resolver import AgentResolver


def build_core_agent_executor(
    agent_resolver: AgentResolver,
    agent_context_builder: AgentContextBuilder,
    agent_handler_registry: AgentHandlerRegistry,
) -> AgentExecutor:
    """Build an AgentExecutor from explicitly injected collaborators."""

    return AgentExecutor(
        agent_resolver=agent_resolver,
        agent_context_builder=agent_context_builder,
        agent_handler_registry=agent_handler_registry,
    )

"""Factory for safe specialized-agent delegation."""

from __future__ import annotations

from core.agent_context import AgentContextBuilder
from core.agent_delegation import AgentDelegationService
from core.agent_executor import AgentExecutor
from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolver


def build_core_agent_delegation_service(
    *,
    agent_registry: AgentRegistry,
    agent_resolver: AgentResolver,
    agent_context_builder: AgentContextBuilder,
    agent_executor: AgentExecutor,
) -> AgentDelegationService:
    """Build AgentDelegationService from explicit shared collaborators."""

    return AgentDelegationService(
        agent_registry=agent_registry,
        agent_resolver=agent_resolver,
        agent_context_builder=agent_context_builder,
        agent_executor=agent_executor,
    )

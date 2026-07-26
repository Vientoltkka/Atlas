"""Factory for multi-chain agent delegation coordination."""

from __future__ import annotations

from core.agent_context import AgentContextBuilder
from core.agent_delegation import AgentDelegationService
from core.agent_delegation_chain import AgentDelegationChainService
from core.agent_delegation_coordinator import AgentDelegationCoordinator
from core.agent_executor import AgentExecutor
from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolver


def build_core_agent_delegation_coordinator(
    *,
    agent_registry: AgentRegistry,
    agent_resolver: AgentResolver,
    agent_context_builder: AgentContextBuilder,
    agent_executor: AgentExecutor,
    agent_delegation_service: AgentDelegationService,
    agent_delegation_chain_service: AgentDelegationChainService,
) -> AgentDelegationCoordinator:
    """Build AgentDelegationCoordinator from explicit shared collaborators."""

    return AgentDelegationCoordinator(
        agent_registry=agent_registry,
        agent_resolver=agent_resolver,
        agent_context_builder=agent_context_builder,
        agent_executor=agent_executor,
        agent_delegation_service=agent_delegation_service,
        agent_delegation_chain_service=agent_delegation_chain_service,
    )

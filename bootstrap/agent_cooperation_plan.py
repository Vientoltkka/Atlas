"""Factory for declarative cooperation plans."""

from __future__ import annotations

from core.agent_cooperation_plan import AgentCooperationPlanner
from core.agent_context import AgentContextBuilder
from core.agent_delegation import AgentDelegationService
from core.agent_delegation_chain import AgentDelegationChainService
from core.agent_delegation_coordinator import AgentDelegationCoordinator
from core.agent_executor import AgentExecutor
from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolver
from core.multi_agent import MultiAgentCoordinator
from core.skill_system import SkillSystem


def build_core_agent_cooperation_planner(
    *,
    agent_registry: AgentRegistry,
    agent_resolver: AgentResolver,
    agent_context_builder: AgentContextBuilder,
    agent_executor: AgentExecutor,
    agent_delegation_service: AgentDelegationService,
    agent_delegation_chain_service: AgentDelegationChainService,
    agent_delegation_coordinator: AgentDelegationCoordinator,
    multi_agent_coordinator: MultiAgentCoordinator,
    skill_system: SkillSystem,
) -> AgentCooperationPlanner:
    """Build one planner from explicit shared collaborators without execution."""

    return AgentCooperationPlanner(
        agent_registry=agent_registry,
        agent_resolver=agent_resolver,
        agent_context_builder=agent_context_builder,
        agent_executor=agent_executor,
        agent_delegation_service=agent_delegation_service,
        agent_delegation_chain_service=agent_delegation_chain_service,
        agent_delegation_coordinator=agent_delegation_coordinator,
        multi_agent_coordinator=multi_agent_coordinator,
        skill_system=skill_system,
    )

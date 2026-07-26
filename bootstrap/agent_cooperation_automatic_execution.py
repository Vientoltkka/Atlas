"""Factory for controlled automatic cooperation execution."""

from __future__ import annotations

from core.agent_context import AgentContextBuilder
from core.agent_cooperation_automatic_execution import (
    AgentCooperationAutomaticExecutionService,
)
from core.agent_cooperation_automatic_planner import AgentCooperationAutomaticPlanner
from core.agent_cooperation_plan import AgentCooperationPlanner
from core.agent_delegation import AgentDelegationService
from core.agent_delegation_chain import AgentDelegationChainService
from core.agent_delegation_coordinator import AgentDelegationCoordinator
from core.agent_executor import AgentExecutor
from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolver
from core.multi_agent import MultiAgentCoordinator, MultiAgentResolver
from core.skill_system import SkillSystem


def build_core_agent_cooperation_automatic_execution_service(
    *,
    agent_registry: AgentRegistry,
    agent_resolver: AgentResolver,
    agent_context_builder: AgentContextBuilder,
    agent_executor: AgentExecutor,
    skill_system: SkillSystem,
    agent_delegation_service: AgentDelegationService,
    agent_delegation_chain_service: AgentDelegationChainService,
    agent_delegation_coordinator: AgentDelegationCoordinator,
    multi_agent_resolver: MultiAgentResolver,
    multi_agent_coordinator: MultiAgentCoordinator,
    agent_cooperation_planner: AgentCooperationPlanner,
    agent_cooperation_automatic_planner: AgentCooperationAutomaticPlanner,
) -> AgentCooperationAutomaticExecutionService:
    """Build the service from shared dependencies without planning or execution."""

    return AgentCooperationAutomaticExecutionService(
        agent_registry=agent_registry,
        agent_resolver=agent_resolver,
        agent_context_builder=agent_context_builder,
        agent_executor=agent_executor,
        skill_system=skill_system,
        agent_delegation_service=agent_delegation_service,
        agent_delegation_chain_service=agent_delegation_chain_service,
        agent_delegation_coordinator=agent_delegation_coordinator,
        multi_agent_resolver=multi_agent_resolver,
        multi_agent_coordinator=multi_agent_coordinator,
        agent_cooperation_planner=agent_cooperation_planner,
        agent_cooperation_automatic_planner=agent_cooperation_automatic_planner,
    )

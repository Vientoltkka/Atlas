"""Factory for controlled cooperation-plan replanning proposals."""

from __future__ import annotations

from core.agent_cooperation_plan import AgentCooperationPlanner
from core.agent_plan_replanner import AgentPlanReplanner, build_core_agent_plan_replanner
from core.agent_plan_supervisor import AgentPlanSupervisor
from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolver


def build_agent_plan_replanner(
    *,
    agent_registry: AgentRegistry,
    agent_resolver: AgentResolver,
    agent_cooperation_planner: AgentCooperationPlanner,
    agent_plan_supervisor: AgentPlanSupervisor,
) -> AgentPlanReplanner:
    """Build the pure replanner from shared collaborators."""

    return build_core_agent_plan_replanner(
        agent_registry=agent_registry,
        agent_resolver=agent_resolver,
        agent_cooperation_planner=agent_cooperation_planner,
        agent_plan_supervisor=agent_plan_supervisor,
    )


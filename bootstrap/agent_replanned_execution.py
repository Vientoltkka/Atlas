"""Factory for controlled replanned execution."""

from __future__ import annotations

from core.agent_cooperation_plan import AgentCooperationPlanner
from core.agent_replanned_execution import (
    AgentReplannedExecutionService,
    build_core_agent_replanned_execution_service,
)
from core.agent_plan_replanner import AgentPlanReplanner
from core.agent_plan_supervisor import AgentPlanSupervisor
from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolver


def build_agent_replanned_execution_service(
    *,
    agent_registry: AgentRegistry,
    agent_resolver: AgentResolver,
    agent_cooperation_planner: AgentCooperationPlanner,
    agent_plan_supervisor: AgentPlanSupervisor,
    agent_plan_replanner: AgentPlanReplanner,
) -> AgentReplannedExecutionService:
    """Build the controlled service with shared existing components."""

    return build_core_agent_replanned_execution_service(
        agent_registry=agent_registry,
        agent_resolver=agent_resolver,
        agent_cooperation_planner=agent_cooperation_planner,
        agent_plan_supervisor=agent_plan_supervisor,
        agent_plan_replanner=agent_plan_replanner,
    )


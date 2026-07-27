"""Factory for deterministic cooperation-plan supervision."""

from __future__ import annotations

from core.agent_plan_supervisor import AgentPlanSupervisor


def build_core_agent_plan_supervisor() -> AgentPlanSupervisor:
    """Build the pure supervisor without executing plans or agents."""

    return AgentPlanSupervisor()


def build_agent_plan_supervisor() -> AgentPlanSupervisor:
    """Build the pure supervisor without executing plans or agents."""

    return build_core_agent_plan_supervisor()

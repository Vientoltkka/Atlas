"""Factory for the isolated capability orchestrator."""

from __future__ import annotations

from core.capability_orchestrator import CapabilityOrchestrator, Observer
from core.capability_planner import CapabilityPlanner
from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_plan_validator import ExecutionPlanValidator
from core.goal_driven_execution import GoalDrivenExecutionController


def build_core_capability_orchestrator(
    capability_planner: CapabilityPlanner,
    execution_plan_validator: ExecutionPlanValidator,
    execution_plan_executor: ExecutionPlanExecutor,
    *,
    goal_driven_controller: GoalDrivenExecutionController | None = None,
    observer: Observer | None = None,
) -> CapabilityOrchestrator:
    """Build a capability orchestrator from explicitly injected collaborators."""

    return CapabilityOrchestrator(
        capability_planner,
        execution_plan_validator,
        execution_plan_executor,
        goal_driven_controller=goal_driven_controller,
        observer=observer,
    )

"""Factory for bounded goal-driven execution."""

from __future__ import annotations

from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_plan_validator import ExecutionPlanValidator
from core.execution_replanner import ExecutionReplanner
from core.goal_driven_execution import GoalDrivenExecutionController, GoalDrivenObserver
from core.goal_verifier import GoalVerifier


def build_goal_driven_execution_controller(
    execution_plan_validator: ExecutionPlanValidator,
    execution_plan_executor: ExecutionPlanExecutor,
    *,
    goal_verifier: GoalVerifier | None = None,
    execution_replanner: ExecutionReplanner | None = None,
    observer: GoalDrivenObserver | None = None,
) -> GoalDrivenExecutionController:
    """Build a bounded goal-driven execution controller from injected collaborators."""

    return GoalDrivenExecutionController(
        execution_plan_validator,
        execution_plan_executor,
        goal_verifier=goal_verifier,
        execution_replanner=execution_replanner,
        observer=observer,
    )

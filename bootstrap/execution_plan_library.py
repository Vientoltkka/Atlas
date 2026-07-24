"""Composition helpers for Atlas execution-plan libraries."""

from __future__ import annotations

from core.execution_plan_library import ExecutionPlanLibrary
from core.execution_plan_output import ExecutionPlanOutput
from core.execution_plan_registry import ExecutionPlanReference
from core.execution_variable_reference import ExecutionVariableReference
from core.planner import ExecutionPlan, ExecutionStep
from core.step_output_reference import StepOutputReference
from core.workflow_discovery import WorkflowDiscoveryService


def build_core_execution_plan_library() -> ExecutionPlanLibrary | None:
    """Return Atlas core workflows when stable production definitions exist."""
    return ExecutionPlanLibrary(
        "atlas.core",
        (
            _project_tree_workflow(),
        ),
        version="1.0",
        title="Atlas core workflows",
        description="Stable production workflows backed by real Atlas tools.",
    )


def build_workflow_discovery_service() -> WorkflowDiscoveryService:
    """Build the pure deterministic workflow discovery service."""
    return WorkflowDiscoveryService()


def _project_tree_workflow():
    plan = ExecutionPlan(
        goal="Show the structure of a project or directory.",
        ordered_steps=(
            ExecutionStep(
                "project_tree",
                "Return the Python project tree for the requested path.",
                "project_tree",
                arguments={"path": ExecutionVariableReference("path")},
            ),
        ),
        estimated_steps=1,
        required_tools=("project_tree",),
        detected_risks=(),
        requires_confirmation=False,
        output=ExecutionPlanOutput(
            {
                "path": ExecutionVariableReference("path"),
                "tree": StepOutputReference("project_tree"),
            }
        ),
    )
    from core.execution_plan_library import WorkflowDefinition

    return WorkflowDefinition(
        reference=ExecutionPlanReference("project.tree.show", "1.0"),
        plan=plan,
        title="Show project tree",
        description="Return the Python file structure for an explicit local project or directory path.",
        category="project.analysis",
        tags=("project_tree", "filesystem", "read_only"),
    )

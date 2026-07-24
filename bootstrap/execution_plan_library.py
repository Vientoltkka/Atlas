"""Composition helpers for Atlas execution-plan libraries."""

from __future__ import annotations

from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
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
            _directory_list_workflow(),
        ),
        version="1.0",
        title="Atlas core workflows",
        description="Stable production workflows backed by real Atlas tools.",
    )


def build_workflow_discovery_service() -> WorkflowDiscoveryService:
    """Build the pure deterministic workflow discovery service."""
    return WorkflowDiscoveryService()


def _project_tree_workflow() -> WorkflowDefinition:
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
                "directory_path": ExecutionVariableReference("path"),
                "tree": StepOutputReference("project_tree"),
            }
        ),
    )
    return WorkflowDefinition(
        reference=ExecutionPlanReference("project.tree.show", "1.0"),
        plan=plan,
        title="Show project tree",
        description="Return the Python file structure for an explicit local project or directory path.",
        category="project.analysis",
        tags=("project_tree", "filesystem", "read_only"),
    )


def _directory_list_workflow() -> WorkflowDefinition:
    plan = ExecutionPlan(
        goal="List the direct entries of a directory.",
        ordered_steps=(
            ExecutionStep(
                "list_directory",
                "Return the direct entries for the requested directory path.",
                "list_directory",
                arguments={"path": ExecutionVariableReference("directory_path")},
            ),
        ),
        estimated_steps=1,
        required_tools=("list_directory",),
        detected_risks=(),
        requires_confirmation=False,
        output=ExecutionPlanOutput(
            {
                "directory_path": ExecutionVariableReference("directory_path"),
                "entries": StepOutputReference("list_directory"),
            }
        ),
    )
    return WorkflowDefinition(
        reference=ExecutionPlanReference("directory.list", "1.0"),
        plan=plan,
        title="List directory entries",
        description="Return sorted direct entry names for an explicit local directory path.",
        category="filesystem.directory",
        tags=("list_directory", "filesystem", "directory", "read_only"),
    )

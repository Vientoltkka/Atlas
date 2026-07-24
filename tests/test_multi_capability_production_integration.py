from __future__ import annotations

from pathlib import Path

import pytest

from bootstrap.bootstrap import Bootstrap
from core.atlas_request_classifier import StructuredInput
from core.atlas_router import AtlasRoutingStatus
from core.capability_execution_service import CapabilityExecutionStatus
from core.execution_variable_reference import ExecutionVariableReference


PROJECT_TREE_CAPABILITY_ID = "workflow.atlas.core.project.tree.show.1.0"
PROJECT_TREE_REFERENCE_ID = "project.tree.show"
DIRECTORY_LIST_REFERENCE_ID = "directory.list"


@pytest.fixture(scope="module")
def bootstrapped_atlas():
    return Bootstrap.build()


def _project(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    (tmp_path / "zeta.py").write_text("print('zeta')\n", encoding="utf-8")
    return tmp_path


def _capability_result(orchestrator, payload: dict[str, object]):
    return orchestrator.route_structured_input(
        StructuredInput(
            route="capability",
            payload=payload,
        )
    )


def test_single_capability_execution_remains_on_existing_path(tmp_path: Path, bootstrapped_atlas) -> None:
    project = _project(tmp_path)

    result = _capability_result(
        bootstrapped_atlas,
        {
            "capability_id": PROJECT_TREE_CAPABILITY_ID,
            "required_inputs": ["path"],
            "required_outputs": ["tree"],
            "path": str(project),
        },
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.capability_result is not None
    assert result.capability_result.status is CapabilityExecutionStatus.COMPLETED
    assert result.capability_result.selected_capability is not None
    assert result.capability_result.selected_capability.capability_id == PROJECT_TREE_CAPABILITY_ID
    selected_plan = result.capability_result.orchestration_result.selected_plan
    assert selected_plan is not None
    assert len(selected_plan.ordered_steps) == 1
    assert selected_plan.ordered_steps[0].tool == "project_tree"


def test_multi_step_request_uses_multi_capability_planner_automatically(
    tmp_path: Path,
    bootstrapped_atlas,
) -> None:
    project = _project(tmp_path)

    result = _capability_result(
        bootstrapped_atlas,
        {
            "required_inputs": ["path"],
            "required_outputs": ["entries"],
            "path": str(project),
        },
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.capability_result is not None
    assert result.capability_result.status is CapabilityExecutionStatus.COMPLETED
    assert result.capability_result.output == {"entries": ("README.md", "pkg", "zeta.py")}
    orchestration = result.capability_result.orchestration_result
    assert orchestration.planning_decision is None
    assert orchestration.selected_plan is not None
    assert tuple(step.subplan_ref.plan_id for step in orchestration.selected_plan.ordered_steps) == (
        PROJECT_TREE_REFERENCE_ID,
        DIRECTORY_LIST_REFERENCE_ID,
    )


def test_multi_step_execution_reuses_output_binding_variables(tmp_path: Path, bootstrapped_atlas) -> None:
    project = _project(tmp_path)

    result = _capability_result(
        bootstrapped_atlas,
        {
            "required_inputs": ["path"],
            "required_outputs": ["entries"],
            "path": str(project),
        },
    )

    assert result.capability_result is not None
    selected_plan = result.capability_result.orchestration_result.selected_plan
    assert selected_plan is not None
    first, second = selected_plan.ordered_steps
    assert first.output_binding is not None
    assert first.output_binding.variable_name == "capability_1_output"
    assert second.arguments["directory_path"] == ExecutionVariableReference(
        "capability_1_output",
        ("directory_path",),
    )


def test_single_step_candidate_does_not_use_composed_plan(tmp_path: Path, bootstrapped_atlas) -> None:
    project = _project(tmp_path)

    result = _capability_result(
        bootstrapped_atlas,
        {
            "required_inputs": ["directory_path"],
            "required_outputs": ["entries"],
            "directory_path": str(project),
        },
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.capability_result is not None
    orchestration = result.capability_result.orchestration_result
    assert orchestration.planning_decision is not None
    assert orchestration.selected_plan is not None
    assert len(orchestration.selected_plan.ordered_steps) == 1
    assert orchestration.selected_plan.ordered_steps[0].tool == "list_directory"


def test_multi_step_missing_initial_input_returns_structured_error(bootstrapped_atlas) -> None:
    result = _capability_result(
        bootstrapped_atlas,
        {
            "required_inputs": ["path"],
            "required_outputs": ["entries"],
        },
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.capability_result is not None
    assert result.capability_result.status is CapabilityExecutionStatus.EXECUTION_FAILED
    assert result.capability_result.error_code == "PARAMETER_RESOLUTION_FAILED"

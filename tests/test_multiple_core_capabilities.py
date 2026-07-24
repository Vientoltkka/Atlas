from __future__ import annotations

from pathlib import Path

import pytest

from bootstrap.bootstrap import Bootstrap
from bootstrap.execution_plan_library import build_core_execution_plan_library
from core.atlas_request_classifier import StructuredInput
from core.atlas_router import AtlasRoutingStatus
from core.capability_execution_service import CapabilityExecutionStatus
from core.capability_planner import CapabilityPlanningRequest, CapabilityPlanningStatus
from core.capability_resolver import CapabilityResolutionRequest, CapabilityType, WorkflowCapabilityProvider
from core.execution_plan_registry import ExecutionPlanReference


PROJECT_TREE_CAPABILITY_ID = "workflow.atlas.core.project.tree.show.1.0"
DIRECTORY_LIST_CAPABILITY_ID = "workflow.atlas.core.directory.list.1.0"
PROJECT_TREE_REFERENCE = ExecutionPlanReference("project.tree.show", "1.0")
DIRECTORY_LIST_REFERENCE = ExecutionPlanReference("directory.list", "1.0")


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


def test_core_library_contains_multiple_real_workflows() -> None:
    library = build_core_execution_plan_library()
    assert library is not None

    references = tuple(workflow.reference for workflow in library.workflows())

    assert references == (PROJECT_TREE_REFERENCE, DIRECTORY_LIST_REFERENCE)
    assert library.get(PROJECT_TREE_REFERENCE).plan.required_tools == ("project_tree",)
    assert library.get(DIRECTORY_LIST_REFERENCE).plan.required_tools == ("list_directory",)


def test_workflow_provider_discovers_both_core_capabilities() -> None:
    library = build_core_execution_plan_library()
    assert library is not None

    capabilities = WorkflowCapabilityProvider((library,)).list_capabilities()
    by_id = {capability.capability_id: capability for capability in capabilities}

    assert tuple(by_id) == (PROJECT_TREE_CAPABILITY_ID, DIRECTORY_LIST_CAPABILITY_ID)
    assert by_id[PROJECT_TREE_CAPABILITY_ID].input_names == ("path",)
    assert by_id[PROJECT_TREE_CAPABILITY_ID].output_names == ("path", "directory_path", "tree")
    assert by_id[DIRECTORY_LIST_CAPABILITY_ID].input_names == ("directory_path",)
    assert by_id[DIRECTORY_LIST_CAPABILITY_ID].output_names == ("directory_path", "entries")


def test_resolver_returns_different_candidates_by_filters(bootstrapped_atlas) -> None:
    planner = bootstrapped_atlas._capability_execution_service._capability_orchestrator._capability_planner  # type: ignore[attr-defined]
    resolver = planner._resolver

    by_id = resolver.resolve(
        CapabilityResolutionRequest(
            capability_types=(CapabilityType.WORKFLOW,),
            required_capability_ids=(DIRECTORY_LIST_CAPABILITY_ID,),
        )
    )
    by_category = resolver.resolve(CapabilityResolutionRequest(required_categories=("filesystem.directory",)))
    by_tag = resolver.resolve(CapabilityResolutionRequest(required_tags=("list_directory",)))
    by_input = resolver.resolve(CapabilityResolutionRequest(required_inputs=("directory_path",)))
    by_output = resolver.resolve(CapabilityResolutionRequest(desired_outputs=("entries",)))
    tree_by_output = resolver.resolve(CapabilityResolutionRequest(desired_outputs=("tree",)))

    assert by_id.candidates[0].capability.capability_id == DIRECTORY_LIST_CAPABILITY_ID
    assert by_category.candidates[0].capability.capability_id == DIRECTORY_LIST_CAPABILITY_ID
    assert by_tag.candidates[0].capability.capability_id == DIRECTORY_LIST_CAPABILITY_ID
    assert by_input.candidates[0].capability.capability_id == DIRECTORY_LIST_CAPABILITY_ID
    assert by_output.candidates[0].capability.capability_id == DIRECTORY_LIST_CAPABILITY_ID
    assert tree_by_output.candidates[0].capability.capability_id == PROJECT_TREE_CAPABILITY_ID


def test_planner_selects_each_workflow_deterministically(bootstrapped_atlas) -> None:
    planner = bootstrapped_atlas._capability_execution_service._capability_orchestrator._capability_planner  # type: ignore[attr-defined]

    tree = planner.plan(
        CapabilityPlanningRequest(
            "Show project tree",
            capability_id=PROJECT_TREE_CAPABILITY_ID,
            required_inputs=("path",),
            required_outputs=("tree",),
        )
    )
    directory = planner.plan(
        CapabilityPlanningRequest(
            "List directory",
            capability_id=DIRECTORY_LIST_CAPABILITY_ID,
            required_inputs=("directory_path",),
            required_outputs=("entries",),
        )
    )

    assert tree.status is CapabilityPlanningStatus.SELECTED
    assert tree.selected_workflow_reference == PROJECT_TREE_REFERENCE
    assert directory.status is CapabilityPlanningStatus.SELECTED
    assert directory.selected_workflow_reference == DIRECTORY_LIST_REFERENCE


def test_project_tree_e2e_still_executes_with_multiple_capabilities(
    tmp_path: Path,
    bootstrapped_atlas,
) -> None:
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
    assert result.capability_result.output == {
        "path": str(project),
        "directory_path": str(project),
        "tree": ("pkg\\alpha.py", "zeta.py"),
    }


def test_directory_list_e2e_executes_second_real_capability(
    tmp_path: Path,
    bootstrapped_atlas,
) -> None:
    project = _project(tmp_path)

    result = _capability_result(
        bootstrapped_atlas,
        {
            "capability_id": DIRECTORY_LIST_CAPABILITY_ID,
            "required_inputs": ["directory_path"],
            "required_outputs": ["entries"],
            "directory_path": str(project),
        },
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.capability_result is not None
    assert result.capability_result.status is CapabilityExecutionStatus.COMPLETED
    assert result.capability_result.selected_capability is not None
    assert result.capability_result.selected_capability.capability_id == DIRECTORY_LIST_CAPABILITY_ID
    assert result.capability_result.output == {
        "directory_path": str(project),
        "entries": ("README.md", "pkg", "zeta.py"),
    }
    execution = result.capability_result.orchestration_result.execution_result
    assert execution is not None
    assert execution.step_results[0].tool_name == "list_directory"


def test_missing_required_input_is_rejected_for_second_capability(bootstrapped_atlas) -> None:
    result = _capability_result(
        bootstrapped_atlas,
        {
            "capability_id": DIRECTORY_LIST_CAPABILITY_ID,
            "required_inputs": ["directory_path"],
            "required_outputs": ["entries"],
        },
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.capability_result is not None
    assert result.capability_result.error_code == "PARAMETER_RESOLUTION_FAILED"


def test_unknown_capability_id_returns_structured_no_candidate_result(bootstrapped_atlas) -> None:
    result = _capability_result(
        bootstrapped_atlas,
        {
            "capability_id": "workflow.atlas.core.missing.1.0",
            "required_inputs": ["directory_path"],
            "directory_path": ".",
        },
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.capability_result is not None
    assert result.capability_result.status is CapabilityExecutionStatus.NO_CAPABILITY_CANDIDATES


def test_ambiguous_request_stops_before_execution(tmp_path: Path, bootstrapped_atlas) -> None:
    project = _project(tmp_path)

    result = _capability_result(
        bootstrapped_atlas,
        {
            "path": str(project),
            "directory_path": str(project),
        },
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.capability_result is not None
    assert result.capability_result.status in {
        CapabilityExecutionStatus.CAPABILITY_AMBIGUOUS,
        CapabilityExecutionStatus.WORKFLOW_AMBIGUOUS,
    }
    assert result.capability_result.execution_status is None


def test_outputs_are_isolated_between_capability_executions(tmp_path: Path, bootstrapped_atlas) -> None:
    project = _project(tmp_path)

    tree = _capability_result(
        bootstrapped_atlas,
        {
            "capability_id": PROJECT_TREE_CAPABILITY_ID,
            "required_inputs": ["path"],
            "required_outputs": ["tree"],
            "path": str(project),
        },
    )
    directory = _capability_result(
        bootstrapped_atlas,
        {
            "capability_id": DIRECTORY_LIST_CAPABILITY_ID,
            "required_inputs": ["directory_path"],
            "required_outputs": ["entries"],
            "directory_path": str(project),
        },
    )

    assert tree.capability_result is not None
    assert directory.capability_result is not None
    assert "tree" in tree.capability_result.output
    assert "entries" not in tree.capability_result.output
    assert "entries" in directory.capability_result.output
    assert "tree" not in directory.capability_result.output

from __future__ import annotations

from pathlib import Path

import pytest

from bootstrap.bootstrap import Bootstrap
from bootstrap.execution_plan_library import build_core_execution_plan_library
from core.atlas_request_classifier import StructuredInput
from core.atlas_router import AtlasRoutingStatus
from core.capability_execution_service import CapabilityExecutionStatus
from core.capability_planner import CapabilityPlanningRequest, CapabilityPlanningStatus
from core.capability_resolver import (
    CapabilityResolutionRequest,
    CapabilityResolver,
    CapabilityType,
    WorkflowCapabilityProvider,
)
from core.execution_plan_registry import ExecutionPlanReference, ExecutionPlanRegistry


PROJECT_TREE_CAPABILITY_ID = "workflow.atlas.core.project.tree.show.1.0"
PROJECT_TREE_REFERENCE = ExecutionPlanReference("project.tree.show", "1.0")
DIRECTORY_LIST_REFERENCE = ExecutionPlanReference("directory.list", "1.0")


@pytest.fixture(scope="module")
def bootstrapped_atlas():
    return Bootstrap.build()


def test_core_library_contains_project_tree_workflow() -> None:
    library = build_core_execution_plan_library()

    assert library is not None
    workflow = library.get(PROJECT_TREE_REFERENCE)
    assert workflow.title == "Show project tree"
    assert workflow.category == "project.analysis"
    assert workflow.tags == ("project_tree", "filesystem", "read_only")
    assert workflow.plan.required_tools == ("project_tree",)
    assert workflow.plan.output is not None


def test_core_project_tree_workflow_installs_in_registry() -> None:
    library = build_core_execution_plan_library()
    registry = ExecutionPlanRegistry()

    assert library is not None
    install = library.install(registry)

    assert install.installed == (PROJECT_TREE_REFERENCE, DIRECTORY_LIST_REFERENCE)
    assert registry.contains(PROJECT_TREE_REFERENCE.plan_id, version=PROJECT_TREE_REFERENCE.version)


def test_workflow_provider_resolver_and_planner_select_project_tree(bootstrapped_atlas) -> None:
    orchestrator = bootstrapped_atlas
    service = orchestrator._capability_execution_service  # type: ignore[attr-defined]
    capability_orchestrator = service._capability_orchestrator  # type: ignore[union-attr]
    planner = capability_orchestrator._capability_planner
    resolver = planner._resolver
    registry = planner._registry

    library = build_core_execution_plan_library()
    assert library is not None
    provider_capability = WorkflowCapabilityProvider((library,)).list_capabilities()[0]
    resolution = resolver.resolve(
        CapabilityResolutionRequest(
            capability_types=(CapabilityType.WORKFLOW,),
            required_capability_ids=(PROJECT_TREE_CAPABILITY_ID,),
            required_inputs=("path",),
            desired_outputs=("tree",),
        )
    )
    decision = planner.plan(
        CapabilityPlanningRequest(
            "Show project tree",
            capability_id=PROJECT_TREE_CAPABILITY_ID,
            required_inputs=("path",),
            preferred_workflow_reference=PROJECT_TREE_REFERENCE,
        )
    )

    assert provider_capability.capability_id == PROJECT_TREE_CAPABILITY_ID
    assert provider_capability.input_names == ("path",)
    assert resolution.matched_capabilities == 1
    assert isinstance(resolver, CapabilityResolver)
    assert registry is not None
    assert registry.contains(PROJECT_TREE_REFERENCE.plan_id, version=PROJECT_TREE_REFERENCE.version)
    assert decision.status is CapabilityPlanningStatus.SELECTED
    assert decision.selected_workflow_reference == PROJECT_TREE_REFERENCE


def test_project_tree_e2e_from_structured_input_uses_real_bootstrap(
    tmp_path: Path,
    bootstrapped_atlas,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
    (tmp_path / "pkg" / "notes.txt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_alpha.py").write_text("def test_alpha(): pass\n", encoding="utf-8")
    orchestrator = bootstrapped_atlas

    result = orchestrator.route_structured_input(
        StructuredInput(
            route="capability",
            kind="project_tree",
            payload={
                "capability_id": PROJECT_TREE_CAPABILITY_ID,
                "preferred_workflow_reference": {
                    "plan_id": PROJECT_TREE_REFERENCE.plan_id,
                    "version": PROJECT_TREE_REFERENCE.version,
                },
                "required_inputs": ["path"],
                "required_outputs": ["tree"],
                "path": str(tmp_path),
            },
            request_id="project-tree-e2e",
        )
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.capability_result is not None
    assert result.capability_result.status is CapabilityExecutionStatus.COMPLETED
    assert result.capability_result.selected_capability is not None
    assert result.capability_result.selected_capability.capability_id == PROJECT_TREE_CAPABILITY_ID
    assert result.capability_result.output == {
        "path": str(tmp_path),
        "directory_path": str(tmp_path),
        "tree": ("pkg\\alpha.py", "tests\\test_alpha.py"),
    }
    execution = result.capability_result.orchestration_result.execution_result
    assert execution is not None
    assert execution.step_results[0].tool_name == "project_tree"


def test_project_tree_payload_without_path_fails_structurally(bootstrapped_atlas) -> None:
    orchestrator = bootstrapped_atlas

    result = orchestrator.route_structured_input(
        StructuredInput(
            route="capability",
            payload={
                "capability_id": PROJECT_TREE_CAPABILITY_ID,
                "preferred_workflow_reference": {
                    "plan_id": PROJECT_TREE_REFERENCE.plan_id,
                    "version": PROJECT_TREE_REFERENCE.version,
                },
                "required_inputs": ["path"],
            },
        )
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.capability_result is not None
    assert result.capability_result.error_code == "PARAMETER_RESOLUTION_FAILED"


def test_project_tree_payload_with_wrong_path_type_fails_structurally(bootstrapped_atlas) -> None:
    orchestrator = bootstrapped_atlas

    result = orchestrator.route_structured_input(
        StructuredInput(
            route="capability",
            payload={
                "capability_id": PROJECT_TREE_CAPABILITY_ID,
                "preferred_workflow_reference": {
                    "plan_id": PROJECT_TREE_REFERENCE.plan_id,
                    "version": PROJECT_TREE_REFERENCE.version,
                },
                "required_inputs": ["path"],
                "path": 123,
            },
        )
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.capability_result is not None
    assert result.capability_result.error_code == "TOOL_SCHEMA_VALIDATION_FAILED"


def test_project_tree_nonexistent_path_returns_structured_error(
    tmp_path: Path,
    bootstrapped_atlas,
) -> None:
    orchestrator = bootstrapped_atlas
    missing = tmp_path / "missing"

    result = orchestrator.route_structured_input(
        StructuredInput(
            route="capability",
            payload={
                "capability_id": PROJECT_TREE_CAPABILITY_ID,
                "preferred_workflow_reference": {
                    "plan_id": PROJECT_TREE_REFERENCE.plan_id,
                    "version": PROJECT_TREE_REFERENCE.version,
                },
                "required_inputs": ["path"],
                "path": str(missing),
            },
        )
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.capability_result is not None
    assert result.capability_result.error_code == "TOOL_EXCEPTION"


def test_process_prompt_is_not_modified_by_project_tree_capability() -> None:
    source = Path("core/orchestrator.py").read_text(encoding="utf-8")

    assert "route_structured_input(" not in source[source.index("    def process_prompt("):source.index("    def execute_capability(")]


def test_project_tree_capability_uses_no_llm_or_fake_tools() -> None:
    library = build_core_execution_plan_library()
    assert library is not None
    workflow = library.get(PROJECT_TREE_REFERENCE)
    source = Path("bootstrap/execution_plan_library.py").read_text(encoding="utf-8")

    assert workflow.plan.required_tools == ("project_tree",)
    assert "PromptClient" not in source
    assert "fake" not in source.casefold()
    assert "eval(" not in source
    assert "exec(" not in source

from __future__ import annotations

from pathlib import Path

import pytest

from bootstrap.capability_resolver import build_core_capability_resolver
from bootstrap.execution_plan_library import build_core_execution_plan_library
from core.capability_resolver import (
    CapabilityProviderError,
    CapabilityResolutionRequest,
    CapabilityResolver,
    CapabilityType,
    ConflictingCapabilityDefinitionError,
    WorkflowCapabilityProvider,
)
from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
from core.execution_plan_registry import ExecutionPlanReference
from core.execution_variable_reference import ExecutionVariableReference
from core.planner import ExecutionPlan, ExecutionStep


def _plan(tool: str = "project_tree", *, input_name: str = "path") -> ExecutionPlan:
    return ExecutionPlan(
        goal=f"Run {tool}.",
        ordered_steps=(
            ExecutionStep(
                f"run_{tool}",
                f"Run {tool}.",
                tool,
                arguments={"path": ExecutionVariableReference(input_name)},
            ),
        ),
        estimated_steps=1,
        required_tools=(tool,),
        detected_risks=(),
        requires_confirmation=False,
    )


def _workflow(
    plan_id: str,
    *,
    title: str | None = None,
    version: str = "1.0",
    tags: tuple[str, ...] = ("filesystem",),
) -> WorkflowDefinition:
    return WorkflowDefinition(
        reference=ExecutionPlanReference(plan_id, version),
        plan=_plan(),
        title=title or plan_id.replace(".", " ").title(),
        description="Read-only workflow discovered from an execution-plan library.",
        category="project.analysis",
        tags=tags,
    )


def _capability_ids(provider: WorkflowCapabilityProvider) -> tuple[str, ...]:
    return tuple(capability.capability_id for capability in provider.list_capabilities())


def test_discovers_multiple_capabilities_from_execution_plan_library() -> None:
    library = ExecutionPlanLibrary(
        "atlas.core",
        (_workflow("project.tree.show"), _workflow("project.tree.python")),
        version="1.0",
    )

    capabilities = WorkflowCapabilityProvider((library,)).list_capabilities()

    assert _capability_ids(WorkflowCapabilityProvider((library,))) == (
        "workflow.atlas.core.project.tree.show.1.0",
        "workflow.atlas.core.project.tree.python.1.0",
    )
    assert all(capability.capability_type is CapabilityType.WORKFLOW for capability in capabilities)
    assert all(capability.input_names == ("path",) for capability in capabilities)


def test_added_workflows_are_discovered_without_resolver_changes() -> None:
    base = ExecutionPlanLibrary("atlas.core", (_workflow("project.tree.show"),), version="1.0")
    expanded = ExecutionPlanLibrary(
        "atlas.core",
        (_workflow("project.tree.show"), _workflow("project.tree.python")),
        version="1.0",
    )

    assert _capability_ids(WorkflowCapabilityProvider((base,))) == (
        "workflow.atlas.core.project.tree.show.1.0",
    )
    assert _capability_ids(WorkflowCapabilityProvider((expanded,))) == (
        "workflow.atlas.core.project.tree.show.1.0",
        "workflow.atlas.core.project.tree.python.1.0",
    )


def test_removed_workflows_disappear_from_discovery() -> None:
    expanded = ExecutionPlanLibrary(
        "atlas.core",
        (_workflow("project.tree.show"), _workflow("project.tree.python")),
        version="1.0",
    )
    reduced = ExecutionPlanLibrary("atlas.core", (_workflow("project.tree.python"),), version="1.0")

    assert "workflow.atlas.core.project.tree.show.1.0" in _capability_ids(
        WorkflowCapabilityProvider((expanded,))
    )
    assert "workflow.atlas.core.project.tree.show.1.0" not in _capability_ids(
        WorkflowCapabilityProvider((reduced,))
    )


def test_duplicate_capability_ids_are_rejected_by_resolver() -> None:
    first = ExecutionPlanLibrary("atlas.core", (_workflow("project.tree.show", title="First"),), version="1.0")
    second = ExecutionPlanLibrary("atlas.core", (_workflow("project.tree.show", title="Second"),), version="1.0")
    resolver = CapabilityResolver((WorkflowCapabilityProvider((first, second)),))

    with pytest.raises(ConflictingCapabilityDefinitionError):
        resolver.resolve(CapabilityResolutionRequest())


def test_versions_are_discovered_distinctly_and_invalid_versions_fail() -> None:
    versioned = ExecutionPlanLibrary(
        "atlas.core",
        (_workflow("project.tree.show", version="1.0"), _workflow("project.tree.show", version="2.0")),
        version="1.0",
    )

    assert _capability_ids(WorkflowCapabilityProvider((versioned,))) == (
        "workflow.atlas.core.project.tree.show.1.0",
        "workflow.atlas.core.project.tree.show.2.0",
    )
    with pytest.raises(Exception):
        _workflow("project.tree.bad", version="latest/range")


def test_invalid_capabilities_from_library_fail_structurally() -> None:
    class InvalidWorkflowLibrary(ExecutionPlanLibrary):
        def workflows(self):  # type: ignore[override]
            return ("bad",)

    library = InvalidWorkflowLibrary("atlas.invalid", (), allow_empty=True)

    with pytest.raises(CapabilityProviderError):
        WorkflowCapabilityProvider((library,)).list_capabilities()
    with pytest.raises(CapabilityProviderError):
        WorkflowCapabilityProvider((object(),))  # type: ignore[arg-type]


def test_empty_execution_plan_library_discovers_no_capabilities() -> None:
    library = ExecutionPlanLibrary("atlas.empty", (), allow_empty=True)

    assert WorkflowCapabilityProvider((library,)).list_capabilities() == ()
    assert CapabilityResolver((WorkflowCapabilityProvider((library,)),)).resolve(
        CapabilityResolutionRequest()
    ).matched_capabilities == 0


def test_resolver_uses_library_as_capability_source_when_workflows_exist() -> None:
    library = build_core_execution_plan_library()
    assert library is not None

    resolver = build_core_capability_resolver(
        tool_registry=object(),  # type: ignore[arg-type]
        execution_plan_libraries=(library,),
    )
    result = resolver.resolve(CapabilityResolutionRequest(enabled_only=False))

    assert result.matched_capabilities == len(library.workflows())
    assert all(candidate.capability.capability_type is CapabilityType.WORKFLOW for candidate in result.candidates)


def test_capability_resolver_has_no_hardcoded_workflow_capability_ids() -> None:
    source = Path("core/capability_resolver.py").read_text(encoding="utf-8")

    assert "project.tree.show" not in source
    assert "workflow.atlas.core" not in source
    assert "eval(" not in source
    assert "exec(" not in source
    assert "importlib" not in source

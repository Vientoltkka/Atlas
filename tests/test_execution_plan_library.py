from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from bootstrap.execution_plan_library import build_core_execution_plan_library
from core.execution_plan_library import (
    DuplicateWorkflowDefinitionError,
    ExecutionPlanLibrary,
    ExecutionPlanLibraryConflictError,
    ExecutionPlanLibraryError,
    ExecutionPlanLibraryInstallResult,
    ExecutionPlanLibraryNotInstalledError,
    ExecutionPlanLibraryTooLargeError,
    InvalidExecutionPlanLibraryError,
    InvalidExecutionPlanLibraryIdError,
    InvalidExecutionPlanLibraryVersionError,
    InvalidWorkflowCategoryError,
    InvalidWorkflowDescriptionError,
    InvalidWorkflowDefinitionError,
    InvalidWorkflowTagError,
    InvalidWorkflowTitleError,
    MAX_LIBRARY_WORKFLOWS,
    MAX_WORKFLOW_DESCRIPTION_LENGTH,
    MAX_WORKFLOW_TAGS,
    MAX_WORKFLOW_TITLE_LENGTH,
    WorkflowDefinition,
)
from core.execution_plan_registry import ExecutionPlanReference, ExecutionPlanRegistry
from core.execution_plan_validator import ExecutionPlanValidator, plan_signature
from core.planner import ExecutionPlan, ExecutionStep


def _step(step_id: str, tool: str = "direct_response") -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        description=f"Run {step_id}.",
        tool=tool,
    )


def _plan(step_id: str = "step_1") -> ExecutionPlan:
    return ExecutionPlan(
        goal=f"Run {step_id}.",
        ordered_steps=(_step(step_id),),
        estimated_steps=1,
        required_tools=(),
        detected_risks=(),
        requires_confirmation=False,
    )


def _workflow(
    plan_id: str = "project.inspect",
    *,
    version: str | None = "1.0",
    plan: ExecutionPlan | None = None,
    title: str = "Inspect project",
    description: str = "Read and summarize a local project structure.",
    category: str = "project.analysis",
    tags: tuple[str, ...] = ("inspection", "filesystem"),
    enabled: bool = True,
) -> WorkflowDefinition:
    return WorkflowDefinition(
        reference=ExecutionPlanReference(plan_id, version),
        plan=plan or _plan(plan_id.replace(".", "_")),
        title=title,
        description=description,
        category=category,
        tags=tags,
        enabled=enabled,
    )


def test_workflow_definition_validates_and_is_immutable() -> None:
    workflow = _workflow(
        title="  Inspect Python project  ",
        description="Line one.\nLine two.",
        category="PROJECT.Analysis",
        tags=("Inspection", "filesystem"),
    )

    assert workflow.title == "Inspect Python project"
    assert workflow.description == "Line one.\nLine two."
    assert workflow.category == "project.analysis"
    assert workflow.tags == ("inspection", "filesystem")
    assert workflow.enabled is True
    with pytest.raises(FrozenInstanceError):
        workflow.title = "Other"  # type: ignore[misc]


def test_workflow_definition_rejects_invalid_reference_and_plan() -> None:
    with pytest.raises(InvalidWorkflowDefinitionError):
        WorkflowDefinition(  # type: ignore[arg-type]
            reference="project.inspect",
            plan=_plan(),
            title="Inspect",
            description="Description.",
            category="project",
        )
    with pytest.raises(InvalidWorkflowDefinitionError):
        WorkflowDefinition(  # type: ignore[arg-type]
            reference=ExecutionPlanReference("project.inspect"),
            plan=object(),
            title="Inspect",
            description="Description.",
            category="project",
        )
    with pytest.raises(InvalidWorkflowDefinitionError):
        _workflow(plan=replace(_plan(), estimated_steps=99))


@pytest.mark.parametrize("title", ["", "   ", "Bad\x01title"])
def test_workflow_title_rejects_empty_whitespace_and_control(title: str) -> None:
    with pytest.raises(InvalidWorkflowTitleError):
        _workflow(title=title)


def test_workflow_title_limit_is_enforced() -> None:
    assert _workflow(title="A" * MAX_WORKFLOW_TITLE_LENGTH).title == (
        "A" * MAX_WORKFLOW_TITLE_LENGTH
    )
    with pytest.raises(InvalidWorkflowTitleError):
        _workflow(title="A" * (MAX_WORKFLOW_TITLE_LENGTH + 1))


def test_workflow_description_validation() -> None:
    assert _workflow(description="Line one.\nLine two.").description == (
        "Line one.\nLine two."
    )
    for description in ("", "   ", "Bad\x01description"):
        with pytest.raises(InvalidWorkflowDescriptionError):
            _workflow(description=description)
    with pytest.raises(InvalidWorkflowDescriptionError):
        _workflow(description="A" * (MAX_WORKFLOW_DESCRIPTION_LENGTH + 1))


@pytest.mark.parametrize("category", ["", "project analysis", "project/analysis", "../project"])
def test_workflow_category_rejects_invalid_identifiers(category: str) -> None:
    with pytest.raises(InvalidWorkflowCategoryError):
        _workflow(category=category)


def test_workflow_tags_validate_normalize_preserve_order_and_reject_duplicates() -> None:
    workflow = _workflow(tags=("First", "second-tag", "third.tag"))
    assert workflow.tags == ("first", "second-tag", "third.tag")

    for tags in (("",), ("first", "first"), ("bad tag",), ("bad/tag",)):
        with pytest.raises(InvalidWorkflowTagError):
            _workflow(tags=tags)
    with pytest.raises(InvalidWorkflowTagError):
        _workflow(tags=tuple(f"tag-{index}" for index in range(MAX_WORKFLOW_TAGS + 1)))
    with pytest.raises(InvalidWorkflowTagError):
        _workflow(tags=["not", "tuple"])  # type: ignore[arg-type]


def test_enabled_false_remains_catalog_visible() -> None:
    workflow = _workflow(enabled=False)
    library = ExecutionPlanLibrary("atlas.core", (workflow,), version="1.0")

    assert library.workflows() == (workflow,)
    assert library.enabled_workflows() == ()
    assert library.disabled_workflows() == (workflow,)


def test_library_validates_identity_version_size_duplicates_and_tuple_copy() -> None:
    original = [_workflow("project.inspect")]
    library = ExecutionPlanLibrary(
        "atlas.core",
        original,
        version="1.0",
        title=" Core workflows ",
        description=" Reusable workflows. ",
    )
    original.append(_workflow("project.other"))

    assert library.library_id == "atlas.core"
    assert library.version == "1.0"
    assert library.title == "Core workflows"
    assert library.description == "Reusable workflows."
    assert library.workflows() == (_workflow("project.inspect"),)

    with pytest.raises(InvalidExecutionPlanLibraryIdError):
        ExecutionPlanLibrary("atlas/core", (_workflow(),))
    with pytest.raises(InvalidExecutionPlanLibraryVersionError):
        ExecutionPlanLibrary("atlas.core", (_workflow(),), version="latest/range")
    with pytest.raises(InvalidExecutionPlanLibraryVersionError):
        ExecutionPlanLibrary("atlas.core", (_workflow(),), version="latest")
    with pytest.raises(InvalidExecutionPlanLibraryError):
        ExecutionPlanLibrary("atlas.core", ())
    with pytest.raises(DuplicateWorkflowDefinitionError):
        ExecutionPlanLibrary("atlas.core", (_workflow(), _workflow()))
    with pytest.raises(ExecutionPlanLibraryTooLargeError):
        ExecutionPlanLibrary(
            "atlas.core",
            tuple(_workflow(f"project.workflow_{index}") for index in range(MAX_LIBRARY_WORKFLOWS + 1)),
        )


def test_library_allows_empty_only_when_explicit() -> None:
    library = ExecutionPlanLibrary("atlas.empty", (), allow_empty=True)

    assert library.workflows() == ()


def test_library_queries_are_exact_deterministic_and_pure() -> None:
    first = _workflow(
        "project.inspect",
        category="project.analysis",
        tags=("inspection", "filesystem"),
    )
    second = _workflow(
        "project.test",
        category="coding",
        tags=("tests", "verification", "filesystem"),
    )
    third = _workflow(
        "project.disabled",
        category="coding",
        tags=("tests",),
        enabled=False,
    )
    library = ExecutionPlanLibrary("atlas.core", (first, second, third), version="1.0")

    assert library.contains(first.reference) is True
    assert library.contains(ExecutionPlanReference("missing")) is False
    assert library.get(second.reference) is second
    with pytest.raises(ExecutionPlanLibraryError):
        library.get(ExecutionPlanReference("missing"))
    assert library.find_by_category("coding") == (second, third)
    assert library.find_by_tag("filesystem") == (first, second)
    assert library.search(category="coding") == (second, third)
    assert library.search(tags=("tests", "verification")) == (second,)
    assert library.search(enabled=False) == (third,)
    assert library.search(category="coding", tags=("tests",), enabled=True) == (second,)
    assert library.workflows() == (first, second, third)


def test_install_registers_only_enabled_and_returns_structured_result() -> None:
    enabled = _workflow("project.inspect")
    disabled = _workflow("project.disabled", enabled=False)
    library = ExecutionPlanLibrary("atlas.core", (enabled, disabled), version="1.0")
    registry = ExecutionPlanRegistry()

    result = library.install(registry)

    assert isinstance(result, ExecutionPlanLibraryInstallResult)
    assert result.library_id == "atlas.core"
    assert result.library_version == "1.0"
    assert result.installed == (enabled.reference,)
    assert result.replaced == ()
    assert result.skipped_disabled == (disabled.reference,)
    assert result.atomic is True
    assert registry.resolve(enabled.reference) is enabled.plan
    assert registry.contains(disabled.reference.plan_id, version=disabled.reference.version) is False


def test_install_detects_collision_and_atomic_true_leaves_registry_unchanged() -> None:
    existing = _plan("existing")
    workflow = _workflow("project.inspect", plan=_plan("new"))
    registry = ExecutionPlanRegistry()
    registry.register(
        workflow.reference.plan_id,
        existing,
        version=workflow.reference.version,
    )
    library = ExecutionPlanLibrary("atlas.core", (workflow,), version="1.0")

    with pytest.raises(ExecutionPlanLibraryConflictError):
        library.install(registry, replace=False, atomic=True)

    assert registry.resolve(workflow.reference) is existing


def test_install_replace_true_replaces_and_reports_replaced() -> None:
    original = _plan("original")
    workflow = _workflow("project.inspect", plan=_plan("replacement"))
    registry = ExecutionPlanRegistry()
    registry.register(workflow.reference.plan_id, original, version=workflow.reference.version)
    library = ExecutionPlanLibrary("atlas.core", (workflow,), version="1.0")

    result = library.install(registry, replace=True)

    assert result.installed == ()
    assert result.replaced == (workflow.reference,)
    assert registry.resolve(workflow.reference) is workflow.plan


def test_atomic_true_rolls_back_partial_install_and_restores_original(monkeypatch) -> None:
    original = _plan("original")
    first = _workflow("project.first")
    second = _workflow("project.second")
    third = _workflow("project.third", plan=_plan("third_new"))
    registry = ExecutionPlanRegistry()
    registry.register(third.reference.plan_id, original, version=third.reference.version)
    library = ExecutionPlanLibrary("atlas.core", (first, second, third), version="1.0")
    real_register_entry = registry.register_entry

    def fail_on_second(entry, *, replace=False):
        if entry.reference == second.reference:
            raise RuntimeError("boom")
        return real_register_entry(entry, replace=replace)

    monkeypatch.setattr(registry, "register_entry", fail_on_second)

    with pytest.raises(Exception) as error:
        library.install(registry, replace=True, atomic=True)

    assert getattr(error.value, "rollback_performed", None) is True
    assert registry.contains(first.reference.plan_id, version=first.reference.version) is False
    assert registry.resolve(third.reference) is original


def test_atomic_false_reports_partial_install_without_rollback(monkeypatch) -> None:
    first = _workflow("project.first")
    second = _workflow("project.second")
    registry = ExecutionPlanRegistry()
    library = ExecutionPlanLibrary("atlas.core", (first, second), version="1.0")
    real_register_entry = registry.register_entry

    def fail_on_second(entry, *, replace=False):
        if entry.reference == second.reference:
            raise RuntimeError("boom")
        return real_register_entry(entry, replace=replace)

    monkeypatch.setattr(registry, "register_entry", fail_on_second)

    result = library.install(registry, atomic=False)

    assert result.installed == (first.reference,)
    assert result.failed_reference == second.reference
    assert result.rollback_performed is False
    assert registry.resolve(first.reference) is first.plan
    assert registry.contains(second.reference.plan_id, version=second.reference.version) is False


def test_uninstall_removes_matching_enabled_workflows_and_skips_disabled() -> None:
    enabled = _workflow("project.inspect")
    disabled = _workflow("project.disabled", enabled=False)
    library = ExecutionPlanLibrary("atlas.core", (enabled, disabled), version="1.0")
    registry = ExecutionPlanRegistry()
    library.install(registry)

    result = library.uninstall(registry)

    assert result.removed == (enabled.reference,)
    assert result.missing == ()
    assert result.conflicted == ()
    assert result.skipped_disabled == (disabled.reference,)
    assert registry.contains(enabled.reference.plan_id, version=enabled.reference.version) is False


def test_uninstall_reports_missing_and_strict_missing_fails() -> None:
    workflow = _workflow("project.inspect")
    library = ExecutionPlanLibrary("atlas.core", (workflow,), version="1.0")
    registry = ExecutionPlanRegistry()

    assert library.uninstall(registry).missing == (workflow.reference,)
    with pytest.raises(ExecutionPlanLibraryNotInstalledError):
        library.uninstall(registry, strict=True)


def test_uninstall_detects_signature_conflict_and_does_not_remove_external_plan() -> None:
    workflow = _workflow("project.inspect", plan=_plan("original"))
    changed = _plan("changed")
    library = ExecutionPlanLibrary("atlas.core", (workflow,), version="1.0")
    registry = ExecutionPlanRegistry()
    library.install(registry)
    registry.register(workflow.reference.plan_id, changed, version=workflow.reference.version, replace=True)

    result = library.uninstall(registry)

    assert result.conflicted == (workflow.reference,)
    assert registry.resolve(workflow.reference) is changed
    with pytest.raises(ExecutionPlanLibraryConflictError):
        library.uninstall(registry, strict=True)


def test_library_has_no_mutable_installation_state_across_registries() -> None:
    workflow = _workflow("project.inspect")
    library = ExecutionPlanLibrary("atlas.core", (workflow,), version="1.0")
    first = ExecutionPlanRegistry()
    second = ExecutionPlanRegistry()

    library.install(first)
    library.install(second)
    library.uninstall(first)

    assert first.contains(workflow.reference.plan_id, version=workflow.reference.version) is False
    assert second.resolve(workflow.reference) is workflow.plan


def test_library_signature_is_deterministic_and_changes_for_catalog_inputs() -> None:
    base_plan = _plan("base")
    workflow = _workflow("project.inspect", plan=base_plan)
    base = ExecutionPlanLibrary("atlas.core", (workflow,), version="1.0")
    same = ExecutionPlanLibrary("atlas.core", (_workflow("project.inspect", plan=base_plan),), version="1.0")

    assert base.library_signature() == same.library_signature()
    assert base.library_signature() != ExecutionPlanLibrary(
        "atlas.core",
        (workflow, _workflow("project.other")),
        version="1.0",
    ).library_signature()
    assert base.library_signature() != ExecutionPlanLibrary(
        "atlas.core",
        (_workflow("project.other", plan=base_plan),),
        version="1.0",
    ).library_signature()
    assert base.library_signature() != ExecutionPlanLibrary(
        "atlas.core",
        (_workflow("project.inspect", plan=base_plan, title="Other title"),),
        version="1.0",
    ).library_signature()
    assert base.library_signature() != ExecutionPlanLibrary(
        "atlas.core",
        (_workflow("project.inspect", plan=base_plan, category="coding"),),
        version="1.0",
    ).library_signature()
    assert base.library_signature() != ExecutionPlanLibrary(
        "atlas.core",
        (_workflow("project.inspect", plan=base_plan, tags=("other",)),),
        version="1.0",
    ).library_signature()
    assert base.library_signature() != ExecutionPlanLibrary(
        "atlas.core",
        (_workflow("project.inspect", plan=base_plan, enabled=False),),
        version="1.0",
    ).library_signature()
    assert base.library_signature() != ExecutionPlanLibrary(
        "atlas.core",
        (_workflow("project.inspect", plan=_plan("changed")),),
        version="1.0",
    ).library_signature()


def test_registry_subplans_outputs_validator_executor_and_bootstrap_remain_compatible() -> None:
    child = _plan("child")
    registry = ExecutionPlanRegistry()
    registry.register("project.child", child)
    parent = ExecutionPlan(
        goal="Run parent.",
        ordered_steps=(
            ExecutionStep(
                id="run_child",
                description="Run child.",
                tool=None,
                subplan_ref=ExecutionPlanReference("project.child"),
            ),
        ),
        estimated_steps=1,
        required_tools=(),
        detected_risks=(),
        requires_confirmation=False,
    )

    assert ExecutionPlanValidator(plan_registry=registry).validate(parent).is_valid is True
    assert plan_signature(parent)
    assert registry.resolve(ExecutionPlanReference("project.child")) is child
    library = build_core_execution_plan_library()
    assert library is not None
    assert library.contains(ExecutionPlanReference("project.tree.show", "1.0"))


def test_library_module_does_not_use_forbidden_runtime_features() -> None:
    source = Path("core/execution_plan_library.py").read_text(encoding="utf-8")

    for forbidden in ("eval(", "exec(", "__import__", "importlib", "pickle", "requests", "urllib"):
        assert forbidden not in source

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bootstrap.execution_plan_library import build_workflow_discovery_service
from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
from core.execution_plan_registry import ExecutionPlanReference, ExecutionPlanRegistry
from core.execution_plan_validator import ExecutionPlanValidator
from core.planner import ExecutionPlan, ExecutionStep
from core.workflow_discovery import (
    CATEGORY_MATCH_SCORE,
    ENABLED_BONUS_SCORE,
    EXACT_REFERENCE_SCORE,
    PREFERRED_TAG_MATCH_SCORE,
    REQUIRED_TAG_MATCH_SCORE,
    TITLE_TERM_MATCH_SCORE,
    ConflictingWorkflowCandidateError,
    InvalidWorkflowDiscoveryCategoryError,
    InvalidWorkflowDiscoveryLibraryIdError,
    InvalidWorkflowDiscoveryReferenceError,
    InvalidWorkflowDiscoveryRequestError,
    InvalidWorkflowDiscoveryTagError,
    InvalidWorkflowDiscoveryTitleTermError,
    MAX_DISCOVERY_LIMIT,
    MAX_DISCOVERY_TITLE_TERM_LENGTH,
    WorkflowDiscoveryLimitError,
    WorkflowDiscoveryRejectionCode,
    WorkflowDiscoveryRequest,
    WorkflowDiscoveryService,
    WorkflowMatchReasonCode,
    workflow_discovery_request_signature,
)


def _step(step_id: str, tool: str = "direct_response") -> ExecutionStep:
    return ExecutionStep(step_id, f"Run {step_id}.", tool)


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
    plan_id: str,
    *,
    version: str | None = "1.0",
    title: str = "Inspect Python project",
    description: str = "Description that should not be searched.",
    category: str = "project",
    tags: tuple[str, ...] = ("inspection",),
    enabled: bool = True,
    plan: ExecutionPlan | None = None,
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


def _library(
    library_id: str,
    workflows: tuple[WorkflowDefinition, ...],
    *,
    version: str | None = "1.0",
    allow_empty: bool = False,
) -> ExecutionPlanLibrary:
    return ExecutionPlanLibrary(library_id, workflows, version=version, allow_empty=allow_empty)


def test_request_empty_valid_immutable_and_normalizes_iterables_to_tuples() -> None:
    request = WorkflowDiscoveryRequest(
        categories=["Project"],
        required_tags=["Inspection"],
        preferred_tags=["Python"],
        excluded_tags=["Remote"],
        references=[ExecutionPlanReference("project.inspect")],
        library_ids=["atlas.core"],
        title_terms=["  PYTHON   Project  "],
    )

    assert request.categories == ("project",)
    assert request.required_tags == ("inspection",)
    assert request.preferred_tags == ("python",)
    assert request.excluded_tags == ("remote",)
    assert request.references == (ExecutionPlanReference("project.inspect"),)
    assert request.library_ids == ("atlas.core",)
    assert request.title_terms == ("python project",)
    assert WorkflowDiscoveryRequest().enabled_only is True
    with pytest.raises(FrozenInstanceError):
        request.limit = 3  # type: ignore[misc]


def test_request_rejects_invalid_categories_and_duplicates() -> None:
    for categories in ([""], ["bad category"], ["project/project"], ["project", "project"]):
        with pytest.raises(InvalidWorkflowDiscoveryCategoryError):
            WorkflowDiscoveryRequest(categories=categories)


def test_request_validates_tag_groups_and_cross_group_duplicates() -> None:
    assert WorkflowDiscoveryRequest(
        required_tags=("tests",),
        preferred_tags=("verification",),
        excluded_tags=("remote",),
    )
    for kwargs in (
        {"required_tags": ("",)},
        {"preferred_tags": ("bad tag",)},
        {"excluded_tags": ("remote/unsafe",)},
        {"required_tags": ("tests", "tests")},
        {"required_tags": ("tests",), "preferred_tags": ("tests",)},
        {"preferred_tags": ("remote",), "excluded_tags": ("remote",)},
    ):
        with pytest.raises(InvalidWorkflowDiscoveryTagError):
            WorkflowDiscoveryRequest(**kwargs)


def test_request_validates_references_library_ids_title_terms_score_and_limit() -> None:
    with pytest.raises(InvalidWorkflowDiscoveryReferenceError):
        WorkflowDiscoveryRequest(references=("project.inspect",))  # type: ignore[arg-type]
    with pytest.raises(InvalidWorkflowDiscoveryReferenceError):
        WorkflowDiscoveryRequest(
            references=(
                ExecutionPlanReference("project.inspect"),
                ExecutionPlanReference("project.inspect"),
            )
        )
    with pytest.raises(InvalidWorkflowDiscoveryLibraryIdError):
        WorkflowDiscoveryRequest(library_ids=("atlas/core",))
    with pytest.raises(InvalidWorkflowDiscoveryTitleTermError):
        WorkflowDiscoveryRequest(title_terms=("",))
    with pytest.raises(InvalidWorkflowDiscoveryTitleTermError):
        WorkflowDiscoveryRequest(title_terms=("a" * (MAX_DISCOVERY_TITLE_TERM_LENGTH + 1),))
    for minimum_score in (-1, True):
        with pytest.raises(WorkflowDiscoveryLimitError):
            WorkflowDiscoveryRequest(minimum_score=minimum_score)  # type: ignore[arg-type]
    for limit in (0, -1, MAX_DISCOVERY_LIMIT + 1, True):
        with pytest.raises(WorkflowDiscoveryLimitError):
            WorkflowDiscoveryRequest(limit=limit)  # type: ignore[arg-type]
    assert WorkflowDiscoveryRequest(limit=MAX_DISCOVERY_LIMIT).limit == MAX_DISCOVERY_LIMIT


def test_service_accepts_zero_libraries_and_empty_library() -> None:
    service = WorkflowDiscoveryService()
    empty = service.discover(WorkflowDiscoveryRequest(), ())

    assert empty.candidates == ()
    assert empty.rejections == ()
    assert empty.scanned_libraries == 0
    assert empty.scanned_workflows == 0
    assert empty.has_matches is False
    assert empty.best_score is None
    assert empty.top_candidates == ()
    assert empty.is_ambiguous is False

    library = _library("atlas.empty", (), allow_empty=True)
    result = service.discover(WorkflowDiscoveryRequest(), (library,))
    assert result.scanned_libraries == 1
    assert result.scanned_workflows == 0


def test_discovery_enabled_policy_reference_library_category_and_tag_filters() -> None:
    enabled = _workflow("project.inspect", category="project", tags=("inspection", "filesystem"))
    disabled = _workflow("project.disabled", category="project", tags=("inspection",), enabled=False)
    coding = _workflow("project.tests", category="coding", tags=("tests", "verification"))
    library = _library("atlas.core", (enabled, disabled, coding))
    service = WorkflowDiscoveryService()

    assert service.discover(WorkflowDiscoveryRequest(), (library,)).candidates == (service.discover(WorkflowDiscoveryRequest(), (library,)).candidates)
    assert [c.workflow.reference for c in service.discover(WorkflowDiscoveryRequest(), (library,)).candidates] == [
        coding.reference,
        enabled.reference,
    ]
    assert [c.workflow.reference for c in service.discover(WorkflowDiscoveryRequest(enabled_only=False), (library,)).candidates] == [
        coding.reference,
        enabled.reference,
        disabled.reference,
    ]
    assert service.discover(
        WorkflowDiscoveryRequest(references=(coding.reference,)),
        (library,),
    ).candidates[0].workflow.reference == coding.reference
    assert service.discover(
        WorkflowDiscoveryRequest(library_ids=("other.library",)),
        (library,),
    ).candidates == ()
    assert {c.workflow.reference for c in service.discover(
        WorkflowDiscoveryRequest(categories=("project", "coding")),
        (library,),
    ).candidates} == {enabled.reference, coding.reference}
    assert service.discover(
        WorkflowDiscoveryRequest(required_tags=("tests", "verification")),
        (library,),
    ).candidates[0].workflow.reference == coding.reference
    assert service.discover(
        WorkflowDiscoveryRequest(excluded_tags=("filesystem",)),
        (library,),
    ).candidates[0].workflow.reference == coding.reference


def test_scoring_weights_reasons_and_filter_before_score() -> None:
    workflow = _workflow(
        "project.tests",
        title="Run Python tests",
        category="coding",
        tags=("tests", "verification", "python"),
    )
    request = WorkflowDiscoveryRequest(
        references=(workflow.reference,),
        categories=("coding",),
        required_tags=("tests",),
        preferred_tags=("verification", "python"),
        title_terms=("python",),
    )
    candidate = WorkflowDiscoveryService().discover(request, (_library("atlas.core", (workflow,)),)).candidates[0]

    expected = (
        EXACT_REFERENCE_SCORE
        + CATEGORY_MATCH_SCORE
        + REQUIRED_TAG_MATCH_SCORE
        + (2 * PREFERRED_TAG_MATCH_SCORE)
        + TITLE_TERM_MATCH_SCORE
        + ENABLED_BONUS_SCORE
    )
    assert candidate.score == expected
    assert sum(reason.score for reason in candidate.reasons) == candidate.score
    assert [reason.code for reason in candidate.reasons] == [
        WorkflowMatchReasonCode.EXACT_REFERENCE,
        WorkflowMatchReasonCode.CATEGORY_MATCH,
        WorkflowMatchReasonCode.REQUIRED_TAG_MATCH,
        WorkflowMatchReasonCode.PREFERRED_TAG_MATCH,
        WorkflowMatchReasonCode.PREFERRED_TAG_MATCH,
        WorkflowMatchReasonCode.TITLE_TERM_MATCH,
        WorkflowMatchReasonCode.ENABLED_BONUS,
    ]
    no_match = WorkflowDiscoveryService().discover(
        WorkflowDiscoveryRequest(categories=("research",), preferred_tags=("verification",)),
        (_library("atlas.core", (workflow,)),),
    )
    assert no_match.candidates == ()


def test_title_terms_casefold_collapse_spaces_no_fuzzy_and_no_description_search() -> None:
    workflow = _workflow(
        "project.inspect",
        title="Inspect   PYTHON   Project",
        description="This description mentions hiddenneedle.",
    )
    service = WorkflowDiscoveryService()
    library = _library("atlas.core", (workflow,))

    threshold = TITLE_TERM_MATCH_SCORE + ENABLED_BONUS_SCORE
    assert service.discover(
        WorkflowDiscoveryRequest(title_terms=("python project",), minimum_score=threshold),
        (library,),
    ).candidates
    assert service.discover(
        WorkflowDiscoveryRequest(title_terms=("pythn",), minimum_score=threshold),
        (library,),
    ).candidates == ()
    assert service.discover(
        WorkflowDiscoveryRequest(title_terms=("hiddenneedle",), minimum_score=threshold),
        (library,),
    ).candidates == ()


def test_minimum_score_limit_truncated_order_and_determinism() -> None:
    low = _workflow("project.low", category="project", tags=("inspection",))
    mid = _workflow("project.mid", category="coding", tags=("tests",))
    high = _workflow("project.high", category="coding", tags=("tests", "verification"))
    request = WorkflowDiscoveryRequest(
        categories=("coding",),
        preferred_tags=("verification",),
        minimum_score=CATEGORY_MATCH_SCORE + ENABLED_BONUS_SCORE,
        limit=1,
    )
    result = WorkflowDiscoveryService().discover(
        request,
        (_library("atlas.z", (mid,)), _library("atlas.a", (low, high))),
    )

    assert [candidate.workflow.reference for candidate in result.candidates] == [high.reference]
    assert result.truncated is True
    assert result.matched_workflows == 2
    assert result.rejected_workflows == 1
    repeated = WorkflowDiscoveryService().discover(
        request,
        (_library("atlas.a", (low, high)), _library("atlas.z", (mid,))),
    )
    assert result.candidates == repeated.candidates


def test_tie_break_order_and_ambiguity_properties() -> None:
    b = _workflow("project.b", category="coding", tags=("tests",))
    a = _workflow("project.a", category="coding", tags=("tests",))
    result = WorkflowDiscoveryService().discover(
        WorkflowDiscoveryRequest(categories=("coding",), required_tags=("tests",)),
        (_library("atlas.core", (b, a)),),
    )

    assert [candidate.workflow.reference for candidate in result.candidates] == [a.reference, b.reference]
    assert result.has_matches is True
    assert result.best_score == CATEGORY_MATCH_SCORE + REQUIRED_TAG_MATCH_SCORE + ENABLED_BONUS_SCORE
    assert result.top_candidates == result.candidates
    assert result.is_ambiguous is True
    single = WorkflowDiscoveryService().discover(
        WorkflowDiscoveryRequest(references=(a.reference,)),
        (_library("atlas.core", (a,)),),
    )
    assert single.is_ambiguous is False


def test_duplicate_reference_same_signature_deduplicates_with_sources() -> None:
    plan = _plan("same")
    first = _workflow("project.same", plan=plan, title="B title")
    second = _workflow("project.same", plan=plan, title="A title")
    result = WorkflowDiscoveryService().discover(
        WorkflowDiscoveryRequest(enabled_only=False),
        (_library("atlas.z", (first,)), _library("atlas.a", (second,))),
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.workflow.title == "A title"
    assert [source.library_id for source in candidate.source_libraries] == ["atlas.a", "atlas.z"]


def test_duplicate_reference_different_signature_fails() -> None:
    first = _workflow("project.same", plan=_plan("first"))
    second = _workflow("project.same", plan=_plan("second"))

    with pytest.raises(ConflictingWorkflowCandidateError):
        WorkflowDiscoveryService().discover(
            WorkflowDiscoveryRequest(),
            (_library("atlas.a", (first,)), _library("atlas.b", (second,))),
        )


def test_rejections_are_optional_structured_and_sorted() -> None:
    category = _workflow("project.category", category="project", tags=("inspection",))
    missing_tag = _workflow("project.missing", category="coding", tags=("verification",))
    excluded = _workflow("project.excluded", category="coding", tags=("tests", "remote"))
    disabled = _workflow("project.disabled", category="coding", tags=("tests",), enabled=False)
    library = _library("atlas.core", (category, missing_tag, excluded, disabled))
    request = WorkflowDiscoveryRequest(
        categories=("coding",),
        required_tags=("tests",),
        excluded_tags=("remote",),
        include_rejections=True,
    )
    result = WorkflowDiscoveryService().discover(request, (library,))

    reasons_by_plan = {
        rejection.reference.plan_id: rejection.reasons for rejection in result.rejections
    }
    assert WorkflowDiscoveryRejectionCode.CATEGORY_MISMATCH in reasons_by_plan["project.category"]
    assert WorkflowDiscoveryRejectionCode.MISSING_REQUIRED_TAG in reasons_by_plan["project.missing"]
    assert WorkflowDiscoveryRejectionCode.EXCLUDED_TAG in reasons_by_plan["project.excluded"]
    assert WorkflowDiscoveryRejectionCode.DISABLED in reasons_by_plan["project.disabled"]
    assert WorkflowDiscoveryService().discover(
        WorkflowDiscoveryRequest(categories=("missing",)),
        (library,),
    ).rejections == ()


def test_rejection_for_below_minimum_and_library_filter() -> None:
    workflow = _workflow("project.inspect")
    result = WorkflowDiscoveryService().discover(
        WorkflowDiscoveryRequest(minimum_score=999, include_rejections=True),
        (_library("atlas.core", (workflow,)),),
    )

    assert result.rejections[0].reasons == (WorkflowDiscoveryRejectionCode.BELOW_MINIMUM_SCORE,)
    library_filtered = WorkflowDiscoveryService().discover(
        WorkflowDiscoveryRequest(library_ids=("atlas.other",), include_rejections=True),
        (_library("atlas.core", (workflow,)),),
    )
    assert library_filtered.rejections[0].reasons == (WorkflowDiscoveryRejectionCode.LIBRARY_FILTERED,)


def test_result_request_candidate_reasons_are_immutable_and_inputs_are_defensively_copied() -> None:
    workflows = [_workflow("project.inspect")]
    library = _library("atlas.core", tuple(workflows))
    request = WorkflowDiscoveryRequest(categories=["project"])
    result = WorkflowDiscoveryService().discover(request, [library])
    workflows.append(_workflow("project.other"))

    assert len(result.candidates) == 1
    with pytest.raises(FrozenInstanceError):
        result.truncated = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.candidates[0].score = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.candidates[0].reasons[0].score = 0  # type: ignore[misc]


def test_repeated_calls_signature_and_bootstrap_builder_are_deterministic() -> None:
    request = WorkflowDiscoveryRequest(categories=("project",), title_terms=("Inspect",))
    library = _library("atlas.core", (_workflow("project.inspect"),))
    service = build_workflow_discovery_service()

    assert isinstance(service, WorkflowDiscoveryService)
    assert service.discover(request, (library,)) == service.discover(request, (library,))
    assert workflow_discovery_request_signature(request) == workflow_discovery_request_signature(
        WorkflowDiscoveryRequest(categories=("project",), title_terms=("inspect",))
    )
    assert workflow_discovery_request_signature(request) != workflow_discovery_request_signature(
        WorkflowDiscoveryRequest(categories=("coding",), title_terms=("inspect",))
    )


def test_discovery_does_not_use_registry_validator_or_executor_and_existing_components_still_work() -> None:
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
    library = _library("atlas.core", (_workflow("project.inspect"),))

    assert WorkflowDiscoveryService().discover(WorkflowDiscoveryRequest(), (library,)).candidates
    assert ExecutionPlanValidator(plan_registry=registry).validate(parent).is_valid is True
    assert registry.resolve(ExecutionPlanReference("project.child")) is child
    assert library.install(ExecutionPlanRegistry()).installed == (ExecutionPlanReference("project.inspect", "1.0"),)


def test_workflow_discovery_module_does_not_use_forbidden_runtime_features() -> None:
    source = Path("core/workflow_discovery.py").read_text(encoding="utf-8")

    for forbidden in ("eval(", "exec(", "__import__", "importlib", "pickle", "requests", "urllib", "embedding", "fuzzy"):
        assert forbidden not in source

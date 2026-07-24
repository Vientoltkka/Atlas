from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from bootstrap.workflow_selector import build_core_workflow_selector
from core.capability_resolver import (
    CapabilityCandidate,
    CapabilityDefinition,
    CapabilityMatchReason,
    CapabilityMatchReasonCode,
    CapabilityResolutionRequest,
    CapabilityResolutionResult,
    CapabilityResolver,
    CapabilityType,
    ToolCapabilitySource,
    WorkflowCapabilityProvider,
    WorkflowCapabilitySource,
)
from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
from core.execution_plan_registry import ExecutionPlanReference
from core.planner import ExecutionPlan, ExecutionStep
from core.workflow_discovery import WorkflowLibraryReference
from core.workflow_selector import (
    MAX_WORKFLOW_SELECTION_CANDIDATES,
    PREFERRED_CATEGORY_BONUS,
    PREFERRED_LIBRARY_BONUS,
    PREFERRED_REFERENCE_BONUS,
    PREFERRED_TAG_BONUS,
    ConflictingWorkflowSelectionCandidateError,
    InvalidWorkflowSelectionPolicyError,
    InvalidWorkflowSelectionRequestError,
    WorkflowSelectionPolicy,
    WorkflowSelectionReasonCode,
    WorkflowSelectionRequest,
    WorkflowSelectionStatus,
    WorkflowSelector,
    workflow_selection_policy_signature,
    workflow_selection_request_signature,
)
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "demo.tool"

    @property
    def description(self) -> str:
        return "Demo tool."

    def execute(self, context: ToolContext) -> Any:
        del context
        self.calls += 1
        return "executed"


class StaticProvider:
    def __init__(self, *capabilities: CapabilityDefinition) -> None:
        self._capabilities = capabilities

    def list_capabilities(self) -> tuple[CapabilityDefinition, ...]:
        return self._capabilities


def _step(step_id: str = "step_1") -> ExecutionStep:
    return ExecutionStep(step_id, f"Run {step_id}.", "read_file")


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Run workflow.",
        ordered_steps=(_step(),),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
    )


def _workflow_definition(
    plan_id: str,
    *,
    category: str = "project.analysis",
    tags: tuple[str, ...] = ("inspection",),
    enabled: bool = True,
) -> WorkflowDefinition:
    return WorkflowDefinition(
        reference=ExecutionPlanReference(plan_id, "1.0"),
        plan=_plan(),
        title=f"Workflow {plan_id}",
        description="Safe workflow description.",
        category=category,
        tags=tags,
        enabled=enabled,
    )


def _workflow_capability(
    plan_id: str,
    *,
    library_id: str = "atlas.core",
    version: str | None = "1.0",
    score: int = 10,
    category: str = "project.analysis",
    tags: tuple[str, ...] = ("inspection",),
    enabled: bool = True,
) -> CapabilityCandidate:
    capability = CapabilityDefinition(
        capability_id=f"workflow.{library_id}.{plan_id}.{version or 'unversioned'}",
        capability_type=CapabilityType.WORKFLOW,
        title=f"Workflow {plan_id}",
        description="Safe public workflow description.",
        categories=("workflow", category),
        tags=tags,
        input_names=("read_file",),
        output_names=("result",),
        enabled=enabled,
        source_reference=WorkflowCapabilitySource(
            WorkflowLibraryReference(library_id, version),
            ExecutionPlanReference(plan_id, version),
        ),
    )
    return CapabilityCandidate(
        capability=capability,
        score=score,
        reasons=(
            CapabilityMatchReason(
                CapabilityMatchReasonCode.ENABLED_BONUS,
                None,
                score,
            ),
        ),
    )


def _tool_candidate(score: int = 99) -> CapabilityCandidate:
    capability = CapabilityDefinition(
        capability_id="tool.demo",
        capability_type=CapabilityType.TOOL,
        title="Demo tool",
        description="Safe public tool description.",
        categories=("tool",),
        tags=("demo",),
        input_names=(),
        output_names=(),
        enabled=True,
        source_reference=ToolCapabilitySource("demo.tool"),
    )
    return CapabilityCandidate(
        capability=capability,
        score=score,
        reasons=(CapabilityMatchReason(CapabilityMatchReasonCode.ENABLED_BONUS, None, score),),
    )


def _resolution(*candidates: CapabilityCandidate) -> CapabilityResolutionResult:
    top_score = max((candidate.score for candidate in candidates), default=None)
    return CapabilityResolutionResult(
        request=CapabilityResolutionRequest(enabled_only=False),
        candidates=tuple(candidates),
        rejected=(),
        scanned_capabilities=len(candidates),
        matched_capabilities=len(candidates),
        truncated=False,
        ambiguous=(
            top_score is not None
            and sum(1 for candidate in candidates if candidate.score == top_score) >= 2
        ),
        top_score=top_score,
    )


def test_selects_unique_workflow_and_rejects_tool_candidates() -> None:
    workflow = _workflow_capability("project.inspect", score=20)
    result = WorkflowSelector().select(
        WorkflowSelectionRequest(_resolution(_tool_candidate(99), workflow))
    )

    assert result.status is WorkflowSelectionStatus.SELECTED
    assert result.selected_candidate is not None
    assert result.selected_candidate.candidate == workflow
    assert result.base_score == 20
    assert result.policy_bonus == 0
    assert result.final_score == 20
    assert result.total_input_candidates == 2
    assert result.total_workflow_candidates == 1
    assert result.rejected_candidates[0].reason_code is WorkflowSelectionReasonCode.NON_WORKFLOW_REJECTED
    assert WorkflowSelectionReasonCode.UNIQUE_TOP_SELECTED in {reason.code for reason in result.reasons}


def test_no_workflow_candidates_and_disabled_candidates_are_explicit() -> None:
    no_workflows = WorkflowSelector().select(WorkflowSelectionRequest(_resolution(_tool_candidate())))
    assert no_workflows.status is WorkflowSelectionStatus.NO_CANDIDATES
    assert no_workflows.total_workflow_candidates == 0

    disabled = WorkflowSelector().select(
        WorkflowSelectionRequest(_resolution(_workflow_capability("project.disabled", enabled=False)))
    )
    assert disabled.status is WorkflowSelectionStatus.NO_CANDIDATES
    assert disabled.rejected_candidates[0].reason_code is WorkflowSelectionReasonCode.DISABLED_REJECTED

    allowed = WorkflowSelector().select(
        WorkflowSelectionRequest(
            _resolution(_workflow_capability("project.disabled", enabled=False)),
            WorkflowSelectionPolicy(enabled_only=False),
        )
    )
    assert allowed.status is WorkflowSelectionStatus.SELECTED


def test_minimum_score_ambiguous_and_tiebreak_policy() -> None:
    low = _workflow_capability("project.low", score=3)
    below = WorkflowSelector().select(
        WorkflowSelectionRequest(_resolution(low), WorkflowSelectionPolicy(minimum_score=4))
    )
    assert below.status is WorkflowSelectionStatus.BELOW_MINIMUM_SCORE
    assert below.reasons[0].code is WorkflowSelectionReasonCode.BELOW_MINIMUM_SCORE

    first = _workflow_capability("project.a", score=10)
    second = _workflow_capability("project.b", score=10)
    ambiguous = WorkflowSelector().select(WorkflowSelectionRequest(_resolution(second, first)))
    assert ambiguous.status is WorkflowSelectionStatus.AMBIGUOUS
    assert len(ambiguous.ambiguous_candidates) == 2

    selected = WorkflowSelector().select(
        WorkflowSelectionRequest(
            _resolution(second, first),
            WorkflowSelectionPolicy(require_unique_top_score=False),
        )
    )
    assert selected.status is WorkflowSelectionStatus.SELECTED
    assert selected.selected_candidate is not None
    assert selected.selected_candidate.candidate == first


def test_library_category_and_tag_filters() -> None:
    core = _workflow_capability(
        "project.core",
        library_id="atlas.core",
        category="project.analysis",
        tags=("inspection", "safe"),
    )
    other = _workflow_capability(
        "project.other",
        library_id="atlas.other",
        category="project.other",
        tags=("remote",),
    )
    selector = WorkflowSelector()

    assert selector.select(
        WorkflowSelectionRequest(_resolution(core, other), WorkflowSelectionPolicy(allowed_library_ids=("atlas.core",)))
    ).selected_candidate.candidate == core  # type: ignore[union-attr]
    assert selector.select(
        WorkflowSelectionRequest(_resolution(core, other), WorkflowSelectionPolicy(excluded_library_ids=("atlas.core",)))
    ).selected_candidate.candidate == other  # type: ignore[union-attr]
    assert selector.select(
        WorkflowSelectionRequest(_resolution(core, other), WorkflowSelectionPolicy(allowed_categories=("project.analysis",)))
    ).selected_candidate.candidate == core  # type: ignore[union-attr]
    assert selector.select(
        WorkflowSelectionRequest(_resolution(core, other), WorkflowSelectionPolicy(excluded_categories=("project.analysis",)))
    ).selected_candidate.candidate == other  # type: ignore[union-attr]
    assert selector.select(
        WorkflowSelectionRequest(_resolution(core, other), WorkflowSelectionPolicy(required_tags=("safe",)))
    ).selected_candidate.candidate == core  # type: ignore[union-attr]
    assert selector.select(
        WorkflowSelectionRequest(_resolution(core, other), WorkflowSelectionPolicy(excluded_tags=("safe",)))
    ).selected_candidate.candidate == other  # type: ignore[union-attr]


def test_preferences_and_explicit_reference_add_visible_policy_bonus() -> None:
    preferred = _workflow_capability(
        "project.preferred",
        library_id="atlas.preferred",
        category="project.analysis",
        tags=("inspection", "fast"),
        score=10,
    )
    other = _workflow_capability("project.other", score=10)
    policy = WorkflowSelectionPolicy(
        require_unique_top_score=False,
        preferred_library_ids=("atlas.preferred",),
        preferred_categories=("project.analysis",),
        preferred_tags=("fast",),
    )
    request = WorkflowSelectionRequest(
        _resolution(other, preferred),
        policy,
        preferred_reference=ExecutionPlanReference("project.preferred", "1.0"),
    )

    result = WorkflowSelector().select(request)

    assert result.status is WorkflowSelectionStatus.SELECTED
    assert result.selected_candidate is not None
    assert result.selected_candidate.candidate == preferred
    assert result.selected_candidate.policy_bonus == (
        PREFERRED_REFERENCE_BONUS
        + PREFERRED_LIBRARY_BONUS
        + PREFERRED_CATEGORY_BONUS
        + PREFERRED_TAG_BONUS
    )
    assert {
        WorkflowSelectionReasonCode.PREFERRED_REFERENCE_MATCH,
        WorkflowSelectionReasonCode.PREFERRED_LIBRARY_MATCH,
        WorkflowSelectionReasonCode.PREFERRED_CATEGORY_MATCH,
        WorkflowSelectionReasonCode.PREFERRED_TAG_MATCH,
    }.issubset({reason.code for reason in result.selected_candidate.reasons})


def test_order_is_deterministic_independent_from_input_order() -> None:
    a = _workflow_capability("project.a", score=10)
    b = _workflow_capability("project.b", score=10)
    policy = WorkflowSelectionPolicy(require_unique_top_score=False)
    first = WorkflowSelector().select(WorkflowSelectionRequest(_resolution(b, a), policy))
    second = WorkflowSelector().select(WorkflowSelectionRequest(_resolution(a, b), policy))

    assert first.selected_candidate is not None
    assert second.selected_candidate is not None
    assert first.selected_candidate.candidate.capability.capability_id == "workflow.atlas.core.project.a.1.0"
    assert first.selected_candidate.candidate == second.selected_candidate.candidate


def test_duplicates_identical_conflicting_and_invalid_scores() -> None:
    candidate = _workflow_capability("project.same", score=10)
    duplicate = _workflow_capability("project.same", score=10)
    result = WorkflowSelector().select(WorkflowSelectionRequest(_resolution(candidate, duplicate)))
    assert result.status is WorkflowSelectionStatus.SELECTED
    assert result.total_rejected == 1
    assert result.rejected_candidates[0].reason_code is WorkflowSelectionReasonCode.DUPLICATE_IDENTICAL

    conflicting = _workflow_capability("project.same", score=10, tags=("other",))
    with pytest.raises(ConflictingWorkflowSelectionCandidateError):
        WorkflowSelector().select(WorkflowSelectionRequest(_resolution(candidate, conflicting)))

    for bad_score in (float("nan"), float("inf"), True):
        invalid = _workflow_capability("project.invalid", score=1)
        object.__setattr__(invalid, "score", bad_score)
        broken = CapabilityResolutionResult(
            request=CapabilityResolutionRequest(),
            candidates=(invalid,),
            rejected=(),
            scanned_capabilities=1,
            matched_capabilities=1,
            truncated=False,
            ambiguous=False,
            top_score=bad_score,  # type: ignore[arg-type]
        )
        selected = WorkflowSelector().select(WorkflowSelectionRequest(broken))
        assert selected.status is WorkflowSelectionStatus.INVALID_INPUT


def test_policy_validation_limits_contradictions_immutability_and_defensive_copy() -> None:
    policy = WorkflowSelectionPolicy(
        allowed_library_ids=[" Atlas.Core ", "atlas.core"],  # type: ignore[arg-type]
        preferred_tags=[" Fast "],  # type: ignore[arg-type]
    )
    assert policy.allowed_library_ids == ("atlas.core",)
    assert policy.preferred_tags == ("fast",)
    with pytest.raises(FrozenInstanceError):
        policy.minimum_score = 1  # type: ignore[misc]

    for kwargs in (
        {"minimum_score": True},
        {"maximum_candidates_considered": 0},
        {"maximum_candidates_considered": MAX_WORKFLOW_SELECTION_CANDIDATES + 1},
        {"allowed_library_ids": ("atlas.core",), "excluded_library_ids": ("atlas.core",)},
        {"allowed_categories": ("project",), "excluded_categories": ("project",)},
        {"required_tags": ("safe",), "excluded_tags": ("safe",)},
    ):
        with pytest.raises(InvalidWorkflowSelectionPolicyError):
            WorkflowSelectionPolicy(**kwargs)

    metadata = {"nested": {"value": 1}, "items": ["a"]}
    request = WorkflowSelectionRequest(_resolution(_workflow_capability("project.a")), metadata=metadata)
    metadata["nested"]["value"] = 2  # type: ignore[index]
    metadata["items"].append("b")  # type: ignore[union-attr]
    assert request.metadata["nested"]["value"] == 1  # type: ignore[index]
    assert request.metadata["items"] == ("a",)
    with pytest.raises(TypeError):
        request.metadata["x"] = 1  # type: ignore[index]
    with pytest.raises(InvalidWorkflowSelectionRequestError):
        WorkflowSelectionRequest(_resolution(_workflow_capability("project.a")), metadata={"bad": lambda: None})


def test_signatures_are_deterministic_and_result_is_explainable() -> None:
    policy_a = WorkflowSelectionPolicy(
        allowed_library_ids=("atlas.core", "atlas.extra"),
        preferred_tags=("fast", "safe"),
    )
    policy_b = WorkflowSelectionPolicy(
        allowed_library_ids=("atlas.extra", "atlas.core"),
        preferred_tags=("safe", "fast"),
    )
    candidate = _workflow_capability("project.a", tags=("fast", "safe"))
    request_a = WorkflowSelectionRequest(_resolution(candidate), policy_a, metadata={"trace": "stable"})
    request_b = WorkflowSelectionRequest(_resolution(candidate), policy_b, metadata={"trace": "stable"})

    assert workflow_selection_policy_signature(policy_a) == workflow_selection_policy_signature(policy_b)
    assert workflow_selection_request_signature(request_a) == workflow_selection_request_signature(request_b)
    result = WorkflowSelector().select(request_a)
    assert result.policy_signature == workflow_selection_policy_signature(policy_a)
    assert result.request_signature == workflow_selection_request_signature(request_a)
    assert result.reasons
    assert result.considered_candidates[0].reasons


def test_does_not_mutate_resolution_result_execute_workflows_or_use_llm_and_bootstrap_works() -> None:
    tool = SpyTool()
    registry = ToolRegistry()
    registry.register(tool)
    workflow = _workflow_definition("project.real")
    library = ExecutionPlanLibrary("atlas.core", (workflow,), version="1.0")
    capabilities = WorkflowCapabilityProvider((library,)).list_capabilities()
    resolution = CapabilityResolver((StaticProvider(*capabilities),)).resolve(
        CapabilityResolutionRequest(enabled_only=False)
    )
    before = resolution

    selector, policy = build_core_workflow_selector()
    result = selector.select(WorkflowSelectionRequest(resolution, policy))

    assert result.status is WorkflowSelectionStatus.SELECTED
    assert resolution == before
    assert tool.calls == 0
    assert isinstance(selector, WorkflowSelector)
    assert isinstance(policy, WorkflowSelectionPolicy)

    source = __import__("pathlib").Path("core/workflow_selector.py").read_text(encoding="utf-8")
    for forbidden in (
        "eval(",
        "exec(",
        "__import__",
        "importlib",
        "pickle",
        "import requests",
        "import urllib",
        "PromptClient",
        "CapabilityResolver(",
        "ExecutionPlanExecutor(",
        ".execute(",
        "open(",
    ):
        assert forbidden not in source

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from bootstrap.capability_planner import build_core_capability_planner
from core.capability_planner import (
    MAX_CAPABILITY_PLANNING_CANDIDATES,
    CapabilityPlanner,
    CapabilityPlanningDecision,
    CapabilityPlanningRequest,
    CapabilityPlanningStatus,
    InvalidCapabilityPlanningRequestError,
    capability_planning_request_signature,
)
from core.capability_resolver import (
    CapabilityCandidate,
    CapabilityDefinition,
    CapabilityMatchReason,
    CapabilityMatchReasonCode,
    CapabilityResolutionRequest,
    CapabilityResolutionResult,
    CapabilityResolver,
    CapabilityResolverError,
    CapabilityType,
    ToolCapabilitySource,
    WorkflowCapabilityProvider,
    WorkflowCapabilitySource,
)
from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
from core.execution_plan_registry import ExecutionPlanReference
from core.planner import ExecutionPlan, ExecutionStep, Planner
from core.workflow_discovery import WorkflowLibraryReference
from core.workflow_selector import WorkflowSelectionPolicy, WorkflowSelectionRequest, WorkflowSelectionStatus, WorkflowSelector
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


class CountingResolver(CapabilityResolver):
    def __init__(self, result: CapabilityResolutionResult | None = None, *, fail: bool = False) -> None:
        super().__init__(())
        self.result = result
        self.fail = fail
        self.calls = 0
        self.requests: list[CapabilityResolutionRequest] = []

    def resolve(self, request: CapabilityResolutionRequest) -> CapabilityResolutionResult:
        self.calls += 1
        self.requests.append(request)
        if self.fail:
            raise CapabilityResolverError("resolver failed")
        assert self.result is not None
        return self.result


class CountingSelector(WorkflowSelector):
    def __init__(self, status: WorkflowSelectionStatus = WorkflowSelectionStatus.SELECTED) -> None:
        self.calls = 0
        self.requests: list[WorkflowSelectionRequest] = []
        self.status = status

    def select(self, request: WorkflowSelectionRequest):
        self.calls += 1
        self.requests.append(request)
        if self.status is WorkflowSelectionStatus.SELECTED:
            return super().select(request)
        return super().select(
            WorkflowSelectionRequest(
                request.resolution_result,
                WorkflowSelectionPolicy(
                    minimum_score=999 if self.status is WorkflowSelectionStatus.BELOW_MINIMUM_SCORE else request.policy.minimum_score,
                    require_unique_top_score=True,
                    enabled_only=request.policy.enabled_only,
                ),
                preferred_reference=request.preferred_reference,
                metadata=request.metadata,
            )
        )


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


def _workflow(
    plan_id: str = "project.inspect",
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


def _workflow_candidate(
    workflow: WorkflowDefinition,
    *,
    library_id: str = "atlas.core",
    library_version: str | None = "1.0",
    score: int = 10,
) -> CapabilityCandidate:
    capability = CapabilityDefinition(
        capability_id=f"workflow.{library_id}.{workflow.reference.plan_id}.{workflow.reference.version}",
        capability_type=CapabilityType.WORKFLOW,
        title=workflow.title,
        description=workflow.description,
        categories=("workflow", workflow.category),
        tags=workflow.tags,
        input_names=workflow.plan.required_tools,
        output_names=("result",),
        enabled=workflow.enabled,
        source_reference=WorkflowCapabilitySource(
            WorkflowLibraryReference(library_id, library_version),
            workflow.reference,
        ),
    )
    return CapabilityCandidate(
        capability,
        score,
        (CapabilityMatchReason(CapabilityMatchReasonCode.ENABLED_BONUS, None, score),),
    )


def _tool_candidate(score: int = 10) -> CapabilityCandidate:
    capability = CapabilityDefinition(
        capability_id="tool.demo",
        capability_type=CapabilityType.TOOL,
        title="Demo tool",
        description="Safe tool description.",
        categories=("tool",),
        tags=("demo",),
        input_names=(),
        output_names=(),
        enabled=True,
        source_reference=ToolCapabilitySource("demo.tool"),
    )
    return CapabilityCandidate(
        capability,
        score,
        (CapabilityMatchReason(CapabilityMatchReasonCode.ENABLED_BONUS, None, score),),
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


def _planner_for(
    result: CapabilityResolutionResult,
    library: ExecutionPlanLibrary,
    selector: CountingSelector | None = None,
) -> tuple[CapabilityPlanner, CountingResolver, CountingSelector]:
    resolver = CountingResolver(result)
    active_selector = selector or CountingSelector()
    return (
        CapabilityPlanner(resolver, active_selector, execution_plan_libraries=(library,)),
        resolver,
        active_selector,
    )


def test_request_valid_immutable_normalizes_defensively_and_signature_is_order_independent() -> None:
    metadata = {"nested": {"value": 1}, "items": ["a"]}
    first = CapabilityPlanningRequest(
        "  Inspect   project  ",
        capability_id=" WORKFLOW.Atlas.Core.Project.Inspect.1.0 ",
        required_categories=["Project.Analysis", "project.analysis"],  # type: ignore[arg-type]
        excluded_categories=["Remote"],  # type: ignore[arg-type]
        required_tags=["Inspection"],  # type: ignore[arg-type]
        preferred_tags=["Fast", "Safe"],  # type: ignore[arg-type]
        required_inputs=["Read_File"],  # type: ignore[arg-type]
        required_outputs=["Result"],  # type: ignore[arg-type]
        preferred_workflow_reference=ExecutionPlanReference("project.inspect", "1.0"),
        metadata=metadata,
    )
    second = CapabilityPlanningRequest(
        "Inspect project",
        capability_id="workflow.atlas.core.project.inspect.1.0",
        required_categories=("project.analysis",),
        excluded_categories=("remote",),
        required_tags=("inspection",),
        preferred_tags=("safe", "fast"),
        required_inputs=("read_file",),
        required_outputs=("result",),
        preferred_workflow_reference=ExecutionPlanReference("project.inspect", "1.0"),
        metadata={"items": ("a",), "nested": {"value": 1}},
    )
    metadata["nested"]["value"] = 2  # type: ignore[index]
    metadata["items"].append("b")  # type: ignore[union-attr]

    assert first.objective == "Inspect project"
    assert first.required_categories == ("project.analysis",)
    assert first.metadata["nested"]["value"] == 1  # type: ignore[index]
    assert capability_planning_request_signature(first) == capability_planning_request_signature(second)
    with pytest.raises(FrozenInstanceError):
        first.objective = "Other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.metadata["x"] = 1  # type: ignore[index]


def test_invalid_request_and_limits_are_rejected_or_reported() -> None:
    for kwargs in (
        {"objective": ""},
        {"objective": "x", "required_categories": ("",)},
        {"objective": "x", "required_categories": ("a",), "excluded_categories": ("a",)},
        {"objective": "x", "minimum_capability_score": True},
        {"objective": "x", "maximum_candidates": MAX_CAPABILITY_PLANNING_CANDIDATES + 1},
        {"objective": "x", "metadata": {"bad": lambda: None}},
    ):
        with pytest.raises(InvalidCapabilityPlanningRequestError):
            CapabilityPlanningRequest(**kwargs)

    decision = CapabilityPlanner(CountingResolver(_resolution()), CountingSelector()).plan(object())  # type: ignore[arg-type]
    assert decision.status is CapabilityPlanningStatus.INVALID_REQUEST


def test_resolver_without_candidates_or_failure_does_not_call_selector() -> None:
    empty = CountingResolver(_resolution())
    selector = CountingSelector()
    decision = CapabilityPlanner(empty, selector).plan(CapabilityPlanningRequest("Inspect project"))

    assert decision.status is CapabilityPlanningStatus.NO_CAPABILITY_CANDIDATES
    assert selector.calls == 0

    failing = CountingResolver(fail=True)
    failed = CapabilityPlanner(failing, selector).plan(CapabilityPlanningRequest("Inspect project"))
    assert failed.status is CapabilityPlanningStatus.RESOLUTION_FAILED


def test_resolver_ambiguous_non_workflows_and_non_workflow_top_are_explicit() -> None:
    ambiguous = _resolution(_tool_candidate(10), _tool_candidate(10))
    selector = CountingSelector()
    decision = CapabilityPlanner(CountingResolver(ambiguous), selector).plan(CapabilityPlanningRequest("Inspect project"))
    assert decision.status is CapabilityPlanningStatus.CAPABILITY_AMBIGUOUS
    assert selector.calls == 0

    single_tool = CapabilityPlanner(CountingResolver(_resolution(_tool_candidate())), selector).plan(
        CapabilityPlanningRequest("Inspect project")
    )
    assert single_tool.status is CapabilityPlanningStatus.INCOMPATIBLE_CAPABILITY


def test_selector_states_are_mapped_and_libraries_are_not_read_without_selection() -> None:
    workflow = _workflow()
    candidate = _workflow_candidate(workflow)
    missing_library = ExecutionPlanLibrary("atlas.other", (_workflow("project.other"),), version="1.0")

    no_candidates = CapabilityPlanner(
        CountingResolver(_resolution(candidate)),
        CountingSelector(),
        execution_plan_libraries=(missing_library,),
    ).plan(CapabilityPlanningRequest("Inspect project", required_tags=("missing",)))
    assert no_candidates.status is CapabilityPlanningStatus.NO_WORKFLOW_CANDIDATES

    below, _resolver, selector = _planner_for(
        _resolution(candidate),
        missing_library,
        CountingSelector(WorkflowSelectionStatus.BELOW_MINIMUM_SCORE),
    )
    below_decision = below.plan(CapabilityPlanningRequest("Inspect project"))
    assert below_decision.status is CapabilityPlanningStatus.WORKFLOW_BELOW_MINIMUM_SCORE
    assert selector.calls == 1


def test_selector_ambiguous_and_deterministic_selected_workflow_plan_recovery() -> None:
    first = _workflow("project.a")
    second = _workflow("project.b")
    library = ExecutionPlanLibrary("atlas.core", (first, second), version="1.0")
    ambiguous_planner, _resolver, _selector = _planner_for(
        _resolution(_workflow_candidate(second), _workflow_candidate(first)),
        library,
    )

    ambiguous = ambiguous_planner.plan(CapabilityPlanningRequest("Inspect project"))
    assert ambiguous.status is CapabilityPlanningStatus.WORKFLOW_AMBIGUOUS

    selected = ambiguous_planner.plan(CapabilityPlanningRequest("Inspect project", require_unique_workflow=False))
    assert selected.status is CapabilityPlanningStatus.SELECTED
    assert selected.selected_workflow is first
    assert selected.selected_workflow_reference == ExecutionPlanReference("project.a", "1.0")
    assert selected.plan is first.plan
    assert selected.library_id == "atlas.core"
    assert selected.plan_id == "project.a"
    assert selected.version == "1.0"
    assert selected.plan_signature


def test_correct_selection_with_real_workflow_capability_provider_and_preferred_reference() -> None:
    first = _workflow("project.a", tags=("inspection", "safe"))
    second = _workflow("project.b", tags=("inspection", "fast"))
    library = ExecutionPlanLibrary("atlas.core", (first, second), version="1.0")
    resolver = CapabilityResolver((WorkflowCapabilityProvider((library,)),))
    planner = CapabilityPlanner(resolver, WorkflowSelector(), execution_plan_libraries=(library,))

    decision = planner.plan(
        CapabilityPlanningRequest(
            "Inspect project",
            required_categories=("workflow",),
            required_tags=("inspection",),
            preferred_tags=("fast",),
            preferred_workflow_reference=ExecutionPlanReference("project.b", "1.0"),
            require_unique_workflow=False,
        )
    )

    assert decision.status is CapabilityPlanningStatus.SELECTED
    assert decision.selected_capability is not None
    assert decision.selected_capability.capability_type is CapabilityType.WORKFLOW
    assert decision.selected_workflow is second
    assert decision.plan is second.plan


def test_unresolvable_and_inconsistent_workflow_reference_are_statuses() -> None:
    workflow = _workflow("project.a")
    candidate = _workflow_candidate(workflow)
    missing = CapabilityPlanner(CountingResolver(_resolution(candidate)), CountingSelector()).plan(
        CapabilityPlanningRequest("Inspect project")
    )
    assert missing.status is CapabilityPlanningStatus.WORKFLOW_NOT_RESOLVABLE

    wrong_library = ExecutionPlanLibrary("atlas.core", (_workflow("project.other"),), version="1.0", allow_empty=False)
    inconsistent = CapabilityPlanner(
        CountingResolver(_resolution(candidate)),
        CountingSelector(),
        execution_plan_libraries=(wrong_library,),
    ).plan(CapabilityPlanningRequest("Inspect project"))
    assert inconsistent.status is CapabilityPlanningStatus.WORKFLOW_NOT_RESOLVABLE


def test_planner_explicit_integration_is_compatible_and_does_not_execute() -> None:
    workflow = _workflow("project.a")
    library = ExecutionPlanLibrary("atlas.core", (workflow,), version="1.0")
    capability_planner, _resolver, _selector = _planner_for(_resolution(_workflow_candidate(workflow)), library)
    planner = Planner(capability_planner=capability_planner)
    tool = SpyTool()
    registry = ToolRegistry()
    registry.register(tool)

    decision = planner.plan_with_capabilities(CapabilityPlanningRequest("Inspect project"))

    assert isinstance(decision, CapabilityPlanningDecision)
    assert decision.status is CapabilityPlanningStatus.SELECTED
    assert decision.plan is workflow.plan
    assert tool.calls == 0
    assert Planner().create_plan("hola").task == "chat"
    with pytest.raises(RuntimeError):
        Planner().plan_with_capabilities(CapabilityPlanningRequest("Inspect project"))


def test_bootstrap_factory_and_source_forbidden_runtime_features() -> None:
    workflow = _workflow("project.a")
    library = ExecutionPlanLibrary("atlas.core", (workflow,), version="1.0")
    planner = build_core_capability_planner(
        capability_resolver=CapabilityResolver((WorkflowCapabilityProvider((library,)),)),
        workflow_selector=WorkflowSelector(),
        execution_plan_libraries=(library,),
    )
    assert isinstance(planner, CapabilityPlanner)

    source = Path("core/capability_planner.py").read_text(encoding="utf-8")
    planner_source = Path("core/planner.py").read_text(encoding="utf-8")
    for text in (source, planner_source):
        for forbidden in (
            "ToolExecutor",
            "ExecutionPlanExecutor",
            "subprocess",
            "import requests",
            "http",
            "logging",
            "print(",
            "eval(",
            "exec(",
            "pickle",
            "PromptClient",
            "ask_messages",
        ):
            assert forbidden not in text

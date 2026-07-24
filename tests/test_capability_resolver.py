from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from bootstrap.capability_resolver import build_core_capability_resolver
from core.capability_resolver import (
    CAPABILITY_ID_MATCH_SCORE,
    CAPABILITY_TYPE_MATCH_SCORE,
    DESIRED_OUTPUT_MATCH_SCORE,
    ENABLED_BONUS_SCORE,
    MAX_CAPABILITY_RESULTS,
    PREFERRED_TAG_MATCH_SCORE,
    REQUIRED_CATEGORY_MATCH_SCORE,
    REQUIRED_INPUT_MATCH_SCORE,
    REQUIRED_TAG_MATCH_SCORE,
    TITLE_TERM_MATCH_SCORE,
    CapabilityCandidate,
    CapabilityDefinition,
    CapabilityMatchReasonCode,
    CapabilityProviderError,
    CapabilityRejectionCode,
    CapabilityResolutionRequest,
    CapabilityResolver,
    CapabilityType,
    CapabilityValidationError,
    ConflictingCapabilityDefinitionError,
    InvalidCapabilityResolutionRequestError,
    ToolCapabilityProvider,
    ToolCapabilitySource,
    WorkflowCapabilityProvider,
    WorkflowCapabilitySource,
    capability_resolution_request_signature,
)
from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
from core.execution_plan_registry import ExecutionPlanReference
from core.planner import ExecutionPlan, ExecutionStep
from core.workflow_discovery import WorkflowLibraryReference
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext
from tools.tool_schema import ToolArgumentsSchema, ToolParameterSchema


class SpyTool(BaseTool):
    def __init__(
        self,
        name: str = "demo.tool",
        *,
        description: str = "Demo tool.",
        requires_confirmation: bool = False,
    ) -> None:
        self._name = name
        self._description = description
        self._requires_confirmation = requires_confirmation
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def requires_confirmation(self) -> bool:
        return self._requires_confirmation

    def execute(self, context: ToolContext) -> Any:
        del context
        self.calls += 1
        return "executed"


class StaticProvider:
    def __init__(self, *capabilities: CapabilityDefinition) -> None:
        self._capabilities = capabilities

    def list_capabilities(self) -> tuple[CapabilityDefinition, ...]:
        return self._capabilities


class FailingProvider:
    def list_capabilities(self) -> tuple[CapabilityDefinition, ...]:
        raise RuntimeError("boom")


class InvalidProvider:
    def list_capabilities(self) -> tuple[CapabilityDefinition, ...]:
        return ("bad",)  # type: ignore[return-value]


def _capability(
    capability_id: str = "tool.demo",
    *,
    capability_type: CapabilityType = CapabilityType.TOOL,
    title: str = "Demo capability",
    categories: tuple[str, ...] = ("tool", "demo"),
    tags: tuple[str, ...] = ("demo",),
    input_names: tuple[str, ...] = ("path",),
    output_names: tuple[str, ...] = ("result",),
    enabled: bool = True,
    source_reference: ToolCapabilitySource | WorkflowCapabilitySource | None = None,
    metadata: dict[str, object] | None = None,
) -> CapabilityDefinition:
    source = source_reference
    if source is None:
        source = (
            ToolCapabilitySource(capability_id.removeprefix("tool."))
            if capability_type is CapabilityType.TOOL
            else WorkflowCapabilitySource(
                WorkflowLibraryReference("atlas.core", "1.0"),
                ExecutionPlanReference(capability_id.removeprefix("workflow."), "1.0"),
            )
        )
    return CapabilityDefinition(
        capability_id=capability_id,
        capability_type=capability_type,
        title=title,
        description="Safe public description.",
        categories=categories,
        tags=tags,
        input_names=input_names,
        output_names=output_names,
        enabled=enabled,
        source_reference=source,
        metadata={} if metadata is None else metadata,
    )


def _step(step_id: str, tool: str = "read_file") -> ExecutionStep:
    return ExecutionStep(step_id, f"Run {step_id}.", tool)


def _plan(required_tools: tuple[str, ...] = ("read_file",)) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Run workflow.",
        ordered_steps=(_step("step_1"),),
        estimated_steps=1,
        required_tools=required_tools,
        detected_risks=(),
        requires_confirmation=False,
    )


def _workflow(
    plan_id: str = "project.inspect",
    *,
    title: str = "Inspect project",
    category: str = "project.analysis",
    tags: tuple[str, ...] = ("inspection", "filesystem"),
    enabled: bool = True,
) -> WorkflowDefinition:
    return WorkflowDefinition(
        reference=ExecutionPlanReference(plan_id, "1.0"),
        plan=_plan(),
        title=title,
        description="Inspect project with a reusable workflow.",
        category=category,
        tags=tags,
        enabled=enabled,
    )


def test_capability_definition_validates_is_immutable_and_defensively_copies() -> None:
    metadata = {"flag": True, "nested": {"count": 1}, "items": ["a"]}
    capability = _capability(
        " Tool.Demo ",
        categories=[" Tool ", "Demo"],  # type: ignore[arg-type]
        tags=["Demo", "Demo"],  # type: ignore[arg-type]
        input_names=[" Path "],  # type: ignore[arg-type]
        metadata=metadata,
    )
    metadata["nested"]["count"] = 2  # type: ignore[index]
    metadata["items"].append("b")  # type: ignore[union-attr]

    assert capability.capability_id == "tool.demo"
    assert capability.categories == ("tool", "demo")
    assert capability.tags == ("demo",)
    assert capability.input_names == ("path",)
    assert capability.metadata["nested"]["count"] == 1  # type: ignore[index]
    assert capability.metadata["items"] == ("a",)
    with pytest.raises(FrozenInstanceError):
        capability.enabled = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        capability.metadata["x"] = 1  # type: ignore[index]


def test_capability_definition_rejects_invalid_ids_sources_and_unsafe_metadata() -> None:
    with pytest.raises(CapabilityValidationError):
        _capability("")
    with pytest.raises(CapabilityValidationError):
        _capability("tool.demo", source_reference=WorkflowCapabilitySource(WorkflowLibraryReference("atlas.core", "1.0"), ExecutionPlanReference("p")))
    for value in (float("nan"), float("inf"), lambda: None, object):
        with pytest.raises(CapabilityValidationError):
            _capability(metadata={"bad": value})


def test_resolution_request_normalizes_deduplicates_validates_and_is_immutable() -> None:
    request = CapabilityResolutionRequest(
        capability_types=["tool", CapabilityType.TOOL],  # type: ignore[list-item]
        required_capability_ids=[" Tool.Demo ", "tool.demo"],
        required_categories=[" Tool "],
        excluded_categories=["Remote"],
        required_tags=["Read"],
        preferred_tags=["Fast"],
        required_inputs=["Path"],
        desired_outputs=["Result"],
        title_terms=["  Demo   Tool "],
        limit=2,
    )

    assert request.capability_types == (CapabilityType.TOOL,)
    assert request.required_capability_ids == ("tool.demo",)
    assert request.required_categories == ("tool",)
    assert request.excluded_categories == ("remote",)
    assert request.title_terms == ("demo tool",)
    with pytest.raises(FrozenInstanceError):
        request.limit = 3  # type: ignore[misc]

    for kwargs in (
        {"required_categories": [""]},
        {"title_terms": [""]},
        {"minimum_score": True},
        {"limit": MAX_CAPABILITY_RESULTS + 1},
        {"enabled_only": "yes"},
    ):
        with pytest.raises((InvalidCapabilityResolutionRequestError, CapabilityValidationError)):
            CapabilityResolutionRequest(**kwargs)


def test_tool_capability_provider_reads_registered_tools_without_execution() -> None:
    tool = SpyTool("demo.tool", requires_confirmation=True)
    registry = ToolRegistry()
    registry.register(
        tool,
        arguments_schema=ToolArgumentsSchema(
            parameters=(ToolParameterSchema("path", str, required=True),),
        ),
    )

    capabilities = ToolCapabilityProvider(registry).list_capabilities()

    assert len(capabilities) == 1
    capability = capabilities[0]
    assert capability.capability_type is CapabilityType.TOOL
    assert capability.capability_id == "tool.demo.tool"
    assert capability.categories == ("tool", "demo")
    assert "confirmation_required" in capability.tags
    assert capability.input_names == ("path",)
    assert capability.source_reference == ToolCapabilitySource("demo.tool")
    assert tool.calls == 0


def test_workflow_capability_provider_reads_real_workflows_without_installing() -> None:
    workflow = _workflow()
    library = ExecutionPlanLibrary("atlas.core", (workflow,), version="1.0")

    capabilities = WorkflowCapabilityProvider((library,)).list_capabilities()

    assert len(capabilities) == 1
    capability = capabilities[0]
    assert capability.capability_type is CapabilityType.WORKFLOW
    assert capability.categories == ("workflow", "project.analysis")
    assert capability.tags == ("inspection", "filesystem")
    assert capability.input_names == ()
    assert capability.source_reference == WorkflowCapabilitySource(
        WorkflowLibraryReference("atlas.core", "1.0"),
        workflow.reference,
    )


def test_resolver_filters_scores_reasons_and_rejections() -> None:
    selected = _capability(
        "tool.read_file",
        title="Read local file",
        categories=("tool", "filesystem"),
        tags=("read", "local", "fast"),
        input_names=("path",),
        output_names=("result",),
    )
    disabled = _capability("tool.disabled", enabled=False)
    other = _capability("workflow.other", capability_type=CapabilityType.WORKFLOW, categories=("workflow",), tags=("read",))
    request = CapabilityResolutionRequest(
        capability_types=(CapabilityType.TOOL,),
        required_capability_ids=("tool.read_file",),
        required_categories=("filesystem",),
        excluded_categories=("remote",),
        required_tags=("read",),
        preferred_tags=("fast",),
        required_inputs=("path",),
        desired_outputs=("result",),
        title_terms=("local file",),
        include_rejections=True,
    )

    result = CapabilityResolver((StaticProvider(selected, disabled, other),)).resolve(request)

    expected = (
        CAPABILITY_ID_MATCH_SCORE
        + CAPABILITY_TYPE_MATCH_SCORE
        + REQUIRED_CATEGORY_MATCH_SCORE
        + REQUIRED_TAG_MATCH_SCORE
        + REQUIRED_INPUT_MATCH_SCORE
        + DESIRED_OUTPUT_MATCH_SCORE
        + PREFERRED_TAG_MATCH_SCORE
        + TITLE_TERM_MATCH_SCORE
        + ENABLED_BONUS_SCORE
    )
    assert result.scanned_capabilities == 3
    assert result.matched_capabilities == 1
    assert result.top_score == expected
    assert result.candidates[0].score == expected
    assert sum(reason.score for reason in result.candidates[0].reasons) == expected
    assert [reason.code for reason in result.candidates[0].reasons] == [
        CapabilityMatchReasonCode.CAPABILITY_ID_MATCH,
        CapabilityMatchReasonCode.CAPABILITY_TYPE_MATCH,
        CapabilityMatchReasonCode.REQUIRED_CATEGORY_MATCH,
        CapabilityMatchReasonCode.REQUIRED_TAG_MATCH,
        CapabilityMatchReasonCode.REQUIRED_INPUT_MATCH,
        CapabilityMatchReasonCode.DESIRED_OUTPUT_MATCH,
        CapabilityMatchReasonCode.PREFERRED_TAG_MATCH,
        CapabilityMatchReasonCode.TITLE_TERM_MATCH,
        CapabilityMatchReasonCode.ENABLED_BONUS,
    ]
    assert {rejection.reason_code for rejection in result.rejected} == {
        CapabilityRejectionCode.CAPABILITY_ID_MISMATCH,
        CapabilityRejectionCode.TYPE_MISMATCH,
    }


def test_filters_cover_type_categories_tags_inputs_enabled_minimum_score_and_limit() -> None:
    first = _capability("tool.a", categories=("tool", "filesystem"), tags=("read", "fast"), title="Alpha reader")
    second = _capability("tool.b", categories=("tool", "filesystem"), tags=("read",), title="Beta reader")
    disabled = _capability("tool.c", categories=("tool", "filesystem"), tags=("read",), enabled=False)
    workflow = _capability("workflow.a", capability_type=CapabilityType.WORKFLOW, categories=("workflow", "filesystem"), tags=("read",))
    resolver = CapabilityResolver((StaticProvider(first, second, disabled, workflow),))

    assert resolver.resolve(CapabilityResolutionRequest(capability_types=(CapabilityType.WORKFLOW,))).candidates[0].capability == workflow
    assert resolver.resolve(CapabilityResolutionRequest(required_categories=("filesystem",), excluded_categories=("workflow",))).matched_capabilities == 2
    assert resolver.resolve(CapabilityResolutionRequest(required_tags=("fast",))).candidates[0].capability == first
    assert resolver.resolve(CapabilityResolutionRequest(required_inputs=("missing",))).candidates == ()
    assert resolver.resolve(CapabilityResolutionRequest(enabled_only=False)).matched_capabilities == 4
    high = resolver.resolve(CapabilityResolutionRequest(preferred_tags=("fast",), minimum_score=PREFERRED_TAG_MATCH_SCORE + ENABLED_BONUS_SCORE))
    assert high.candidates == (CapabilityCandidate(first, PREFERRED_TAG_MATCH_SCORE + ENABLED_BONUS_SCORE, high.candidates[0].reasons),)
    limited = resolver.resolve(CapabilityResolutionRequest(limit=1, enabled_only=False))
    assert limited.truncated is True
    assert len(limited.candidates) == 1


def test_title_terms_are_literal_casefold_substrings_only() -> None:
    capability = _capability(title="Inspect   PYTHON   Project")
    resolver = CapabilityResolver((StaticProvider(capability),))

    assert resolver.resolve(CapabilityResolutionRequest(title_terms=("python project",), minimum_score=6)).candidates
    assert resolver.resolve(CapabilityResolutionRequest(title_terms=("pythn",), minimum_score=6)).candidates == ()


def test_order_deduplicates_identical_capabilities_conflicts_and_ambiguity() -> None:
    b = _capability("tool.b", tags=("same",), source_reference=ToolCapabilitySource("b.tool"))
    a = _capability("tool.a", tags=("same",), source_reference=ToolCapabilitySource("a.tool"))
    result = CapabilityResolver((StaticProvider(b, a),)).resolve(CapabilityResolutionRequest(required_tags=("same",)))

    assert [candidate.capability.capability_id for candidate in result.candidates] == ["tool.a", "tool.b"]
    assert result.ambiguous is True

    duplicate = _capability("tool.a", tags=("same",), source_reference=ToolCapabilitySource("a.tool"))
    assert CapabilityResolver((StaticProvider(a, duplicate),)).resolve(CapabilityResolutionRequest()).matched_capabilities == 1

    conflicting = _capability("tool.a", title="Other title", source_reference=ToolCapabilitySource("a.tool"))
    with pytest.raises(ConflictingCapabilityDefinitionError):
        CapabilityResolver((StaticProvider(a, conflicting),)).resolve(CapabilityResolutionRequest())


def test_provider_failures_invalid_data_limits_and_bootstrap_builder() -> None:
    with pytest.raises(CapabilityProviderError):
        CapabilityResolver((FailingProvider(),)).resolve(CapabilityResolutionRequest())
    with pytest.raises(CapabilityProviderError):
        CapabilityResolver((InvalidProvider(),)).resolve(CapabilityResolutionRequest())
    with pytest.raises(CapabilityProviderError):
        CapabilityResolver(tuple(StaticProvider() for _ in range(17)))

    registry = ToolRegistry()
    registry.register(SpyTool("demo.tool"))
    resolver = build_core_capability_resolver(tool_registry=registry)
    assert resolver.resolve(CapabilityResolutionRequest()).matched_capabilities == 1


def test_signature_is_deterministic_for_semantically_equivalent_requests() -> None:
    first = CapabilityResolutionRequest(
        required_categories=("filesystem", "tool"),
        preferred_tags=("fast", "read"),
        title_terms=("Read File",),
    )
    second = CapabilityResolutionRequest(
        required_categories=("tool", "filesystem"),
        preferred_tags=("read", "fast"),
        title_terms=(" read   file ",),
    )

    assert capability_resolution_request_signature(first) == capability_resolution_request_signature(second)
    assert capability_resolution_request_signature(first) != capability_resolution_request_signature(
        CapabilityResolutionRequest(required_categories=("workflow",))
    )


def test_absence_of_execution_io_llm_and_real_registry_library_compatibility() -> None:
    tool = SpyTool("demo.tool")
    registry = ToolRegistry()
    registry.register(tool)
    library = ExecutionPlanLibrary("atlas.core", (_workflow(),), version="1.0")
    resolver = build_core_capability_resolver(
        tool_registry=registry,
        execution_plan_libraries=(library,),
    )

    result = resolver.resolve(CapabilityResolutionRequest(enabled_only=False))

    assert result.matched_capabilities == 2
    assert tool.calls == 0
    assert Bootstrap.build_tool_registry().list()
    assert ToolCapabilityProvider(Bootstrap.build_tool_registry()).list_capabilities()
    assert WorkflowCapabilityProvider((library,)).list_capabilities()

    source = Path("core/capability_resolver.py").read_text(encoding="utf-8")
    for forbidden in (
        "eval(",
        "exec(",
        "__import__",
        "importlib",
        "pickle",
        "import requests",
        "import urllib",
        "PromptClient",
        "ExecutionPlanExecutor(",
        ".execute(",
        "open(",
    ):
        assert forbidden not in source

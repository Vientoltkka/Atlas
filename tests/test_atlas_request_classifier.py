from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from bootstrap.atlas_request_adapter import build_core_atlas_request_adapter
from bootstrap.atlas_request_classifier import build_core_atlas_request_classifier
from bootstrap.atlas_request_normalizer import build_core_atlas_request_normalizer
from bootstrap.atlas_router import build_core_atlas_router
from bootstrap.bootstrap import Bootstrap
from bootstrap.capability_execution_service import build_capability_execution_service
from bootstrap.capability_orchestrator import build_core_capability_orchestrator
from bootstrap.capability_planner import build_core_capability_planner
from bootstrap.capability_resolver import build_core_capability_resolver
from bootstrap.workflow_selector import build_core_workflow_selector
from core.atlas_request_adapter import AtlasRequestAdapter
from core.atlas_request_classifier import (
    AtlasRequestClassificationStatus,
    AtlasRequestClassifier,
    InvalidStructuredInputError,
    StructuredInput,
    atlas_request_classification_signature,
)
from core.atlas_request_normalizer import AtlasRequestNormalizer
from core.atlas_router import AtlasRouteType, AtlasRoutingResult, AtlasRoutingStatus
from core.capability_execution_service import CapabilityExecutionService
from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
from core.execution_plan_registry import ExecutionPlanReference
from core.execution_plan_validator import ExecutionPlanValidator
from core.orchestrator import AtlasOrchestrator
from core.planner import ExecutionPlan, ExecutionStep
from core.router import Router
from memory.conversation import ConversationMemory
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(self, output: Any = "ok") -> None:
        self._output = output
        self.calls = 0

    @property
    def name(self) -> str:
        return "demo.tool"

    @property
    def description(self) -> str:
        return "Safe classifier test tool."

    def execute(self, context: ToolContext) -> Any:
        del context
        self.calls += 1
        return self._output


class CountingAdapter(AtlasRequestAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def adapt(self, request):
        self.calls += 1
        return super().adapt(request)


class CountingRouter:
    def __init__(self, result: AtlasRoutingResult | None = None) -> None:
        self.result = result or AtlasRoutingResult(
            AtlasRoutingStatus.COMPLETED,
            AtlasRouteType.CAPABILITY,
            output={"ok": True},
        )
        self.calls = 0
        self.requests = []

    def route(self, request):
        self.calls += 1
        self.requests.append(request)
        return self.result


class ChatAgent:
    name = "chat"
    generated_path = None

    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, model: str, messages):
        del model, messages
        self.calls += 1
        return "respuesta anterior"


def _step() -> ExecutionStep:
    return ExecutionStep("step_1", "Execute demo workflow.", "demo.tool")


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Execute demo workflow.",
        ordered_steps=(_step(),),
        estimated_steps=1,
        required_tools=("demo.tool",),
        detected_risks=(),
        requires_confirmation=False,
    )


def _workflow(plan: ExecutionPlan) -> WorkflowDefinition:
    return WorkflowDefinition(
        reference=ExecutionPlanReference("workflow.demo", "1.0"),
        plan=plan,
        title="Demo workflow",
        description="Safe workflow for classifier tests.",
        category="demo",
        tags=("demo",),
    )


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _real_capability_service(registry: ToolRegistry, library: ExecutionPlanLibrary) -> CapabilityExecutionService:
    resolver = build_core_capability_resolver(
        tool_registry=registry,
        execution_plan_libraries=(library,),
    )
    selector, _policy = build_core_workflow_selector()
    planner = build_core_capability_planner(
        capability_resolver=resolver,
        workflow_selector=selector,
        execution_plan_libraries=(library,),
    )
    orchestrator = build_core_capability_orchestrator(
        planner,
        ExecutionPlanValidator(registry),
        ExecutionPlanExecutor(registry),
    )
    return build_capability_execution_service(orchestrator)


def _atlas_orchestrator(*, classifier=None, adapter=None, router=None, normalizer="default"):
    agent = ChatAgent()
    registry = SimpleNamespace(get=lambda name: agent if name == "chat" else None)
    atlas_request_normalizer = AtlasRequestNormalizer() if normalizer == "default" else normalizer
    return AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=registry,
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        atlas_request_classifier=classifier,
        atlas_request_adapter=adapter,
        atlas_request_normalizer=atlas_request_normalizer,
        atlas_router=router,
    ), agent


def test_classifies_capability_id() -> None:
    result = AtlasRequestClassifier().classify(StructuredInput(capability_id="workflow.demo"))

    assert result.status is AtlasRequestClassificationStatus.CLASSIFIED
    assert result.route_type is AtlasRouteType.CAPABILITY
    assert result.structured_request is not None
    assert result.structured_request.payload["capability_id"] == "workflow.demo"  # type: ignore[index]


@pytest.mark.parametrize(
    ("route", "expected"),
    (
        ("CAPABILITY", AtlasRouteType.CAPABILITY),
        ("tool", AtlasRouteType.TOOL),
        (AtlasRouteType.WORKFLOW, AtlasRouteType.WORKFLOW),
        ("agent", AtlasRouteType.AGENT),
        ("conversation", AtlasRouteType.CONVERSATION),
        ("unknown", AtlasRouteType.UNKNOWN),
    ),
)
def test_classifies_explicit_route(route, expected: AtlasRouteType) -> None:
    result = AtlasRequestClassifier().classify(StructuredInput(route=route))

    assert result.status is AtlasRequestClassificationStatus.CLASSIFIED
    assert result.route_type is expected


def test_kind_workflow_and_tool_name_are_explicit_rules() -> None:
    by_kind = AtlasRequestClassifier().classify(StructuredInput(kind="TOOL"))
    by_workflow = AtlasRequestClassifier().classify(StructuredInput(workflow_id="workflow.demo"))
    by_tool = AtlasRequestClassifier().classify(StructuredInput(tool_name="demo.tool"))

    assert by_kind.route_type is AtlasRouteType.TOOL
    assert by_workflow.route_type is AtlasRouteType.WORKFLOW
    assert by_tool.route_type is AtlasRouteType.TOOL


def test_unknown_when_no_rule_matches() -> None:
    result = AtlasRequestClassifier().classify(StructuredInput(payload={"safe": "value"}))

    assert result.status is AtlasRequestClassificationStatus.UNKNOWN
    assert result.structured_request is None
    assert result.error_code == "UNKNOWN_ROUTE"


def test_invalid_route_is_structured_failure() -> None:
    result = AtlasRequestClassifier().classify(StructuredInput(route="missing"))

    assert result.status is AtlasRequestClassificationStatus.INVALID_ROUTE
    assert result.error_code == "INVALID_ROUTE"


def test_valid_payload_and_metadata_are_accepted() -> None:
    structured_input = StructuredInput(
        route="tool",
        payload={"nested": {"items": [1, True, None]}},
        metadata={"source": "test"},
    )

    result = AtlasRequestClassifier().classify(structured_input)

    assert result.status is AtlasRequestClassificationStatus.CLASSIFIED
    assert result.structured_request is not None
    assert result.structured_request.payload["nested"]["items"] == (1, True, None)  # type: ignore[index]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"payload": {"bad": object()}},
        {"payload": {"bad": float("nan")}},
        {"payload": {"bad": float("inf")}},
        {"metadata": {"api_key": "secret"}},
        {"payload": {"api_token": "secret"}},
    ),
)
def test_invalid_payload_metadata_nan_infinity_and_secrets(kwargs) -> None:
    with pytest.raises(InvalidStructuredInputError):
        StructuredInput(route="tool", **kwargs)


def test_copies_payload_and_metadata_defensively() -> None:
    payload = {"items": ["a"]}
    metadata = {"trace": "one"}
    structured_input = StructuredInput(route="tool", payload=payload, metadata=metadata)
    payload["items"].append("b")
    metadata["trace"] = "two"

    assert isinstance(structured_input.payload, MappingProxyType)
    assert structured_input.payload["items"] == ("a",)  # type: ignore[index]
    assert structured_input.metadata["trace"] == "one"  # type: ignore[index]
    with pytest.raises(TypeError):
        structured_input.payload["x"] = "y"  # type: ignore[index]


def test_rejects_depth_collection_and_string_limits() -> None:
    deep = {"x": {"x": {"x": {"x": {"x": {"x": {"x": {"x": {"x": "too deep"}}}}}}}}}
    too_many = {f"k{i}": i for i in range(65)}
    too_long = "x" * 501

    with pytest.raises(InvalidStructuredInputError):
        StructuredInput(route="tool", payload=deep)
    with pytest.raises(InvalidStructuredInputError):
        StructuredInput(route="tool", payload=too_many)
    with pytest.raises(InvalidStructuredInputError):
        StructuredInput(route="tool", payload={"value": too_long})


def test_signature_is_deterministic() -> None:
    first = StructuredInput(
        route="capability",
        payload={"metadata": {"b": "2", "a": "1"}, "required_tags": ["demo"]},
        metadata={"z": 1, "a": True},
        request_id="r-1",
    )
    second = StructuredInput(
        route=AtlasRouteType.CAPABILITY,
        payload={"required_tags": ["demo"], "metadata": {"a": "1", "b": "2"}},
        metadata={"a": True, "z": 1},
        request_id="r-1",
    )

    assert atlas_request_classification_signature(first) == atlas_request_classification_signature(second)


def test_classifier_does_not_call_adapter() -> None:
    adapter = CountingAdapter()
    result = AtlasRequestClassifier().classify(StructuredInput(route="tool", payload={"safe": "value"}))

    assert result.status is AtlasRequestClassificationStatus.CLASSIFIED
    assert adapter.calls == 0


def test_route_structured_input_calls_adapter_and_router_once() -> None:
    adapter = CountingAdapter()
    router = CountingRouter()
    orchestrator, agent = _atlas_orchestrator(
        classifier=AtlasRequestClassifier(),
        adapter=adapter,
        router=router,
    )

    result = orchestrator.route_structured_input(StructuredInput(route="tool", payload={"safe": "value"}))

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert adapter.calls == 1
    assert router.calls == 1
    assert agent.calls == 0


def test_route_structured_input_without_classifier_is_compatible() -> None:
    adapter = CountingAdapter()
    router = CountingRouter()
    orchestrator, _agent = _atlas_orchestrator(classifier=None, adapter=adapter, router=router)

    result = orchestrator.route_structured_input(StructuredInput(route="tool"))

    assert result.status is AtlasRoutingStatus.SERVICE_UNAVAILABLE
    assert result.error_code == "ATLAS_REQUEST_CLASSIFIER_UNAVAILABLE"
    assert adapter.calls == 0
    assert router.calls == 0


def test_route_structured_input_without_normalizer_is_compatible() -> None:
    adapter = CountingAdapter()
    router = CountingRouter()
    orchestrator, _agent = _atlas_orchestrator(
        classifier=AtlasRequestClassifier(),
        adapter=adapter,
        router=router,
        normalizer=None,
    )

    result = orchestrator.route_structured_input(StructuredInput(route="tool"))

    assert result.status is AtlasRoutingStatus.SERVICE_UNAVAILABLE
    assert result.error_code == "ATLAS_REQUEST_NORMALIZER_UNAVAILABLE"
    assert adapter.calls == 0
    assert router.calls == 0


def test_bootstrap_injects_one_classifier_instance() -> None:
    orchestrator = Bootstrap.build()

    assert isinstance(orchestrator._atlas_request_classifier, AtlasRequestClassifier)  # type: ignore[attr-defined]
    assert isinstance(orchestrator._atlas_request_normalizer, AtlasRequestNormalizer)  # type: ignore[attr-defined]


def test_route_structured_input_unknown_does_not_call_adapter_or_router() -> None:
    adapter = CountingAdapter()
    router = CountingRouter()
    orchestrator, _agent = _atlas_orchestrator(
        classifier=AtlasRequestClassifier(),
        adapter=adapter,
        router=router,
    )

    result = orchestrator.route_structured_input(StructuredInput(payload={"safe": "value"}))

    assert result.status is AtlasRoutingStatus.INVALID_REQUEST
    assert result.error_code == "UNKNOWN_ROUTE"
    assert adapter.calls == 0
    assert router.calls == 0


def test_route_capability_e2e_from_structured_input() -> None:
    tool = SpyTool(output={"final": "ok"})
    registry = _registry(tool)
    library = ExecutionPlanLibrary("atlas.test", (_workflow(_plan()),), version="1.0")
    service = _real_capability_service(registry, library)
    router = build_core_atlas_router(capability_execution_service=service)
    orchestrator, _agent = _atlas_orchestrator(
        classifier=build_core_atlas_request_classifier(),
        adapter=build_core_atlas_request_adapter(),
        router=router,
    )

    result = orchestrator.route_structured_input(
        StructuredInput(
            capability_id="workflow.atlas.test.workflow.demo.1.0",
            payload={"preferred_workflow_reference": {"plan_id": "workflow.demo", "version": "1.0"}},
            request_id="r-cap",
        )
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert dict(result.output) == {"final": "ok"}
    assert tool.calls == 1


@pytest.mark.parametrize("route", ("tool", "workflow", "conversation"))
def test_unsupported_routes_remain_not_executable(route: str) -> None:
    router = build_core_atlas_router(capability_execution_service=None)
    orchestrator, _agent = _atlas_orchestrator(
        classifier=AtlasRequestClassifier(),
        adapter=AtlasRequestAdapter(),
        router=router,
    )

    result = orchestrator.route_structured_input(StructuredInput(route=route, payload={"safe": "value"}))

    assert result.status is AtlasRoutingStatus.ROUTE_UNAVAILABLE


def test_agent_route_with_unsupported_payload_through_classifier_flow_is_invalid() -> None:
    router = build_core_atlas_router(capability_execution_service=None)
    orchestrator, _agent = _atlas_orchestrator(
        classifier=AtlasRequestClassifier(),
        adapter=AtlasRequestAdapter(),
        router=router,
    )

    result = orchestrator.route_structured_input(StructuredInput(route="agent", payload={"safe": "value"}))

    assert result.status is AtlasRoutingStatus.INVALID_REQUEST
    assert result.error_code == "INVALID_AGENT_PAYLOAD"


def test_invalid_input_object_returns_structured_failure() -> None:
    result = AtlasRequestClassifier().classify(object())  # type: ignore[arg-type]

    assert result.status is AtlasRequestClassificationStatus.INVALID_INPUT
    assert result.error_code == "INVALID_INPUT"

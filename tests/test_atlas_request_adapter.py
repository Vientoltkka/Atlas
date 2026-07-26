from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from bootstrap.atlas_request_adapter import build_core_atlas_request_adapter
from bootstrap.atlas_router import build_core_atlas_router
from bootstrap.bootstrap import Bootstrap
from bootstrap.capability_execution_service import build_capability_execution_service
from bootstrap.capability_orchestrator import build_core_capability_orchestrator
from bootstrap.capability_planner import build_core_capability_planner
from bootstrap.capability_resolver import build_core_capability_resolver
from bootstrap.workflow_selector import build_core_workflow_selector
from core.atlas_request_adapter import (
    AtlasRequestAdaptationStatus,
    AtlasRequestAdapter,
    InvalidStructuredAtlasRequestError,
    StructuredAtlasRequest,
    structured_atlas_request_signature,
)
from core.atlas_router import AtlasRouteType, AtlasRoutingResult, AtlasRoutingStatus
from core.capability_execution_service import (
    CapabilityExecutionRequest,
    CapabilityExecutionResult,
    CapabilityExecutionService,
    CapabilityExecutionStatus,
)
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
        return "Safe request adapter test tool."

    def execute(self, context: ToolContext) -> Any:
        del context
        self.calls += 1
        return self._output


class CountingRouter:
    def __init__(self, result: AtlasRoutingResult | None = None) -> None:
        self.result = result or AtlasRoutingResult(
            AtlasRoutingStatus.COMPLETED,
            AtlasRouteType.CONVERSATION,
            output={"ok": True},
        )
        self.calls = 0
        self.requests = []

    def route(self, request):
        self.calls += 1
        self.requests.append(request)
        return self.result


class CountingCapabilityService(CapabilityExecutionService):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: CapabilityExecutionRequest) -> CapabilityExecutionResult:
        del request
        self.calls += 1
        return CapabilityExecutionResult(CapabilityExecutionStatus.COMPLETED, output={"ok": True})


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
        description="Safe workflow for request adapter tests.",
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


def _atlas_orchestrator(*, adapter=None, router=None):
    agent = ChatAgent()
    registry = SimpleNamespace(get=lambda name: agent if name == "chat" else None)
    return AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=registry,
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        atlas_request_adapter=adapter,
        atlas_router=router,
    ), agent


@pytest.mark.parametrize("route_type", tuple(AtlasRouteType))
def test_valid_adaptation_for_each_route_type(route_type: AtlasRouteType) -> None:
    payload = {"objective": "Execute demo"} if route_type is AtlasRouteType.CAPABILITY else {"safe": "value"}
    result = AtlasRequestAdapter().adapt(StructuredAtlasRequest(route_type, payload=payload, request_id="r-1"))

    assert result.status is AtlasRequestAdaptationStatus.ADAPTED
    assert result.routing_request is not None
    assert result.routing_request.route_type is route_type
    assert result.routing_request.request_id == "r-1"


def test_route_type_enum_and_valid_string_are_normalized() -> None:
    enum_result = AtlasRequestAdapter().adapt(StructuredAtlasRequest(AtlasRouteType.TOOL))
    string_result = AtlasRequestAdapter().adapt(StructuredAtlasRequest("CAPABILITY", payload={"objective": "Run"}))

    assert enum_result.routing_request is not None
    assert enum_result.routing_request.route_type is AtlasRouteType.TOOL
    assert string_result.routing_request is not None
    assert string_result.routing_request.route_type is AtlasRouteType.CAPABILITY


@pytest.mark.parametrize("route_type", ("", "missing"))
def test_invalid_route_type_returns_structured_failure(route_type: str) -> None:
    result = AtlasRequestAdapter().adapt(StructuredAtlasRequest(route_type))

    assert result.status is AtlasRequestAdaptationStatus.INVALID_ROUTE_TYPE
    assert result.routing_request is None
    assert result.error_code == "INVALID_ROUTE_TYPE"


def test_absent_payload_and_nested_payload_are_valid() -> None:
    absent = AtlasRequestAdapter().adapt(StructuredAtlasRequest(AtlasRouteType.CONVERSATION))
    nested = AtlasRequestAdapter().adapt(
        StructuredAtlasRequest(AtlasRouteType.TOOL, payload={"a": {"b": [1, True, None]}})
    )

    assert absent.status is AtlasRequestAdaptationStatus.ADAPTED
    assert absent.routing_request is not None
    assert absent.routing_request.payload is None
    assert nested.status is AtlasRequestAdaptationStatus.ADAPTED
    assert nested.routing_request is not None
    assert nested.routing_request.payload["a"]["b"] == (1, True, None)  # type: ignore[index]


def test_payload_and_metadata_are_copied_defensively() -> None:
    payload = {"items": ["a"]}
    metadata = {"trace": "one"}
    request = StructuredAtlasRequest(AtlasRouteType.TOOL, payload=payload, metadata=metadata)
    payload["items"].append("b")
    metadata["trace"] = "two"

    assert isinstance(request.payload, MappingProxyType)
    assert request.payload["items"] == ("a",)  # type: ignore[index]
    assert request.metadata["trace"] == "one"  # type: ignore[index]
    with pytest.raises(TypeError):
        request.payload["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_rejects_nan_and_infinity(bad: float) -> None:
    with pytest.raises(InvalidStructuredAtlasRequestError):
        StructuredAtlasRequest(AtlasRouteType.TOOL, payload={"bad": bad})


@pytest.mark.parametrize("bad", (object(), lambda: None, AtlasRequestAdapter))
def test_rejects_arbitrary_objects(bad: object) -> None:
    with pytest.raises(InvalidStructuredAtlasRequestError):
        StructuredAtlasRequest(AtlasRouteType.TOOL, payload={"bad": bad})


def test_rejects_depth_node_collection_and_string_limits() -> None:
    deep = {"x": {"x": {"x": {"x": {"x": {"x": {"x": {"x": {"x": "too deep"}}}}}}}}}
    too_many_items = {f"k{i}": i for i in range(65)}
    too_long = "x" * 501

    with pytest.raises(InvalidStructuredAtlasRequestError):
        StructuredAtlasRequest(AtlasRouteType.TOOL, payload=deep)
    with pytest.raises(InvalidStructuredAtlasRequestError):
        StructuredAtlasRequest(AtlasRouteType.TOOL, payload=too_many_items)
    with pytest.raises(InvalidStructuredAtlasRequestError):
        StructuredAtlasRequest(AtlasRouteType.TOOL, payload={"value": too_long})


def test_signature_is_deterministic_with_mapping_order() -> None:
    first = StructuredAtlasRequest(
        "capability",
        payload={"required_tags": ["demo"], "metadata": {"b": "2", "a": "1"}},
        request_id="r-1",
        metadata={"z": 1, "a": True},
    )
    second = StructuredAtlasRequest(
        AtlasRouteType.CAPABILITY,
        payload={"metadata": {"a": "1", "b": "2"}, "required_tags": ["demo"]},
        request_id="r-1",
        metadata={"a": True, "z": 1},
    )

    assert structured_atlas_request_signature(first) == structured_atlas_request_signature(second)


def test_adapter_does_not_call_router() -> None:
    router = CountingRouter()
    result = AtlasRequestAdapter().adapt(StructuredAtlasRequest(AtlasRouteType.TOOL, payload={"safe": "value"}))

    assert result.status is AtlasRequestAdaptationStatus.ADAPTED
    assert router.calls == 0


def test_route_structured_request_routes_once_after_successful_adaptation() -> None:
    router = CountingRouter()
    orchestrator, agent = _atlas_orchestrator(adapter=AtlasRequestAdapter(), router=router)

    result = orchestrator.route_structured_request(
        StructuredAtlasRequest(AtlasRouteType.TOOL, payload={"safe": "value"}, request_id="r-1")
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert router.calls == 1
    assert router.requests[0].route_type is AtlasRouteType.TOOL
    assert agent.calls == 0


def test_failed_adaptation_does_not_call_router() -> None:
    router = CountingRouter()
    orchestrator, _agent = _atlas_orchestrator(adapter=AtlasRequestAdapter(), router=router)

    result = orchestrator.route_structured_request(StructuredAtlasRequest("missing", request_id="r-1"))

    assert result.status is AtlasRoutingStatus.INVALID_REQUEST
    assert result.error_code == "INVALID_ROUTE_TYPE"
    assert router.calls == 0


def test_missing_adapter_is_compatible_and_does_not_call_router() -> None:
    router = CountingRouter()
    orchestrator, _agent = _atlas_orchestrator(adapter=None, router=router)

    result = orchestrator.route_structured_request(StructuredAtlasRequest(AtlasRouteType.TOOL, request_id="r-1"))

    assert result.status is AtlasRoutingStatus.SERVICE_UNAVAILABLE
    assert result.error_code == "ATLAS_REQUEST_ADAPTER_UNAVAILABLE"
    assert router.calls == 0


def test_bootstrap_injects_one_adapter_instance() -> None:
    orchestrator = Bootstrap.build()

    assert isinstance(orchestrator._atlas_request_adapter, AtlasRequestAdapter)  # type: ignore[attr-defined]


def test_capability_route_e2e_from_structured_request_to_completed() -> None:
    tool = SpyTool(output={"final": "ok"})
    registry = _registry(tool)
    library = ExecutionPlanLibrary("atlas.test", (_workflow(_plan()),), version="1.0")
    service = _real_capability_service(registry, library)
    router = build_core_atlas_router(capability_execution_service=service)
    orchestrator, _agent = _atlas_orchestrator(adapter=build_core_atlas_request_adapter(), router=router)

    result = orchestrator.route_structured_request(
        StructuredAtlasRequest(
            "CAPABILITY",
            payload={
                "capability_id": "workflow.atlas.test.workflow.demo.1.0",
                "preferred_workflow_reference": {"plan_id": "workflow.demo", "version": "1.0"},
            },
            request_id="r-cap",
        )
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert dict(result.output) == {"final": "ok"}
    assert tool.calls == 1


@pytest.mark.parametrize(
    "route_type",
    (AtlasRouteType.CONVERSATION, AtlasRouteType.TOOL, AtlasRouteType.WORKFLOW),
)
def test_non_capability_routes_still_do_not_execute(route_type: AtlasRouteType) -> None:
    router = build_core_atlas_router(capability_execution_service=None)
    orchestrator, _agent = _atlas_orchestrator(adapter=AtlasRequestAdapter(), router=router)

    result = orchestrator.route_structured_request(StructuredAtlasRequest(route_type, payload={"safe": "value"}))

    assert result.status is AtlasRoutingStatus.ROUTE_UNAVAILABLE
    assert result.error_code == "ROUTE_UNAVAILABLE"


def test_agent_route_with_unsupported_payload_through_adapter_is_invalid() -> None:
    router = build_core_atlas_router(capability_execution_service=None)
    orchestrator, _agent = _atlas_orchestrator(adapter=AtlasRequestAdapter(), router=router)

    result = orchestrator.route_structured_request(
        StructuredAtlasRequest(AtlasRouteType.AGENT, payload={"safe": "value"})
    )

    assert result.status is AtlasRoutingStatus.INVALID_REQUEST
    assert result.error_code == "INVALID_AGENT_PAYLOAD"


def test_unknown_route_does_not_execute_capability_and_invalid_route_does_not_route() -> None:
    service = CountingCapabilityService()
    router = build_core_atlas_router(capability_execution_service=service)
    orchestrator, _agent = _atlas_orchestrator(adapter=AtlasRequestAdapter(), router=router)

    unknown = orchestrator.route_structured_request(StructuredAtlasRequest(AtlasRouteType.UNKNOWN))
    invalid = orchestrator.route_structured_request(StructuredAtlasRequest("not-a-route"))

    assert unknown.status is AtlasRoutingStatus.UNKNOWN_ROUTE
    assert service.calls == 0
    assert invalid.status is AtlasRoutingStatus.INVALID_REQUEST
    assert service.calls == 0


def test_secrets_and_payloads_are_not_exposed_in_failures() -> None:
    with pytest.raises(InvalidStructuredAtlasRequestError):
        StructuredAtlasRequest(AtlasRouteType.TOOL, payload={"api_key": "secret"})

    result = AtlasRequestAdapter().adapt(StructuredAtlasRequest("not-a-route", payload={"safe": "value"}))
    routed = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=SimpleNamespace(get=lambda _name: None),
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        atlas_request_adapter=AtlasRequestAdapter(),
        atlas_router=CountingRouter(),
    ).route_structured_request(StructuredAtlasRequest("not-a-route", payload={"safe": "value"}))

    assert result.message == "route_type is invalid."
    assert "safe" not in repr(result)
    assert "secret" not in repr(routed)

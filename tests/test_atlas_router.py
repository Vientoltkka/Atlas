from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from bootstrap.atlas_router import build_core_atlas_router
from bootstrap.capability_execution_service import build_capability_execution_service
from bootstrap.capability_orchestrator import build_core_capability_orchestrator
from bootstrap.capability_planner import build_core_capability_planner
from bootstrap.capability_resolver import build_core_capability_resolver
from bootstrap.workflow_selector import build_core_workflow_selector
from core.atlas_router import (
    AtlasRouteType,
    AtlasRouter,
    AtlasRoutingRequest,
    AtlasRoutingResult,
    AtlasRoutingStatus,
    InvalidAtlasRoutingRequestError,
    atlas_routing_request_signature,
)
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
    def __init__(self, output: Any = "ok", *, fail: bool = False) -> None:
        self._output = output
        self._fail = fail
        self.calls = 0

    @property
    def name(self) -> str:
        return "demo.tool"

    @property
    def description(self) -> str:
        return "Safe Atlas router test tool."

    def execute(self, context: ToolContext) -> Any:
        del context
        self.calls += 1
        if self._fail:
            raise RuntimeError("tool failed")
        return self._output


class CountingCapabilityService(CapabilityExecutionService):
    def __init__(
        self,
        result: CapabilityExecutionResult | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.result = result or CapabilityExecutionResult(
            CapabilityExecutionStatus.COMPLETED,
            output={"ok": True},
        )
        self.fail = fail
        self.calls = 0
        self.requests: list[CapabilityExecutionRequest] = []

    def execute(self, request: CapabilityExecutionRequest) -> CapabilityExecutionResult:
        self.calls += 1
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("api_token=secret")
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
        description="Safe workflow for Atlas router tests.",
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


def _atlas_orchestrator(atlas_router=None):
    agent = ChatAgent()
    registry = SimpleNamespace(get=lambda name: agent if name == "chat" else None)
    return AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=registry,
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        atlas_router=atlas_router,
    ), agent


def test_capability_route_completed_delegates_once_and_returns_safe_output() -> None:
    capability_request = CapabilityExecutionRequest(metadata={"source": "test"})
    service = CountingCapabilityService(
        CapabilityExecutionResult(
            CapabilityExecutionStatus.COMPLETED,
            output={"public": "ok", "api_token": "hidden"},
            message="done",
        )
    )
    router = AtlasRouter(service)
    result = router.route(
        AtlasRoutingRequest(
            AtlasRouteType.CAPABILITY,
            payload=capability_request,
            request_id="request-1",
            metadata={"trace": "safe"},
        )
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.route_type is AtlasRouteType.CAPABILITY
    assert dict(result.output) == {"public": "ok", "api_token": "[redacted]"}
    assert result.capability_result is service.result
    assert result.request_id == "request-1"
    assert service.calls == 1
    assert service.requests == [capability_request]


def test_capability_non_completed_status_is_preserved_in_capability_result() -> None:
    service = CountingCapabilityService(
        CapabilityExecutionResult(
            CapabilityExecutionStatus.PLAN_VALIDATION_FAILED,
            error_code="PLAN_VALIDATION_FAILED",
            message="Selected capability plan did not pass validation.",
        )
    )

    result = AtlasRouter(service).route(
        AtlasRoutingRequest(AtlasRouteType.CAPABILITY, CapabilityExecutionRequest())
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.capability_result is not None
    assert result.capability_result.status is CapabilityExecutionStatus.PLAN_VALIDATION_FAILED
    assert result.error_code == "PLAN_VALIDATION_FAILED"


def test_missing_capability_service_and_wrong_payload_are_structured_results() -> None:
    missing = AtlasRouter().route(
        AtlasRoutingRequest(AtlasRouteType.CAPABILITY, CapabilityExecutionRequest())
    )
    wrong_payload = AtlasRouter(CountingCapabilityService()).route(
        AtlasRoutingRequest(AtlasRouteType.CAPABILITY, payload={"kind": "capability"})
    )

    assert missing.status is AtlasRoutingStatus.SERVICE_UNAVAILABLE
    assert missing.error_code == "CAPABILITY_EXECUTION_SERVICE_UNAVAILABLE"
    assert wrong_payload.status is AtlasRoutingStatus.INVALID_REQUEST
    assert wrong_payload.error_code == "INVALID_CAPABILITY_PAYLOAD"


@pytest.mark.parametrize(
    "route_type",
    (
        AtlasRouteType.CONVERSATION,
        AtlasRouteType.TOOL,
        AtlasRouteType.WORKFLOW,
        AtlasRouteType.AGENT,
    ),
)
def test_unavailable_routes_do_not_call_capability_service(route_type: AtlasRouteType) -> None:
    service = CountingCapabilityService()
    result = AtlasRouter(service).route(AtlasRoutingRequest(route_type, payload={"safe": "value"}))

    assert result.status is AtlasRoutingStatus.ROUTE_UNAVAILABLE
    assert result.error_code == "ROUTE_UNAVAILABLE"
    assert service.calls == 0


def test_unknown_route_and_invalid_request_are_structured() -> None:
    router = AtlasRouter(CountingCapabilityService())

    unknown = router.route(AtlasRoutingRequest(AtlasRouteType.UNKNOWN))
    invalid = router.route(object())  # type: ignore[arg-type]

    assert unknown.status is AtlasRoutingStatus.UNKNOWN_ROUTE
    assert unknown.error_code == "UNKNOWN_ROUTE"
    assert invalid.status is AtlasRoutingStatus.INVALID_REQUEST
    assert invalid.route_type is AtlasRouteType.UNKNOWN


def test_invalid_payload_metadata_and_sensitive_values_are_rejected() -> None:
    with pytest.raises(InvalidAtlasRoutingRequestError):
        AtlasRoutingRequest("missing")
    with pytest.raises(InvalidAtlasRoutingRequestError):
        AtlasRoutingRequest(AtlasRouteType.TOOL, payload={"api_key": "secret"})
    with pytest.raises(InvalidAtlasRoutingRequestError):
        AtlasRoutingRequest(AtlasRouteType.TOOL, metadata={"credential": "secret"})
    with pytest.raises(InvalidAtlasRoutingRequestError):
        AtlasRoutingRequest(AtlasRouteType.TOOL, payload=object())


def test_service_exception_returns_sanitized_internal_error() -> None:
    result = AtlasRouter(CountingCapabilityService(fail=True)).route(
        AtlasRoutingRequest(AtlasRouteType.CAPABILITY, CapabilityExecutionRequest())
    )

    assert result.status is AtlasRoutingStatus.INTERNAL_ERROR
    assert result.error_code == "INTERNAL_ERROR"
    assert "secret" not in repr(result)
    assert "api_token" not in repr(result)


def test_signature_is_deterministic_and_uses_safe_structure() -> None:
    first = AtlasRoutingRequest(
        "capability",
        payload=CapabilityExecutionRequest(required_tags=("demo",), metadata={"b": "2", "a": "1"}),
        request_id="abc",
        metadata={"z": 1, "a": True},
    )
    second = AtlasRoutingRequest(
        AtlasRouteType.CAPABILITY,
        payload=CapabilityExecutionRequest(required_tags=("demo",), metadata={"a": "1", "b": "2"}),
        request_id="abc",
        metadata={"a": True, "z": 1},
    )
    changed = AtlasRoutingRequest(
        AtlasRouteType.CAPABILITY,
        payload=CapabilityExecutionRequest(required_tags=("demo",)),
        request_id="other",
        metadata={"a": True, "z": 1},
    )

    assert atlas_routing_request_signature(first) == atlas_routing_request_signature(second)
    assert atlas_routing_request_signature(first) != atlas_routing_request_signature(changed)


def test_request_and_result_copy_payloads_defensively() -> None:
    payload = {"items": ["a"]}
    request = AtlasRoutingRequest(AtlasRouteType.TOOL, payload=payload)
    payload["items"].append("b")
    result = AtlasRoutingResult(AtlasRoutingStatus.ROUTE_UNAVAILABLE, AtlasRouteType.TOOL, output={"items": ["x"]})

    assert request.payload["items"] == ("a",)  # type: ignore[index]
    with pytest.raises(TypeError):
        request.payload["new"] = "value"  # type: ignore[index]
    assert result.output["items"] == ("x",)  # type: ignore[index]


def test_bootstrap_factory_and_orchestrator_optional_integration() -> None:
    service = CountingCapabilityService()
    router = build_core_atlas_router(capability_execution_service=service)
    orchestrator, agent = _atlas_orchestrator(router)
    request = AtlasRoutingRequest(AtlasRouteType.CAPABILITY, CapabilityExecutionRequest())

    routed = orchestrator.route_request(request)
    chat = orchestrator.process_prompt("hola", confirm=lambda _prompt: "")

    assert isinstance(router, AtlasRouter)
    assert routed.status is AtlasRoutingStatus.COMPLETED
    assert service.calls == 1
    assert chat == "respuesta anterior"
    assert agent.calls == 1


def test_orchestrator_without_atlas_router_returns_service_unavailable() -> None:
    orchestrator, agent = _atlas_orchestrator(None)
    result = orchestrator.route_request(
        AtlasRoutingRequest(AtlasRouteType.CAPABILITY, CapabilityExecutionRequest(), request_id="r-1")
    )

    assert result.status is AtlasRoutingStatus.SERVICE_UNAVAILABLE
    assert result.route_type is AtlasRouteType.CAPABILITY
    assert result.request_id == "r-1"
    assert result.error_code == "ATLAS_ROUTER_UNAVAILABLE"
    assert agent.calls == 0


def test_e2e_capability_route_runs_real_chain_to_completed() -> None:
    tool = SpyTool(output={"final": "ok"})
    registry = _registry(tool)
    library = ExecutionPlanLibrary("atlas.test", (_workflow(_plan()),), version="1.0")
    service = _real_capability_service(registry, library)
    router = build_core_atlas_router(capability_execution_service=service)

    result = router.route(
        AtlasRoutingRequest(
            AtlasRouteType.CAPABILITY,
            CapabilityExecutionRequest(
                capability_id="workflow.atlas.test.workflow.demo.1.0",
                preferred_workflow_reference=ExecutionPlanReference("workflow.demo", "1.0"),
            ),
        )
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert dict(result.output) == {"final": "ok"}
    assert result.capability_result is not None
    assert result.capability_result.execution_status == "completed"
    assert tool.calls == 1

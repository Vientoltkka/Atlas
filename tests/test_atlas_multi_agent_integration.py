from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

from bootstrap.agent_system import build_core_agent_system
from bootstrap.atlas_router import build_core_atlas_router
from core.agent_context import AgentContext
from core.agent_executor import AgentExecutionStatus
from core.agent_registry import AgentCapabilities, AgentContextPolicy, AgentDefinition, AgentPermissions, AgentType
from core.atlas_request_adapter import AtlasRequestAdapter
from core.atlas_request_classifier import AtlasRequestClassifier, StructuredInput
from core.atlas_request_normalizer import AtlasRequestNormalizer
from core.atlas_router import AtlasAgentRoutingRequest, AtlasRouteType, AtlasRouter, AtlasRoutingRequest, AtlasRoutingStatus
from core.multi_agent import MultiAgentExecutionStatus, MultiAgentFailurePolicy
from core.orchestrator import AtlasOrchestrator
from core.router import Router
from memory.conversation import ConversationMemory


class Handler:
    calls: list[str] = []

    def __init__(self, agent_id: str, *, fail: bool = False) -> None:
        self._agent_id = agent_id
        self.fail = fail

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        Handler.calls.append(self._agent_id)
        if self.fail:
            raise RuntimeError("password leaked")
        return {"agent_id": self._agent_id, "value": context.structured_input.get("value", "ok")}


class ChatAgent:
    name = "chat"
    generated_path = None

    def run(self, *, model: str, messages):
        del model, messages
        return "chat"


def _definition(agent_id: str, *, capabilities: tuple[str, ...] = ("agent.inspect",)) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=AgentType.GENERAL,
        name=agent_id,
        description="Atlas multi-agent integration test agent.",
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_shared_context=True),
    )


def _system(definitions: tuple[AgentDefinition, ...], handlers: tuple[Handler, ...]):
    result = build_core_agent_system()
    assert result.system is not None
    system = result.system
    for definition in definitions:
        system.agent_registry.register(definition)
    for handler in handlers:
        system.agent_handler_registry.register(handler)
    return system


def _orchestrator(router: AtlasRouter) -> AtlasOrchestrator:
    chat = ChatAgent()
    registry = SimpleNamespace(get=lambda name: chat if name == "chat" else None)
    return AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=registry,
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        atlas_router=router,
        atlas_request_adapter=AtlasRequestAdapter(),
        atlas_request_classifier=AtlasRequestClassifier(),
        atlas_request_normalizer=AtlasRequestNormalizer(),
    )


def setup_function() -> None:
    Handler.calls = []


def test_multi_agent_enabled_routes_to_coordinator_and_aggregates_output() -> None:
    router = build_core_atlas_router(
        agent_system=_system(
            (_definition("agent.a"), _definition("agent.b")),
            (Handler("agent.a"), Handler("agent.b")),
        )
    )

    result = router.route(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest(
                multi_agent_enabled=True,
                required_agent_ids=("agent.a", "agent.b"),
                payload={"value": "ok"},
                min_agents=2,
                max_agents=2,
            ),
        )
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.multi_agent_result is not None
    assert result.multi_agent_result.status is MultiAgentExecutionStatus.SUCCESS
    assert result.output["team"] == ("agent.a", "agent.b")  # type: ignore[index]
    assert Handler.calls == ["agent.a", "agent.b"]


def test_multi_agent_enabled_false_preserves_single_agent_automatic_selection() -> None:
    router = build_core_atlas_router(
        agent_system=_system(
            (_definition("agent.a", capabilities=("code.edit",)), _definition("agent.b")),
            (Handler("agent.a"), Handler("agent.b")),
        )
    )

    result = router.route(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest(required_capabilities=("code.edit",), multi_agent_enabled=False),
        )
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.agent_result is not None
    assert result.multi_agent_result is None
    assert Handler.calls == ["agent.a"]


def test_explicit_agent_id_preserves_single_agent_flow_even_when_multi_agent_enabled() -> None:
    router = build_core_atlas_router(
        agent_system=_system(
            (_definition("agent.a"), _definition("agent.b")),
            (Handler("agent.a"), Handler("agent.b")),
        )
    )

    result = router.route(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest(agent_id="agent.a", multi_agent_enabled=True, required_agent_ids=("agent.a", "agent.b")),
        )
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.agent_result is not None
    assert result.multi_agent_result is None
    assert Handler.calls == ["agent.a"]


def test_multi_agent_failures_and_continue_policy_are_structured() -> None:
    router = build_core_atlas_router(
        agent_system=_system(
            (_definition("agent.a"), _definition("agent.b")),
            (Handler("agent.a", fail=True), Handler("agent.b")),
        )
    )

    result = router.route(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest(
                multi_agent_enabled=True,
                required_agent_ids=("agent.a", "agent.b"),
                min_agents=2,
                max_agents=2,
                failure_policy=MultiAgentFailurePolicy.CONTINUE_ON_FAILURE,
            ),
        )
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.multi_agent_result is not None
    assert result.multi_agent_result.status is MultiAgentExecutionStatus.PARTIAL_SUCCESS
    assert Handler.calls == ["agent.a", "agent.b"]
    assert "password" not in repr(result)


def test_missing_team_and_agent_system_absent_are_structured() -> None:
    no_team = build_core_atlas_router(
        agent_system=_system((_definition("agent.a"),), (Handler("agent.a"),))
    ).route(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest(multi_agent_enabled=True, required_agent_ids=("agent.a", "agent.missing")),
        )
    )
    unavailable = AtlasRouter().route(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest(multi_agent_enabled=True, required_agent_ids=("agent.a", "agent.b")),
        )
    )

    assert no_team.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert no_team.multi_agent_result is not None
    assert no_team.multi_agent_result.status is MultiAgentExecutionStatus.NO_MATCHING_TEAM
    assert unavailable.status is AtlasRoutingStatus.SERVICE_UNAVAILABLE
    assert unavailable.error_code == "AGENT_SYSTEM_UNAVAILABLE"


def test_events_metrics_and_shared_agent_system_instances() -> None:
    system = _system((_definition("agent.a"), _definition("agent.b")), (Handler("agent.a"), Handler("agent.b")))
    result = build_core_atlas_router(agent_system=system).route(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest(multi_agent_enabled=True, required_agent_ids=("agent.a", "agent.b")),
        )
    )

    event_names = {event.name for event in result.events}
    assert system.multi_agent_coordinator is not None
    assert result.metrics["multi_agent_executions_requested"] == 1
    assert result.metrics["multi_agent_steps_started"] == 2
    assert "multi_agent_execution_requested" in event_names
    assert "multi_agent_step_succeeded" in event_names
    assert "multi_agent_execution_completed" in event_names


def test_e2e_from_structured_input_to_multi_agent_executor() -> None:
    router = build_core_atlas_router(
        agent_system=_system(
            (_definition("agent.a"), _definition("agent.b")),
            (Handler("agent.a"), Handler("agent.b")),
        )
    )

    result = _orchestrator(router).route_structured_input(
        StructuredInput(
            route="agent",
            payload={
                "multi_agent_enabled": True,
                "required_agent_ids": ("agent.a", "agent.b"),
                "payload": {"value": "e2e"},
                "min_agents": 2,
                "max_agents": 2,
            },
        )
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.multi_agent_result is not None
    assert result.output["outputs"]["agent.a"]["value"] == "e2e"  # type: ignore[index]
    assert Handler.calls == ["agent.a", "agent.b"]

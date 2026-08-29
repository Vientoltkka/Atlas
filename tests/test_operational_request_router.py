from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agents.registry import AgentRegistry
from bootstrap.bootstrap import Bootstrap
from core.operational_request_router import (
    OperationalRequestRouter,
    OperationalRouterConfig,
    RequestRoute,
    RouteDecision,
    RoutingConfigurationError,
    SystemCommand,
    MemoryOperation,
)
from core.orchestrator import AtlasOrchestrator
from core.request_gateway import (
    RequestExecutionContext,
    RequestGateway,
    RequestSource,
    RequestSafetyContext,
)
from core.router import Router
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry


NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _gateway() -> RequestGateway:
    return RequestGateway(clock=lambda: NOW, id_generator=lambda: "request-1")


def _router(
    *,
    tools: ToolRegistry | None = None,
    agents: AgentRegistry | None = None,
    config: OperationalRouterConfig | None = None,
) -> OperationalRequestRouter:
    return OperationalRequestRouter(
        tool_registry=tools,
        agent_registry=agents,
        config=config,
        clock=lambda: NOW,
    )


def _classify(text: str, **kwargs):
    request = _gateway().from_text(text, **kwargs)
    return _router(**kwargs.get("router_kwargs", {})).classify(request)


def test_strength_estimation_request_routes_to_existing_training_agent() -> None:
    agents = AgentRegistry()
    agents.register(SimpleNamespace(name="training"))  # type: ignore[arg-type]

    decision = _router(agents=agents).classify(
        _gateway().from_text(
            "Calcula mi 1RM estimado de back squat si he hecho 5 repeticiones con 100 kg."
        )
    )

    assert decision.route is RequestRoute.AGENT_DELEGATION
    assert decision.target_agent_name == "training"

def test_direct_response_for_simple_question() -> None:
    decision = _router().classify(_gateway().from_text("Que es una API?"))

    assert decision.route is RequestRoute.DIRECT_RESPONSE
    assert decision.confidence == 0.6


def test_memory_query_store_retrieve_and_forget() -> None:
    router = _router()

    store = router.classify(_gateway().from_text("recuerda que entreno a las 18:00"))
    retrieve = router.classify(_gateway().from_text("que recuerdas sobre entrenamiento"))
    forget = router.classify(_gateway().from_text("olvida mi horario"))

    assert store.route is RequestRoute.MEMORY_QUERY
    assert store.memory_operation is MemoryOperation.STORE
    assert retrieve.memory_operation is MemoryOperation.RETRIEVE
    assert forget.memory_operation is MemoryOperation.FORGET


def test_single_tool_uses_only_registered_clear_tool() -> None:
    tools = ToolRegistry()
    tool = _Tool("desktop.open_vscode", "Open VS Code application.")
    tools.register(tool)

    decision = _router(tools=tools).classify(_gateway().from_text("abre vscode"))

    assert decision.route is RequestRoute.SINGLE_TOOL
    assert decision.target_tool_name == "desktop.open_vscode"
    assert tool.calls == 0


def test_calendar_list_is_selected_instead_of_list_directory() -> None:
    decision = _router(tools=Bootstrap.build_tool_registry()).classify(
        _gateway().from_text(
            "Lista eventos del calendario entre "
            "2026-08-09T09:00:00+01:00 y 2026-08-09T10:00:00+01:00"
        )
    )

    assert decision.route is RequestRoute.SINGLE_TOOL
    assert decision.target_tool_name == "calendar_list_events"
    assert decision.target_tool_name != "list_directory"


def test_missing_tool_is_not_single_tool() -> None:
    decision = _router().classify(_gateway().from_text("abre vscode"))

    assert decision.route is RequestRoute.UNSUPPORTED
    assert decision.target_tool_name is None


def test_ambiguous_tools_require_clarification() -> None:
    tools = ToolRegistry()
    tools.register(_Tool("desktop.open_alpha", "Open application."))
    tools.register(_Tool("desktop.open_beta", "Open application."))

    decision = _router(tools=tools).classify(_gateway().from_text("abre aplicacion"))

    assert decision.route is RequestRoute.CLARIFICATION_REQUIRED
    assert decision.requires_clarification is True
    assert "herramienta" in decision.clarification_question.lower()


def test_agent_delegation_uses_only_registered_agent() -> None:
    agents = AgentRegistry()
    agent = _Agent("coding", "Programming and Python code assistant.")
    agents.register(agent)

    decision = _router(agents=agents).classify(_gateway().from_text("ayudame con codigo python"))

    assert decision.route is RequestRoute.AGENT_DELEGATION
    assert decision.target_agent_name == "coding"
    assert agent.calls == 0


def test_missing_or_ambiguous_agent_does_not_invent_target() -> None:
    missing = _router().classify(_gateway().from_text("ayudame con codigo python"))
    agents = AgentRegistry()
    agents.register(_Agent("coding", "Programming assistant."))
    agents.register(_Agent("project", "Programming assistant."))
    ambiguous = _router(agents=agents).classify(_gateway().from_text("ayudame con programming"))

    assert missing.route is RequestRoute.DIRECT_RESPONSE
    assert missing.target_agent_name is None
    assert ambiguous.route is RequestRoute.CLARIFICATION_REQUIRED


def test_autonomous_execution_for_sequential_actions_and_multiple_domains() -> None:
    first = _router().classify(_gateway().from_text("abre vscode y despues ejecuta los tests"))
    second = _router().classify(_gateway().from_text("analiza el archivo y luego prepara una correccion"))

    assert first.route is RequestRoute.AUTONOMOUS_EXECUTION
    assert second.route is RequestRoute.AUTONOMOUS_EXECUTION


def test_long_but_simple_question_is_not_autonomous() -> None:
    text = "Explica con detalle y ejemplos que es una interfaz en programacion orientada a objetos"

    decision = _router().classify(_gateway().from_text(text))

    assert decision.route is RequestRoute.DIRECT_RESPONSE


def test_resume_execution_source_context_and_missing_session() -> None:
    resume = _router().classify(
        _gateway().from_resume("session-1", confirmation_response=True)
    )
    missing = _router().classify(_gateway().from_text("continua la ejecucion"))

    assert resume.route is RequestRoute.RESUME_EXECUTION
    assert resume.target_session_id == "session-1"
    assert missing.route is RequestRoute.CLARIFICATION_REQUIRED
    assert "session_id" in missing.clarification_question


def test_system_commands_are_typed_and_unknown_command_is_not_invented() -> None:
    exit_decision = _router().classify(_gateway().from_text("salir"))
    status_decision = _router().classify(_gateway().from_text("estado"))
    unknown = _router().classify(_gateway().from_text("teletransporta atlas"))

    assert exit_decision.route is RequestRoute.SYSTEM_COMMAND
    assert exit_decision.system_command is SystemCommand.EXIT
    assert status_decision.system_command is SystemCommand.STATUS
    assert unknown.route is RequestRoute.DIRECT_RESPONSE
    assert unknown.system_command is None


def test_unsupported_action_and_clarification_question() -> None:
    unsupported = _router().classify(_gateway().from_text("envia un email a Victor"))
    clarification = _router().classify(_gateway().from_text("abre el archivo"))

    assert unsupported.route is RequestRoute.UNSUPPORTED
    assert unsupported.fallback_route is RequestRoute.DIRECT_RESPONSE
    assert clarification.route is RequestRoute.CLARIFICATION_REQUIRED
    assert clarification.clarification_question


def test_precedence_resume_memory_and_autonomous() -> None:
    tools = ToolRegistry()
    tools.register(_Tool("desktop.open_vscode", "Open VS Code application."))
    resume = _router(tools=tools).classify(
        _gateway().from_resume("session-1", content="confirma la ejecucion session-1")
    )
    memory = _router(tools=tools).classify(
        _gateway().from_text("recuerda que abre vscode por la mañana")
    )
    autonomous = _router(tools=tools).classify(
        _gateway().from_text("abre vscode y despues ejecuta los tests")
    )

    assert resume.route is RequestRoute.RESUME_EXECUTION
    assert memory.route is RequestRoute.MEMORY_QUERY
    assert autonomous.route is RequestRoute.AUTONOMOUS_EXECUTION


def test_safety_flags_source_system_and_confirmation() -> None:
    tools = ToolRegistry()
    tools.register(_Tool("desktop.open_vscode", "Open VS Code application.", requires_confirmation=True))
    request = _gateway().from_system(
        "abre vscode",
        safety_context=RequestSafetyContext(user_present=False),
    )

    decision = _router(tools=tools).classify(request)

    assert decision.route is RequestRoute.SINGLE_TOOL
    assert decision.requires_confirmation is True
    assert "side_effects_disabled" in decision.safety_flags
    assert request.safety_context.trusted_source is False


def test_route_decision_invariants_and_determinism() -> None:
    router = _router()
    request = _gateway().from_text("Que es Atlas?")

    first = router.classify(request)
    second = _router().classify(request)

    assert isinstance(first, RouteDecision)
    assert first == second
    assert first.created_at.tzinfo is not None
    assert tuple(first.matched_rules) == first.matched_rules
    with pytest.raises(AttributeError):
        first.route = RequestRoute.UNSUPPORTED  # type: ignore[misc]


def test_config_disables_routes_and_validates_values() -> None:
    direct_disabled = _router(
        config=OperationalRouterConfig(direct_response_enabled=False)
    ).classify(_gateway().from_text("Que es Atlas?"))
    auto_disabled = _router(
        config=OperationalRouterConfig(autonomous_execution_enabled=False)
    ).classify(_gateway().from_text("abre vscode y despues ejecuta los tests"))

    assert direct_disabled.route is RequestRoute.UNSUPPORTED
    assert auto_disabled.route is not RequestRoute.AUTONOMOUS_EXECUTION
    with pytest.raises(RoutingConfigurationError):
        OperationalRouterConfig(confidence_threshold=2)


def test_events_do_not_contain_full_content() -> None:
    router = _router()
    router.classify(_gateway().from_text("texto privado completo"))

    assert router.events[0].event_type == "request_classification_started"
    assert any(event.event_type == "route_selected" for event in router.events)
    assert all(not hasattr(event, "content") for event in router.events)


def test_router_api_compatibility_and_classification() -> None:
    request = _gateway().from_text("Que es Atlas?")
    router = Router(operational_router=_router())

    assert router.classify_request(request).route is RequestRoute.DIRECT_RESPONSE
    assert router.route(SimpleNamespace(task="chat", objective="hola")) == "chat"
    assert router.route_request(request) == "chat"


def test_orchestrator_can_classify_without_planning_or_execution() -> None:
    class FailingPlanner:
        def create_plan(self, _prompt):  # pragma: no cover
            raise AssertionError("planner must not be called")

    agent = _Agent("chat", "General chat.")
    registry = AgentRegistry()
    registry.register(agent)
    orchestrator = AtlasOrchestrator(
        planner=FailingPlanner(),
        router=Router(operational_router=_router(agents=registry)),
        model_manager=SimpleNamespace(choose_model=lambda _agent: "unused"),
        memory=SimpleNamespace(add_user=lambda _prompt: None, add_assistant=lambda _response: None, history=list),
        registry=registry,
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        request_gateway=_gateway(),
    )

    decision = orchestrator.classify_prompt("Que es Atlas?")

    assert decision.route is RequestRoute.DIRECT_RESPONSE
    assert agent.calls == 0


def test_voice_request_classifies_like_text_and_resume_context_is_not_executed() -> None:
    router = _router()
    text = router.classify(_gateway().from_text("Que es Atlas?"))
    voice = router.classify(_gateway().from_voice("Que es Atlas?", confidence=0.8))
    resume = router.classify(
        _gateway().from_text(
            "confirmo",
            execution_context=RequestExecutionContext(
                session_id="session-1",
                confirmation_response=True,
            ),
        )
    )

    assert voice.route is text.route
    assert resume.route is RequestRoute.RESUME_EXECUTION
    assert resume.target_session_id == "session-1"


class _Tool(BaseTool):
    def __init__(
        self,
        name: str,
        description: str,
        *,
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

    def execute(self, context):
        del context
        self.calls += 1
        return "executed"


class _Agent:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.calls = 0

    def run(self, model, messages):
        del model, messages
        self.calls += 1
        return "agent"

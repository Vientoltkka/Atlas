from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agents.registry import AgentRegistry
from bootstrap.bootstrap import Bootstrap
from core.autonomous_execution import (
    AutonomousExecutionOutcome,
    AutonomousExecutionResult,
    ExecutionTrace,
)
from core.execution_supervisor import ExecutionState
from core.operational_request_router import (
    MemoryOperation,
    OperationalRequestRouter,
    RequestRoute,
    RouteDecision,
    SystemCommand,
)
from core.orchestrator import AtlasOrchestrator
from core.planner import Plan
from core.router import Router
from core.operational_route_executor import (
    ClarificationRouteHandler,
    DirectResponseRouteHandler,
    OperationalRouteExecutor,
    RouteExecutionPresenter,
    RouteExecutionResult,
    RouteExecutionStatus,
    UnsupportedRouteHandler,
    build_default_route_handlers,
)
from core.request_gateway import (
    RequestExecutionContext,
    RequestGateway,
    RequestSafetyContext,
)
from tools.argument_schema import (
    ArgumentField,
    ArgumentSchema,
    ArgumentSchemaRegistry,
    ArgumentValidator,
)
from tools.base_tool import BaseTool
from tools.calendar.calendar_list_events_tool import (
    CALENDAR_LIST_EVENTS_ARGUMENTS_SCHEMA,
)
from tools.executor import ToolExecutor
from tools.intent_selector import ToolIntentRegistry, ToolSelector
from tools.registry import ToolRegistry
from tools.single_tool_runner import SingleToolRunner
from tools.tool_schema import ToolArgumentsSchema, ToolParameterSchema


NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _gateway(request_id: str = "request-1") -> RequestGateway:
    return RequestGateway(clock=lambda: NOW, id_generator=lambda: request_id)


def _decision(
    route: RequestRoute,
    *,
    request_id: str = "request-1",
    **kwargs,
) -> RouteDecision:
    return RouteDecision(
        request_id=request_id,
        route=route,
        confidence=1.0,
        reason=f"test {route.value}",
        matched_rules=(f"test.{route.value}",),
        created_at=NOW,
        **kwargs,
    )


def _completed_tool_result(tool_name: str, output) -> RouteExecutionResult:
    return RouteExecutionResult(
        request_id="request-1",
        route=RequestRoute.SINGLE_TOOL,
        status=RouteExecutionStatus.COMPLETED,
        output=output,
        error=None,
        started_at=NOW,
        finished_at=NOW,
        duration=0.0,
        target_tool_name=tool_name,
    )


def _executor(**overrides) -> OperationalRouteExecutor:
    values = {
        "direct_responder": lambda request: f"respuesta:{request.content}",
        "memory": _Memory(),
        "tool_registry": None,
        "tool_executor": None,
        "single_tool_runner": None,
        "agent_registry": AgentRegistry(),
        "model_selector": lambda name: f"model:{name}",
        "autonomous_orchestrator": None,
        "execution_supervisor": None,
        "clock": lambda: NOW,
    }
    values.update(overrides)
    return OperationalRouteExecutor(
        build_default_route_handlers(**values),
        clock=lambda: NOW,
    )


def _tool_runtime(*, dangerous: bool = False):
    registry = ToolRegistry()
    tool = _Tool("test.tool", dangerous=dangerous)
    schema = ToolArgumentsSchema(
        parameters=(ToolParameterSchema("value", str, required=True),)
    )
    registry.register(tool, arguments_schema=schema)
    intents = ToolIntentRegistry()
    intents.register("test.action", tool.name)
    selector = ToolSelector(registry, intents)
    schemas = ArgumentSchemaRegistry()
    schemas.register(
        ArgumentSchema(
            "test.action",
            (ArgumentField("value", str, required=True),),
        )
    )
    runner = SingleToolRunner(
        selector,
        ArgumentValidator(schemas),
        ToolExecutor(registry),
    )
    return registry, ToolExecutor(registry), runner, tool


def _autonomous_result(
    outcome: AutonomousExecutionOutcome = AutonomousExecutionOutcome.COMPLETED,
    *,
    session_id: str = "session-1",
    requires_confirmation: bool = False,
) -> AutonomousExecutionResult:
    return AutonomousExecutionResult(
        session_id=session_id,
        outcome=outcome,
        final_state=(
            ExecutionState.COMPLETED
            if outcome is AutonomousExecutionOutcome.COMPLETED
            else ExecutionState.WAITING_CONFIRMATION
            if outcome is AutonomousExecutionOutcome.WAITING_CONFIRMATION
            else ExecutionState.FAILED
        ),
        objective="objetivo",
        original_plan=None,
        active_plan=None,
        completed_step_ids=("step-1",)
        if outcome is AutonomousExecutionOutcome.COMPLETED
        else (),
        requires_confirmation=requires_confirmation,
        summary=f"resultado {outcome.value}",
        trace=ExecutionTrace(),
        started_at=NOW,
        finished_at=NOW,
        duration=0.0,
    )


def test_every_request_route_has_exactly_one_explicit_handler() -> None:
    handlers = build_default_route_handlers(clock=lambda: NOW)

    assert frozenset(handlers) == frozenset(RequestRoute)


def test_direct_response_invokes_only_conversational_handler() -> None:
    calls = []
    request = _gateway().from_text("hola")
    executor = _executor(direct_responder=lambda item: calls.append(item) or "ok")

    result = executor.execute(request, _decision(RequestRoute.DIRECT_RESPONSE))

    assert result.status is RouteExecutionStatus.COMPLETED
    assert result.output == "ok"
    assert calls == [request]


def test_phase_17_4_direct_response_streams_fragments_without_changing_route() -> None:
    request = _gateway().from_voice("explica hyrox")
    streamed: list[str] = []

    def stream_response(_request, _context):
        yield "Primera frase. "
        yield "Segunda frase. "
        yield "No debe consumirse."

    executor = _executor(direct_streaming_responder=stream_response)

    result = executor.execute(
        request,
        _decision(RequestRoute.DIRECT_RESPONSE),
        output_fragment_sink=lambda fragment: streamed.append(fragment)
        or fragment != "Segunda frase. ",
    )

    assert result.status is RouteExecutionStatus.COMPLETED
    assert result.route is RequestRoute.DIRECT_RESPONSE
    assert result.output == "Primera frase. Segunda frase. "
    assert streamed == ["Primera frase. ", "Segunda frase. "]


def test_direct_response_failure_is_safe_and_typed() -> None:
    request = _gateway().from_text("hola")

    def fail(_request):
        raise RuntimeError("private stack detail")

    result = _executor(direct_responder=fail).execute(
        request,
        _decision(RequestRoute.DIRECT_RESPONSE),
    )

    assert result.status is RouteExecutionStatus.FAILED
    assert result.error.code == "OPERATIONAL_ROUTE_EXECUTION_ERROR"
    assert "private stack detail" not in RouteExecutionPresenter().present(result)


def test_memory_store_and_retrieve_use_existing_memory() -> None:
    memory = _Memory()
    store_request = _gateway("store-1").from_text(
        "recuerda que prefiero respuestas breves",
        request_id="store-1",
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )
    store = _executor(memory=memory).execute(
        store_request,
        _decision(
            RequestRoute.MEMORY_QUERY,
            request_id="store-1",
            memory_operation=MemoryOperation.STORE,
        ),
    )
    retrieve_request = _gateway("retrieve-1").from_text(
        "que recuerdas",
        request_id="retrieve-1",
    )
    retrieve = _executor(memory=memory).execute(
        retrieve_request,
        _decision(
            RequestRoute.MEMORY_QUERY,
            request_id="retrieve-1",
            memory_operation=MemoryOperation.RETRIEVE,
        ),
    )

    assert store.status is RouteExecutionStatus.COMPLETED
    assert store.side_effects_performed is True
    assert memory.items == ["prefiero respuestas breves"]
    assert retrieve.status is RouteExecutionStatus.COMPLETED
    assert retrieve.output["items"][0] == "prefiero respuestas breves"


def test_memory_unsupported_operation_does_not_simulate_success() -> None:
    request = _gateway().from_text(
        "olvida algo",
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )
    result = _executor().execute(
        request,
        _decision(
            RequestRoute.MEMORY_QUERY,
            memory_operation=MemoryOperation.FORGET,
        ),
    )

    assert result.status is RouteExecutionStatus.UNSUPPORTED
    assert result.side_effects_performed is False


def test_single_tool_executes_registered_tool_with_validated_metadata_arguments() -> None:
    registry, tool_executor, runner, tool = _tool_runtime()
    request = _gateway().from_text(
        "ejecuta",
        metadata={"tool_arguments": {"value": "real"}},
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )
    result = _executor(
        tool_registry=registry,
        tool_executor=tool_executor,
        single_tool_runner=runner,
    ).execute(
        request,
        _decision(RequestRoute.SINGLE_TOOL, target_tool_name=tool.name),
    )

    assert result.status is RouteExecutionStatus.COMPLETED
    assert result.output == "tool:real"
    assert tool.calls == 1


def test_single_tool_missing_arguments_requires_clarification() -> None:
    registry, tool_executor, runner, tool = _tool_runtime()
    request = _gateway().from_text("ejecuta")

    result = _executor(
        tool_registry=registry,
        tool_executor=tool_executor,
        single_tool_runner=runner,
    ).execute(
        request,
        _decision(RequestRoute.SINGLE_TOOL, target_tool_name=tool.name),
    )

    assert result.status is RouteExecutionStatus.CLARIFICATION_REQUIRED
    assert result.requires_clarification is True
    assert tool.calls == 0


def test_calendar_single_tool_executes_with_normalized_arguments() -> None:
    registry = ToolRegistry()
    tool = _CalendarTool()
    registry.register(tool, arguments_schema=CALENDAR_LIST_EVENTS_ARGUMENTS_SCHEMA)
    tool_executor = ToolExecutor(registry)
    schema_registry = Bootstrap.build_argument_schema_registry()
    runner = SingleToolRunner(
        Bootstrap.build_tool_selector(registry),
        ArgumentValidator(schema_registry),
        tool_executor,
    )
    request = _gateway().from_text(
        "Lista eventos del calendario entre "
        "2026-08-09T09:00:00+01:00 y 2026-08-09T10:00:00+01:00 "
        "max_results=3"
    )

    result = _executor(
        tool_registry=registry,
        tool_executor=tool_executor,
        single_tool_runner=runner,
    ).execute(
        request,
        _decision(RequestRoute.SINGLE_TOOL, target_tool_name=tool.name),
    )

    assert result.status is RouteExecutionStatus.COMPLETED
    assert result.output == {
        "time_min": "2026-08-09T09:00:00+01:00",
        "time_max": "2026-08-09T10:00:00+01:00",
        "max_results": 3,
    }
    assert tool.calls == 1


def test_calendar_single_tool_without_range_requires_clarification() -> None:
    registry = ToolRegistry()
    tool = _CalendarTool()
    registry.register(tool, arguments_schema=CALENDAR_LIST_EVENTS_ARGUMENTS_SCHEMA)

    result = _executor(
        tool_registry=registry,
        tool_executor=ToolExecutor(registry),
    ).execute(
        _gateway().from_text("Lista eventos del calendario"),
        _decision(RequestRoute.SINGLE_TOOL, target_tool_name=tool.name),
    )

    assert result.status is RouteExecutionStatus.CLARIFICATION_REQUIRED
    assert result.output["missing_information"] == ("time_min", "time_max")
    assert tool.calls == 0


def test_single_tool_unknown_target_fails_without_execution() -> None:
    registry, tool_executor, runner, tool = _tool_runtime()
    request = _gateway().from_text("ejecuta")

    result = _executor(
        tool_registry=registry,
        tool_executor=tool_executor,
        single_tool_runner=runner,
    ).execute(
        request,
        _decision(RequestRoute.SINGLE_TOOL, target_tool_name="missing.tool"),
    )

    assert result.status is RouteExecutionStatus.FAILED
    assert result.error.code == "ROUTE_TARGET_UNAVAILABLE"
    assert tool.calls == 0


def test_dangerous_single_tool_waits_and_resume_uses_existing_confirmation() -> None:
    registry, tool_executor, runner, tool = _tool_runtime(dangerous=True)
    executor = _executor(
        tool_registry=registry,
        tool_executor=tool_executor,
        single_tool_runner=runner,
    )
    request = _gateway().from_voice(
        "ejecuta",
        metadata={"tool_arguments": {"value": "real"}},
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )
    waiting = executor.execute(
        request,
        _decision(
            RequestRoute.SINGLE_TOOL,
            target_tool_name=tool.name,
            requires_confirmation=True,
        ),
    )

    assert waiting.status is RouteExecutionStatus.WAITING_CONFIRMATION
    assert waiting.execution_reference
    assert tool.calls == 0

    resume = _gateway("request-2").from_resume(
        waiting.execution_reference,
        confirmation_response=True,
        request_id="request-2",
    )
    completed = executor.execute(
        resume,
        _decision(
            RequestRoute.RESUME_EXECUTION,
            request_id="request-2",
            target_session_id=waiting.execution_reference,
        ),
    )

    assert completed.status is RouteExecutionStatus.COMPLETED
    assert tool.calls == 1


def test_side_effects_disabled_rejects_tool() -> None:
    registry, tool_executor, runner, tool = _tool_runtime()
    request = _gateway().from_text(
        "ejecuta",
        metadata={"tool_arguments": {"value": "real"}},
    )
    result = _executor(
        tool_registry=registry,
        tool_executor=tool_executor,
        single_tool_runner=runner,
    ).execute(
        request,
        _decision(RequestRoute.SINGLE_TOOL, target_tool_name=tool.name),
    )

    assert result.status is RouteExecutionStatus.REJECTED
    assert tool.calls == 0


def test_agent_delegation_invokes_only_registered_specialized_agent() -> None:
    registry = AgentRegistry()
    agent = _Agent("coding")
    registry.register(agent)
    request = _gateway().from_text("revisa el codigo")

    result = _executor(agent_registry=registry).execute(
        request,
        _decision(RequestRoute.AGENT_DELEGATION, target_agent_name="coding"),
    )

    assert result.status is RouteExecutionStatus.COMPLETED
    assert result.output == "agent:coding"
    assert agent.calls == 1


def test_agent_delegation_missing_or_chat_target_fails() -> None:
    registry = AgentRegistry()
    chat = _Agent("chat")
    registry.register(chat)
    request = _gateway().from_text("hola")

    missing = _executor(agent_registry=registry).execute(
        request,
        _decision(RequestRoute.AGENT_DELEGATION, target_agent_name="missing"),
    )
    chat_result = _executor(agent_registry=registry).execute(
        request,
        _decision(RequestRoute.AGENT_DELEGATION, target_agent_name="chat"),
    )

    assert missing.status is RouteExecutionStatus.FAILED
    assert chat_result.status is RouteExecutionStatus.FAILED
    assert chat.calls == 0


def test_autonomous_execution_delegates_to_existing_facade_and_preserves_session() -> None:
    facade = _Autonomous()
    request = _gateway().from_text(
        "haz dos cosas",
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )

    result = _executor(autonomous_orchestrator=facade).execute(
        request,
        _decision(RequestRoute.AUTONOMOUS_EXECUTION),
    )

    assert result.status is RouteExecutionStatus.COMPLETED
    assert result.session_id == "session-1"
    assert facade.objectives == ["haz dos cosas"]


def test_autonomous_waiting_confirmation_and_failure_are_preserved() -> None:
    waiting_facade = _Autonomous(
        _autonomous_result(
            AutonomousExecutionOutcome.WAITING_CONFIRMATION,
            requires_confirmation=True,
        )
    )
    failed_facade = _Autonomous(
        _autonomous_result(AutonomousExecutionOutcome.FAILED)
    )
    first = _gateway("first").from_text(
        "haz dos cosas",
        request_id="first",
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )
    second = _gateway("second").from_text(
        "haz dos cosas",
        request_id="second",
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )

    waiting = _executor(autonomous_orchestrator=waiting_facade).execute(
        first,
        _decision(RequestRoute.AUTONOMOUS_EXECUTION, request_id="first"),
    )
    failed = _executor(autonomous_orchestrator=failed_facade).execute(
        second,
        _decision(RequestRoute.AUTONOMOUS_EXECUTION, request_id="second"),
    )

    assert waiting.status is RouteExecutionStatus.WAITING_CONFIRMATION
    assert waiting.requires_confirmation is True
    assert failed.status is RouteExecutionStatus.FAILED


def test_resume_requires_session_and_preserves_confirmation_response() -> None:
    facade = _Autonomous()
    missing_request = _gateway().from_text("continua")
    missing = _executor(autonomous_orchestrator=facade).execute(
        missing_request,
        _decision(RequestRoute.RESUME_EXECUTION),
    )
    resume_request = _gateway("request-2").from_resume(
        "session-1",
        confirmation_response=False,
        request_id="request-2",
    )
    resumed = _executor(autonomous_orchestrator=facade).execute(
        resume_request,
        _decision(
            RequestRoute.RESUME_EXECUTION,
            request_id="request-2",
            target_session_id="session-1",
        ),
    )

    assert missing.status is RouteExecutionStatus.CLARIFICATION_REQUIRED
    assert resumed.status is RouteExecutionStatus.COMPLETED
    assert facade.resumes == [("session-1", False, None)]


def test_system_commands_are_structured_and_do_not_exit_process() -> None:
    request = _gateway().from_system("salir")
    result = _executor().execute(
        request,
        _decision(RequestRoute.SYSTEM_COMMAND, system_command=SystemCommand.EXIT),
    )

    assert result.status is RouteExecutionStatus.COMPLETED
    assert result.output["signal"] == "exit_requested"


def test_system_status_and_list_use_real_supervisor_api() -> None:
    supervisor = _Supervisor()
    status_request = _gateway("status").from_system("estado", request_id="status")
    list_request = _gateway("list").from_system("lista", request_id="list")
    executor = _executor(execution_supervisor=supervisor)

    status = executor.execute(
        status_request,
        _decision(
            RequestRoute.SYSTEM_COMMAND,
            request_id="status",
            system_command=SystemCommand.STATUS,
        ),
    )
    listed = executor.execute(
        list_request,
        _decision(
            RequestRoute.SYSTEM_COMMAND,
            request_id="list",
            system_command=SystemCommand.LIST_EXECUTIONS,
        ),
    )

    assert status.output["output"]["active"] == 1
    assert listed.output["output"] == ("session-1",)
    assert supervisor.calls == ["overview", "list"]


def test_system_cancel_without_session_requires_clarification() -> None:
    request = _gateway().from_system(
        "cancela ejecucion",
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )

    result = _executor(autonomous_orchestrator=_Autonomous()).execute(
        request,
        _decision(
            RequestRoute.SYSTEM_COMMAND,
            system_command=SystemCommand.CANCEL_EXECUTION,
        ),
    )

    assert result.status is RouteExecutionStatus.CLARIFICATION_REQUIRED
    assert "session_id" in result.clarification_question


def test_read_only_status_is_not_cached_by_idempotency_ledger() -> None:
    supervisor = _Supervisor()
    request = _gateway().from_system("estado")
    decision = _decision(
        RequestRoute.SYSTEM_COMMAND,
        system_command=SystemCommand.STATUS,
    )
    executor = _executor(execution_supervisor=supervisor)

    executor.execute(request, decision)
    executor.execute(request, decision)

    assert supervisor.calls == ["overview", "overview"]


def test_clarification_and_unsupported_never_execute_fallbacks() -> None:
    calls = []
    executor = _executor(direct_responder=lambda request: calls.append(request))
    clarification_request = _gateway("clarify").from_text("abre", request_id="clarify")
    unsupported_request = _gateway("unsupported").from_text(
        "teletransporta",
        request_id="unsupported",
    )

    clarification = executor.execute(
        clarification_request,
        _decision(
            RequestRoute.CLARIFICATION_REQUIRED,
            request_id="clarify",
            requires_clarification=True,
            clarification_question="Que quieres abrir?",
        ),
    )
    unsupported = executor.execute(
        unsupported_request,
        _decision(
            RequestRoute.UNSUPPORTED,
            request_id="unsupported",
            fallback_route=RequestRoute.DIRECT_RESPONSE,
        ),
    )

    assert clarification.status is RouteExecutionStatus.CLARIFICATION_REQUIRED
    assert unsupported.status is RouteExecutionStatus.UNSUPPORTED
    assert calls == []


def test_mismatched_request_decision_is_typed_failure() -> None:
    request = _gateway().from_text("hola")

    result = _executor().execute(
        request,
        _decision(RequestRoute.DIRECT_RESPONSE, request_id="other"),
    )

    assert result.status is RouteExecutionStatus.FAILED
    assert result.error.code == "INVALID_ROUTE_DECISION_EXECUTION"


def test_resume_session_mismatch_and_source_incompatibility_do_not_reclassify() -> None:
    resume = _gateway().from_resume("session-1")
    mismatch = _executor(autonomous_orchestrator=_Autonomous()).execute(
        resume,
        _decision(
            RequestRoute.RESUME_EXECUTION,
            target_session_id="session-2",
        ),
    )
    incompatible = _executor().execute(
        resume,
        _decision(RequestRoute.DIRECT_RESPONSE),
    )

    assert mismatch.error.code == "INVALID_ROUTE_DECISION_EXECUTION"
    assert incompatible.error.code == "INVALID_ROUTE_DECISION_EXECUTION"


def test_missing_and_duplicate_handler_configuration_are_rejected() -> None:
    request = _gateway().from_text("hola")
    executor = OperationalRouteExecutor({}, clock=lambda: NOW)

    missing = executor.execute(request, _decision(RequestRoute.DIRECT_RESPONSE))

    assert missing.error.code == "ROUTE_HANDLER_NOT_CONFIGURED"
    with pytest.raises(ValueError):
        OperationalRouteExecutor(
            {
                RequestRoute.DIRECT_RESPONSE: DirectResponseRouteHandler(
                    lambda _request: "ok",
                    clock=lambda: NOW,
                )
            },
            clock=lambda: NOW,
        ).register_handler(
            RequestRoute.DIRECT_RESPONSE,
            DirectResponseRouteHandler(lambda _request: "other", clock=lambda: NOW),
        )


def test_effectful_request_id_is_idempotent_but_new_id_executes() -> None:
    registry, tool_executor, runner, tool = _tool_runtime()
    executor = _executor(
        tool_registry=registry,
        tool_executor=tool_executor,
        single_tool_runner=runner,
    )
    request = _gateway().from_text(
        "ejecuta",
        metadata={"tool_arguments": {"value": "one"}},
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )
    decision = _decision(RequestRoute.SINGLE_TOOL, target_tool_name=tool.name)

    first = executor.execute(request, decision)
    duplicate = executor.execute(request, decision)
    new_request = _gateway("request-2").from_text(
        "ejecuta",
        request_id="request-2",
        metadata={"tool_arguments": {"value": "two"}},
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )
    second = executor.execute(
        new_request,
        _decision(
            RequestRoute.SINGLE_TOOL,
            request_id="request-2",
            target_tool_name=tool.name,
        ),
    )

    assert duplicate is first
    assert second.status is RouteExecutionStatus.COMPLETED
    assert tool.calls == 2
    assert any(event.event_type == "duplicate_request_detected" for event in executor.events)


def test_zero_timeout_fails_before_handler_is_called() -> None:
    calls = []
    request = _gateway().from_text(
        "hola",
        execution_context=RequestExecutionContext(requested_timeout=0),
    )

    result = _executor(
        direct_responder=lambda item: calls.append(item),
    ).execute(request, _decision(RequestRoute.DIRECT_RESPONSE))

    assert result.status is RouteExecutionStatus.FAILED
    assert result.error.code == "ROUTE_EXECUTION_TIMEOUT"
    assert calls == []


def test_external_route_flag_respects_external_call_and_sensitive_data_policy() -> None:
    calls = []
    decision = _decision(
        RequestRoute.DIRECT_RESPONSE,
        safety_flags=("external_call",),
    )
    disabled = _gateway().from_text("hola")
    sensitive = _gateway("request-2").from_text(
        "hola",
        request_id="request-2",
        safety_context=RequestSafetyContext(
            allow_external_calls=True,
            contains_sensitive_data=True,
        ),
    )

    first = _executor(direct_responder=lambda item: calls.append(item)).execute(
        disabled,
        decision,
    )
    second = _executor(direct_responder=lambda item: calls.append(item)).execute(
        sensitive,
        _decision(
            RequestRoute.DIRECT_RESPONSE,
            request_id="request-2",
            safety_flags=("external_call",),
        ),
    )

    assert first.status is RouteExecutionStatus.REJECTED
    assert second.status is RouteExecutionStatus.REJECTED
    assert calls == []


def test_orchestrator_process_prompt_and_process_request_use_gateway_router_executor() -> None:
    seen = []
    gateway = _gateway()
    route_executor = _executor(
        direct_responder=lambda request: seen.append(request) or "ok",
    )
    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(
            create_plan=lambda _prompt: (_ for _ in ()).throw(
                AssertionError("legacy planner must not be called")
            )
        ),
        router=Router(operational_router=OperationalRequestRouter(clock=lambda: NOW)),
        model_manager=SimpleNamespace(choose_model=lambda _agent: "unused"),
        memory=_Memory(),
        registry=AgentRegistry(),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        request_gateway=gateway,
        operational_route_executor=route_executor,
    )

    prompt_result = orchestrator.process_prompt_result("hola")
    existing = gateway.from_text("otro", request_id="request-2")
    request_result = orchestrator.process_request(existing)

    assert prompt_result.status is RouteExecutionStatus.COMPLETED
    assert request_result.status is RouteExecutionStatus.COMPLETED
    assert seen[1] is existing
    assert orchestrator.present_route_execution(prompt_result) == "ok"


def test_orchestrator_voice_result_uses_same_flow_and_rejects_empty_transcription() -> None:
    seen = []
    route_executor = _executor(
        direct_responder=lambda request: seen.append(request) or "ok",
    )
    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda text: Plan("chat", text)),
        router=Router(operational_router=OperationalRequestRouter(clock=lambda: NOW)),
        model_manager=SimpleNamespace(choose_model=lambda _agent: "unused"),
        memory=_Memory(),
        registry=AgentRegistry(),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        request_gateway=_gateway(),
        operational_route_executor=route_executor,
    )

    result = orchestrator.process_voice_prompt_result("hola", confidence=0.9)

    assert result.status is RouteExecutionStatus.COMPLETED
    assert seen[0].source.value == "voice"
    with pytest.raises(ValueError):
        orchestrator.process_voice_prompt_result("   ")


def test_orchestrator_voice_direct_timeout_is_not_presented_as_success() -> None:
    def timed_out(_request):
        raise TimeoutError("native Ollama timeout")

    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda text: Plan("chat", text)),
        router=Router(operational_router=OperationalRequestRouter(clock=lambda: NOW)),
        model_manager=SimpleNamespace(choose_model=lambda _agent: "unused"),
        memory=_Memory(),
        registry=AgentRegistry(),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        request_gateway=_gateway(),
        operational_route_executor=_executor(direct_responder=timed_out),
    )

    with pytest.raises(TimeoutError, match="agotó su timeout"):
        orchestrator.process_voice_prompt("hola", confirm=lambda _prompt: "")


def test_result_models_events_and_trace_are_immutable_and_content_safe() -> None:
    request = _gateway().from_text("contenido privado completo")
    executor = _executor()
    result = executor.execute(request, _decision(RequestRoute.DIRECT_RESPONSE))

    assert result.started_at.tzinfo is not None
    assert result.trace[0].sequence == 1
    assert result.trace[1].sequence == 2
    assert all(not hasattr(event, "content") for event in executor.events)
    with pytest.raises(AttributeError):
        result.status = RouteExecutionStatus.FAILED  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.metadata["x"] = "y"  # type: ignore[index]


@pytest.mark.parametrize(
    ("route", "status", "expected"),
    [
        (RequestRoute.DIRECT_RESPONSE, RouteExecutionStatus.COMPLETED, "ok"),
        (RequestRoute.UNSUPPORTED, RouteExecutionStatus.UNSUPPORTED, "No puedo"),
        (
            RequestRoute.CLARIFICATION_REQUIRED,
            RouteExecutionStatus.CLARIFICATION_REQUIRED,
            "Que falta?",
        ),
    ],
)
def test_presenter_returns_safe_readable_text(route, status, expected) -> None:
    request = _gateway().from_text("hola")
    if route is RequestRoute.DIRECT_RESPONSE:
        result = _executor(direct_responder=lambda _request: "ok").execute(
            request,
            _decision(route),
        )
    elif route is RequestRoute.UNSUPPORTED:
        result = _executor().execute(request, _decision(route))
    else:
        result = _executor().execute(
            request,
            _decision(
                route,
                requires_clarification=True,
                clarification_question="Que falta?",
            ),
        )

    assert result.status is status
    assert expected in RouteExecutionPresenter().present(result)


def test_presenter_shows_empty_calendar_result() -> None:
    result = _completed_tool_result("calendar_list_events", {"events": []})

    assert RouteExecutionPresenter().present(result) == (
        "No hay eventos en el rango solicitado."
    )


def test_presenter_shows_normalized_calendar_events() -> None:
    result = _completed_tool_result(
        "calendar_list_events",
        {
            "events": [
                {
                    "id": "evt-1",
                    "summary": "Revision",
                    "start": "2026-08-10T10:00:00+01:00",
                    "end": "2026-08-10T10:30:00+01:00",
                },
                {
                    "id": "evt-2",
                    "summary": "Planificacion",
                    "start": "2026-08-10T12:00:00+01:00",
                    "end": "2026-08-10T13:00:00+01:00",
                },
            ]
        },
    )

    assert RouteExecutionPresenter().present(result) == (
        "Eventos encontrados:\n"
        "- Revision: 2026-08-10T10:00:00+01:00 - 2026-08-10T10:30:00+01:00\n"
        "- Planificacion: 2026-08-10T12:00:00+01:00 - 2026-08-10T13:00:00+01:00"
    )


def test_presenter_keeps_generic_mapping_fallback_for_other_tools() -> None:
    result = _completed_tool_result("other_tool", {"value": 1})

    assert RouteExecutionPresenter().present(result) == "Operacion completada."


def test_handler_selection_is_deterministic() -> None:
    request = _gateway().from_text("hola")
    first = _executor().execute(request, _decision(RequestRoute.DIRECT_RESPONSE))
    second = _executor().execute(request, _decision(RequestRoute.DIRECT_RESPONSE))

    assert type(_executor().handlers[RequestRoute.DIRECT_RESPONSE]) is type(
        _executor().handlers[RequestRoute.DIRECT_RESPONSE]
    )
    assert first == second


class _Memory:
    def __init__(self) -> None:
        self.items = []

    def add_user(self, value):
        self.items.append(value)

    def history(self):
        return list(self.items)


class _Tool(BaseTool):
    def __init__(self, name: str, *, dangerous: bool = False) -> None:
        self._name = name
        self._dangerous = dangerous
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Test tool."

    @property
    def requires_confirmation(self) -> bool:
        return self._dangerous

    def execute(self, context):
        self.calls += 1
        return f"tool:{context.parameters['value']}"


class _CalendarTool(BaseTool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "calendar_list_events"

    @property
    def description(self) -> str:
        return "List Google Calendar events in a time range."

    def execute(self, context):
        self.calls += 1
        return dict(context.parameters)


class _Agent:
    def __init__(self, name):
        self.name = name
        self.description = "Specialized agent."
        self.calls = 0

    def run(self, *, model, messages):
        assert model == f"model:{self.name}"
        assert messages[0]["role"] == "user"
        self.calls += 1
        return f"agent:{self.name}"


class _Autonomous:
    def __init__(self, result=None):
        self.result = result or _autonomous_result()
        self.objectives = []
        self.resumes = []
        self.cancels = []

    def execute_objective(self, objective, *, planning_context, execution_options):
        assert planning_context["request_id"]
        assert execution_options.allow_automatic_recovery is False
        self.objectives.append(objective)
        return self.result

    def resume_execution(
        self,
        session_id,
        *,
        confirmation,
        recovery_authorization,
    ):
        self.resumes.append((session_id, confirmation, recovery_authorization))
        return self.result

    def cancel_execution(self, session_id):
        self.cancels.append(session_id)
        return _autonomous_result(AutonomousExecutionOutcome.CANCELLED)


class _Supervisor:
    def __init__(self):
        self.calls = []

    def get_overview(self):
        self.calls.append("overview")
        return {"active": 1}

    def list_sessions(self):
        self.calls.append("list")
        return ("session-1",)

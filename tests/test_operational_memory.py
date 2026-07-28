from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agents.registry import AgentRegistry
from core.autonomous_execution import (
    AutonomousExecutionOutcome,
    AutonomousExecutionResult,
    ExecutionTrace,
)
from core.execution_memory_recorder import ExecutionMemoryRecorder
from core.execution_supervisor import ExecutionState
from core.operational_context import OperationalContextBuilder
from core.operational_request_router import (
    MemoryOperation,
    RequestRoute,
    RouteDecision,
)
from core.operational_route_executor import (
    OperationalRouteExecutor,
    RouteExecutionPresenter,
    RouteExecutionStatus,
    build_default_route_handlers,
)
from core.request_gateway import RequestGateway, RequestSafetyContext
from memory.conversation import ConversationMemory
from memory.operational import (
    InvalidMemoryEntryError,
    MemoryCategory,
    MemoryEntryNotFoundError,
    MemoryPolicy,
    SensitiveMemoryRejectedError,
)


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


def _memory_request(
    content: str,
    *,
    request_id: str = "request-1",
    metadata=None,
):
    return _gateway(request_id).from_text(
        content,
        request_id=request_id,
        metadata=metadata or {},
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )


def _executor(
    memory,
    *,
    direct_responder=None,
    agent_registry=None,
    autonomous=None,
    policy=None,
):
    context_builder = OperationalContextBuilder(
        memory,
        policy=policy,
        clock=lambda: NOW,
    )
    return OperationalRouteExecutor(
        build_default_route_handlers(
            direct_responder=direct_responder or (lambda _request: "ok"),
            memory=memory,
            agent_registry=agent_registry or AgentRegistry(),
            model_selector=lambda name: f"model:{name}",
            autonomous_orchestrator=autonomous,
            clock=lambda: NOW,
        ),
        context_builder=context_builder,
        clock=lambda: NOW,
    )


def _execute_memory(
    memory,
    request,
    operation: MemoryOperation,
):
    return _executor(memory).execute(
        request,
        _decision(
            RequestRoute.MEMORY_QUERY,
            request_id=request.request_id,
            memory_operation=operation,
        ),
    )


def test_store_valid_entry_returns_stable_memory_id() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    result = _execute_memory(
        memory,
        _memory_request(
            "recuerda que prefiero respuestas breves",
            metadata={
                "memory_category": "user_preference",
                "tags": ["style"],
            },
        ),
        MemoryOperation.STORE,
    )

    assert result.status is RouteExecutionStatus.COMPLETED
    assert result.output["memory_id"] == "memory-000001"
    assert memory.get_entry("memory-000001").content == "prefiero respuestas breves"
    assert memory.get_entry("memory-000001").category is MemoryCategory.USER_PREFERENCE


def test_store_empty_and_sensitive_content_are_rejected() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    empty = _execute_memory(
        memory,
        _memory_request("recuerda"),
        MemoryOperation.STORE,
    )
    sensitive = _execute_memory(
        memory,
        _memory_request(
            "recuerda que mi dato privado es X",
            metadata={"sensitive": True},
        ),
        MemoryOperation.STORE,
    )

    assert empty.status is RouteExecutionStatus.CLARIFICATION_REQUIRED
    assert sensitive.status is RouteExecutionStatus.REJECTED
    assert memory.list_entries() == ()


def test_credentials_are_rejected_even_if_sensitive_storage_is_enabled() -> None:
    policy = MemoryPolicy(allow_sensitive_storage=True)
    memory = ConversationMemory(policy=policy, clock=lambda: NOW)

    with pytest.raises(SensitiveMemoryRejectedError):
        memory.store_entry(
            "api_key=private-value",
            sensitive=True,
        )


def test_retrieve_relevant_entry_and_empty_result_are_valid() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    memory.store_entry("Atlas usa Python", category=MemoryCategory.PROJECT_FACT)

    found = _execute_memory(
        memory,
        _memory_request("que recuerdas de Python"),
        MemoryOperation.RETRIEVE,
    )
    missing = _execute_memory(
        memory,
        _memory_request("que recuerdas de nutricion", request_id="request-2"),
        MemoryOperation.RETRIEVE,
    )

    assert found.output["count"] == 1
    assert found.output["items"][0]["memory_id"] == "memory-000001"
    assert missing.status is RouteExecutionStatus.COMPLETED
    assert missing.output["items"] == ()


def test_list_excludes_inactive_expired_and_sensitive_entries() -> None:
    policy = MemoryPolicy(allow_sensitive_storage=True)
    memory = ConversationMemory(policy=policy, clock=lambda: NOW)
    active = memory.store_entry("activo")
    forgotten = memory.store_entry("olvidado")
    memory.forget_entry(forgotten.memory_id)
    memory.store_entry("expirado", expires_at=NOW - timedelta(seconds=1))
    memory.store_entry("privado", sensitive=True)

    result = _execute_memory(
        memory,
        _memory_request("lista memoria"),
        MemoryOperation.LIST,
    )

    assert tuple(item["memory_id"] for item in result.output["items"]) == (
        active.memory_id,
    )


def test_forget_by_memory_id_soft_deletes_and_missing_id_fails_safely() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    entry = memory.store_entry("dato temporal")
    forgotten = _execute_memory(
        memory,
        _memory_request(
            f"olvida {entry.memory_id}",
            metadata={"memory_id": entry.memory_id},
        ),
        MemoryOperation.FORGET,
    )
    missing = _execute_memory(
        memory,
        _memory_request(
            "olvida memory-999999",
            request_id="request-2",
            metadata={"memory_id": "memory-999999"},
        ),
        MemoryOperation.FORGET,
    )

    assert forgotten.output["forgotten"] is True
    assert memory.get_entry(entry.memory_id).active is False
    assert missing.status is RouteExecutionStatus.FAILED
    assert missing.error.safe_cause == "MEMORY_ENTRY_NOT_FOUND"


def test_forget_ambiguous_phrase_requires_exact_memory_id() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    memory.store_entry("proyecto Atlas usa Python")
    memory.store_entry("proyecto Atlas usa pytest")

    result = _execute_memory(
        memory,
        _memory_request("olvida proyecto Atlas"),
        MemoryOperation.FORGET,
    )

    assert result.status is RouteExecutionStatus.CLARIFICATION_REQUIRED
    assert "memory_id" in result.clarification_question
    assert len(memory.list_entries()) == 2


def test_update_preserves_identity_and_changes_updated_at() -> None:
    clock = _MutableClock(NOW)
    memory = ConversationMemory(clock=clock)
    entry = memory.store_entry("valor anterior")
    clock.current = NOW + timedelta(minutes=1)

    result = _execute_memory(
        memory,
        _memory_request(
            f"actualiza {entry.memory_id}",
            metadata={
                "memory_id": entry.memory_id,
                "memory_content": "valor nuevo",
            },
        ),
        MemoryOperation.UPDATE,
    )

    updated = memory.get_entry(entry.memory_id)
    assert result.output["memory_id"] == entry.memory_id
    assert updated.content == "valor nuevo"
    assert updated.updated_at > updated.created_at


def test_update_missing_and_sensitive_entry_fail_clearly() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    missing = _execute_memory(
        memory,
        _memory_request(
            "actualiza memory-999999",
            metadata={
                "memory_id": "memory-999999",
                "memory_content": "nuevo",
            },
        ),
        MemoryOperation.UPDATE,
    )
    entry = memory.store_entry("dato")
    sensitive = _execute_memory(
        memory,
        _memory_request(
            f"actualiza {entry.memory_id}",
            request_id="request-2",
            metadata={
                "memory_id": entry.memory_id,
                "memory_content": "dato privado",
                "sensitive": True,
            },
        ),
        MemoryOperation.UPDATE,
    )

    assert missing.status is RouteExecutionStatus.FAILED
    assert sensitive.status is RouteExecutionStatus.REJECTED
    assert memory.get_entry(entry.memory_id).content == "dato"


def test_exact_duplicate_returns_existing_entry_without_second_write() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    first = memory.store_entry(
        "Usar pytest",
        category=MemoryCategory.DECISION,
        tags=("test",),
    )
    second = memory.store_entry(
        "  usar   PYTEST ",
        category=MemoryCategory.DECISION,
        tags=("test",),
    )

    assert second is first
    assert len(memory.list_entries()) == 1
    assert memory.events[-1].event_type == "memory_duplicate_detected"


def test_duplicate_identity_scope_respects_conversation() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    first = memory.store_entry("dato", conversation_id="conversation-1")
    second = memory.store_entry("dato", conversation_id="conversation-2")

    assert first.memory_id != second.memory_id


def test_memory_entry_is_immutable_and_metadata_is_safe() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    entry = memory.store_entry("dato", metadata={"source": ["manual"]})

    with pytest.raises(AttributeError):
        entry.content = "otro"  # type: ignore[misc]
    with pytest.raises(TypeError):
        entry.metadata["x"] = "y"  # type: ignore[index]
    with pytest.raises(InvalidMemoryEntryError):
        memory.store_entry("dato nuevo", metadata={"api_token": "x"})


def test_empty_context_works_without_memory_configuration() -> None:
    request = _gateway().from_text("hola")
    context = OperationalContextBuilder(None, clock=lambda: NOW).build(
        request,
        _decision(RequestRoute.DIRECT_RESPONSE),
    )

    assert context.selected_memory_ids == ()
    assert context.recent_messages == ()
    assert context.total_characters == 0


def test_direct_context_includes_relevant_preference_not_irrelevant_fact() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    preference = memory.store_entry(
        "Prefiero respuestas breves",
        category=MemoryCategory.USER_PREFERENCE,
    )
    memory.store_entry(
        "Mi postre favorito es chocolate",
        category=MemoryCategory.USER_FACT,
    )
    request = _gateway().from_text("Explica Python")
    context = OperationalContextBuilder(memory, clock=lambda: NOW).build(
        request,
        _decision(RequestRoute.DIRECT_RESPONSE),
    )

    assert context.selected_memory_ids == (preference.memory_id,)


def test_agent_context_filters_domain_and_autonomous_context_uses_project_facts() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    project = memory.store_entry(
        "Atlas usa Python",
        category=MemoryCategory.PROJECT_FACT,
        tags=("python",),
    )
    memory.store_entry(
        "Prefiero comida italiana",
        category=MemoryCategory.USER_PREFERENCE,
    )
    agent_request = _gateway().from_text(
        "revisa Python",
        metadata={"tags": ["python"]},
    )
    autonomous_request = _gateway("request-2").from_text(
        "actualiza el proyecto Python y ejecuta tests",
        request_id="request-2",
    )
    builder = OperationalContextBuilder(memory, clock=lambda: NOW)

    agent_context = builder.build(
        agent_request,
        _decision(
            RequestRoute.AGENT_DELEGATION,
            target_agent_name="coding",
        ),
    )
    autonomous_context = builder.build(
        autonomous_request,
        _decision(RequestRoute.AUTONOMOUS_EXECUTION, request_id="request-2"),
    )

    assert agent_context.selected_memory_ids == (project.memory_id,)
    assert autonomous_context.project_context == (project,)


@pytest.mark.parametrize(
    "route",
    [
        RequestRoute.SINGLE_TOOL,
        RequestRoute.RESUME_EXECUTION,
        RequestRoute.SYSTEM_COMMAND,
    ],
)
def test_routes_without_memory_need_do_not_infer_context(route) -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    memory.store_entry("valor ambiguo", category=MemoryCategory.TASK_CONTEXT)
    request = _gateway().from_text("hazlo")
    kwargs = {}
    if route is RequestRoute.SINGLE_TOOL:
        kwargs["target_tool_name"] = "tool"
    if route is RequestRoute.RESUME_EXECUTION:
        kwargs["target_session_id"] = "session-1"

    context = OperationalContextBuilder(memory, clock=lambda: NOW).build(
        request,
        _decision(route, **kwargs),
    )

    assert context.relevant_memories == ()


def test_same_conversation_tags_recency_and_importance_affect_deterministic_order() -> None:
    clock = _MutableClock(NOW - timedelta(days=2))
    memory = ConversationMemory(clock=clock)
    older = memory.store_entry(
        "Python decision antigua",
        category=MemoryCategory.DECISION,
        conversation_id="conversation-1",
        importance=0.9,
        tags=("python",),
    )
    clock.current = NOW - timedelta(days=1)
    newer = memory.store_entry(
        "Python decision reciente",
        category=MemoryCategory.DECISION,
        conversation_id="conversation-1",
        importance=0.5,
        tags=("python",),
    )
    clock.current = NOW
    request = _gateway().from_text(
        "Python",
        conversation_id="conversation-1",
        metadata={"tags": ["python"]},
    )
    context = OperationalContextBuilder(memory, clock=lambda: NOW).build(
        request,
        _decision(RequestRoute.AGENT_DELEGATION, target_agent_name="coding"),
    )

    assert context.selected_memory_ids == (older.memory_id, newer.memory_id)


def test_inactive_expired_and_sensitive_entries_are_not_selected() -> None:
    policy = MemoryPolicy(allow_sensitive_storage=True)
    memory = ConversationMemory(policy=policy, clock=lambda: NOW)
    inactive = memory.store_entry(
        "Python inactivo",
        category=MemoryCategory.PROJECT_FACT,
    )
    memory.forget_entry(inactive.memory_id)
    memory.store_entry(
        "Python expirado",
        category=MemoryCategory.PROJECT_FACT,
        expires_at=NOW - timedelta(seconds=1),
    )
    memory.store_entry(
        "Python privado",
        category=MemoryCategory.PROJECT_FACT,
        sensitive=True,
    )
    request = _gateway().from_text(
        "Python",
        safety_context=RequestSafetyContext(allow_external_calls=True),
    )

    context = OperationalContextBuilder(memory, policy=policy, clock=lambda: NOW).build(
        request,
        _decision(RequestRoute.AGENT_DELEGATION, target_agent_name="coding"),
    )

    assert context.relevant_memories == ()


def test_context_entry_and_character_limits_mark_truncation() -> None:
    policy = MemoryPolicy(max_context_entries=1, max_context_characters=20)
    memory = ConversationMemory(policy=policy, clock=lambda: NOW)
    memory.store_entry(
        "Python uno muy largo",
        category=MemoryCategory.PROJECT_FACT,
    )
    memory.store_entry(
        "Python dos muy largo",
        category=MemoryCategory.PROJECT_FACT,
    )
    request = _gateway().from_text("Python")

    context = OperationalContextBuilder(memory, policy=policy, clock=lambda: NOW).build(
        request,
        _decision(RequestRoute.AUTONOMOUS_EXECUTION),
    )

    assert len(context.relevant_memories) <= 1
    assert context.total_characters <= 20
    assert context.truncated is True


def test_recent_messages_are_limited_ordered_and_do_not_duplicate_current_request() -> None:
    policy = MemoryPolicy(recent_entry_limit=2)
    memory = ConversationMemory(policy=policy, clock=lambda: NOW)
    memory.add_user("uno")
    memory.add_assistant("dos")
    memory.add_user("actual")
    request = _gateway().from_text("actual")

    context = OperationalContextBuilder(memory, policy=policy, clock=lambda: NOW).build(
        request,
        _decision(RequestRoute.DIRECT_RESPONSE),
    )

    assert tuple(message["content"] for message in context.recent_messages) == (
        "uno",
        "dos",
    )


def test_direct_response_receives_context_and_no_memory_keeps_legacy_callable() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    entry = memory.store_entry(
        "Prefiero respuestas breves",
        category=MemoryCategory.USER_PREFERENCE,
    )
    captured = []
    request = _gateway().from_text("Explica Python")
    contextual = _executor(
        memory,
        direct_responder=lambda item, context: captured.append(context) or "ok",
    ).execute(request, _decision(RequestRoute.DIRECT_RESPONSE))
    legacy = _executor(
        None,
        direct_responder=lambda item: f"legacy:{item.content}",
    ).execute(request, _decision(RequestRoute.DIRECT_RESPONSE))

    assert contextual.status is RouteExecutionStatus.COMPLETED
    assert captured[0].selected_memory_ids == (entry.memory_id,)
    assert legacy.output == "legacy:Explica Python"


def test_agent_delegation_transports_only_limited_context() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    entry = memory.store_entry(
        "Atlas usa Python",
        category=MemoryCategory.PROJECT_FACT,
    )
    registry = AgentRegistry()
    agent = _Agent()
    registry.register(agent)
    request = _gateway().from_text("revisa Python")

    result = _executor(memory, agent_registry=registry).execute(
        request,
        _decision(RequestRoute.AGENT_DELEGATION, target_agent_name="coding"),
    )

    assert result.status is RouteExecutionStatus.COMPLETED
    assert entry.content in agent.messages[0]["content"]
    assert result.metadata["operational_context"]["selected_count"] == 1


def test_autonomous_planning_receives_bounded_context_not_whole_memory() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    selected = memory.store_entry(
        "Atlas usa Python",
        category=MemoryCategory.PROJECT_FACT,
    )
    memory.store_entry(
        "Dato culinario irrelevante",
        category=MemoryCategory.USER_FACT,
    )
    autonomous = _Autonomous()
    request = _gateway().from_text(
        "actualiza Python y ejecuta tests",
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )

    result = _executor(memory, autonomous=autonomous).execute(
        request,
        _decision(RequestRoute.AUTONOMOUS_EXECUTION),
    )

    assert result.status is RouteExecutionStatus.COMPLETED
    assert autonomous.context["selected_memory_ids"] == (selected.memory_id,)
    assert "culinario" not in autonomous.context["relevant_context"]


def test_automatic_recording_is_off_by_default_and_opt_in_is_selective() -> None:
    default_memory = ConversationMemory(clock=lambda: NOW)
    default_recorder = ExecutionMemoryRecorder(default_memory, clock=lambda: NOW)
    completed = SimpleNamespace(
        request_id="request-1",
        route=RequestRoute.AUTONOMOUS_EXECUTION,
        status=RouteExecutionStatus.COMPLETED,
        output={"summary": "Proyecto validado"},
    )

    assert default_recorder.record(completed) is None
    assert default_memory.list_entries() == ()

    policy = MemoryPolicy(
        automatic_write_enabled=True,
        allow_execution_result_storage=True,
    )
    memory = ConversationMemory(policy=policy, clock=lambda: NOW)
    recorder = ExecutionMemoryRecorder(memory, policy=policy, clock=lambda: NOW)
    entry = recorder.record(completed)
    failed = SimpleNamespace(
        request_id="request-2",
        route=RequestRoute.AUTONOMOUS_EXECUTION,
        status=RouteExecutionStatus.FAILED,
        output={"summary": "No guardar"},
    )
    conversational = SimpleNamespace(
        request_id="request-3",
        route=RequestRoute.DIRECT_RESPONSE,
        status=RouteExecutionStatus.COMPLETED,
        output="No guardar conversacion",
    )

    assert entry.category is MemoryCategory.EXECUTION_RESULT
    assert recorder.record(failed) is None
    assert recorder.record(conversational) is None
    assert len(memory.list_entries()) == 1


def test_sensitive_automatic_result_is_not_recorded() -> None:
    policy = MemoryPolicy(
        automatic_write_enabled=True,
        allow_execution_result_storage=True,
    )
    memory = ConversationMemory(policy=policy, clock=lambda: NOW)
    recorder = ExecutionMemoryRecorder(memory, policy=policy, clock=lambda: NOW)
    result = SimpleNamespace(
        request_id="request-1",
        route=RequestRoute.AUTONOMOUS_EXECUTION,
        status=RouteExecutionStatus.COMPLETED,
        output={"summary": "token privado"},
    )

    assert recorder.record(result) is None
    assert memory.list_entries() == ()


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (MemoryOperation.STORE, "guardada"),
        (MemoryOperation.RETRIEVE, "recuperaron"),
        (MemoryOperation.LIST, "entradas"),
        (MemoryOperation.FORGET, "olvidada"),
        (MemoryOperation.UPDATE, "actualizada"),
    ],
)
def test_presenter_memory_operations_are_readable(operation, expected) -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    stored = _execute_memory(
        memory,
        _memory_request("recuerda que Atlas usa Python"),
        MemoryOperation.STORE,
    )
    memory_id = stored.output["memory_id"]
    request_id = f"request-{operation.value}"
    if operation is MemoryOperation.STORE:
        result = stored
    elif operation is MemoryOperation.RETRIEVE:
        result = _execute_memory(
            memory,
            _memory_request("que recuerdas de Python", request_id=request_id),
            operation,
        )
    elif operation is MemoryOperation.LIST:
        result = _execute_memory(
            memory,
            _memory_request("lista memoria", request_id=request_id),
            operation,
        )
    elif operation is MemoryOperation.FORGET:
        result = _execute_memory(
            memory,
            _memory_request(
                f"olvida {memory_id}",
                request_id=request_id,
                metadata={"memory_id": memory_id},
            ),
            operation,
        )
    else:
        result = _execute_memory(
            memory,
            _memory_request(
                f"actualiza {memory_id}",
                request_id=request_id,
                metadata={
                    "memory_id": memory_id,
                    "memory_content": "Atlas usa Python 3",
                },
            ),
            operation,
        )

    assert expected in RouteExecutionPresenter().present(result)


def test_events_never_contain_memory_content() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    memory.store_entry("contenido privado no debe aparecer")
    request = _gateway().from_text("contenido")
    builder = OperationalContextBuilder(memory, clock=lambda: NOW)
    builder.build(request, _decision(RequestRoute.MEMORY_QUERY))

    assert all(not hasattr(event, "content") for event in memory.events)
    assert all(not hasattr(event, "content") for event in builder.events)


def test_same_input_memory_policy_and_clock_produce_same_context() -> None:
    def build():
        memory = ConversationMemory(clock=lambda: NOW)
        memory.store_entry("Atlas usa Python", category=MemoryCategory.PROJECT_FACT)
        request = _gateway().from_text("Python")
        return OperationalContextBuilder(memory, clock=lambda: NOW).build(
            request,
            _decision(RequestRoute.AUTONOMOUS_EXECUTION),
        )

    assert build() == build()


def test_text_and_voice_equivalents_select_same_memory_context() -> None:
    memory = ConversationMemory(clock=lambda: NOW)
    memory.store_entry(
        "Prefiero respuestas breves",
        category=MemoryCategory.USER_PREFERENCE,
    )
    gateway = _gateway()
    text = gateway.from_text("Explica Python")
    voice = gateway.from_voice("Explica Python")
    builder = OperationalContextBuilder(memory, clock=lambda: NOW)

    text_context = builder.build(
        text,
        _decision(RequestRoute.DIRECT_RESPONSE),
    )
    voice_context = builder.build(
        voice,
        _decision(RequestRoute.DIRECT_RESPONSE),
    )

    assert text_context.selected_memory_ids == voice_context.selected_memory_ids
    assert text_context.prompt_context() == voice_context.prompt_context()


def test_autonomous_facade_passes_planning_context_to_planner() -> None:
    import tests.test_autonomous_execution as autonomous_fixtures

    plan = autonomous_fixtures._plan((autonomous_fixtures._step("a"),))

    class PlannerSpy(autonomous_fixtures._FixedPlanner):
        def __init__(self, active_plan):
            super().__init__(active_plan)
            self.planning_context = None

        def generate_execution_plan(self, objective, **kwargs):
            self.planning_context = kwargs.get("planning_context")
            return super().generate_execution_plan(objective, **kwargs)

    planner = PlannerSpy(plan)
    result = autonomous_fixtures._orchestrator(
        plan,
        planner=planner,
    ).execute_objective(
        "build",
        planning_context={
            "selected_memory_ids": ("memory-000001",),
            "relevant_context": "Atlas usa Python",
        },
    )

    assert result.outcome is AutonomousExecutionOutcome.COMPLETED
    assert planner.planning_context["selected_memory_ids"] == ("memory-000001",)


def test_two_controlled_writes_do_not_corrupt_shared_memory() -> None:
    memory = ConversationMemory(clock=lambda: NOW)

    with ThreadPoolExecutor(max_workers=2) as pool:
        entries = tuple(
            pool.map(
                lambda value: memory.store_entry(
                    value,
                    category=MemoryCategory.PROJECT_FACT,
                ),
                ("dato uno", "dato dos"),
            )
        )

    assert {entry.memory_id for entry in entries} == {
        "memory-000001",
        "memory-000002",
    }
    assert len(memory.list_entries()) == 2


class _Agent:
    name = "coding"
    description = "Coding agent."

    def __init__(self):
        self.messages = None

    def run(self, *, model, messages):
        assert model == "model:coding"
        self.messages = messages
        return "agent ok"


class _Autonomous:
    def __init__(self):
        self.context = None

    def execute_objective(self, objective, *, planning_context, execution_options):
        del objective, execution_options
        self.context = planning_context
        return AutonomousExecutionResult(
            session_id="session-1",
            outcome=AutonomousExecutionOutcome.COMPLETED,
            final_state=ExecutionState.COMPLETED,
            objective="objetivo",
            original_plan=None,
            active_plan=None,
            completed_step_ids=("step-1",),
            summary="completado",
            trace=ExecutionTrace(),
            started_at=NOW,
            finished_at=NOW,
            duration=0.0,
        )


class _MutableClock:
    def __init__(self, current):
        self.current = current

    def __call__(self):
        return self.current

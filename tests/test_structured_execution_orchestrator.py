from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from agents.registry import AgentRegistry
from core.execution_plan_executor import ExecutionControl, ExecutionPlanExecutor
from core.execution_plan_validator import ExecutionPlanValidator, PlanValidationResult
from core.orchestrator import AtlasOrchestrator
from core.planner import Planner
from core.router import Router
from core.structured_execution import StructuredExecutionCoordinator
from memory.conversation import ConversationMemory
from tools.argument_schema import (
    ArgumentField,
    ArgumentSchema,
    ArgumentSchemaRegistry,
    ArgumentValidator,
)
from tools.base_tool import BaseTool
from tools.executor import ToolExecutor
from tools.intent_selector import ToolIntentRegistry, ToolSelector
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(
        self,
        name: str,
        output: Any,
        calls: list[str],
        *,
        requires_confirmation: bool = False,
        fail: bool = False,
    ) -> None:
        self._name = name
        self._output = output
        self._calls = calls
        self._requires_confirmation = requires_confirmation
        self._fail = fail
        self.contexts: list[ToolContext] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}."

    @property
    def requires_confirmation(self) -> bool:
        return self._requires_confirmation

    def execute(self, context: ToolContext) -> Any:
        self._calls.append(self._name)
        self.contexts.append(context)
        if self._fail:
            raise RuntimeError("fake failure")
        return self._output


class ChatAgentFake:
    name = "chat"
    description = "fake chat"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, model, messages):
        self.calls += 1
        return f"fallback:{model}:{messages[-1]['content']}"


class PlannerSpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def create_plan(self, prompt: str):
        self.calls.append(prompt)
        return SimpleNamespace(task="chat", objective=prompt)


def _structured_parts(
    calls: list[str] | None = None,
    *,
    read_fail: bool = False,
) -> tuple[StructuredExecutionCoordinator, ToolRegistry, Planner]:
    active_calls = calls if calls is not None else []
    registry = ToolRegistry()
    registry.register(SpyTool("read_file", "contenido", active_calls, fail=read_fail))
    registry.register(
        SpyTool("write_file", "written", active_calls, requires_confirmation=True)
    )

    intent_registry = ToolIntentRegistry()
    intent_registry.register("file.read", "read_file")
    intent_registry.register("file.write", "write_file")
    selector = ToolSelector(registry, intent_registry)

    schema_registry = ArgumentSchemaRegistry()
    schema_registry.register(
        ArgumentSchema("file.read", (ArgumentField("path", str, required=True),))
    )
    schema_registry.register(
        ArgumentSchema(
            "file.write",
            (
                ArgumentField("path", str, required=True),
                ArgumentField("content", (str, dict), required=True),
            ),
        )
    )
    validator = ArgumentValidator(schema_registry)
    planner = Planner(
        tool_registry=registry,
        tool_selector=selector,
        schema_registry=schema_registry,
        argument_validator=validator,
    )
    coordinator = StructuredExecutionCoordinator(
        planner=planner,
        validator=ExecutionPlanValidator(),
        executor=ExecutionPlanExecutor(registry, ToolExecutor(registry)),
    )
    return coordinator, registry, planner


def _orchestrator(
    *,
    structured_execution_enabled: bool,
    coordinator: StructuredExecutionCoordinator | None,
    planner: Any | None = None,
) -> tuple[AtlasOrchestrator, ChatAgentFake, PlannerSpy | None]:
    agent = ChatAgentFake()
    registry = AgentRegistry()
    registry.register(agent)
    planner_spy = planner if isinstance(planner, PlannerSpy) else None
    orchestrator = AtlasOrchestrator(
        planner=planner or PlannerSpy(),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=registry,
        write_file=SimpleNamespace(execute=lambda *_args: "write legacy"),
        structured_execution_coordinator=coordinator,
        structured_execution_enabled=structured_execution_enabled,
    )
    return orchestrator, agent, planner_spy


def test_feature_flag_disabled_preserves_previous_flow() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    planner = PlannerSpy()
    orchestrator, agent, planner_spy = _orchestrator(
        structured_execution_enabled=False,
        coordinator=coordinator,
        planner=planner,
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert response.startswith("fallback:model:chat:")
    assert agent.calls == 1
    assert planner_spy is planner
    assert planner.calls == ["Lee README.md"]
    assert calls == []


def test_conversational_request_uses_previous_flow_without_executor() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, agent, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt("hola", confirm=lambda _prompt: "")

    assert response.startswith("fallback:model:chat:")
    assert agent.calls == 1
    assert calls == []


def test_actionable_safe_request_uses_structured_pipeline() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, agent, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert "Ejecucion estructurada completada" in response
    assert agent.calls == 0
    assert calls == ["read_file"]


def test_dangerous_request_blocks_for_confirmation_without_tool_calls() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt(
        "Lee README.md y copia su contenido en resumen.txt",
        confirm=lambda _prompt: "",
    )

    assert "pendiente de confirmacion" in response
    assert "write_file" in response
    assert calls == []


def test_valid_confirmation_executes_the_same_pending_plan_once() -> None:
    calls: list[str] = []
    coordinator, registry, _ = _structured_parts(calls)
    response = coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    assert response.confirmation_token is not None

    confirmed = coordinator.confirm(
        response.confirmation_token,
        objective="Lee README.md y copia su contenido en resumen.txt",
    )

    assert confirmed.status == "completed"
    assert calls == ["read_file", "write_file"]
    assert registry.get("write_file").contexts[0].parameters == {  # type: ignore[attr-defined]
        "path": "resumen.txt",
        "content": "contenido",
    }


def test_changed_objective_after_validation_produces_mismatch() -> None:
    coordinator, _, _ = _structured_parts([])
    response = coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    assert response.confirmation_token is not None

    confirmed = coordinator.confirm(response.confirmation_token, objective="otro objetivo")

    assert confirmed.status == "validation_mismatch"
    assert confirmed.error_code == "VALIDATION_MISMATCH"


def test_validator_invalid_prevents_executor_call() -> None:
    class InvalidValidator:
        def validate(self, plan):
            return PlanValidationResult(
                is_valid=False,
                errors=["invalid"],
                warnings=[],
                requires_confirmation=False,
                status="invalid",
            )

    class ExplodingExecutor:
        def execute(self, *_args, **_kwargs):  # pragma: no cover - must not run
            raise AssertionError("executor must not run")

    _, _, planner = _structured_parts([])
    coordinator = StructuredExecutionCoordinator(
        planner=planner,
        validator=InvalidValidator(),  # type: ignore[arg-type]
        executor=ExplodingExecutor(),  # type: ignore[arg-type]
    )

    response = coordinator.handle("Lee README.md")

    assert response.status == "validation_failed"
    assert response.execution_result is None


def test_unknown_tool_from_structured_provider_is_not_executed() -> None:
    calls: list[str] = []
    _, registry, _ = _structured_parts(calls)
    planner = Planner(
        tool_registry=registry,
        plan_response_provider=lambda _goal, _catalog: (
            '{"goal":"x","steps":[{"id":1,"description":"Delete",'
            '"tool":"delete_file","arguments":{},"dependencies":[]}],'
            '"risks":[],"requires_confirmation":false}'
        ),
    )
    coordinator = StructuredExecutionCoordinator(
        planner=planner,
        validator=ExecutionPlanValidator(),
        executor=ExecutionPlanExecutor(registry),
    )

    response = coordinator.handle("Borra README.md")

    assert response.status == "planning_failed"
    assert response.error_code == "UNKNOWN_TOOL"
    assert calls == []


def test_tool_failure_returns_structured_failure() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls, read_fail=True)

    response = coordinator.handle("Lee README.md")

    assert response.handled is True
    assert response.status == "failed"
    assert response.execution_result is not None
    assert response.execution_result.failed_step == "step_1"
    assert calls == ["read_file"]


def test_interruption_returns_structured_status() -> None:
    coordinator, _, _ = _structured_parts([])

    response = coordinator.handle(
        "Lee README.md",
        control=ExecutionControl(should_stop=lambda: True),
    )

    assert response.status == "interrupted"
    assert response.execution_result is not None
    assert response.execution_result.interrupted is True


def test_cancellation_returns_structured_status() -> None:
    coordinator, _, _ = _structured_parts([])

    response = coordinator.handle(
        "Lee README.md",
        control=ExecutionControl(should_cancel=lambda: True),
    )

    assert response.status == "cancelled"
    assert response.execution_result is not None
    assert response.execution_result.cancelled is True


def test_insufficient_information_uses_fallback() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, agent, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt("Lee este archivo", confirm=lambda _prompt: "")

    assert response.startswith("fallback:model:chat:")
    assert agent.calls == 1
    assert calls == []


def test_real_parsing_error_is_not_hidden_as_fallback() -> None:
    _, registry, _ = _structured_parts([])
    planner = Planner(
        tool_registry=registry,
        plan_response_provider=lambda _goal, _catalog: "{bad json",
    )
    coordinator = StructuredExecutionCoordinator(
        planner=planner,
        validator=ExecutionPlanValidator(),
        executor=ExecutionPlanExecutor(registry),
    )
    orchestrator, agent, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert "No se pudo generar un plan estructurado" in response
    assert agent.calls == 0


def test_orchestrator_confirmation_method_delegates_without_regeneration() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )
    pending = coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    assert pending.confirmation_token is not None

    result = orchestrator.confirm_structured_execution(
        pending.confirmation_token,
        objective="Lee README.md y copia su contenido en resumen.txt",
    )

    assert result.status == "completed"
    assert calls == ["read_file", "write_file"]


def test_plan_signature_change_after_validation_produces_mismatch() -> None:
    coordinator, _, _ = _structured_parts([])
    pending = coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    assert pending.confirmation_token is not None
    stored = coordinator._pending_plans[pending.confirmation_token]  # type: ignore[attr-defined]
    changed_execution = replace(
        stored.execution,
        plan=replace(stored.execution.plan, goal="changed"),
    )
    coordinator._pending_plans[pending.confirmation_token] = replace(  # type: ignore[attr-defined]
        stored,
        execution=changed_execution,
    )

    result = coordinator.confirm(
        pending.confirmation_token,
        objective="Lee README.md y copia su contenido en resumen.txt",
    )

    assert result.status == "validation_mismatch"
    assert result.error_code == "VALIDATION_MISMATCH"


def test_conversational_confirmation_si_executes_pending_plan_once() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )

    pending = orchestrator.process_prompt(
        "Lee README.md y copia su contenido en resumen.txt",
        confirm=lambda _prompt: "",
    )
    response = orchestrator.process_prompt("si", confirm=lambda _prompt: "")
    second = orchestrator.process_prompt("si", confirm=lambda _prompt: "")

    assert "pendiente de confirmacion" in pending
    assert "Plan confirmado" in response
    assert calls == ["read_file", "write_file"]
    assert second.startswith("fallback:model:chat:")
    assert calls == ["read_file", "write_file"]


def test_conversational_confirmation_confirmo_and_adelante_are_accepted() -> None:
    for phrase in ("confirmo", "adelante"):
        calls: list[str] = []
        coordinator, _, _ = _structured_parts(calls)
        orchestrator, _, _ = _orchestrator(
            structured_execution_enabled=True,
            coordinator=coordinator,
        )

        orchestrator.process_prompt(
            "Lee README.md y copia su contenido en resumen.txt",
            confirm=lambda _prompt: "",
        )
        response = orchestrator.process_prompt(phrase, confirm=lambda _prompt: "")

        assert "Plan confirmado" in response
        assert calls == ["read_file", "write_file"]


def test_voice_like_confirmation_phrases_are_accepted() -> None:
    for phrase in ("sí adelante", "vale hazlo"):
        calls: list[str] = []
        coordinator, _, _ = _structured_parts(calls)
        orchestrator, _, _ = _orchestrator(
            structured_execution_enabled=True,
            coordinator=coordinator,
        )
        orchestrator.process_voice_prompt(
            "Lee README.md y copia su contenido en resumen.txt",
            confirm=lambda _prompt: "",
        )

        response = orchestrator.process_voice_prompt(phrase, confirm=lambda _prompt: "")

        assert "Plan confirmado" in response
        assert calls == ["read_file", "write_file"]


def test_rejection_no_and_cancela_cancel_pending_plan_without_tools() -> None:
    for phrase in ("no", "cancela", "no cancela"):
        calls: list[str] = []
        coordinator, _, _ = _structured_parts(calls)
        orchestrator, _, _ = _orchestrator(
            structured_execution_enabled=True,
            coordinator=coordinator,
        )
        orchestrator.process_prompt(
            "Lee README.md y copia su contenido en resumen.txt",
            confirm=lambda _prompt: "",
        )

        response = orchestrator.process_prompt(phrase, confirm=lambda _prompt: "")

        assert "fue cancelado" in response
        assert calls == []


def test_cancellation_invalidates_token_and_later_confirmation_has_no_pending_plan() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )
    orchestrator.process_prompt(
        "Lee README.md y copia su contenido en resumen.txt",
        confirm=lambda _prompt: "",
    )
    orchestrator.process_prompt("cancela", confirm=lambda _prompt: "")

    response = coordinator.confirm_pending()

    assert response.status == "no_pending_execution"
    assert response.error_code == "NO_PENDING_EXECUTION"
    assert calls == []


def test_show_pending_plan_and_risks_keeps_plan_pending_without_tools() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )
    orchestrator.process_prompt(
        "Lee README.md y copia su contenido en resumen.txt",
        confirm=lambda _prompt: "",
    )

    plan = orchestrator.process_prompt("muéstrame el plan", confirm=lambda _prompt: "")
    risks = orchestrator.process_prompt("cuáles son los riesgos", confirm=lambda _prompt: "")
    confirmed = orchestrator.process_prompt("confirmo", confirm=lambda _prompt: "")

    assert "Objetivo:" in plan
    assert "Herramientas:" in plan
    assert "Riesgos:" in risks
    assert "Plan confirmado" in confirmed
    assert calls == ["read_file", "write_file"]


def test_no_pending_confirmation_cancel_or_show_return_no_pending_message() -> None:
    coordinator, _, _ = _structured_parts([])

    assert coordinator.confirm_pending().error_code == "NO_PENDING_EXECUTION"
    assert coordinator.cancel_pending().error_code == "NO_PENDING_EXECUTION"
    assert coordinator.show_pending().error_code == "NO_PENDING_EXECUTION"


def test_ambiguous_and_unrelated_replies_do_not_execute_or_fallback() -> None:
    for phrase in ("quizá", "luego", "cuéntame otra cosa"):
        calls: list[str] = []
        coordinator, _, _ = _structured_parts(calls)
        orchestrator, agent, _ = _orchestrator(
            structured_execution_enabled=True,
            coordinator=coordinator,
        )
        orchestrator.process_prompt(
            "Lee README.md y copia su contenido en resumen.txt",
            confirm=lambda _prompt: "",
        )

        response = orchestrator.process_prompt(phrase, confirm=lambda _prompt: "")

        assert "No he ejecutado nada" in response
        assert calls == []
        assert agent.calls == 0


def test_confirmation_with_goal_change_is_ambiguous_and_does_not_execute() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, agent, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )
    orchestrator.process_prompt(
        "Lee README.md y copia su contenido en resumen.txt",
        confirm=lambda _prompt: "",
    )

    response = orchestrator.process_prompt(
        "sí, pero usa otro archivo",
        confirm=lambda _prompt: "",
    )

    assert "No he ejecutado nada" in response
    assert calls == []
    assert agent.calls == 0


def test_new_actionable_request_while_pending_does_not_bypass_confirmation() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, agent, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )
    orchestrator.process_prompt(
        "Lee README.md y copia su contenido en resumen.txt",
        confirm=lambda _prompt: "",
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert "Hay un plan pendiente" in response
    assert calls == []
    assert agent.calls == 0


def test_disabling_flag_with_pending_plan_cancels_without_execution() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )
    orchestrator.process_prompt(
        "Lee README.md y copia su contenido en resumen.txt",
        confirm=lambda _prompt: "",
    )
    orchestrator._structured_execution_enabled = False  # type: ignore[attr-defined]

    response = orchestrator.process_prompt("confirmo", confirm=lambda _prompt: "")

    assert "fue cancelado" in response
    assert calls == []


def test_pending_messages_do_not_show_token_or_signature() -> None:
    coordinator, _, _ = _structured_parts([])
    response = coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    assert response.confirmation_token is not None

    assert response.confirmation_token not in response.message
    assert "Token" not in response.message
    assert "Firma" not in response.message


def test_confirming_pending_plan_does_not_call_planner_again() -> None:
    class PlannerCounting(Planner):
        def __init__(self, wrapped: Planner) -> None:
            self.wrapped = wrapped
            self.calls = 0

        def generate_execution_plan(self, prompt: str):
            self.calls += 1
            return self.wrapped.generate_execution_plan(prompt)

    calls: list[str] = []
    _, registry, planner = _structured_parts(calls)
    counting = PlannerCounting(planner)
    coordinator = StructuredExecutionCoordinator(
        planner=counting,
        validator=ExecutionPlanValidator(),
        executor=ExecutionPlanExecutor(registry, ToolExecutor(registry)),
    )

    coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    coordinator.confirm_pending()

    assert counting.calls == 1
    assert calls == ["read_file", "write_file"]


def test_arguments_modified_after_validation_produce_mismatch_and_invalidate_plan() -> None:
    coordinator, _, _ = _structured_parts([])
    pending = coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    assert pending.confirmation_token is not None
    stored = coordinator._pending_plans[pending.confirmation_token]  # type: ignore[attr-defined]
    first, second = stored.execution.plan.ordered_steps
    changed_step = replace(
        second,
        arguments={"path": "otro.txt", "content": {"$ref": "steps.step_1.output"}},
    )
    changed_plan = replace(stored.execution.plan, ordered_steps=(first, changed_step))
    coordinator._pending_plans[pending.confirmation_token] = replace(  # type: ignore[attr-defined]
        stored,
        execution=replace(stored.execution, plan=changed_plan),
    )

    result = coordinator.confirm(
        pending.confirmation_token,
        objective="Lee README.md y copia su contenido en resumen.txt",
    )
    second_attempt = coordinator.confirm(pending.confirmation_token)

    assert result.status == "validation_mismatch"
    assert second_attempt.error_code == "NO_PENDING_EXECUTION"

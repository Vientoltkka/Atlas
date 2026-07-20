from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
from typing import Any

from agents.registry import AgentRegistry
from core.execution_plan_executor import (
    ExecutionControl,
    ExecutionPlanExecutor,
    ExecutionProgress,
    ResumableExecutionState,
)
from core.execution_plan_validator import ExecutionPlanValidator, PlanValidationResult
from core.hybrid_execution_planner import StructuredPlanningProgress
from core.orchestrator import AtlasOrchestrator
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult, Planner
from core.resumable_execution_store import JsonResumableExecutionStore
from core.router import Router
from core.structured_execution import StructuredExecutionCoordinator, StructuredExecutionResponse
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


class CoordinatorFake:
    def __init__(
        self,
        response: StructuredExecutionResponse | None = None,
        *,
        raise_keyboard_interrupt: bool = False,
    ) -> None:
        self.response = response or StructuredExecutionResponse(
            handled=True,
            status="planned",
            message="Plan estructurado generado.",
        )
        self.raise_keyboard_interrupt = raise_keyboard_interrupt
        self.calls: list[dict[str, Any]] = []
        self.pending = False
        self.on_handle = None

    def has_pending_execution(self) -> bool:
        return self.pending

    def handle(self, objective: str, **kwargs):
        self.calls.append({"objective": objective, **kwargs})
        if self.on_handle is not None:
            self.on_handle(kwargs)
        if self.raise_keyboard_interrupt:
            raise KeyboardInterrupt
        return self.response

    def cancel_pending(self):
        self.pending = False
        return StructuredExecutionResponse(
            handled=True,
            status="pending_execution_cancelled",
            message="El plan pendiente fue cancelado. No se ejecuto ninguna herramienta.",
        )


def _structured_parts(
    calls: list[str] | None = None,
    *,
    read_fail: bool = False,
    resumable_store: Any | None = None,
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
        resumable_store=resumable_store,
    )
    return coordinator, registry, planner


def _resumable_state_from_response(
    response: StructuredExecutionResponse,
    *,
    objective: str,
    confirmation_granted: bool = False,
) -> ResumableExecutionState:
    assert response.plan is not None
    assert response.validation_result is not None
    assert response.execution_result is not None
    return ResumableExecutionState(
        objective=objective,
        original_plan=response.plan,
        validation_result=response.validation_result,
        validated_plan_signature=response.validation_result.plan_signature,
        completed_step_ids=tuple(response.execution_result.completed_steps),
        pending_step_ids=tuple(response.execution_result.pending_steps),
        failed_step_ids=tuple(response.execution_result.failed_steps),
        interrupted_step_id=response.execution_result.current_step,
        previous_results={
            step.step_id: step.output
            for step in response.execution_result.step_results
            if step.success and step.status == "completed"
        },
        resumable=response.execution_result.resumable,
        interruption_reason=response.execution_result.interruption_reason,
        confirmation_granted=confirmation_granted,
    )


def _orchestrator(
    *,
    structured_execution_enabled: bool,
    coordinator: StructuredExecutionCoordinator | None,
    planner: Any | None = None,
    structured_plan_streaming_enabled: bool = False,
    structured_plan_execution_enabled: bool = False,
    structured_planning_progress_enabled: bool = True,
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
        structured_plan_streaming_enabled=structured_plan_streaming_enabled,
        structured_plan_execution_enabled=structured_plan_execution_enabled,
        structured_planning_progress_enabled=structured_planning_progress_enabled,
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


def test_streaming_flag_false_uses_non_streaming_structured_route() -> None:
    coordinator = CoordinatorFake()
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_streaming_enabled=False,
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert response == "Plan estructurado generado."
    assert coordinator.calls[0]["on_planning_progress"] is None
    assert coordinator.calls[0]["execute_after_planning"] is False


def test_execution_flag_true_allows_structured_route_to_execute() -> None:
    coordinator = CoordinatorFake()
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert response == "Plan estructurado generado."
    assert isinstance(coordinator.calls[0]["control"], ExecutionControl)
    assert callable(coordinator.calls[0]["on_execution_progress"])
    assert coordinator.calls[0]["execute_after_planning"] is True


def test_streaming_flag_true_uses_streaming_structured_route_without_execution() -> None:
    coordinator = CoordinatorFake()
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_streaming_enabled=True,
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert response == "Plan estructurado generado."
    assert callable(coordinator.calls[0]["on_planning_progress"])
    assert isinstance(coordinator.calls[0]["planning_control"], ExecutionControl)
    assert coordinator.calls[0]["execute_after_planning"] is False


def test_structured_execution_disabled_preserves_old_flow_even_if_streaming_flag_true() -> None:
    coordinator = CoordinatorFake()
    planner = PlannerSpy()
    orchestrator, agent, _ = _orchestrator(
        structured_execution_enabled=False,
        structured_plan_streaming_enabled=True,
        coordinator=coordinator,  # type: ignore[arg-type]
        planner=planner,
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert response.startswith("fallback:model:chat:")
    assert coordinator.calls == []
    assert agent.calls == 1
    assert planner.calls == ["Lee README.md"]


def test_streaming_progress_messages_are_safe_and_ordered(capsys) -> None:
    coordinator = CoordinatorFake()

    def emit_progress(kwargs):
        on_progress = kwargs["on_planning_progress"]
        on_progress(StructuredPlanningProgress("preparing", 0, message='{"secret":"x"}'))
        on_progress(StructuredPlanningProgress("waiting_model", 1))
        on_progress(StructuredPlanningProgress("receiving", 2, received_chars=1, chunk_count=1, first_token_received=True, message="{"))
        on_progress(StructuredPlanningProgress("completed", 3, received_chars=10, chunk_count=2, message="}"))

    coordinator.on_handle = emit_progress
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_streaming_enabled=True,
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    output = capsys.readouterr().out
    assert "Preparando el plan" in output
    assert "Esperando al modelo" in output
    assert "El modelo ha comenzado a responder" in output
    assert "Plan generado" in output
    assert "secret" not in output
    assert "{" not in output
    assert output.index("Preparando el plan") < output.index("Esperando al modelo")
    assert output.index("Esperando al modelo") < output.index("El modelo ha comenzado")


def test_first_token_and_receiving_messages_are_not_repeated(monkeypatch, capsys) -> None:
    times = iter([0.0, 0.0, 6.0, 12.0])
    monkeypatch.setattr("core.orchestrator.time.monotonic", lambda: next(times))
    coordinator = CoordinatorFake()

    def emit_progress(kwargs):
        on_progress = kwargs["on_planning_progress"]
        on_progress(StructuredPlanningProgress("receiving", 1, received_chars=1, chunk_count=1, first_token_received=True))
        on_progress(StructuredPlanningProgress("receiving", 6000, received_chars=2, chunk_count=2, first_token_received=True))
        on_progress(StructuredPlanningProgress("receiving", 12000, received_chars=3, chunk_count=3, first_token_received=True))

    coordinator.on_handle = emit_progress
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_streaming_enabled=True,
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    output = capsys.readouterr().out
    assert output.count("El modelo ha comenzado a responder") == 1
    assert output.count("Generando el plan") == 1


def test_progress_messages_can_be_disabled(capsys) -> None:
    coordinator = CoordinatorFake()

    def emit_progress(kwargs):
        assert kwargs["on_planning_progress"] is None

    coordinator.on_handle = emit_progress
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_streaming_enabled=True,
        structured_planning_progress_enabled=False,
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert "Preparando el plan" not in capsys.readouterr().out


def test_keyboard_interrupt_cancels_streaming_plan_and_keeps_orchestrator_usable() -> None:
    coordinator = CoordinatorFake(raise_keyboard_interrupt=True)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_streaming_enabled=True,
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    cancelled = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")
    coordinator.raise_keyboard_interrupt = False
    next_response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert cancelled == "Planificación cancelada."
    assert next_response == "Plan estructurado generado."
    assert len(coordinator.calls) == 2


def test_second_request_during_streaming_is_rejected_without_new_plan() -> None:
    coordinator = CoordinatorFake()
    nested_responses: list[str] = []
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_streaming_enabled=True,
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    def reenter(_kwargs):
        nested_responses.append(
            orchestrator.process_prompt("Lee otro archivo", confirm=lambda _prompt: "")
        )

    coordinator.on_handle = reenter
    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert response == "Plan estructurado generado."
    assert nested_responses == ["Atlas está generando un plan."]
    assert len(coordinator.calls) == 1


def test_actionable_safe_request_uses_structured_pipeline() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, agent, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert "Ejecucion estructurada completada" in response
    assert agent.calls == 0
    assert calls == ["read_file"]


def test_execution_disabled_by_default_returns_plan_without_tool_execution() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, agent, _ = _orchestrator(
        structured_execution_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert "Plan estructurado generado" in response
    assert "La ejecucion no se realizo" in response
    assert agent.calls == 0
    assert calls == []


def test_execution_enabled_shows_safe_progress_messages(capsys) -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    output = capsys.readouterr().out
    assert "Plan validado" in output
    assert "Iniciando ejecuci" in output
    assert "Preparando la ejecuci" in output
    assert "Ejecutando paso 1 de 1" in output
    assert "Paso 1 completado" in output
    assert "Ejecuci" in output and "completada" in output
    assert "plan_signature" not in output
    assert "raw_response" not in output
    assert "confirmation_token" not in output
    assert "Ejecucion estructurada completada" in response
    assert calls == ["read_file"]


def test_retry_progress_messages_are_safe(capsys) -> None:
    coordinator = CoordinatorFake()

    def emit_progress(kwargs):
        on_execution_progress = kwargs["on_execution_progress"]
        on_execution_progress(
            ExecutionProgress(
                "step_retry_scheduled",
                step_index=1,
                total_steps=1,
                attempt_number=2,
                max_attempts=2,
                retry_reason="temporary_unavailable",
            )
        )
        on_execution_progress(
            ExecutionProgress(
                "step_completed_after_retry",
                step_index=1,
                total_steps=1,
                attempt_number=2,
                max_attempts=2,
            )
        )
        on_execution_progress(
            ExecutionProgress(
                "step_retry_exhausted",
                step_index=1,
                total_steps=1,
                attempt_number=2,
                max_attempts=2,
            )
        )

    coordinator.on_handle = emit_progress
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    output = capsys.readouterr().out
    assert "Reintentando paso 1, intento 2 de 2" in output
    assert "Paso 1 se completó tras un reintento" in output
    assert "Paso 1 agotó sus intentos" in output
    assert "plan_signature" not in output
    assert "raw_response" not in output


def test_second_request_during_execution_is_rejected_without_new_plan() -> None:
    coordinator = CoordinatorFake()
    nested_responses: list[str] = []
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    def reenter(kwargs):
        on_execution_progress = kwargs["on_execution_progress"]
        on_execution_progress(ExecutionProgress("preparing", total_steps=1))
        nested_responses.append(
            orchestrator.process_prompt("Lee otro archivo", confirm=lambda _prompt: "")
        )

    coordinator.on_handle = reenter
    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert response == "Plan estructurado generado."
    assert nested_responses == ["Atlas está ejecutando un plan."]
    assert len(coordinator.calls) == 1


def test_streaming_real_coordinator_generates_safe_plan_without_tool_execution() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, agent, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_streaming_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert "Plan estructurado generado" in response
    assert "La ejecucion no se realizo en esta fase" in response
    assert "Ejecucion estructurada completada" not in response
    assert agent.calls == 0
    assert calls == []


def test_dangerous_request_blocks_for_confirmation_without_tool_calls() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt(
        "Lee README.md y copia su contenido en resumen.txt",
        confirm=lambda _prompt: "",
    )

    assert "pendiente de confirmacion" in response
    assert "write_file" in response
    assert calls == []


def test_streaming_confirmation_keeps_pending_plan_without_tool_calls() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_streaming_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt(
        "Lee README.md y copia su contenido en resumen.txt",
        confirm=lambda _prompt: "",
    )

    assert "pendiente de confirmacion" in response
    assert coordinator.has_pending_execution()
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


def test_partial_execution_response_uses_safe_visible_summary() -> None:
    class FixedPlanner:
        def __init__(self, plan: ExecutionPlan) -> None:
            self.plan = plan

        def generate_execution_plan(self, _objective: str, **_kwargs):
            return PlanGenerationResult(
                success=True,
                plan=self.plan,
                generation_attempted=True,
            )

    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(SpyTool("first_tool", {"secret": "raw"}, calls))
    registry.register(SpyTool("second_tool", "unused", calls, fail=True))
    registry.register(SpyTool("third_tool", "unused", calls))
    plan = ExecutionPlan(
        goal="partial fake plan",
        ordered_steps=(
            ExecutionStep("step_1", "first", "first_tool"),
            ExecutionStep("step_2", "second", "second_tool", ("step_1",)),
            ExecutionStep("step_3", "third", "third_tool", ("step_2",)),
        ),
        estimated_steps=3,
        required_tools=("first_tool", "second_tool", "third_tool"),
        detected_risks=(),
        requires_confirmation=False,
    )
    coordinator = StructuredExecutionCoordinator(
        planner=FixedPlanner(plan),  # type: ignore[arg-type]
        validator=ExecutionPlanValidator(),
        executor=ExecutionPlanExecutor(registry, ToolExecutor(registry)),
    )

    response = coordinator.handle("ejecuta parcial")

    assert response.status == "partially_completed"
    assert response.partial_state is not None
    assert response.partial_state.completed_step_ids == ("step_1",)
    assert response.partial_state.failed_step_ids == ("step_2",)
    assert response.partial_state.skipped_step_ids == ("step_3",)
    assert response.message == "\n".join(
        [
            "Atlas: Ejecucion completada parcialmente.",
            "Atlas: Se completaron 1 de 3 pasos.",
            "Atlas: El paso step_2 ha fallado.",
            "Atlas: Quedan 0 pasos pendientes.",
            "Atlas: La ejecucion no puede reanudarse.",
        ]
    )
    assert "raw" not in response.message
    assert calls == ["first_tool", "second_tool"]


def test_interruption_returns_structured_status() -> None:
    coordinator, _, _ = _structured_parts([])

    response = coordinator.handle(
        "Lee README.md",
        control=ExecutionControl(should_stop=lambda: True),
    )

    assert response.status == "interrupted"
    assert response.execution_result is not None
    assert response.execution_result.interrupted is True


def test_interrupted_structured_execution_can_resume_without_regenerating_plan() -> None:
    class PlannerCounting(Planner):
        def __init__(self, wrapped: Planner) -> None:
            self.wrapped = wrapped
            self.calls = 0

        def generate_execution_plan(self, prompt: str, **kwargs):
            self.calls += 1
            return self.wrapped.generate_execution_plan(prompt, **kwargs)

    calls: list[str] = []
    _, registry, planner = _structured_parts(calls)
    counting = PlannerCounting(planner)
    coordinator = StructuredExecutionCoordinator(
        planner=counting,
        validator=ExecutionPlanValidator(),
        executor=ExecutionPlanExecutor(registry, ToolExecutor(registry)),
    )
    pending = coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    assert pending.status == "confirmation_required"

    interrupted = coordinator.confirm_pending(
        control=ExecutionControl(should_stop=lambda: len(calls) >= 1),
    )
    resumed = coordinator.resume_pending_execution()

    assert interrupted.status == "interrupted"
    assert interrupted.execution_result is not None
    assert interrupted.execution_result.resumable is True
    assert coordinator.has_resumable_execution() is False
    assert resumed.status == "completed"
    assert counting.calls == 1
    assert calls == ["read_file", "write_file"]
    assert registry.get("write_file").contexts[0].parameters == {  # type: ignore[attr-defined]
        "path": "resumen.txt",
        "content": "contenido",
    }


def test_interrupted_execution_is_persisted_and_loaded_after_rebuild(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    first_calls: list[str] = []
    store = JsonResumableExecutionStore(state_path)
    coordinator, _, _ = _structured_parts(first_calls, resumable_store=store)
    coordinator.handle("Lee README.md y copia su contenido en resumen.txt")

    interrupted = coordinator.confirm_pending(
        control=ExecutionControl(should_stop=lambda: len(first_calls) >= 1),
    )

    assert interrupted.status == "interrupted"
    assert state_path.exists()

    second_calls: list[str] = []
    rebuilt_store = JsonResumableExecutionStore(state_path)
    rebuilt, registry, _ = _structured_parts(
        second_calls,
        resumable_store=rebuilt_store,
    )
    loaded = rebuilt.load_persisted_resumable_execution()
    resumed = rebuilt.resume_pending_execution()

    assert loaded.status == "resumable_execution_loaded"
    assert resumed.status == "completed"
    assert second_calls == ["write_file"]
    assert registry.get("write_file").contexts[0].parameters == {  # type: ignore[attr-defined]
        "path": "resumen.txt",
        "content": "contenido",
    }
    assert not state_path.exists()


def test_orchestrator_detects_persisted_state_without_executing_tools(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    calls: list[str] = []
    store = JsonResumableExecutionStore(state_path)
    coordinator, _, _ = _structured_parts(calls, resumable_store=store)
    coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    coordinator.confirm_pending(
        control=ExecutionControl(should_stop=lambda: len(calls) >= 1),
    )

    rebuilt_calls: list[str] = []
    rebuilt, _, _ = _structured_parts(
        rebuilt_calls,
        resumable_store=JsonResumableExecutionStore(state_path),
    )
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=rebuilt,
    )

    response = orchestrator._load_persisted_structured_execution()  # type: ignore[attr-defined]

    assert response is not None
    assert response.status == "resumable_execution_loaded"
    assert "puede reanudarse" in response.message
    assert rebuilt_calls == []


def test_orchestrator_resumes_persisted_state_and_deletes_file(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    first_calls: list[str] = []
    coordinator, _, _ = _structured_parts(
        first_calls,
        resumable_store=JsonResumableExecutionStore(state_path),
    )
    coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    coordinator.confirm_pending(
        control=ExecutionControl(should_stop=lambda: len(first_calls) >= 1),
    )
    second_calls: list[str] = []
    rebuilt, _, _ = _structured_parts(
        second_calls,
        resumable_store=JsonResumableExecutionStore(state_path),
    )
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=rebuilt,
    )

    response = orchestrator.process_prompt("reanuda", confirm=lambda _prompt: "")
    second = orchestrator.process_prompt("reanuda", confirm=lambda _prompt: "")

    assert "Ejecucion estructurada completada" in response
    assert second == "No hay ninguna ejecución pendiente que pueda reanudarse."
    assert second_calls == ["write_file"]
    assert not state_path.exists()


def test_tampered_persisted_state_never_reaches_executor(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(
        calls,
        resumable_store=JsonResumableExecutionStore(state_path),
    )
    coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    coordinator.confirm_pending(
        control=ExecutionControl(should_stop=lambda: len(calls) >= 1),
    )
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["validated_plan_signature"] = "tampered"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    rebuilt_calls: list[str] = []
    rebuilt, _, _ = _structured_parts(
        rebuilt_calls,
        resumable_store=JsonResumableExecutionStore(state_path),
    )
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=rebuilt,
    )

    response = orchestrator.process_prompt("reanuda", confirm=lambda _prompt: "")

    assert response == "No se puede reanudar la ejecución."
    assert rebuilt_calls == []
    assert state_path.exists()


def test_discard_resume_phrase_deletes_persisted_state(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(
        calls,
        resumable_store=JsonResumableExecutionStore(state_path),
    )
    coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    coordinator.confirm_pending(
        control=ExecutionControl(should_stop=lambda: len(calls) >= 1),
    )
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt(
        "cancela la ejecución pendiente",
        confirm=lambda _prompt: "",
    )

    assert response == "La ejecución pendiente fue descartada."
    assert not state_path.exists()
    assert calls == ["read_file"]


def test_resumable_state_signature_change_blocks_resume_without_tools() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    interrupted = coordinator.confirm_pending(
        control=ExecutionControl(should_stop=lambda: len(calls) >= 1),
    )
    assert coordinator.resumable_execution() is not None
    changed_state = replace(
        coordinator.resumable_execution(),  # type: ignore[arg-type]
        original_plan=replace(interrupted.plan, goal="changed"),
    )
    coordinator._resumable_execution = changed_state  # type: ignore[attr-defined]

    response = coordinator.resume_pending_execution()

    assert response.status == "rejected"
    assert response.error_code == "VALIDATION_MISMATCH"
    assert calls == ["read_file"]


def test_resume_confirmation_block_keeps_resumable_state_until_valid_confirmation() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    interrupted = coordinator.confirm_pending(
        control=ExecutionControl(should_stop=lambda: len(calls) >= 1),
    )
    state = replace(
        _resumable_state_from_response(
            interrupted,
            objective="Lee README.md y copia su contenido en resumen.txt",
            confirmation_granted=False,
        ),
        confirmation_granted=False,
    )
    coordinator._resumable_execution = state  # type: ignore[attr-defined]

    blocked = coordinator.resume_pending_execution()
    resumed = coordinator.resume_pending_execution(confirmation_granted=True)

    assert blocked.status == "blocked_confirmation"
    assert blocked.requires_confirmation is True
    assert resumed.status == "completed"
    assert calls == ["read_file", "write_file"]


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
        structured_plan_execution_enabled=True,
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
        structured_plan_execution_enabled=True,
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
            structured_plan_execution_enabled=True,
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
            structured_plan_execution_enabled=True,
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
        structured_plan_execution_enabled=True,
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
        structured_plan_execution_enabled=True,
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


def test_conversational_resume_phrases_are_accepted(capsys) -> None:
    for phrase in ("reanuda", "continúa", "continuar ejecución", "sigue con el plan", "retoma", "retoma la ejecución"):
        calls: list[str] = []
        coordinator, _, _ = _structured_parts(calls)
        orchestrator, _, _ = _orchestrator(
            structured_execution_enabled=True,
            structured_plan_execution_enabled=True,
            coordinator=coordinator,
        )
        orchestrator.process_prompt(
            "Lee README.md y copia su contenido en resumen.txt",
            confirm=lambda _prompt: "",
        )
        orchestrator.process_prompt("confirmo", confirm=lambda _prompt: "")
        coordinator._resumable_execution = None  # type: ignore[attr-defined]
        coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
        coordinator.confirm_pending(
            control=ExecutionControl(should_stop=lambda: len(calls) >= 3),
        )

        response = orchestrator.process_prompt(phrase, confirm=lambda _prompt: "")

        assert "Ejecucion estructurada completada" in response
        assert calls[-1] == "write_file"

    output = capsys.readouterr().out
    assert "Reanudando ejecuci" in output
    assert "Se conservar" in output
    assert "Continuando desde el paso 2 de 2" in output


def test_resume_without_resumable_execution_returns_safe_message() -> None:
    coordinator, _, _ = _structured_parts([])
    orchestrator, agent, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=coordinator,
    )

    response = orchestrator.process_prompt("reanuda", confirm=lambda _prompt: "")

    assert response == "No hay ninguna ejecución pendiente que pueda reanudarse."
    assert agent.calls == 0


def test_new_request_does_not_resume_interrupted_execution_implicitly() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=coordinator,
    )
    coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    coordinator.confirm_pending(control=ExecutionControl(should_stop=lambda: len(calls) >= 1))

    response = orchestrator.process_prompt("Lee README.md", confirm=lambda _prompt: "")

    assert "Ejecucion estructurada completada" in response
    assert calls == ["read_file", "read_file"]
    assert coordinator.has_resumable_execution() is True


def test_double_resume_does_not_duplicate_tools() -> None:
    calls: list[str] = []
    coordinator, _, _ = _structured_parts(calls)
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=coordinator,
    )
    coordinator.handle("Lee README.md y copia su contenido en resumen.txt")
    coordinator.confirm_pending(control=ExecutionControl(should_stop=lambda: len(calls) >= 1))

    first = orchestrator.process_prompt("reanuda", confirm=lambda _prompt: "")
    second = orchestrator.process_prompt("reanuda", confirm=lambda _prompt: "")

    assert "Ejecucion estructurada completada" in first
    assert second == "No hay ninguna ejecución pendiente que pueda reanudarse."
    assert calls == ["read_file", "write_file"]


def test_resume_request_during_execution_is_rejected() -> None:
    coordinator, _, _ = _structured_parts([])
    orchestrator, _, _ = _orchestrator(
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        coordinator=coordinator,
    )
    orchestrator._structured_execution_active = True  # type: ignore[attr-defined]

    response = orchestrator.process_prompt("reanuda", confirm=lambda _prompt: "")

    assert response == "Atlas está ejecutando un plan."


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

        def generate_execution_plan(self, prompt: str, **kwargs):
            self.calls += 1
            return self.wrapped.generate_execution_plan(prompt, **kwargs)

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

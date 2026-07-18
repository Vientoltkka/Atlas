from __future__ import annotations

from pathlib import Path
from typing import Any

from bootstrap.bootstrap import Bootstrap
from core.orchestrator import AtlasOrchestrator
from tools.execution_coordinator import (
    ExecutionCoordinationResult,
    ExecutionCoordinationStatus,
)
from tools.execution_decision import ExecutionDecision, ExecutionMode
from use_cases.execution_conversation import ExecutionConversationController


class _PlannerFake:
    def create_plan(self, prompt: str):
        return {"prompt": prompt}


class _RouterFake:
    def route(self, plan):
        return "chat"


class _ModelManagerFake:
    def choose_model(self, agent_name: str) -> str:
        return "fake-model"


class _MemoryFake:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def history(self) -> list[dict[str, str]]:
        return self.messages.copy()


class _AgentFake:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, model: str, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        return "respuesta conversacional existente"


class _RegistryFake:
    def __init__(self, agent: _AgentFake) -> None:
        self._agent = agent

    def get(self, name: str):
        if name == "chat":
            return self._agent
        return None


class _WriteFileFake:
    def execute(self, path: str, content: str) -> str:
        return "ok"


def _controller() -> ExecutionConversationController:
    return ExecutionConversationController(Bootstrap.build_execution_coordinator())


def test_direct_response_keeps_conversational_fallback() -> None:
    controller = _controller()

    outcome = controller.handle("Hola")

    assert outcome.direct_response_required is True
    assert outcome.text == ""
    assert outcome.result.status == ExecutionCoordinationStatus.DIRECT_RESPONSE_REQUIRED


def test_orchestrator_uses_existing_conversation_for_direct_response() -> None:
    agent = _AgentFake()
    orchestrator = AtlasOrchestrator(
        planner=_PlannerFake(),
        router=_RouterFake(),
        model_manager=_ModelManagerFake(),
        memory=_MemoryFake(),
        registry=_RegistryFake(agent),
        write_file=_WriteFileFake(),
        execution_conversation=_controller(),
    )

    response = orchestrator.process_prompt("Explicame Clean Architecture", confirm=input)

    assert response == "respuesta conversacional existente"
    assert agent.calls == 1


def test_text_loop_uses_execution_controller_before_other_handlers(
    monkeypatch,
    capsys,
) -> None:
    class _ExecutionConversationFake:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def handle(self, prompt: str):
            self.calls.append(prompt)
            return type(
                "Outcome",
                (),
                {
                    "direct_response_required": False,
                    "text": "resultado de herramienta",
                },
            )()

    class _VoiceConversationFake:
        def execute(self, **kwargs):
            raise AssertionError("voice handler must not run first")

    execution = _ExecutionConversationFake()
    orchestrator = AtlasOrchestrator(
        planner=_PlannerFake(),
        router=_RouterFake(),
        model_manager=_ModelManagerFake(),
        memory=_MemoryFake(),
        registry=_RegistryFake(_AgentFake()),
        write_file=_WriteFileFake(),
        execution_conversation=execution,
        voice_conversation=_VoiceConversationFake(),
    )
    prompts = iter(["Lee README.md", "salir"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(prompts))

    orchestrator.start()

    output = capsys.readouterr().out
    assert execution.calls == ["Lee README.md"]
    assert "resultado de herramienta" in output


def test_read_file_executes_and_presents_content() -> None:
    outcome = _controller().handle("Lee README.md")

    assert outcome.direct_response_required is False
    assert outcome.result.status == ExecutionCoordinationStatus.EXECUTED
    assert "Atlas" in outcome.text
    assert "ToolRunResult" not in outcome.text


def test_missing_information_asks_for_required_field() -> None:
    outcome = _controller().handle("Lee este archivo")

    assert outcome.result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert "path" in outcome.text
    assert outcome.result.executed is False


def test_ambiguous_request_asks_for_clarification() -> None:
    outcome = _controller().handle("Escribe algo en un archivo")

    assert outcome.result.status == ExecutionCoordinationStatus.AMBIGUOUS_REQUEST
    assert "path" in outcome.text
    assert "content" in outcome.text
    assert outcome.result.executed is False


def test_unsupported_request_is_not_conversational() -> None:
    outcome = _controller().handle("Borra README.md")

    assert outcome.direct_response_required is False
    assert outcome.result.status == ExecutionCoordinationStatus.UNSUPPORTED
    assert "no dispone" in outcome.text


def test_write_confirmation_yes_executes_once(tmp_path: Path) -> None:
    target = tmp_path / "prueba.txt"
    controller = _controller()

    pending = controller.handle(f"Escribe hola en {target}")

    assert pending.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert target.exists() is False
    assert controller.pending_confirmation_id is not None
    assert str(pending.result.confirmation_id) not in pending.text

    confirmed = controller.handle("si")
    repeated = controller.handle("si")

    assert confirmed.result.status == ExecutionCoordinationStatus.EXECUTED
    assert target.read_text(encoding="utf-8") == "hola"
    assert controller.pending_confirmation_id is None
    assert repeated.result.status is ExecutionCoordinationStatus.UNSUPPORTED
    assert repeated.result.executed is False
    assert target.read_text(encoding="utf-8") == "hola"


def test_write_confirmation_no_cancels_without_creating_file(tmp_path: Path) -> None:
    target = tmp_path / "prueba.txt"
    controller = _controller()

    controller.handle(f"Escribe hola en {target}")
    cancelled = controller.handle("no")

    assert cancelled.result.status == ExecutionCoordinationStatus.CANCELLED
    assert "cancelada" in cancelled.text.lower()
    assert target.exists() is False
    assert controller.pending_confirmation_id is None


def test_ambiguous_confirmation_keeps_pending_confirmation(tmp_path: Path) -> None:
    target = tmp_path / "prueba.txt"
    controller = _controller()

    controller.handle(f"Escribe hola en {target}")
    ambiguous = controller.handle("quizas")

    assert ambiguous.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert controller.pending_confirmation_id is not None
    assert target.exists() is False
    assert "Responde si/s" in ambiguous.text

    confirmed = controller.handle("s")

    assert confirmed.result.status == ExecutionCoordinationStatus.EXECUTED
    assert target.read_text(encoding="utf-8") == "hola"


def test_new_command_while_pending_must_resolve_confirmation_first(tmp_path: Path) -> None:
    target = tmp_path / "prueba.txt"
    controller = _controller()

    controller.handle(f"Escribe hola en {target}")
    response = controller.handle("Lee README.md")

    assert response.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert controller.pending_confirmation_id is not None
    assert target.exists() is False
    assert "confirmacion" in response.text.lower()


def test_chain_pauses_and_confirmation_presents_final_result(tmp_path: Path) -> None:
    target = tmp_path / "resumen.txt"
    controller = _controller()

    pending = controller.handle(f"Lee README.md y copia su contenido en {target}")

    assert pending.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert pending.result.executed is True
    assert target.exists() is False

    confirmed = controller.handle("s")

    assert confirmed.result.status == ExecutionCoordinationStatus.EXECUTED
    assert target.exists() is True
    assert "Cadena completada" in confirmed.text
    assert "ToolChainResult" not in confirmed.text


def test_empty_input_is_safe() -> None:
    outcome = _controller().handle("   ")

    assert outcome.direct_response_required is True
    assert outcome.result.executed is False


def test_structured_result_remains_available_for_debugging() -> None:
    controller = _controller()

    outcome = controller.handle("Lee este archivo")

    assert controller.last_result is outcome.result


def test_presenter_handles_failed_result_without_internal_dump() -> None:
    class _FailingCoordinator:
        def execute(self, prompt: str) -> ExecutionCoordinationResult:
            return ExecutionCoordinationResult(
                status=ExecutionCoordinationStatus.FAILED,
                mode=ExecutionMode.SINGLE_TOOL,
                decision=ExecutionDecision(
                    mode=ExecutionMode.SINGLE_TOOL,
                    reason="forced",
                    confidence=1.0,
                ),
                proposal=None,
                execution_result=None,
                message="forced failure",
                executed=False,
            )

        def confirm(self, confirmation_id: str, response: str) -> Any:
            raise AssertionError("no pending confirmation")

    outcome = ExecutionConversationController(_FailingCoordinator()).handle("haz algo")

    assert outcome.text == "No he podido completar la operacion."
    assert "ExecutionCoordinationResult" not in outcome.text

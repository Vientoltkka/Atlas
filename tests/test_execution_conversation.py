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
from tools.tool_chain_proposal_builder import StructuredToolChainProposal
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
    assert outcome.text == "Que archivo quieres leer?"
    assert outcome.result.executed is False


def test_ambiguous_request_asks_for_clarification() -> None:
    outcome = _controller().handle("Escribe algo en un archivo")

    assert outcome.result.status == ExecutionCoordinationStatus.AMBIGUOUS_REQUEST
    assert outcome.text == "Que contenido quieres escribir y en que archivo?"
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


def test_incomplete_read_accepts_next_turn_path_and_executes() -> None:
    controller = _controller()

    pending = controller.handle("Lee este archivo")
    completed = controller.handle("README.md")

    assert pending.result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert controller.pending_clarification is None
    assert completed.result.status == ExecutionCoordinationStatus.EXECUTED
    assert "Atlas" in completed.text


def test_incomplete_directory_list_accepts_next_turn_path() -> None:
    controller = _controller()

    pending = controller.handle("Lista esta carpeta")
    completed = controller.handle("tools")

    assert pending.result.status == ExecutionCoordinationStatus.AMBIGUOUS_REQUEST
    assert completed.result.status == ExecutionCoordinationStatus.EXECUTED
    assert "- execution_conversation.py" not in completed.text
    assert "- base_tool.py" in completed.text


def test_write_missing_path_uses_existing_content_and_asks_confirmation(tmp_path: Path) -> None:
    target = tmp_path / "notas.txt"
    controller = _controller()

    pending = controller.handle("Escribe hola")
    completed = controller.handle(f"en {target}")

    assert pending.result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert target.exists() is False
    assert completed.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert controller.pending_clarification is None
    assert controller.pending_confirmation_id is not None
    assert target.exists() is False


def test_write_missing_content_uses_next_turn_content(tmp_path: Path) -> None:
    target = tmp_path / "notas.txt"
    controller = _controller()

    pending = controller.handle(f"Escribe en {target}")
    completed = controller.handle("hola mundo")

    assert pending.result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert completed.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert target.exists() is False


def test_write_ambiguous_path_and_content_can_be_completed_together(tmp_path: Path) -> None:
    target = tmp_path / "notas.txt"
    controller = _controller()

    pending = controller.handle("Escribe algo en un archivo")
    completed = controller.handle(f"hola en {target}")

    assert pending.result.status == ExecutionCoordinationStatus.AMBIGUOUS_REQUEST
    assert completed.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert controller.pending_clarification is None
    assert controller.pending_confirmation_id is not None
    assert target.exists() is False


def test_partial_clarification_keeps_only_remaining_information(tmp_path: Path) -> None:
    target = tmp_path / "notas.txt"
    controller = _controller()

    controller.handle("Escribe algo en un archivo")
    partial = controller.handle(f"en {target}")

    assert partial.result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert controller.pending_clarification is not None
    assert controller.pending_clarification.requested_fields == ("content",)
    assert target.exists() is False


def test_irrelevant_clarification_does_not_execute_or_clear_state(tmp_path: Path) -> None:
    target = tmp_path / "notas.txt"
    controller = _controller()

    controller.handle(f"Escribe en {target}")
    irrelevant = controller.handle("   ???   ")

    assert irrelevant.result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert controller.pending_clarification is not None
    assert controller.pending_confirmation_id is None
    assert target.exists() is False


def test_empty_clarification_keeps_state(tmp_path: Path) -> None:
    target = tmp_path / "notas.txt"
    controller = _controller()

    controller.handle(f"Escribe en {target}")
    empty = controller.handle("   ")

    assert empty.result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert controller.pending_clarification is not None
    assert target.exists() is False


def test_cancel_clarification_clears_state_without_side_effects(tmp_path: Path) -> None:
    target = tmp_path / "notas.txt"
    controller = _controller()

    controller.handle(f"Escribe en {target}")
    cancelled = controller.handle("olvidalo")

    assert cancelled.result.status == ExecutionCoordinationStatus.CANCELLED
    assert controller.pending_clarification is None
    assert controller.pending_confirmation_id is None
    assert target.exists() is False


def test_new_order_during_clarification_is_not_mixed(tmp_path: Path) -> None:
    target = tmp_path / "notas.txt"
    controller = _controller()

    controller.handle(f"Escribe en {target}")
    response = controller.handle("Borra README.md")

    assert response.result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert controller.pending_clarification is not None
    assert target.exists() is False
    assert "cancelar" in response.text


def test_clarification_and_confirmation_do_not_coexist(tmp_path: Path) -> None:
    target = tmp_path / "notas.txt"
    controller = _controller()

    controller.handle("Escribe hola")
    completed = controller.handle(f"en {target}")

    assert completed.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert controller.pending_clarification is None
    assert controller.pending_confirmation_id is not None


def test_completed_write_confirmation_creates_file_once(tmp_path: Path) -> None:
    target = tmp_path / "notas.txt"
    controller = _controller()

    controller.handle("Escribe hola")
    controller.handle(f"en {target}")
    confirmed = controller.handle("si")
    repeated = controller.handle("si")

    assert confirmed.result.status == ExecutionCoordinationStatus.EXECUTED
    assert target.read_text(encoding="utf-8") == "hola"
    assert repeated.result.executed is False
    assert target.read_text(encoding="utf-8") == "hola"


def test_incomplete_chain_can_be_completed_and_pauses_before_write(tmp_path: Path) -> None:
    target = tmp_path / "resumen.txt"
    controller = _controller()

    pending = controller.handle("Lee este archivo y guardalo en otro")
    completed = controller.handle(f"lee README.md y guardalo en {target}")

    assert pending.result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert completed.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert completed.result.executed is True
    assert target.exists() is False
    assert controller.pending_clarification is None
    assert controller.pending_confirmation_id is not None


def test_chain_destination_can_be_completed_without_repeating_after_confirmation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "resumen.txt"
    controller = _controller()

    controller.handle("Lee README.md y guardalo en otro archivo")
    pending = controller.handle(str(target))

    assert pending.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert target.exists() is False
    confirmed = controller.handle("s")

    assert confirmed.result.status == ExecutionCoordinationStatus.EXECUTED
    assert target.exists() is True
    assert "Cadena completada" in confirmed.text


def test_direct_response_still_works_without_pending_clarification() -> None:
    outcome = _controller().handle("Hola")

    assert outcome.direct_response_required is True
    assert outcome.result.status == ExecutionCoordinationStatus.DIRECT_RESPONSE_REQUIRED


def test_regression_chain_clarification_full_answer_stays_tool_chain(
    tmp_path: Path,
) -> None:
    target = tmp_path / "resumen.txt"
    controller = _controller()

    pending = controller.handle("Lee este archivo y guárdalo en otro")

    assert pending.result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert controller.pending_clarification is not None
    assert controller.pending_clarification.mode is ExecutionMode.TOOL_CHAIN

    clarified = controller.handle(f"lee README.md y guárdalo en {target}")

    assert clarified.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert clarified.result.mode is ExecutionMode.TOOL_CHAIN
    assert clarified.result.execution_result is not None
    assert clarified.result.execution_result.execution_count == 1
    assert clarified.result.execution_result.steps[0].step_id == "read"
    assert clarified.result.execution_result.steps[0].result.execution_count == 1
    assert clarified.result.execution_result.steps[1].step_id == "write"
    assert clarified.result.execution_result.steps[1].result.execution_count == 0
    assert target.exists() is False

    confirmed = controller.handle("s")

    assert confirmed.result.status == ExecutionCoordinationStatus.EXECUTED
    assert confirmed.result.execution_result is not None
    assert confirmed.result.execution_result.steps[0].result.execution_count == 1
    assert confirmed.result.execution_result.steps[1].result.execution_count == 1
    assert target.exists() is True
    assert target.read_text(encoding="utf-8") == Path("README.md").read_text(encoding="utf-8")


def test_regression_partial_chain_keeps_read_and_completes_only_write_path(
    tmp_path: Path,
) -> None:
    target = tmp_path / "resumen.txt"
    controller = _controller()

    first = controller.handle("Lee README.md y guárdalo en otro")
    clarified = controller.handle(str(target))

    assert first.result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert first.result.mode is ExecutionMode.TOOL_CHAIN
    assert clarified.result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert clarified.result.mode is ExecutionMode.TOOL_CHAIN
    assert clarified.result.execution_result.steps[0].result.validated_arguments["path"] == "README.md"
    assert clarified.result.execution_result.steps[1].result.validated_arguments["path"] == str(target)
    assert target.exists() is False


def test_regression_pending_chain_preserves_original_proposal_candidates_and_fields() -> None:
    controller = _controller()

    outcome = controller.handle("Lee este archivo y guárdalo en otro")
    pending = controller.pending_clarification

    assert outcome.text == "Que archivo quieres leer y en que archivo quieres guardarlo?"
    assert pending is not None
    assert pending.mode is ExecutionMode.TOOL_CHAIN
    assert isinstance(pending.proposal, StructuredToolChainProposal)
    assert pending.candidate_tools == ("file.read", "file.write")
    assert pending.requested_fields == ("read.path", "write.path")
    assert pending.original_text == "Lee este archivo y guárdalo en otro"


def test_regression_chain_clarification_never_degrades_to_single_tool(
    tmp_path: Path,
) -> None:
    target = tmp_path / "resumen.txt"
    controller = _controller()

    controller.handle("Lee este archivo y guárdalo en otro")
    clarified = controller.handle(f"lee README.md y guárdalo en {target}")

    assert clarified.result.mode is ExecutionMode.TOOL_CHAIN
    assert clarified.result.status != ExecutionCoordinationStatus.EXECUTED
    assert "Atlas" not in clarified.text.splitlines()[0]


def test_regression_chain_clarification_question_is_natural() -> None:
    outcome = _controller().handle("Lee este archivo y guárdalo en otro")

    assert outcome.text == "Que archivo quieres leer y en que archivo quieres guardarlo?"
    assert "path" not in outcome.text

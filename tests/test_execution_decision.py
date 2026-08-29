from __future__ import annotations

from pathlib import Path

from bootstrap.bootstrap import Bootstrap
from tools.execution_decision import (
    ExecutionDecision,
    ExecutionDecisionEngine,
    ExecutionMode,
)


def _engine() -> ExecutionDecisionEngine:
    return Bootstrap.build_execution_decision_engine()


def test_conversational_request_uses_direct_response() -> None:
    decision = _engine().decide("Hola, ¿cómo estás?")

    assert decision.mode == ExecutionMode.DIRECT_RESPONSE
    assert decision.candidate_tools == ()


def test_knowledge_question_uses_direct_response_even_with_and() -> None:
    decision = _engine().decide("Explícame qué son Git y GitHub")

    assert decision.mode == ExecutionMode.DIRECT_RESPONSE
    assert decision.candidate_tools == ()


def test_file_read_request_uses_single_tool() -> None:
    decision = _engine().decide("Lee el archivo README.md")

    assert decision.mode == ExecutionMode.SINGLE_TOOL
    assert decision.candidate_tools == ("file.read",)
    assert decision.required_capabilities == ("read_file",)


def test_directory_list_request_uses_single_tool() -> None:
    decision = _engine().decide("Lista los archivos de la carpeta tools")

    assert decision.mode == ExecutionMode.SINGLE_TOOL
    assert decision.candidate_tools == ("directory.list",)


def test_calendar_list_request_uses_calendar_intent() -> None:
    decision = _engine().decide(
        "Lista eventos del calendario entre "
        "2026-08-09T09:00:00+01:00 y 2026-08-09T10:00:00+01:00"
    )

    assert decision.mode == ExecutionMode.SINGLE_TOOL
    assert decision.candidate_tools == ("calendar.events.list",)


def test_natural_tomorrow_request_uses_only_calendar_intent() -> None:
    decision = _engine().decide("Qué tengo mañana")

    assert decision.mode == ExecutionMode.SINGLE_TOOL
    assert decision.candidate_tools == ("calendar.events.list",)

def test_file_write_request_uses_single_tool() -> None:
    decision = _engine().decide("Escribe hola en resumen.txt")

    assert decision.mode == ExecutionMode.SINGLE_TOOL
    assert decision.candidate_tools == ("file.write",)


def test_named_window_text_request_uses_desktop_type_only() -> None:
    decision = _engine().decide("Escribe ORBE E2E en la ventana de WPS")

    assert decision.mode == ExecutionMode.SINGLE_TOOL
    assert decision.candidate_tools == ("desktop.text.type",)
    assert all(not tool.startswith("file.") for tool in decision.candidate_tools)
    assert "file.write" not in decision.candidate_tools


def test_read_then_write_request_uses_tool_chain() -> None:
    decision = _engine().decide("Lee README.md y copia su contenido en resumen.txt")

    assert decision.mode == ExecutionMode.TOOL_CHAIN
    assert decision.candidate_tools == ("file.read", "file.write")


def test_two_desktop_actions_use_tool_chain() -> None:
    decision = _engine().decide("Abre VS Code y escribe una línea en un archivo")

    assert decision.mode == ExecutionMode.TOOL_CHAIN
    assert "desktop.application.open" in decision.candidate_tools
    assert (
        "desktop.text.type" in decision.candidate_tools
        or "file.write" in decision.candidate_tools
    )


def test_candidate_tools_belong_to_registered_intents() -> None:
    selector = Bootstrap.build_tool_selector()
    decision = _engine().decide("Lee README.md y copia su contenido en resumen.txt")

    assert set(decision.candidate_tools).issubset(set(selector.supported_intents()))


def test_classifier_never_invents_delete_tool() -> None:
    selector = Bootstrap.build_tool_selector()
    decision = _engine().decide("Borra este archivo")

    assert "file.delete" not in decision.candidate_tools
    assert set(decision.candidate_tools).issubset(set(selector.supported_intents()))


def test_classifier_does_not_execute_tools() -> None:
    decision = _engine().decide("Lee el archivo README.md")

    assert decision.mode == ExecutionMode.SINGLE_TOOL
    assert decision.metadata is None


def test_classifier_does_not_create_or_modify_files(tmp_path: Path) -> None:
    target = tmp_path / "resumen.txt"

    decision = _engine().decide(f"Escribe hola en {target}")

    assert decision.mode == ExecutionMode.SINGLE_TOOL
    assert target.exists() is False


def test_confidence_is_normalized() -> None:
    prompts = (
        "Hola",
        "Lee README.md",
        "Lee README.md y copia su contenido en resumen.txt",
        "",
    )

    for prompt in prompts:
        decision = _engine().decide(prompt)
        assert 0.0 <= decision.confidence <= 1.0


def test_result_is_always_structured() -> None:
    decision = _engine().decide("Haz algo mágico con mi ordenador")

    assert isinstance(decision, ExecutionDecision)
    assert isinstance(decision.mode, ExecutionMode)
    assert isinstance(decision.reason, str)
    assert isinstance(decision.candidate_tools, tuple)
    assert isinstance(decision.required_capabilities, tuple)


def test_empty_input_returns_safe_direct_response() -> None:
    decision = _engine().decide("   ")

    assert decision.mode == ExecutionMode.DIRECT_RESPONSE
    assert decision.confidence == 0.4
    assert decision.candidate_tools == ()


def test_ambiguous_request_does_not_activate_dangerous_tool() -> None:
    decision = _engine().decide("Haz algo mágico con mi ordenador")

    assert decision.mode == ExecutionMode.DIRECT_RESPONSE
    assert decision.candidate_tools == ()


def test_bootstrap_builds_execution_decision_engine() -> None:
    engine = Bootstrap.build_execution_decision_engine()

    assert isinstance(engine, ExecutionDecisionEngine)

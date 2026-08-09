from __future__ import annotations

from pathlib import Path

from bootstrap.bootstrap import Bootstrap
from core.operational_request_router import RequestRoute
from services.file_service import FileService
from tools.executor import ToolExecutor


def _configure_runtime(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("ATLAS_HYBRID_PLANNING_ENABLED", raising=False)
    monkeypatch.delenv("ATLAS_STRUCTURED_PLAN_EXECUTION_ENABLED", raising=False)
    monkeypatch.delenv("ATLAS_EXECUTION_PERSISTENCE_ENABLED", raising=False)
    monkeypatch.delenv("ATLAS_STRUCTURED_PLAN_PROVIDER_ENABLED", raising=False)
    monkeypatch.setenv("ATLAS_EXECUTION_HISTORY_PATH", str(tmp_path / "sessions"))
    monkeypatch.setenv(
        "ATLAS_EXECUTION_STATE_PATH",
        str(tmp_path / "execution_state.json"),
    )
    return Bootstrap.build()


def _install_chat_responder(monkeypatch, orchestrator, responder):
    chat_agent = orchestrator._registry.get("chat")
    assert chat_agent is not None
    monkeypatch.setattr(chat_agent._client, "check_model_health", lambda _model: None)
    monkeypatch.setattr(chat_agent._client, "ask", responder)


def test_structured_result_becomes_bounded_conversational_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    orchestrator = _configure_runtime(monkeypatch, tmp_path)
    calls: list[list[dict[str, str]]] = []

    def respond(*, model, messages):
        del model
        calls.append([dict(message) for message in messages])
        prompt = messages[-1]["content"]
        if "Resume" in prompt:
            return "Atlas es un sistema operativo personal basado en agentes."
        return "El archivo leido fue README.md."

    _install_chat_responder(monkeypatch, orchestrator, respond)

    read_response = orchestrator.process_prompt(
        "Lee README.md",
        confirm=lambda _prompt: "",
    )
    summary = orchestrator.process_prompt(
        "Resume brevemente lo que acabas de leer.",
        confirm=lambda _prompt: "",
    )
    file_answer = orchestrator.process_prompt(
        "¿Qué archivo has leído?",
        confirm=lambda _prompt: "",
    )

    assert read_response.startswith(
        "Ejecucion completada, pero no hay evidencia suficiente "
        "para verificar el objetivo."
    )
    assert "Estrategia:" not in read_response
    assert "Autorizacion:" not in read_response
    assert summary.startswith("Atlas es")
    assert file_answer == "El archivo leido fue README.md."
    assert len(calls) == 2
    context = "\n".join(message["content"] for message in calls[0])
    assert "Contexto de ejecucion:" in context
    assert "Objetivo: Lee README.md" in context
    assert "Resultado: # Atlas" in context
    assert orchestrator.classify_prompt(
        "¿Qué archivo has leído?"
    ).route is RequestRoute.AGENT_DELEGATION
    detail = orchestrator.last_structured_execution_response
    assert detail is not None
    assert detail.operational_report is not None
    assert "Estrategia:" in detail.operational_report.to_text()


def test_last_error_is_available_and_next_request_recovers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    orchestrator = _configure_runtime(monkeypatch, tmp_path)
    calls: list[list[dict[str, str]]] = []

    def respond(*, model, messages):
        del model
        calls.append([dict(message) for message in messages])
        return "El error anterior fue que el archivo no existe."

    _install_chat_responder(monkeypatch, orchestrator, respond)
    failed = orchestrator.process_prompt(
        "Lee __fase_15_6_missing__.txt",
        confirm=lambda _prompt: "",
    )
    recovered = orchestrator.process_prompt(
        "¿Cuál fue el error anterior?",
        confirm=lambda _prompt: "",
    )
    succeeded = orchestrator.process_prompt(
        "Lee README.md",
        confirm=lambda _prompt: "",
    )

    assert failed.startswith("No pude completar la accion.")
    assert "Traceback" not in failed
    assert recovered == "El error anterior fue que el archivo no existe."
    error_context = "\n".join(message["content"] for message in calls[0])
    assert "Estado: FAILED" in error_context
    assert "__fase_15_6_missing__.txt" in error_context
    assert "Error:" in error_context
    assert succeeded.startswith(
        "Ejecucion completada, pero no hay evidencia suficiente "
        "para verificar el objetivo."
    )


def test_twenty_direct_turns_use_bounded_existing_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    orchestrator = _configure_runtime(monkeypatch, tmp_path)
    message_counts: list[int] = []
    counter = 0

    def respond(*, model, messages):
        nonlocal counter
        del model
        counter += 1
        message_counts.append(len(messages))
        return f"Respuesta diaria {counter}."

    _install_chat_responder(monkeypatch, orchestrator, respond)
    responses = [
        orchestrator.process_prompt(
            f"¿Qué significa el numero {index}?",
            confirm=lambda _prompt: "",
        )
        for index in range(1, 21)
    ]

    assert responses == [f"Respuesta diaria {index}." for index in range(1, 21)]
    assert counter == 20
    assert max(message_counts) <= 13
    assert len(orchestrator._memory.history()) == 40
    assert all("Traceback" not in response for response in responses)


def test_text_loop_shows_one_processing_status_for_slow_direct_turn(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    orchestrator = _configure_runtime(monkeypatch, tmp_path)

    def respond(*, model, messages):
        del model, messages
        return "Respuesta general."

    _install_chat_responder(monkeypatch, orchestrator, respond)
    inputs = iter(("¿Cuál es la capital de Francia?", "salir"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    orchestrator.start()

    output = capsys.readouterr().out
    assert output.count("Procesando...") == 1
    assert output.count("Respuesta general.") == 1
    assert "Traceback" not in output


def test_ambiguous_reference_in_new_session_requests_clarification(
    monkeypatch,
    tmp_path: Path,
) -> None:
    orchestrator = _configure_runtime(monkeypatch, tmp_path)

    def forbidden_response(**_kwargs):
        raise AssertionError("ambiguous reference must not reach the model")

    _install_chat_responder(monkeypatch, orchestrator, forbidden_response)
    response = orchestrator.process_prompt(
        "Repítelo",
        confirm=lambda _prompt: "",
    )

    assert response == "Necesito que aclares a que archivo, resultado o error te refieres."
    assert orchestrator._memory.history()[-1]["content"] == response


def test_capability_questions_use_real_registry_without_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    orchestrator = _configure_runtime(monkeypatch, tmp_path)

    def forbidden_execution(*_args, **_kwargs):
        raise AssertionError("capability query must not execute a tool")

    monkeypatch.setattr(ToolExecutor, "execute", forbidden_execution)

    overview = orchestrator.process_prompt(
        "¿Qué puedes hacer?",
        confirm=lambda _prompt: "",
    )
    voice = orchestrator.process_prompt(
        "¿Tienes voz?",
        confirm=lambda _prompt: "",
    )
    read = orchestrator.process_prompt(
        "¿Puedes leer archivos?",
        confirm=lambda _prompt: "",
    )
    write = orchestrator.process_prompt(
        "¿Puedes escribir archivos?",
        confirm=lambda _prompt: "",
    )

    assert "Herramientas registradas y disponibles: 41" in overview
    assert "Voz: opcional" in overview
    assert "no configurada" in voice
    assert "read_file esta disponible" in read
    assert "write_file esta disponible" in write
    assert "Requiere confirmacion explicita" in write


def test_confirmation_is_consumed_once_and_cancel_does_not_execute(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    orchestrator = _configure_runtime(monkeypatch, tmp_path)

    def respond(*, model, messages):
        del model, messages
        return "No hay una confirmacion pendiente."

    _install_chat_responder(monkeypatch, orchestrator, respond)
    confirmed_target = tmp_path / "confirmed.txt"
    cancelled_target = tmp_path / "cancelled.txt"

    pending = orchestrator.process_prompt(
        "Escribe hola en confirmed.txt",
        confirm=lambda _prompt: "",
    )
    unrelated = orchestrator.process_prompt(
        "¿Qué herramientas tienes?",
        confirm=lambda _prompt: "",
    )
    confirmed = orchestrator.process_prompt("confirmo", confirm=lambda _prompt: "")
    original = confirmed_target.read_text(encoding="utf-8")
    repeated = orchestrator.process_prompt("confirmo", confirm=lambda _prompt: "")

    assert "pendiente de confirmacion" in pending
    assert unrelated.startswith("Herramientas disponibles (41):")
    assert confirmed_target.exists()
    assert confirmed_target.read_text(encoding="utf-8") == original
    assert "No hay una confirmacion pendiente" in repeated
    assert confirmed_target.read_text(encoding="utf-8") == original
    assert confirmed.startswith(
        "Plan confirmado. Ejecucion completada, pero no hay evidencia "
        "suficiente para verificar el objetivo."
    )

    second_pending = orchestrator.process_prompt(
        "Escribe adios en cancelled.txt",
        confirm=lambda _prompt: "",
    )
    cancelled = orchestrator.process_prompt("olvidalo", confirm=lambda _prompt: "")

    assert "pendiente de confirmacion" in second_pending
    assert "cancelado" in cancelled.lower()
    assert not cancelled_target.exists()


def test_sensitive_file_result_is_not_kept_in_conversation_memory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    secret = "api_key=sk-project-secret-value"
    Path("sensitive.txt").write_text(secret, encoding="utf-8")
    orchestrator = _configure_runtime(monkeypatch, tmp_path)

    response = orchestrator.process_prompt(
        "Lee sensitive.txt",
        confirm=lambda _prompt: "",
    )
    history = "\n".join(item["content"] for item in orchestrator._memory.history())

    assert secret not in response
    assert secret not in history
    assert "[redacted]" in history

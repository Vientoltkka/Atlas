from __future__ import annotations

import pytest

from bootstrap.bootstrap import Bootstrap
from tools.gmail.gmail_request_parser import extract_gmail_arguments
from tools.gmail.gmail_service import (
    GmailConfigurationError,
    GmailService,
)
from tools.gmail.gmail_tools import (
    GmailListTool,
    GmailReadTool,
    GmailSendTool,
)
from tools.tool_context import ToolContext


def _tool_context(parameters: dict[str, object]) -> ToolContext:
    return ToolContext(parameters=parameters)


def test_gmail_send_always_requires_confirmation_and_declares_permission() -> None:
    tool = GmailSendTool()

    assert tool.requires_confirmation is True
    assert tool.required_permissions == ("email.send",)


def test_gmail_list_and_read_are_read_only() -> None:
    assert GmailListTool().requires_confirmation is False
    assert GmailListTool().required_permissions == ()
    assert GmailReadTool().requires_confirmation is False
    assert GmailReadTool().required_permissions == ()


def test_missing_credentials_fail_with_setup_instructions(monkeypatch) -> None:
    monkeypatch.delenv("ATLAS_GMAIL_ADDRESS", raising=False)
    monkeypatch.delenv("ATLAS_GMAIL_APP_PASSWORD", raising=False)
    service = GmailService()

    with pytest.raises(GmailConfigurationError) as list_error:
        service.list_messages()
    assert "ATLAS_GMAIL_ADDRESS" in str(list_error.value)

    with pytest.raises(GmailConfigurationError) as send_error:
        service.send_message(to="pepe@gmail.com", subject="Hola", body="Que tal")
    assert "ATLAS_GMAIL_APP_PASSWORD" in str(send_error.value)


def test_gmail_argument_extraction() -> None:
    assert extract_gmail_arguments(
        "Envia un email a pepe@gmail.com con asunto Hola y cuerpo Que tal",
        "gmail.messages.send",
    ) == {
        "to": "pepe@gmail.com",
        "subject": "Hola",
        "body": "Que tal",
    }
    assert extract_gmail_arguments(
        "Muestrame mis ultimos correos",
        "gmail.messages.list",
    ) == {}
    assert extract_gmail_arguments(
        "Muestrame mis ultimos 10 correos",
        "gmail.messages.list",
    ) == {"max_results": 10}
    assert extract_gmail_arguments(
        "Lee el correo de pepe@gmail.com",
        "gmail.messages.read",
    ) == {"sender": "pepe@gmail.com"}
    assert extract_gmail_arguments(
        "Lee el correo de Maria",
        "gmail.messages.read",
    ) == {"sender": "maria"}


def test_gmail_list_uses_injected_service_and_presents_summary() -> None:
    class _FakeService:
        def list_messages(self, max_results: int = 5):
            del max_results
            return [
                {
                    "id": "12",
                    "from": "Ana <ana@example.com>",
                    "subject": "Reunion",
                    "date": "Fri, 04 Sep 2026 10:00:00 +0100",
                    "snippet": "Confirmamos la reunion de manana.",
                }
            ]

    outcome = GmailListTool(_FakeService()).execute(
        _tool_context({"max_results": 5})
    )

    assert outcome["messages"][0]["subject"] == "Reunion"


def test_gmail_read_by_sender_uses_injected_service() -> None:
    class _FakeService:
        def read_message(self, message_id=None, sender=None):
            del message_id
            assert sender == "pepe@gmail.com"
            return {
                "id": "9",
                "from": "pepe@gmail.com",
                "subject": "Informe",
                "date": "Fri, 04 Sep 2026 09:00:00 +0100",
                "body": "Contenido del informe.",
            }

    outcome = GmailReadTool(_FakeService()).execute(
        _tool_context({"sender": "pepe@gmail.com"})
    )

    assert outcome["body"] == "Contenido del informe."


def test_gmail_read_without_target_fails_controlled() -> None:
    tool = GmailReadTool()

    with pytest.raises(ValueError) as error:
        tool.execute(_tool_context({}))

    assert "id del mensaje" in str(error.value)


def test_gmail_send_uses_injected_service() -> None:
    sent: list[tuple[str, str, str]] = []

    class _FakeService:
        def send_message(self, to: str, subject: str, body: str):
            sent.append((to, subject, body))
            return {"sent_to": to, "subject": subject}

    outcome = GmailSendTool(_FakeService()).execute(
        _tool_context(
            {
                "to": "pepe@gmail.com",
                "subject": "Hola",
                "body": "Que tal",
            }
        )
    )

    assert outcome == {"sent_to": "pepe@gmail.com", "subject": "Hola"}
    assert sent == [("pepe@gmail.com", "Hola", "Que tal")]


def _configure_runtime(monkeypatch, tmp_path) -> None:
    for variable in (
        "ATLAS_HYBRID_PLANNING_ENABLED",
        "ATLAS_STRUCTURED_PLAN_EXECUTION_ENABLED",
        "ATLAS_EXECUTION_PERSISTENCE_ENABLED",
        "ATLAS_STRUCTURED_PLAN_PROVIDER_ENABLED",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ATLAS_EXECUTION_HISTORY_PATH", str(tmp_path / "sessions"))
    monkeypatch.setenv(
        "ATLAS_EXECUTION_STATE_PATH",
        str(tmp_path / "execution_state.json"),
    )


def test_gmail_list_without_credentials_returns_setup_instructions(
    monkeypatch,
    tmp_path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    monkeypatch.delenv("ATLAS_GMAIL_ADDRESS", raising=False)
    monkeypatch.delenv("ATLAS_GMAIL_APP_PASSWORD", raising=False)
    orchestrator = Bootstrap.build()

    response = orchestrator.process_prompt(
        "Muestrame mis ultimos correos",
        confirm=lambda _prompt: "",
    )

    assert "Gmail no esta configurado" in response
    assert "ATLAS_GMAIL_ADDRESS" in response


def test_gmail_list_presents_recent_messages_with_fake_service(
    monkeypatch,
    tmp_path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    monkeypatch.setenv("ATLAS_GMAIL_ADDRESS", "demo@gmail.com")
    monkeypatch.setenv("ATLAS_GMAIL_APP_PASSWORD", "demo-app-password")
    monkeypatch.setattr(
        GmailService,
        "list_messages",
        lambda self, max_results=5: [
            {
                "id": "12",
                "from": "Ana <ana@example.com>",
                "subject": "Reunion",
                "date": "Fri, 04 Sep 2026 10:00:00 +0100",
                "snippet": "Confirmamos la reunion de manana.",
            }
        ],
    )
    orchestrator = Bootstrap.build()

    response = orchestrator.process_prompt(
        "Muestrame mis ultimos correos",
        confirm=lambda _prompt: "",
    )

    assert response.startswith("Últimos correos:")
    assert "[12] Reunion — Ana <ana@example.com>" in response
    assert "Confirmamos la reunion de manana." in response


def test_gmail_send_requires_explicit_confirmation_and_cancel_never_sends(
    monkeypatch,
    tmp_path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    monkeypatch.setenv("ATLAS_GMAIL_ADDRESS", "demo@gmail.com")
    monkeypatch.setenv("ATLAS_GMAIL_APP_PASSWORD", "demo-app-password")
    monkeypatch.setattr(
        GmailService,
        "send_message",
        lambda self, to, subject, body: {"sent_to": to, "subject": subject},
    )
    orchestrator = Bootstrap.build()
    prompt = (
        "Envia un email a pepe@gmail.com con asunto Hola y cuerpo Que tal"
    )

    pending = orchestrator.process_prompt(prompt, confirm=lambda _prompt: "")
    cancelled = orchestrator.process_prompt("cancela", confirm=lambda _prompt: "")

    assert "Voy a enviar un email a pepe@gmail.com" in pending
    assert "Deseas continuar?" in pending
    assert "operacion cancelada" in cancelled.lower()


def test_gmail_send_confirmed_once_sends_exactly_once(
    monkeypatch,
    tmp_path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    monkeypatch.setenv("ATLAS_GMAIL_ADDRESS", "demo@gmail.com")
    monkeypatch.setenv("ATLAS_GMAIL_APP_PASSWORD", "demo-app-password")
    calls: list[tuple[str, str, str]] = []

    def fake_send(self, to: str, subject: str, body: str):
        calls.append((to, subject, body))
        return {"sent_to": to, "subject": subject}

    monkeypatch.setattr(GmailService, "send_message", fake_send)
    orchestrator = Bootstrap.build()
    prompt = (
        "Envia un email a pepe@gmail.com con asunto Hola y cuerpo Que tal"
    )

    pending = orchestrator.process_prompt(prompt, confirm=lambda _prompt: "")
    confirmed = orchestrator.process_prompt("confirmo", confirm=lambda _prompt: "")
    repeated = orchestrator.process_prompt("confirmo", confirm=lambda _prompt: "")

    assert "Voy a enviar un email a pepe@gmail.com" in pending
    assert "Email enviado a pepe@gmail.com" in confirmed
    assert calls == [("pepe@gmail.com", "Hola", "Que tal")]
    assert confirmed != repeated


def test_gmail_read_by_sender_presents_body_with_fake_service(
    monkeypatch,
    tmp_path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    monkeypatch.setenv("ATLAS_GMAIL_ADDRESS", "demo@gmail.com")
    monkeypatch.setenv("ATLAS_GMAIL_APP_PASSWORD", "demo-app-password")

    def fake_read(self, message_id=None, sender=None):
        del message_id
        assert sender == "pepe@gmail.com"
        return {
            "id": "9",
            "from": "pepe@gmail.com",
            "subject": "Informe",
            "date": "Fri, 04 Sep 2026 09:00:00 +0100",
            "body": "Contenido del informe.",
        }

    monkeypatch.setattr(GmailService, "read_message", fake_read)
    orchestrator = Bootstrap.build()

    response = orchestrator.process_prompt(
        "Lee el correo de pepe@gmail.com",
        confirm=lambda _prompt: "",
    )

    assert response.startswith("Correo de pepe@gmail.com — Informe")
    assert "Contenido del informe." in response

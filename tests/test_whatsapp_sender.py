"""Tests for WhatsAppGraphSender retry/backoff behavior (Phase 3, Block 3)."""

from __future__ import annotations

import pytest

from channels.whatsapp_sender import PermanentDeliveryError, WhatsAppGraphSender


def make_sender(transport, sleeps=None, max_attempts=3) -> WhatsAppGraphSender:
    return WhatsAppGraphSender(
        access_token="token",
        phone_number_id="pnid",
        transport=transport,
        max_attempts=max_attempts,
        sleeper=lambda seconds: sleeps.append(seconds) if sleeps is not None else None,
    )


def test_permanent_4xx_single_call_no_retry() -> None:
    calls: list[tuple] = []

    def transport(url, headers, payload):
        calls.append((url, headers, payload))
        return 400, "bad request"

    sender = make_sender(transport)
    with pytest.raises(PermanentDeliveryError):
        sender.send_text("34600", "hola")
    assert len(calls) == 1


def test_server_error_retries_until_exhausted() -> None:
    calls: list[int] = []

    def transport(url, headers, payload):
        calls.append(1)
        return 500, "server error"

    sender = make_sender(transport, max_attempts=3)
    with pytest.raises(RuntimeError):
        sender.send_text("34600", "hola")
    assert len(calls) == 3


def test_timeout_error_retries_until_exhausted() -> None:
    calls: list[int] = []

    def transport(url, headers, payload):
        calls.append(1)
        raise TimeoutError("network down")

    sender = make_sender(transport, max_attempts=3)
    with pytest.raises(TimeoutError):
        sender.send_text("34600", "hola")
    assert len(calls) == 3


def test_success_on_second_attempt() -> None:
    statuses = iter([500, 200])

    def transport(url, headers, payload):
        return next(statuses), ""

    sender = make_sender(transport)
    sender.send_text("34600", "hola")


def test_429_treated_as_transient() -> None:
    statuses = iter([429, 200])

    def transport(url, headers, payload):
        return next(statuses), ""

    sender = make_sender(transport)
    sender.send_text("34600", "hola")


def test_backoff_intervals_are_exponential() -> None:
    def transport(url, headers, payload):
        raise TimeoutError("down")

    sleeps: list[float] = []
    sender = make_sender(transport, sleeps=sleeps, max_attempts=3)
    with pytest.raises(TimeoutError):
        sender.send_text("34600", "hola")
    assert sleeps == [0.5, 1.0]


def test_max_attempts_respected() -> None:
    calls: list[int] = []

    def transport(url, headers, payload):
        calls.append(1)
        return 503, "unavailable"

    sender = make_sender(transport, max_attempts=5)
    with pytest.raises(RuntimeError):
        sender.send_text("34600", "hola")
    assert len(calls) == 5


def test_invalid_configuration_rejected() -> None:
    with pytest.raises(ValueError):
        make_sender(lambda *a: (200, ""), max_attempts=0)


def test_webhook_notifies_user_after_sender_exhausts_retries() -> None:
    from fastapi.testclient import TestClient

    from channels.app import create_webhook_app
    from channels.webhook_idempotency import IdempotencyStore
    from core.agent_executor import AgentExecutionResult, AgentExecutionStatus

    calls: list[int] = []

    def failing_transport(url, headers, payload):
        calls.append(1)
        return 500, "error"

    class ExhaustingSender:
        def __init__(self) -> None:
            self.inner = WhatsAppGraphSender(
                access_token="token",
                phone_number_id="pnid",
                transport=failing_transport,
            )
            self.sent: list[tuple[str, str]] = []

        def send_text(self, recipient_id: str, body: str) -> None:
            if body != "Se ha producido un error procesando tu mensaje.":
                self.inner.send_text(recipient_id, body)
                return
            self.sent.append((recipient_id, body))

    def executor(request):
        return AgentExecutionResult(
            status=AgentExecutionStatus.COMPLETED,
            request_signature="sig",
            output={"text": "respuesta"},
        )

    sender = ExhaustingSender()
    client = TestClient(
        create_webhook_app(
            executor_fn=executor,
            store=IdempotencyStore(),
            sender=sender,
            verify_token="tok",
        )
    )
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": "wamid.retry",
                                    "from": "34600",
                                    "type": "text",
                                    "text": {"body": "Hola"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(calls) == 3
    assert sender.sent and sender.sent[0][1].startswith("Se ha producido")

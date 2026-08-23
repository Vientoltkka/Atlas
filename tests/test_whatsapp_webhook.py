"""Tests for the Atlas WhatsApp webhook (Phase 2)."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from channels.app import create_webhook_app
from channels.webhook_idempotency import IdempotencyStore
from core.agent_executor import AgentExecutionResult, AgentExecutionStatus


VERIFY_TOKEN = "atlas-verify-token"


class FakeStore:
    """Thread-safe in-memory idempotency store mirroring IdempotencyStore."""

    def __init__(self) -> None:
        self._reserved: set[str] = set()
        self._lock = threading.Lock()

    def check_and_reserve(self, event_id: str) -> bool:
        with self._lock:
            if event_id in self._reserved:
                return False
            self._reserved.add(event_id)
            return True


class FakeSender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send_text(self, recipient_id: str, body: str) -> None:
        self.sent.append((recipient_id, body))


def make_executor(results: list[AgentExecutionResult] | None = None):
    calls: list[Any] = []

    def execute(request):
        calls.append(request)
        if results:
            return results.pop(0)
        return AgentExecutionResult(
            status=AgentExecutionStatus.COMPLETED,
            request_signature="sig",
            correlation_id=request.correlation_id,
            output={"text": "Respuesta Atlas"},
        )

    execute.calls = calls  # type: ignore[attr-defined]
    return execute


def text_payload(wamid: str = "wamid.test001", sender: str = "34600111222", body: str = "Hola Atlas") -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "entry1",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "34600000000", "phone_number_id": "pnid"},
                            "contacts": [{"profile": {"name": "Usuario"}, "wa_id": sender}],
                            "messages": [
                                {"id": wamid, "from": sender, "timestamp": "1", "type": "text", "text": {"body": body}}
                            ],
                        },
                    }
                ],
            }
        ],
    }


def make_client(executor=None, store=None, sender=None, verify_token: str = VERIFY_TOKEN):
    return TestClient(
        create_webhook_app(
            executor_fn=executor or make_executor(),
            store=store or FakeStore(),
            sender=sender or FakeSender(),
            verify_token=verify_token,
        )
    )


def test_get_verification_ok() -> None:
    client = make_client()
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub_mode": "subscribe",
            "hub_verify_token": VERIFY_TOKEN,
            "hub_challenge": "challenge123",
        },
    )
    assert response.status_code == 200
    assert response.text == "challenge123"


def test_get_verification_wrong_token() -> None:
    client = make_client()
    response = client.get(
        "/webhook/whatsapp",
        params={"hub_mode": "subscribe", "hub_verify_token": "wrong", "hub_challenge": "x"},
    )
    assert response.status_code == 403


def test_post_text_message_full_flow() -> None:
    executor = make_executor()
    sender = FakeSender()
    client = make_client(executor=executor, sender=sender)
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200
    assert len(executor.calls) == 1
    request = executor.calls[0]
    assert request.user_input == "Hola Atlas"
    assert request.correlation_id.startswith("wa-")
    assert len(request.correlation_id) <= 128
    assert request.session_id.startswith("whatsapp-wa_")
    assert sender.sent == [("34600111222", "Respuesta Atlas")]


def test_post_json_malformed_returns_400() -> None:
    client = make_client()
    response = client.post(
        "/webhook/whatsapp",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400


def test_post_empty_payload_returns_200() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    assert client.post("/webhook/whatsapp", json={}).status_code == 200
    assert client.post("/webhook/whatsapp", json={"foo": "bar"}).status_code == 200
    assert client.post("/webhook/whatsapp", json=[1, 2, 3]).status_code == 200
    assert len(executor.calls) == 0


def test_post_status_ack_ignored() -> None:
    executor = make_executor()
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "statuses": [{"id": "wamid.s1", "status": "delivered"}],
                        }
                    }
                ]
            }
        ]
    }
    response = make_client(executor=executor).post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 0


def test_unsupported_message_type_sends_courtesy() -> None:
    executor = make_executor()
    sender = FakeSender()
    payload = text_payload(wamid="wamid.audio1")
    payload["entry"][0]["changes"][0]["value"]["messages"][0] = {
        "id": "wamid.audio1",
        "from": "34600111222",
        "type": "audio",
        "audio": {"id": "x"},
    }
    response = make_client(executor=executor, sender=sender).post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 0
    assert sender.sent and "texto" in sender.sent[0][1]


def test_duplicate_wamid_executes_once() -> None:
    executor = make_executor()
    store = FakeStore()
    sender = FakeSender()
    client = make_client(executor=executor, store=store, sender=sender)
    first = client.post("/webhook/whatsapp", json=text_payload())
    second = client.post("/webhook/whatsapp", json=text_payload())
    assert first.status_code == 200 and second.status_code == 200
    assert len(executor.calls) == 1
    assert len(sender.sent) == 1


def test_concurrent_duplicates_execute_exactly_once() -> None:
    executor = make_executor()

    class SlowFakeStore(FakeStore):
        """Widens the race window inside check_and_reserve."""

        def check_and_reserve(self, event_id: str) -> bool:
            with self._lock:
                if event_id in self._reserved:
                    return False
                import time

                time.sleep(0.2)
                self._reserved.add(event_id)
                return True

    store = SlowFakeStore()
    sender = FakeSender()
    client = make_client(executor=executor, store=store, sender=sender)
    responses: list[Any] = []
    lock = threading.Lock()

    def post() -> None:
        response = client.post("/webhook/whatsapp", json=text_payload(wamid="wamid.race"))
        with lock:
            responses.append(response)

    threads = [threading.Thread(target=post) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(r.status_code == 200 for r in responses)
    assert len(executor.calls) == 1
    assert len(sender.sent) == 1


def test_executor_failure_returns_200_and_notifies_user() -> None:
    def failing_executor(request):
        raise RuntimeError("boom")

    sender = FakeSender()
    client = make_client(executor=failing_executor, sender=sender)
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200
    assert sender.sent and "error" in sender.sent[0][1].lower()


def test_infrastructure_failure_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    from channels import whatsapp_webhook as module

    def broken_extract(payload: Any) -> Any:
        raise ConnectionError("storage down")

    monkeypatch.setattr(module, "_extract_change_value", broken_extract)
    client = TestClient(
        create_webhook_app(
            executor_fn=make_executor(),
            store=FakeStore(),
            sender=FakeSender(),
            verify_token=VERIFY_TOKEN,
        )
    )
    with pytest.raises(ConnectionError):
        client.post("/webhook/whatsapp", json=text_payload())


def test_sender_body_never_contains_access_token() -> None:
    sender = FakeSender()
    client = make_client(sender=sender)
    client.post("/webhook/whatsapp", json=text_payload())
    serialized = repr(sender.sent) + repr(client.app.routes)
    assert "secret-access-token" not in serialized


def test_idempotency_store_restart_allows_reprocessing_documented() -> None:
    store_a = IdempotencyStore()
    assert store_a.check_and_reserve("wamid.x") is True
    assert store_a.check_and_reserve("wamid.x") is False
    store_b = IdempotencyStore()
    assert store_b.check_and_reserve("wamid.x") is True


def test_correlation_id_length_limit() -> None:
    long_wamid = "wamid." + "A" * 300
    executor = make_executor()
    client = make_client(executor=executor)
    client.post("/webhook/whatsapp", json=text_payload(wamid=long_wamid))
    assert executor.calls
    correlation_id = executor.calls[0].correlation_id
    assert correlation_id is not None and len(correlation_id) <= 128

"""Tests for WhatsApp delivery status handling (V4.0-F1)."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from channels.app import create_webhook_app
from channels.webhook_idempotency import IdempotencyStore
from channels.whatsapp_metrics import (
    MESSAGES_RECEIVED,
    STATUS_DELIVERED,
    STATUS_FAILED,
    STATUS_READ,
    STATUS_SENT,
    WhatsAppMetricsRecorder,
)
from core.agent_executor import AgentExecutionResult, AgentExecutionStatus

VERIFY_TOKEN = "atlas-verify-token"


class FakeStore:
    def check_and_reserve(self, event_id: str) -> bool:
        return True


class FakeSender:
    def send_text(self, recipient_id: str, body: str) -> None:
        pass


def make_executor():
    def execute(request):
        return AgentExecutionResult(
            status=AgentExecutionStatus.COMPLETED,
            request_signature="sig",
            correlation_id=request.correlation_id,
            output={"text": "Respuesta Atlas"},
        )

    return execute


def status_payload(status: str = "delivered", wamid: str = "wamid.s1") -> dict:
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "statuses": [
                                {
                                    "id": wamid,
                                    "status": status,
                                    "recipient_id": "34600111222",
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


def make_client(recorder=None):
    app = create_webhook_app(
        executor_fn=make_executor(),
        store=FakeStore(),
        sender=FakeSender(),
        verify_token=VERIFY_TOKEN,
        recorder=recorder,
    )
    return TestClient(app)


def test_status_sent_counted() -> None:
    recorder = WhatsAppMetricsRecorder()
    response = make_client(recorder).post("/webhook/whatsapp", json=status_payload("sent"))
    assert response.status_code == 200
    assert recorder.value(STATUS_SENT) == 1


def test_status_delivered_counted() -> None:
    recorder = WhatsAppMetricsRecorder()
    response = make_client(recorder).post("/webhook/whatsapp", json=status_payload("delivered"))
    assert response.status_code == 200
    assert recorder.value(STATUS_DELIVERED) == 1


def test_status_read_counted() -> None:
    recorder = WhatsAppMetricsRecorder()
    response = make_client(recorder).post("/webhook/whatsapp", json=status_payload("read"))
    assert response.status_code == 200
    assert recorder.value(STATUS_READ) == 1


def test_status_failed_counted_and_no_wamid_in_response() -> None:
    recorder = WhatsAppMetricsRecorder()
    payload = status_payload("failed", wamid="wamid.secret-xyz")
    payload["entry"][0]["changes"][0]["value"]["statuses"][0]["errors"] = [
        {"code": 131047, "title": "Re-engagement message"}
    ]
    response = make_client(recorder).post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert response.text == ""
    assert recorder.value(STATUS_FAILED) == 1


def test_multiple_acks_counted_independently() -> None:
    recorder = WhatsAppMetricsRecorder()
    client = make_client(recorder)
    client.post("/webhook/whatsapp", json=status_payload("sent"))
    client.post("/webhook/whatsapp", json=status_payload("delivered"))
    client.post("/webhook/whatsapp", json=status_payload("read"))
    snapshot = recorder.snapshot()
    assert snapshot[STATUS_SENT] == 1
    assert snapshot[STATUS_DELIVERED] == 1
    assert snapshot[STATUS_READ] == 1
    assert snapshot[MESSAGES_RECEIVED] == 0


def test_malformed_status_payload_is_ignored() -> None:
    recorder = WhatsAppMetricsRecorder()
    client = make_client(recorder)
    payloads = [
        {"entry": [{"changes": [{"value": {"statuses": []}}]}]},
        {"entry": [{"changes": [{"value": {"statuses": ["not-a-mapping"]}}]}]},
        {"entry": [{"changes": [{"value": {"statuses": [{"id": "wamid.1"}]}}]}]},
        {"entry": [{"changes": [{"value": {"statuses": [123]}}]}]},
    ]
    for payload in payloads:
        response = client.post("/webhook/whatsapp", json=payload)
        assert response.status_code == 200
    assert recorder.snapshot() == {event: 0 for event in recorder.snapshot()}


def test_unknown_status_is_ignored_without_error() -> None:
    recorder = WhatsAppMetricsRecorder()
    response = make_client(recorder).post(
        "/webhook/whatsapp", json=status_payload("queued")
    )
    assert response.status_code == 200
    assert all(value == 0 for value in recorder.snapshot().values())


def test_recorder_observes_acks_end_to_end() -> None:
    recorder = WhatsAppMetricsRecorder()
    client = make_client(recorder)
    client.post("/webhook/whatsapp", json=status_payload("delivered"))
    client.post("/webhook/whatsapp", json=status_payload("failed"))
    assert recorder.value(STATUS_DELIVERED) == 1
    assert recorder.value(STATUS_FAILED) == 1


class BrokenRecorder:
    def record(self, event: str) -> None:
        raise RuntimeError("metrics down")


def test_broken_recorder_still_returns_200() -> None:
    client = make_client(BrokenRecorder())
    for code in ("sent", "delivered", "read", "failed"):
        response = client.post("/webhook/whatsapp", json=status_payload(code))
        assert response.status_code == 200


def test_status_case_insensitive() -> None:
    recorder = WhatsAppMetricsRecorder()
    response = make_client(recorder).post(
        "/webhook/whatsapp", json=status_payload("Delivered")
    )
    assert response.status_code == 200
    assert recorder.value(STATUS_DELIVERED) == 1

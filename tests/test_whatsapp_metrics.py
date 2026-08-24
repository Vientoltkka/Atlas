"""Tests for WhatsApp channel observability (F5.1)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from channels.whatsapp_channel import WhatsAppChannel

from channels.app import create_webhook_app
from channels.whatsapp_metrics import (
    AUDIO_RECEIVED,
    CHANNEL_ERRORS,
    MESSAGES_DUPLICATED,
    MESSAGES_FAILED,
    MESSAGES_RECEIVED,
    VOICE_REPLIES,
    WhatsAppMetricsRecorder,
)
from channels.whatsapp_webhook import build_webhook_router
from core.agent_executor import AgentExecutionResult, AgentExecutionStatus


VERIFY_TOKEN = "atlas-verify-token"


class FakeStore:
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
    def send_text(self, recipient_id: str, body: str) -> None:
        pass

    def upload_media(self, file_path: str, mime_type: str) -> str:
        return "media-123"

    def send_audio(self, recipient_id: str, media_id: str) -> None:
        pass


class FakeRenderer:
    temp_dir: str | None = None

    def render(self, text: str) -> Path:
        path = Path(self.temp_dir or ".") / "reply.ogg"
        path.write_bytes(b"OggS-fake")
        return path


def make_executor():
    def execute(request):
        return AgentExecutionResult(
            status=AgentExecutionStatus.COMPLETED,
            request_signature="sig",
            correlation_id=request.correlation_id,
            output={"text": "Respuesta Atlas"},
        )

    return execute


def text_payload(wamid: str = "wamid.m1", body: str = "Hola Atlas") -> dict:
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": wamid,
                                    "from": "34600111222",
                                    "timestamp": "1",
                                    "type": "text",
                                    "text": {"body": body},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


def audio_payload(wamid: str = "wamid.a1") -> dict:
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": wamid,
                                    "from": "34600111222",
                                    "type": "audio",
                                    "audio": {"id": "media-x"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


class FakeTranscriber:
    def transcribe_media_id(self, media_id: str) -> str:
        return "audio transcrito"


def make_client(recorder=None, executor=None, sender=None, transcriber=None):
    app = create_webhook_app(
        executor_fn=executor or make_executor(),
        store=FakeStore(),
        sender=sender or FakeSender(),
        verify_token=VERIFY_TOKEN,
        transcriber=transcriber,
        recorder=recorder,
    )
    return TestClient(app)


def test_counts_received_messages() -> None:
    recorder = WhatsAppMetricsRecorder()
    client = make_client(recorder=recorder)
    client.post("/webhook/whatsapp", json=text_payload(wamid="wamid.r1"))
    assert recorder.value(MESSAGES_RECEIVED) == 1
    client.post("/webhook/whatsapp", json=text_payload(wamid="wamid.r2"))
    assert recorder.value(MESSAGES_RECEIVED) == 2


def test_counts_duplicates() -> None:
    recorder = WhatsAppMetricsRecorder()
    client = make_client(recorder=recorder)
    client.post("/webhook/whatsapp", json=text_payload())
    client.post("/webhook/whatsapp", json=text_payload())
    assert recorder.value(MESSAGES_RECEIVED) == 2
    assert recorder.value(MESSAGES_DUPLICATED) == 1


def test_counts_failures() -> None:
    recorder = WhatsAppMetricsRecorder()

    def failing_executor(request):
        raise RuntimeError("boom")

    client = make_client(recorder=recorder, executor=failing_executor)
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200
    assert recorder.value(MESSAGES_FAILED) == 1
    assert recorder.value(CHANNEL_ERRORS) == 0


def test_counts_audio_messages() -> None:
    recorder = WhatsAppMetricsRecorder()
    client = make_client(
        recorder=recorder,
        executor=make_executor(),
        transcriber=FakeTranscriber(),
    )
    response = client.post("/webhook/whatsapp", json=audio_payload())
    assert response.status_code == 200
    assert recorder.value(AUDIO_RECEIVED) == 1
    assert recorder.value(MESSAGES_RECEIVED) == 1


def test_counts_voice_replies() -> None:
    recorder = WhatsAppMetricsRecorder()
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(
        build_webhook_router(
            channel=WhatsAppChannel(),
            executor_fn=make_executor(),
            sender=FakeSender(),
            verify_token=VERIFY_TOKEN,
            store=FakeStore(),
            voice_renderer=FakeRenderer(),
            recorder=recorder,
        )
    )
    client = TestClient(app)
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200
    assert recorder.value(VOICE_REPLIES) == 1
    assert recorder.value(MESSAGES_RECEIVED) == 1
    assert recorder.value(MESSAGES_FAILED) == 0


def test_independent_events_do_not_interfere() -> None:
    recorder = WhatsAppMetricsRecorder()
    client = make_client(recorder=recorder)
    client.post("/webhook/whatsapp", json=text_payload(wamid="wamid.i1"))
    client.post("/webhook/whatsapp", json=text_payload(wamid="wamid.i1"))
    snapshot = recorder.snapshot()
    assert snapshot[MESSAGES_RECEIVED] == 2
    assert snapshot[MESSAGES_DUPLICATED] == 1
    assert snapshot[AUDIO_RECEIVED] == 0
    assert snapshot[MESSAGES_FAILED] == 0


def test_thread_safety_basic() -> None:
    recorder = WhatsAppMetricsRecorder()

    def hammer() -> None:
        for _ in range(200):
            recorder.record(MESSAGES_RECEIVED)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert recorder.value(MESSAGES_RECEIVED) == 1600


def test_recorder_is_injected_into_app() -> None:
    recorder = WhatsAppMetricsRecorder()
    client = make_client(recorder=recorder)
    client.post("/webhook/whatsapp", json=text_payload())
    # The same instance observes the flow end to end.
    assert recorder.value(MESSAGES_RECEIVED) == 1


def test_broken_recorder_does_not_break_processing() -> None:
    class BrokenRecorder:
        def record(self, event: str) -> None:
            raise RuntimeError("metrics down")

    client = make_client(recorder=BrokenRecorder())
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200


def test_metrics_representation_has_no_secrets_or_content() -> None:
    recorder = WhatsAppMetricsRecorder()
    client = make_client(recorder=recorder)
    secret_body = "CONFIDENTIAL-secret-access-token-XYZ"
    client.post("/webhook/whatsapp", json=text_payload(body=secret_body))
    serialized = repr(recorder) + repr(recorder.snapshot())
    assert "secret" not in serialized.lower()
    assert "CONFIDENTIAL" not in serialized
    assert "34600111222" not in serialized

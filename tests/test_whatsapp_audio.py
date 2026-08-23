"""Tests for WhatsApp audio transcription (Phase 3, Block 4, Function 3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from fastapi.testclient import TestClient

from channels.app import create_webhook_app
from channels.webhook_idempotency import IdempotencyStore
from channels.whatsapp_audio import (
    AudioTranscriptionError,
    WhatsAppAudioTranscriber,
)
from core.agent_executor import AgentExecutionResult, AgentExecutionStatus


@pytest.fixture()
def temp_dir():
    import shutil
    import tempfile

    path = tempfile.mkdtemp(prefix="atlas-audio-test-")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@dataclass
class FakeResult:
    text: str


class FakeProvider:
    name = "fake"

    def __init__(self, text: str = "hola atlas", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[tuple[np.ndarray, int]] = []

    def transcribe(self, samples, sample_rate):
        self.calls.append((samples, sample_rate))
        if self.fail:
            raise AudioTranscriptionError("no transcript")
        return FakeResult(text=self.text)


def _tiny_wav_bytes() -> bytes:
    """Minimal valid mono 16 kHz WAV with ~0.1 s of silence."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 1_600)
    return buffer.getvalue()


def make_transcriber(provider=None, media_bytes: bytes | None = None, temp_dir=None, **kwargs):
    from channels.whatsapp_media import WhatsAppMediaDownloader

    if media_bytes is None:
        media_bytes = _tiny_wav_bytes()

    def transport(url: str, headers: dict):
        if url.endswith("/media123"):
            assert headers["Authorization"] == "Bearer secret-token"
            return 200, json_metadata()
        return 200, media_bytes

    def json_metadata():
        return (
            '{"url": "https://lookaside.example/dl", '
            '"mime_type": "audio/mpeg", "file_size": %d}' % len(media_bytes)
        )

    downloader = WhatsAppMediaDownloader(
        access_token="secret-token",
        base_url="https://graph.example/v21.0",
        transport=transport,
        allowed_mime_types=frozenset({"audio/ogg", "audio/mpeg"}),
        temp_root=temp_dir,
    )
    return WhatsAppAudioTranscriber(
        downloader=downloader,
        provider=provider or FakeProvider(),
        **kwargs,
    )


def test_valid_audio_downloads_and_transcribes(temp_dir) -> None:
    provider = FakeProvider(text="hola atlas")
    transcriber = make_transcriber(provider, temp_dir=temp_dir)
    text = transcriber.transcribe_media_id("media123")
    assert text == "hola atlas"
    assert len(provider.calls) == 1
    samples, rate = provider.calls[0]
    assert isinstance(samples, np.ndarray)
    assert rate == 16_000


def test_temp_file_removed_after_processing(temp_dir) -> None:
    transcriber = make_transcriber(FakeProvider(), temp_dir=temp_dir)
    transcriber.transcribe_media_id("media123")
    import pathlib

    assert list(pathlib.Path(temp_dir).glob("atlas-media-*")) == []


def test_invalid_media_id_rejected(temp_dir) -> None:
    transcriber = make_transcriber(temp_dir=temp_dir)
    with pytest.raises(Exception):
        transcriber.transcribe_media_id("")


def test_download_failure_propagates_controlled(temp_dir) -> None:
    def transport(url, headers):
        return 500, ""

    from channels.whatsapp_media import MediaDownloadError, WhatsAppMediaDownloader

    downloader = WhatsAppMediaDownloader(
        access_token="secret-token",
        transport=transport,
    )
    transcriber = WhatsAppAudioTranscriber(
        downloader=downloader,
        provider=FakeProvider(),
    )
    with pytest.raises(MediaDownloadError):
        transcriber.transcribe_media_id("media123")


def test_transcription_failure_is_controlled(temp_dir) -> None:
    transcriber = make_transcriber(FakeProvider(fail=True), temp_dir=temp_dir)
    with pytest.raises(AudioTranscriptionError):
        transcriber.transcribe_media_id("media123")


def test_empty_transcript_is_controlled_error(temp_dir) -> None:
    transcriber = make_transcriber(FakeProvider(text=""), temp_dir=temp_dir)
    with pytest.raises(AudioTranscriptionError):
        transcriber.transcribe_media_id("media123")


# ---------------------------------------------------------------------------
# Webhook integration
# ---------------------------------------------------------------------------


def audio_payload(wamid: str = "wamid.audio1") -> dict:
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
                                    "audio": {"id": "media123", "mime_type": "audio/ogg"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


def make_client(transcriber, executor, sender=None) -> TestClient:
    return TestClient(
        create_webhook_app(
            executor_fn=executor,
            store=IdempotencyStore(),
            sender=sender or FakeWebhookSender(),
            verify_token="tok",
            transcriber=transcriber,
        )
    )


class FakeWebhookSender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send_text(self, recipient_id: str, body: str) -> None:
        self.sent.append((recipient_id, body))


def _executor(calls: list):
    def execute(request):
        calls.append(request)
        return AgentExecutionResult(
            status=AgentExecutionStatus.COMPLETED,
            request_signature="sig",
            correlation_id=request.correlation_id,
            output={"text": "respuesta"},
        )

    return execute


def test_webhook_audio_full_flow_executes_atlas_with_transcript() -> None:
    calls: list = []
    sender = FakeWebhookSender()

    class StubTranscriber:
        def transcribe_media_id(self, media_id: str) -> str:
            assert media_id == "media123"
            return "pon la alarma"

    client = make_client(StubTranscriber(), _executor(calls), sender)
    response = client.post("/webhook/whatsapp", json=audio_payload())
    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0].user_input == "pon la alarma"
    assert sender.sent and sender.sent[0][1] == "respuesta"


def test_webhook_duplicate_audio_wamid_executes_once() -> None:
    calls: list = []
    sender = FakeWebhookSender()

    class StubTranscriber:
        def transcribe_media_id(self, media_id: str) -> str:
            return "pon la alarma"

    client = make_client(StubTranscriber(), _executor(calls), sender)
    first = client.post("/webhook/whatsapp", json=audio_payload())
    second = client.post("/webhook/whatsapp", json=audio_payload())
    assert first.status_code == 200 and second.status_code == 200
    assert len(calls) == 1
    assert len(sender.sent) == 1


class StubTranscriberAlwaysFails:
    def transcribe_media_id(self, media_id: str) -> str:
        raise AudioTranscriptionError("boom")


def test_webhook_transcription_failure_sends_courtesy_without_execution() -> None:
    calls: list = []
    sender = FakeWebhookSender()
    client = make_client(StubTranscriberAlwaysFails(), _executor(calls), sender)
    response = client.post("/webhook/whatsapp", json=audio_payload())
    assert response.status_code == 200
    assert len(calls) == 0
    assert sender.sent and "entender" in sender.sent[0][1]


def test_webhook_no_secrets_in_sender_messages(temp_dir) -> None:
    calls: list = []
    sender = FakeWebhookSender()
    client = make_client(
        make_transcriber(temp_dir=temp_dir), _executor(calls), sender
    )
    client.post("/webhook/whatsapp", json=audio_payload())
    serialized = repr(sender.sent)
    assert "secret-token" not in serialized
    assert "media123" not in serialized


def test_regression_image_caption_still_works() -> None:
    calls: list = []
    sender = FakeWebhookSender()
    client = make_client(StubTranscriberAlwaysFails(), _executor(calls), sender)
    payload = {
        "entry": [{"changes": [{"value": {"messages": [
            {
                "id": "wamid.img",
                "from": "34600",
                "type": "image",
                "image": {"id": "img1", "caption": "Que hago hoy?"},
            }
        ]}}]}]
    }
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0].user_input == "Que hago hoy?"


def test_regression_document_unaffected() -> None:
    calls: list = []
    sender = FakeWebhookSender()
    client = make_client(StubTranscriberAlwaysFails(), _executor(calls), sender)
    payload = {
        "entry": [{"changes": [{"value": {"messages": [
            {"id": "wamid.doc", "from": "34600", "type": "document", "document": {"id": "d"}}
        ]}}]}]
    }
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(calls) == 0

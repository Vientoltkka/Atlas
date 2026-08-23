"""Tests for WhatsApp voice replies (Phase 3, Block 4, Function 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from channels.app import create_webhook_app
from channels.webhook_idempotency import IdempotencyStore
from channels.whatsapp_sender import PermanentDeliveryError, WhatsAppGraphSender
from channels.whatsapp_voice_reply import (
    VoiceReplyError,
    WhatsAppVoiceReplyRenderer,
)
from core.agent_executor import AgentExecutionResult, AgentExecutionStatus


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEngine:
    """Mimics pyttsx3 engine: save_to_file writes a minimal valid WAV."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.saved: list[tuple[str, str]] = []

    def save_to_file(self, text: str, filename: str) -> None:
        if self.fail:
            raise RuntimeError("sapi down")
        self.saved.append((text, filename))
        _write_tiny_wav(filename)

    def runAndWait(self) -> None:
        pass


class FakeRenderer:
    """Stands in for WhatsAppVoiceReplyRenderer in webhook tests."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.rendered: list[str] = []
        self.files: list[Path] = []

    def render(self, text: str) -> Path:
        self.rendered.append(text)
        if self.fail:
            raise VoiceReplyError("render failed")
        path = Path(self.temp_dir or ".") / "reply.ogg"
        path.write_bytes(b"OggS-fake")
        self.files.append(path)
        return path

    temp_dir: str | None = None


class RecordingSender:
    def __init__(
        self,
        *,
        upload_fail: bool = False,
        audio_fail: bool = False,
        text_fail: bool = False,
        permanent_upload: bool = False,
    ) -> None:
        self.sent_text: list[tuple[str, str]] = []
        self.sent_audio: list[tuple[str, str]] = []
        self.uploads: list[tuple[str, str]] = []
        self.upload_fail = upload_fail
        self.audio_fail = audio_fail
        self.text_fail = text_fail
        self.permanent_upload = permanent_upload

    def send_text(self, recipient_id: str, body: str) -> None:
        if self.text_fail:
            raise RuntimeError("text down")
        self.sent_text.append((recipient_id, body))

    def upload_media(self, file_path: str, mime_type: str) -> str:
        self.uploads.append((file_path, mime_type))
        if self.upload_fail:
            raise RuntimeError("upload transient")
        if self.permanent_upload:
            raise PermanentDeliveryError("rejected")
        return "media-123"

    def send_audio(self, recipient_id: str, media_id: str) -> None:
        if self.audio_fail:
            raise RuntimeError("audio transient")
        self.sent_audio.append((recipient_id, media_id))


def _write_tiny_wav(filename: str) -> None:
    import io
    import wave
    import contextlib

    buffer = io.BytesIO()
    with contextlib.closing(wave.open(buffer, "wb")) as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 1_600)
    Path(filename).write_bytes(buffer.getvalue())


@pytest.fixture()
def temp_dir():
    import shutil
    import tempfile

    path = tempfile.mkdtemp(prefix="atlas-voice-test-")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def make_renderer(temp_dir, fail_engine=False, converter=None):
    return WhatsAppVoiceReplyRenderer(
        engine_factory=lambda: FakeEngine(fail=fail_engine),
        temp_root=temp_dir,
        converter=converter,
    )


# ---------------------------------------------------------------------------
# Renderer unit tests
# ---------------------------------------------------------------------------


def test_render_produces_ogg_and_cleans_wav(temp_dir) -> None:
    def real_converter(wav_path: Path, ogg_path: Path) -> None:
        # Real PyAV conversion from the generated WAV.
        import av

        with av.open(str(wav_path)) as source:
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16_000)
            with av.open(str(ogg_path), "w", format="ogg") as target:
                stream = target.add_stream("libopus", rate=16_000)
                for frame in source.decode(source.streams.audio[0]):
                    for resampled in resampler.resample(frame):
                        for packet in stream.encode(resampled):
                            target.mux(packet)

    renderer = make_renderer(temp_dir, converter=real_converter)
    result = renderer.render("Respuesta de Atlas")
    assert result.exists()
    assert result.suffix == ".ogg"
    assert result.stat().st_size > 0
    leftovers = [p for p in Path(temp_dir).glob("atlas-voice-*") if p != result]
    assert all(p.suffix == ".ogg" for p in leftovers)


def test_render_failure_leaves_no_temporaries(temp_dir) -> None:
    renderer = make_renderer(temp_dir, fail_engine=True)
    with pytest.raises(VoiceReplyError):
        renderer.render("texto")
    assert list(Path(temp_dir).glob("atlas-voice-*")) == []


def test_conversion_failure_is_controlled(temp_dir) -> None:
    def broken(wav_path: Path, ogg_path: Path) -> None:
        raise RuntimeError("codec error")

    renderer = make_renderer(temp_dir, converter=broken)
    with pytest.raises(VoiceReplyError):
        renderer.render("texto")
    assert list(Path(temp_dir).glob("atlas-voice-*")) == []


def test_empty_text_rejected(temp_dir) -> None:
    renderer = make_renderer(temp_dir)
    with pytest.raises(VoiceReplyError):
        renderer.render("   ")


def test_max_bytes_enforced(temp_dir) -> None:
    renderer = make_renderer(temp_dir)
    renderer._max_bytes = 10
    with pytest.raises(VoiceReplyError):
        renderer.render("texto largo que genera un wav de kilobytes")


# ---------------------------------------------------------------------------
# Sender upload/send_audio
# ---------------------------------------------------------------------------


def test_upload_success_returns_media_id(temp_dir) -> None:
    uploads: list[tuple] = []

    def upload_transport(url, headers, data, files):
        uploads.append((url, headers["Authorization"]))
        return 200, '{"id": "media-abc"}'

    sender = WhatsAppGraphSender(
        access_token="secret-token",
        phone_number_id="pnid",
        upload_transport=upload_transport,
    )
    media_id = sender.upload_media(str(_make_file(temp_dir)), "audio/ogg")
    assert media_id == "media-abc"
    assert uploads[0][1].startswith("Bearer ")


def _make_file(temp_dir) -> Path:
    path = Path(temp_dir) / "audio.ogg"
    path.write_bytes(b"OggS-fake")
    return path


def test_upload_permanent_4xx_single_attempt(temp_dir) -> None:
    calls: list[int] = []

    def upload_transport(url, headers, data, files):
        calls.append(1)
        return 403, ""

    sender = WhatsAppGraphSender(
        access_token="t", phone_number_id="p", upload_transport=upload_transport
    )
    with pytest.raises(PermanentDeliveryError):
        sender.upload_media(str(_make_file(temp_dir)), "audio/ogg")
    assert len(calls) == 1


def test_upload_transient_retries(temp_dir) -> None:
    calls: list[int] = []

    def upload_transport(url, headers, data, files):
        calls.append(1)
        return 500, ""

    sender = WhatsAppGraphSender(
        access_token="t",
        phone_number_id="p",
        upload_transport=upload_transport,
        sleeper=lambda s: None,
    )
    with pytest.raises(RuntimeError):
        sender.upload_media(str(_make_file(temp_dir)), "audio/ogg")
    assert len(calls) == 3


def test_send_audio_success_and_payload() -> None:
    payloads: list[dict] = []

    def transport(url, headers, payload):
        payloads.append(payload)
        return 200, ""

    sender = WhatsAppGraphSender(
        access_token="secret-token", phone_number_id="pnid", transport=transport
    )
    sender.send_audio("34600", "media-abc")
    assert payloads[0]["type"] == "audio"
    assert payloads[0]["audio"]["id"] == "media-abc"


def test_send_audio_empty_media_id_rejected() -> None:
    sender = WhatsAppGraphSender(access_token="t", phone_number_id="p")
    with pytest.raises(PermanentDeliveryError):
        sender.send_audio("34600", "  ")


# ---------------------------------------------------------------------------
# Webhook integration
# ---------------------------------------------------------------------------


def text_payload() -> dict:
    return {
        "entry": [{"changes": [{"value": {"messages": [
            {"id": "wamid.v1", "from": "34600", "type": "text", "text": {"body": "Hola"}}
        ]}}]}]
    }


def make_client(sender, renderer=None) -> TestClient:
    def executor(request):
        return AgentExecutionResult(
            status=AgentExecutionStatus.COMPLETED,
            request_signature="sig",
            correlation_id=request.correlation_id,
            output={"text": "respuesta atlas"},
        )

    return TestClient(
        create_webhook_app(
            executor_fn=executor,
            store=IdempotencyStore(),
            sender=sender,
            verify_token="tok",
            transcriber=None,
            voice_renderer=renderer,
        )
    )


def test_flag_off_sends_only_text() -> None:
    sender = RecordingSender()
    client = make_client(sender, renderer=None)
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200
    assert len(sender.sent_text) == 1
    assert not sender.sent_audio


def test_flag_on_full_flow_audio_without_duplicate_text(temp_dir) -> None:
    renderer = FakeRenderer()
    renderer.temp_dir = temp_dir
    sender = RecordingSender()
    client = make_client(sender, renderer=renderer)
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200
    assert len(sender.sent_audio) == 1
    assert sender.sent_audio[0][1] == "media-123"
    assert len(sender.sent_text) == 0
    assert renderer.rendered and renderer.rendered[0] == "respuesta atlas"


@pytest.mark.parametrize(
    ("kwargs"),
    [
        {"upload_fail": True},
        {"permanent_upload": True},
        {"audio_fail": True},
    ],
)
def test_fallback_to_text_on_any_failure(kwargs, temp_dir) -> None:
    renderer = FakeRenderer()
    renderer.temp_dir = temp_dir
    sender = RecordingSender(**kwargs)
    client = make_client(sender, renderer=renderer)
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200
    assert len(sender.sent_text) == 1
    assert sender.sent_text[0][1] == "respuesta atlas"


def test_renderer_failure_fallback_to_text() -> None:
    renderer = FakeRenderer(fail=True)
    sender = RecordingSender()
    client = make_client(sender, renderer=renderer)
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200
    assert len(sender.sent_text) == 1
    assert not sender.uploads


def test_no_secrets_in_outbound_calls(temp_dir) -> None:
    renderer = FakeRenderer()
    renderer.temp_dir = temp_dir
    sender = RecordingSender()
    client = make_client(sender, renderer=renderer)
    client.post("/webhook/whatsapp", json=text_payload())
    serialized = repr(sender.sent_audio) + repr(sender.sent_text) + repr(renderer.rendered)
    assert "secret-token" not in serialized


def test_temporary_reply_file_cleaned_after_send(temp_dir) -> None:
    renderer = FakeRenderer()
    renderer.temp_dir = temp_dir
    sender = RecordingSender()
    client = make_client(sender, renderer=renderer)
    client.post("/webhook/whatsapp", json=text_payload())
    assert all(not path.exists() for path in renderer.files)


def test_regression_inbound_image_caption_unaffected() -> None:
    sender = RecordingSender()
    client = make_client(sender)

    def executor_capture(calls: list):
        def execute(request):
            calls.append(request.user_input)
            return AgentExecutionResult(
                status=AgentExecutionStatus.COMPLETED,
                request_signature="sig",
                output={"text": "ok"},
            )
        return execute

    calls: list = []
    client2 = TestClient(
        create_webhook_app(
            executor_fn=executor_capture(calls),
            store=IdempotencyStore(),
            sender=sender,
            verify_token="tok",
        )
    )
    payload = {
        "entry": [{"changes": [{"value": {"messages": [
            {"id": "wamid.img9", "from": "34600", "type": "image",
             "image": {"id": "i", "caption": "Que hago?"}}
        ]}}]}]
    }
    response = client2.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert calls == ["Que hdo?".replace("hdo", "hago")]


def test_regression_inbound_idempotency_with_voice_on() -> None:
    renderer = FakeRenderer()
    renderer.temp_dir = "."
    sender = RecordingSender()
    client = make_client(sender, renderer=renderer)
    first = client.post("/webhook/whatsapp", json=text_payload())
    second = client.post("/webhook/whatsapp", json=text_payload())
    assert first.status_code == 200 and second.status_code == 200
    assert len(sender.sent_audio) == 1

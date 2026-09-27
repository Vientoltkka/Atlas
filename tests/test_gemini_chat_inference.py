from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from core.request_gateway import RequestAttachment
from models import chat_inference as chat_inference_module
from models.chat_inference import (
    ChatInferenceError,
    GeminiChatInferenceProvider,
    OllamaChatInferenceProvider,
    default_provider_registry,
)


def _mock_gemini_types(monkeypatch) -> None:
    class FakePart:
        @staticmethod
        def from_bytes(*, data: bytes, mime_type: str):
            return SimpleNamespace(inline_data=SimpleNamespace(data=data, mime_type=mime_type))

    google_genai = ModuleType("google.genai")
    google_genai.types = SimpleNamespace(Part=FakePart)
    monkeypatch.setitem(sys.modules, "google.genai", google_genai)


class FakeGeminiModels:
    def __init__(self, *, response=None, stream_response=None, error: Exception | None = None) -> None:
        self.response = response
        self.stream_response = stream_response
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def generate_content(self, **kwargs):
        self.calls.append(("chat", kwargs))
        if self.error:
            raise self.error
        return self.response

    def generate_content_stream(self, **kwargs):
        self.calls.append(("stream", kwargs))
        if self.error:
            raise self.error
        return self.stream_response


class FakeGeminiClient:
    def __init__(self, models: FakeGeminiModels) -> None:
        self.models = models


def test_gemini_adapts_chat_and_streaming_responses() -> None:
    models = FakeGeminiModels(
        response={"model_version": "gemini-test", "text": "hola"},
        stream_response=iter(
            [
                {"model_version": "gemini-test", "text": "ho"},
                {"model_version": "gemini-test", "text": "la"},
            ]
        ),
    )
    provider = GeminiChatInferenceProvider(
        api_key="test-key",
        default_model="gemini-test",
        client=FakeGeminiClient(models),
    )

    response = provider.chat(
        model="",
        messages=[
            {"role": "system", "content": "responde breve"},
            {"role": "user", "content": "hola"},
        ],
        stream=False,
    )

    assert response == {"model": "gemini-test", "message": {"content": "hola"}}
    assert list(provider.chat(model="", messages=[], stream=True)) == [
        {"model": "gemini-test", "message": {"content": "ho"}},
        {"model": "gemini-test", "message": {"content": "la"}},
    ]
    assert models.calls == [
        (
            "chat",
            {
                "model": "gemini-test",
                "contents": [{"role": "user", "parts": [{"text": "hola"}]}],
                "config": {"system_instruction": "responde breve"},
            },
        ),
        ("stream", {"model": "gemini-test", "contents": [], "config": None}),
    ]
    assert provider.capabilities() == frozenset({"chat", "stream", "health", "remote"})


def test_gemini_normalizes_health_errors() -> None:
    error = httpx.ConnectError("offline")
    provider = GeminiChatInferenceProvider(
        api_key="test-key",
        default_model="gemini-test",
        client=FakeGeminiClient(FakeGeminiModels(error=error)),
    )

    with pytest.raises(ChatInferenceError) as captured:
        provider.health(model="")

    assert captured.value.provider_id == "gemini"
    assert captured.value.model == "gemini-test"
    assert captured.value.__cause__ is error


def test_factory_registers_gemini_only_from_environment(monkeypatch) -> None:
    captured: list[str] = []

    def build_client(api_key: str) -> FakeGeminiClient:
        captured.append(api_key)
        return FakeGeminiClient(FakeGeminiModels())

    monkeypatch.setenv("ATLAS_GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("ATLAS_GEMINI_MODEL", "gemini-test")
    monkeypatch.setattr(chat_inference_module, "_create_gemini_client", build_client)

    registry = default_provider_registry(timeout=15.0, keep_alive="10m", provider_id="gemini")

    assert isinstance(registry.get("ollama"), OllamaChatInferenceProvider)
    assert registry.get("gemini").provider_id == "gemini"
    assert captured == ["test-key"]


def test_gemini_requires_configuration() -> None:
    with pytest.raises(ValueError, match="Gemini api_key is required"):
        GeminiChatInferenceProvider(api_key="")


def test_gemini_vision_sends_text_and_two_images_in_memory(tmp_path, monkeypatch) -> None:
    _mock_gemini_types(monkeypatch)
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"png-bytes")
    second.write_bytes(b"jpg-bytes")
    models = FakeGeminiModels(response={"model_version": "gemini-vision", "text": "analizado"})
    provider = GeminiChatInferenceProvider(
        api_key="test-key",
        default_model="gemini-vision",
        client=FakeGeminiClient(models),
        supports_vision=True,
    )

    response = provider.chat(
        model="",
        messages=[{"role": "user", "content": "Compara estas imágenes."}],
        stream=False,
        attachments=(
            RequestAttachment("one", "first.png", "image/png", local_reference=str(first)),
            RequestAttachment("two", "second.jpg", "image/jpeg", local_reference=str(second)),
        ),
    )

    assert response["message"]["content"] == "analizado"
    request = models.calls[0][1]
    assert request["contents"][0]["parts"][0] == {"text": "Compara estas imágenes."}
    assert [part.inline_data.data for part in request["contents"][0]["parts"][1:]] == [
        b"png-bytes",
        b"jpg-bytes",
    ]
    assert provider.capabilities() >= {"chat", "vision"}


def test_gemini_bioimpedance_vision_adds_cautious_instruction_without_reordering(tmp_path, monkeypatch) -> None:
    _mock_gemini_types(monkeypatch)
    first = tmp_path / "inbody-before.png"
    second = tmp_path / "inbody-after.png"
    first.write_bytes(b"before")
    second.write_bytes(b"after")
    models = FakeGeminiModels(response={"model_version": "gemini-vision", "text": "analizado"})
    provider = GeminiChatInferenceProvider(
        api_key="test-key",
        default_model="gemini-vision",
        client=FakeGeminiClient(models),
        supports_vision=True,
    )
    user_text = "Compara estos dos informes InBody y no guardes datos."

    provider.chat(
        model="",
        messages=[{"role": "user", "content": user_text}],
        stream=False,
        attachments=(
            RequestAttachment("before", first.name, "image/png", local_reference=str(first)),
            RequestAttachment("after", second.name, "image/png", local_reference=str(second)),
        ),
    )

    request = models.calls[0][1]
    assert request["contents"][0]["parts"][0] == {"text": user_text}
    assert [part.inline_data.data for part in request["contents"][0]["parts"][1:]] == [b"before", b"after"]
    instruction = request["config"]["system_instruction"]
    assert "Transcribe los valores visibles" in instruction
    assert "No concluyas ganancia muscular limpia" in instruction
    assert "consentimiento explicito" in instruction


def test_gemini_vision_sdk_error_is_safe(tmp_path, caplog, monkeypatch) -> None:
    _mock_gemini_types(monkeypatch)
    image = tmp_path / "secret.png"
    image.write_bytes(b"private")
    models = FakeGeminiModels(error=RuntimeError("secret path and image bytes"))
    provider = GeminiChatInferenceProvider(
        api_key="test-key",
        default_model="gemini-vision",
        client=FakeGeminiClient(models),
        supports_vision=True,
    )

    with pytest.raises(ChatInferenceError, match="multimodal") as captured:
        provider.chat(
            model="",
            messages=[{"role": "user", "content": "Analiza."}],
            stream=False,
            attachments=(RequestAttachment("one", "secret.png", "image/png", local_reference=str(image)),),
        )

    assert "secret" not in str(captured.value)
    assert "private" not in str(captured.value)
    assert captured.value.__cause__ is None
    diagnostics = [record.getMessage() for record in caplog.records]
    assert diagnostics == [
        "Gemini vision failed | stage=request | exception_type=RuntimeError"
    ]
    assert "secret" not in diagnostics[0]
    assert "private" not in diagnostics[0]
    assert "test-key" not in diagnostics[0]




def test_gemini_missing_sdk_logs_safe_initialization_diagnostic(caplog, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)

    with pytest.raises(RuntimeError, match="google-genai is required"):
        chat_inference_module._create_gemini_client("test-key")

    diagnostics = [record.getMessage() for record in caplog.records]
    assert diagnostics == [
        "Gemini vision failed | stage=sdk_client | exception_type=ModuleNotFoundError"
    ]
    assert "test-key" not in diagnostics[0]


def test_gemini_without_explicit_vision_does_not_send_images(tmp_path) -> None:
    image = tmp_path / "not-sent.png"
    image.write_bytes(b"not-sent")
    models = FakeGeminiModels(response={"model_version": "gemini-text", "text": "texto"})
    provider = GeminiChatInferenceProvider(
        api_key="test-key",
        default_model="gemini-text",
        client=FakeGeminiClient(models),
    )

    with pytest.raises(ChatInferenceError, match="not explicitly enabled"):
        provider.chat(
            model="",
            messages=[{"role": "user", "content": "Analiza."}],
            stream=False,
            attachments=(RequestAttachment("one", "not-sent.png", "image/png", local_reference=str(image)),),
        )

    assert models.calls == []

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from models import chat_inference as chat_inference_module
from models.chat_inference import (
    ChatInferenceError,
    GeminiChatInferenceProvider,
    OllamaChatInferenceProvider,
    default_provider_registry,
)


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

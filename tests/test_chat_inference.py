from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from models import chat_inference as chat_inference_module
from models.chat_inference import (
    ChatInferenceError,
    OllamaChatInferenceProvider,
    OpenAICompatibleChatInferenceProvider,
    default_provider_registry,
)


class FakeCompletions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeOpenAIClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def test_openai_compatible_provider_adapts_chat_response() -> None:
    completions = FakeCompletions(
        {"model": "remote-model", "choices": [{"message": {"content": "hola"}}]}
    )
    provider = OpenAICompatibleChatInferenceProvider(
        provider_id="remote",
        base_url="https://provider.example/v1",
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    response = provider.chat(
        model="remote-model",
        messages=[{"role": "user", "content": "hola"}],
        stream=False,
    )

    assert response == {"model": "remote-model", "message": {"content": "hola"}}
    assert completions.calls == [
        {
            "model": "remote-model",
            "messages": [{"role": "user", "content": "hola"}],
            "stream": False,
        }
    ]


def test_openai_compatible_provider_adapts_streaming_response() -> None:
    completions = FakeCompletions(
        iter(
            [
                {"model": "remote-model", "choices": [{"delta": {"content": "ho"}}]},
                {"model": "remote-model", "choices": [{"delta": {"content": "la"}}]},
            ]
        )
    )
    provider = OpenAICompatibleChatInferenceProvider(
        provider_id="remote",
        base_url="https://provider.example/v1",
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    assert list(provider.chat(model="remote-model", messages=[], stream=True)) == [
        {"model": "remote-model", "message": {"content": "ho"}},
        {"model": "remote-model", "message": {"content": "la"}},
    ]


def test_openai_compatible_provider_health_and_errors_are_normalized() -> None:
    error = httpx.ConnectError("offline")
    completions = FakeCompletions(error=error)
    provider = OpenAICompatibleChatInferenceProvider(
        provider_id="remote",
        base_url="https://provider.example/v1",
        api_key="test-key",
        default_model="default-model",
        client=FakeOpenAIClient(completions),
    )

    with pytest.raises(ChatInferenceError) as captured:
        provider.health(model="")

    assert captured.value.provider_id == "remote"
    assert captured.value.model == "default-model"
    assert captured.value.__cause__ is error
    assert completions.calls == [
        {
            "model": "default-model",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }
    ]


def test_factory_registers_configured_openai_provider_without_exposing_secret(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def build_client(**kwargs):
        captured.update(kwargs)
        return FakeOpenAIClient(FakeCompletions())

    monkeypatch.setattr(chat_inference_module, "OpenAI", build_client)
    registry = default_provider_registry(
        timeout=15.0,
        keep_alive="10m",
        provider_id="remote",
        base_url="https://provider.example/v1",
        api_key="test-key",
        model="default-model",
    )

    assert isinstance(registry.get("ollama"), OllamaChatInferenceProvider)
    assert registry.get("remote").provider_id == "remote"
    assert captured == {"base_url": "https://provider.example/v1", "api_key": "test-key"}
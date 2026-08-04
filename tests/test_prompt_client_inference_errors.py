from __future__ import annotations

import ollama
import pytest

from models import prompt_client as prompt_client_module
from models.prompt_client import InferenceBackendError, PromptClient


class FailingOllamaClient:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def chat(self, **_kwargs):
        raise self._error


@pytest.mark.parametrize(
    "error",
    (
        ollama.ResponseError("backend rejected request", 500),
        ollama.RequestError("backend request failed"),
        TimeoutError("backend timed out"),
        ConnectionError("backend disconnected"),
    ),
)
def test_prompt_client_classifies_only_known_backend_failures(
    monkeypatch,
    error: Exception,
) -> None:
    monkeypatch.setattr(
        prompt_client_module.ollama,
        "Client",
        lambda **_kwargs: FailingOllamaClient(error),
    )
    client = PromptClient()

    with pytest.raises(InferenceBackendError) as captured:
        client.ask_messages("primary:latest", [{"role": "user", "content": "hola"}])

    assert captured.value.model == "primary:latest"
    assert captured.value.__cause__ is error


def test_prompt_client_does_not_reclassify_programming_errors(monkeypatch) -> None:
    error = TypeError("programming error")
    monkeypatch.setattr(
        prompt_client_module.ollama,
        "Client",
        lambda **_kwargs: FailingOllamaClient(error),
    )
    client = PromptClient()

    with pytest.raises(TypeError, match="programming error"):
        list(client.stream_messages("primary:latest", []))


def test_prompt_client_classifies_empty_stream_before_observable_output(
    monkeypatch,
) -> None:
    class EmptyOllamaClient:
        def chat(self, **_kwargs):
            return iter(())

    monkeypatch.setattr(
        prompt_client_module.ollama,
        "Client",
        lambda **_kwargs: EmptyOllamaClient(),
    )
    client = PromptClient()

    with pytest.raises(InferenceBackendError, match="no observable stream content"):
        list(client.stream_messages("primary:latest", []))

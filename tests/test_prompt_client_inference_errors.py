from __future__ import annotations

import pytest

from models import prompt_client as prompt_client_module
from models.chat_inference import ChatInferenceError
from models.prompt_client import InferenceBackendError, PromptClient


class FailingOllamaClient:
    def __init__(self, error: ChatInferenceError) -> None:
        self._error = error

    def chat(self, **_kwargs):
        raise self._error


@pytest.mark.parametrize(
    "reason",
    (
        "backend rejected request",
        "backend request failed",
        "backend timed out",
        "backend disconnected",
    ),
)
def test_prompt_client_preserves_normalized_backend_failure(
    reason: str,
) -> None:
    error = ChatInferenceError(
        provider_id="ollama",
        model="primary:latest",
        reason=reason,
    )
    client = PromptClient(provider=FailingOllamaClient(error))

    with pytest.raises(InferenceBackendError) as captured:
        client.ask_messages("primary:latest", [{"role": "user", "content": "hola"}])

    assert captured.value.model == "primary:latest"
    assert captured.value.reason == reason
    assert captured.value.__cause__ is error
    assert isinstance(captured.value.__cause__, ChatInferenceError)


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

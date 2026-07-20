from __future__ import annotations

from models import prompt_client as prompt_client_module
from models.prompt_client import PromptClient


class _FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, *, model: str, messages: list[dict[str, str]]):
        self.calls.append({"model": model, "messages": messages})
        return {"message": {"content": "{\"status\":\"ok\"}"}}


def test_ask_messages_sends_exact_messages_without_extra_context(monkeypatch) -> None:
    fake_client = _FakeOllamaClient()
    monkeypatch.setattr(prompt_client_module.ollama, "Client", lambda: fake_client)
    client = PromptClient()
    messages = [
        {"role": "system", "content": "structured-planning-v1"},
        {"role": "user", "content": "{\"objective\":\"lee\"}"},
    ]

    response = client.ask_messages(model="planning-model", messages=messages)

    assert response == "{\"status\":\"ok\"}"
    assert fake_client.calls == [
        {
            "model": "planning-model",
            "messages": messages,
        }
    ]
    assert all("Atlas Coding Agent" not in message["content"] for message in fake_client.calls[0]["messages"])
    assert all("Atlas Project Agent" not in message["content"] for message in fake_client.calls[0]["messages"])


def test_ask_keeps_existing_message_route(monkeypatch) -> None:
    fake_client = _FakeOllamaClient()
    monkeypatch.setattr(prompt_client_module.ollama, "Client", lambda: fake_client)
    client = PromptClient()
    messages = [{"role": "system", "content": "Eres Atlas Coding Agent."}]

    response = client.ask(model="coding-model", messages=messages)

    assert response == "{\"status\":\"ok\"}"
    assert fake_client.calls == [
        {
            "model": "coding-model",
            "messages": messages,
        }
    ]

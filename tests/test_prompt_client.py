from __future__ import annotations

from models import prompt_client as prompt_client_module
from models.prompt_client import PromptClient


class _FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.stream_chunks = []

    def chat(self, *, model: str, messages: list[dict[str, str]], stream: bool = False):
        self.calls.append({"model": model, "messages": messages, "stream": stream})
        if stream:
            return iter(self.stream_chunks)
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
            "stream": False,
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
            "stream": False,
        }
    ]


def test_stream_messages_yields_content_in_order_and_ignores_empty_chunks(monkeypatch) -> None:
    fake_client = _FakeOllamaClient()
    fake_client.stream_chunks = [
        {"message": {"content": "{"}},
        {"message": {"content": ""}},
        {"message": {"content": "\"status\""}},
        {"message": {}},
        {"message": {"content": ":\"ok\"}"}},
    ]
    monkeypatch.setattr(prompt_client_module.ollama, "Client", lambda: fake_client)
    client = PromptClient()
    messages = [{"role": "user", "content": "plan"}]

    chunks = list(client.stream_messages(model="planning-model", messages=messages))

    assert chunks == ["{", "\"status\"", ":\"ok\"}"]
    assert fake_client.calls == [
        {
            "model": "planning-model",
            "messages": messages,
            "stream": True,
        }
    ]


def test_stream_messages_does_not_modify_or_extend_messages(monkeypatch) -> None:
    fake_client = _FakeOllamaClient()
    fake_client.stream_chunks = [{"message": {"content": "ok"}}]
    monkeypatch.setattr(prompt_client_module.ollama, "Client", lambda: fake_client)
    client = PromptClient()
    messages = [
        {"role": "system", "content": "structured-planning-v1"},
        {"role": "user", "content": "{\"objective\":\"lee\"}"},
    ]

    assert list(client.stream_messages(model="planning-model", messages=messages)) == ["ok"]

    assert fake_client.calls[0]["messages"] is messages
    assert len(fake_client.calls[0]["messages"]) == 2
    assert all("Atlas Coding Agent" not in message["content"] for message in fake_client.calls[0]["messages"])
    assert all("Atlas Project Agent" not in message["content"] for message in fake_client.calls[0]["messages"])


def test_stream_messages_rejects_malformed_content_chunk(monkeypatch) -> None:
    fake_client = _FakeOllamaClient()
    fake_client.stream_chunks = [{"message": {"content": 3}}]
    monkeypatch.setattr(prompt_client_module.ollama, "Client", lambda: fake_client)
    client = PromptClient()

    try:
        list(client.stream_messages(model="planning-model", messages=[]))
    except ValueError as error:
        assert "Malformed Ollama stream chunk" in str(error)
    else:
        raise AssertionError("Expected malformed stream chunk to raise ValueError")

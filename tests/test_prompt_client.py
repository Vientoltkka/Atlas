from __future__ import annotations

from models import prompt_client as prompt_client_module
from models.prompt_client import PromptClient


class _FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.stream_chunks = []

    def chat(self, *, model: str, messages: list[dict[str, str]], stream: bool = False, keep_alive: str = "10m"):
        self.calls.append({"model": model, "messages": messages, "stream": stream, "keep_alive": keep_alive})
        if stream:
            return iter(self.stream_chunks)
        return {"message": {"content": "{\"status\":\"ok\"}"}}


def test_ask_messages_sends_exact_messages_without_extra_context(monkeypatch) -> None:
    fake_client = _FakeOllamaClient()
    monkeypatch.setattr(prompt_client_module.ollama, "Client", lambda **_kwargs: fake_client)
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
            "keep_alive": "10m",
        }
    ]
    assert all("Atlas Coding Agent" not in message["content"] for message in fake_client.calls[0]["messages"])
    assert all("Atlas Project Agent" not in message["content"] for message in fake_client.calls[0]["messages"])


def test_ask_keeps_existing_message_route(monkeypatch) -> None:
    fake_client = _FakeOllamaClient()
    monkeypatch.setattr(prompt_client_module.ollama, "Client", lambda **_kwargs: fake_client)
    client = PromptClient()
    messages = [{"role": "system", "content": "Eres Atlas Coding Agent."}]

    response = client.ask(model="coding-model", messages=messages)

    assert response == "{\"status\":\"ok\"}"
    assert fake_client.calls == [
        {
            "model": "coding-model",
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
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
    monkeypatch.setattr(prompt_client_module.ollama, "Client", lambda **_kwargs: fake_client)
    client = PromptClient()
    messages = [{"role": "user", "content": "plan"}]

    chunks = list(client.stream_messages(model="planning-model", messages=messages))

    assert chunks == ["{", "\"status\"", ":\"ok\"}"]
    assert fake_client.calls == [
        {
            "model": "planning-model",
            "messages": messages,
            "stream": True,
            "keep_alive": "10m",
        }
    ]


def test_stream_messages_does_not_modify_or_extend_messages(monkeypatch) -> None:
    fake_client = _FakeOllamaClient()
    fake_client.stream_chunks = [{"message": {"content": "ok"}}]
    monkeypatch.setattr(prompt_client_module.ollama, "Client", lambda **_kwargs: fake_client)
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
    monkeypatch.setattr(prompt_client_module.ollama, "Client", lambda **_kwargs: fake_client)
    client = PromptClient()

    try:
        list(client.stream_messages(model="planning-model", messages=[]))
    except ValueError as error:
        assert "Malformed Ollama stream chunk" in str(error)
    else:
        raise AssertionError("Expected malformed stream chunk to raise ValueError")


def test_prompt_client_reports_native_ollama_metrics_and_warm_reuse(
    monkeypatch,
    capsys,
) -> None:
    fake_client = _FakeOllamaClient()
    fake_client.chat = lambda **kwargs: {
        "model": kwargs["model"],
        "message": {"content": "París"},
        "load_duration": 250_000_000,
        "eval_duration": 2_500_000_000,
        "total_duration": 3_000_000_000,
    }
    monkeypatch.setenv("ATLAS_VOICE_METRICS", "1")
    monkeypatch.setattr(prompt_client_module.ollama, "Client", lambda **_kwargs: fake_client)
    client = PromptClient()

    client.ask_messages("glm-local", [{"role": "user", "content": "uno"}])
    client.ask_messages("glm-local", [{"role": "user", "content": "dos"}])

    assert client.last_metrics == {
        "model": "glm-local",
        "load_seconds": 0.25,
        "generation_seconds": 2.5,
        "total_seconds": 3.0,
        "reused_loaded_model": True,
        "keep_alive": "10m",
    }
    output = capsys.readouterr().out
    assert "modelo=glm-local" in output
    assert "carga=0.250s" in output
    assert "generacion=2.500s" in output
    assert "modelo_reutilizado=si" in output

def test_prompt_client_configures_native_ollama_timeout(monkeypatch) -> None:
    captured: dict[str, float] = {}
    fake_client = _FakeOllamaClient()

    def build_client(**kwargs):
        captured.update(kwargs)
        return fake_client

    monkeypatch.setenv("ATLAS_OLLAMA_TIMEOUT", "75")
    monkeypatch.setattr(prompt_client_module.ollama, "Client", build_client)

    PromptClient()

    assert captured == {"timeout": 75.0}
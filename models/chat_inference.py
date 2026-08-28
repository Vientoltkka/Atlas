"""Provider-neutral chat inference contracts and the built-in Ollama adapter."""
from __future__ import annotations
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
import httpx
import ollama

class ChatInferenceError(RuntimeError):
    def __init__(self, provider_id: str, model: str, reason: str):
        self.provider_id, self.model, self.reason = provider_id, model, reason
        super().__init__(reason)

class ChatInferenceProvider(Protocol):
    provider_id: str
    def chat(self, *, model: str, messages: list[dict[str, str]], stream: bool) -> Any: ...
    def health(self, *, model: str) -> Any: ...
    def capabilities(self) -> frozenset[str]: ...

class OllamaChatInferenceProvider:
    provider_id = "ollama"
    def __init__(self, *, timeout: float, keep_alive: str, client: Any = None) -> None:
        self._client = client or ollama.Client(timeout=timeout); self._keep_alive = keep_alive
    def chat(self, *, model: str, messages: list[dict[str, str]], stream: bool) -> Any:
        try: return self._client.chat(model=model, messages=messages, stream=stream, keep_alive=self._keep_alive)
        except (ollama.ResponseError, ollama.RequestError, httpx.RequestError, TimeoutError, ConnectionError) as error: raise ChatInferenceError(self.provider_id, model, str(error)) from error
    def health(self, *, model: str) -> Any:
        try: return self._client.chat(model=model, messages=[{"role":"user","content":"ping"}], stream=False, keep_alive=self._keep_alive, options={"num_predict":1})
        except (ollama.ResponseError, ollama.RequestError, httpx.RequestError, TimeoutError, ConnectionError) as error: raise ChatInferenceError(self.provider_id, model, str(error)) from error
    def capabilities(self) -> frozenset[str]: return frozenset({"chat", "stream", "health", "local"})

@dataclass
class ChatInferenceProviderRegistry:
    providers: dict[str, ChatInferenceProvider]
    def get(self, provider_id: str) -> ChatInferenceProvider:
        try: return self.providers[provider_id]
        except KeyError as error: raise ValueError(f"Unknown chat inference provider: {provider_id}") from error

def default_provider_registry(*, timeout: float, keep_alive: str) -> ChatInferenceProviderRegistry:
    provider = OllamaChatInferenceProvider(timeout=timeout, keep_alive=keep_alive)
    return ChatInferenceProviderRegistry({provider.provider_id: provider})
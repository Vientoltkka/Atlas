"""Provider-neutral chat inference contracts and built-in providers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import os
from typing import Any, Protocol

import httpx
import ollama
from openai import OpenAI, OpenAIError


class ChatInferenceError(RuntimeError):
    def __init__(self, provider_id: str, model: str, reason: str):
        self.provider_id, self.model, self.reason = provider_id, model, reason
        super().__init__(reason)


class ChatInferenceProvider(Protocol):
    provider_id: str

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
    ) -> Any: ...

    def health(self, *, model: str) -> Any: ...

    def capabilities(self) -> frozenset[str]: ...


class OllamaChatInferenceProvider:
    provider_id = "ollama"

    def __init__(self, *, timeout: float, keep_alive: str, client: Any = None) -> None:
        self._client = client or ollama.Client(timeout=timeout)
        self._keep_alive = keep_alive

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
    ) -> Any:
        try:
            return self._client.chat(
                model=model,
                messages=messages,
                stream=stream,
                keep_alive=self._keep_alive,
            )
        except (
            ollama.ResponseError,
            ollama.RequestError,
            httpx.RequestError,
            TimeoutError,
            ConnectionError,
        ) as error:
            raise ChatInferenceError(self.provider_id, model, str(error)) from error

    def health(self, *, model: str) -> Any:
        try:
            return self._client.chat(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                stream=False,
                keep_alive=self._keep_alive,
                options={"num_predict": 1},
            )
        except (
            ollama.ResponseError,
            ollama.RequestError,
            httpx.RequestError,
            TimeoutError,
            ConnectionError,
        ) as error:
            raise ChatInferenceError(self.provider_id, model, str(error)) from error

    def capabilities(self) -> frozenset[str]:
        return frozenset({"chat", "stream", "health", "local"})


class OpenAICompatibleChatInferenceProvider:
    """Adapter for OpenAI-compatible ``/chat/completions`` services."""

    def __init__(
        self,
        *,
        provider_id: str,
        base_url: str,
        api_key: str,
        default_model: str = "",
        client: Any = None,
    ) -> None:
        self.provider_id = provider_id.strip()
        self._base_url = base_url.strip()
        self._api_key = api_key.strip()
        self._default_model = default_model.strip()
        if not self.provider_id:
            raise ValueError("OpenAI-compatible provider_id is required.")
        if not self._base_url:
            raise ValueError("OpenAI-compatible base_url is required.")
        if not self._api_key:
            raise ValueError("OpenAI-compatible api_key is required.")
        self._client = client or OpenAI(base_url=self._base_url, api_key=self._api_key)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
    ) -> Any:
        selected_model = self._select_model(model)
        try:
            response = self._client.chat.completions.create(
                model=selected_model,
                messages=messages,
                stream=stream,
            )
        except (OpenAIError, httpx.RequestError, TimeoutError, ConnectionError) as error:
            raise ChatInferenceError(
                self.provider_id,
                selected_model,
                str(error),
            ) from error
        if stream:
            return self._stream_response(response, selected_model)
        return self._response_payload(response, selected_model, stream=False)

    def health(self, *, model: str) -> Any:
        selected_model = self._select_model(model)
        try:
            response = self._client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                stream=False,
            )
        except (OpenAIError, httpx.RequestError, TimeoutError, ConnectionError) as error:
            raise ChatInferenceError(
                self.provider_id,
                selected_model,
                str(error),
            ) from error
        return self._response_payload(response, selected_model, stream=False)

    def capabilities(self) -> frozenset[str]:
        return frozenset({"chat", "stream", "health", "remote"})

    def _select_model(self, model: str) -> str:
        selected_model = model.strip() or self._default_model
        if not selected_model:
            raise ValueError("An OpenAI-compatible model is required.")
        return selected_model

    def _stream_response(
        self,
        response: Iterator[Any],
        model: str,
    ) -> Iterator[dict[str, Any]]:
        try:
            for chunk in response:
                yield self._response_payload(chunk, model, stream=True)
        except (OpenAIError, httpx.RequestError, TimeoutError, ConnectionError) as error:
            raise ChatInferenceError(self.provider_id, model, str(error)) from error

    @staticmethod
    def _response_payload(
        response: Any,
        requested_model: str,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        choice = _first_choice(response)
        message = _value(choice, "delta" if stream else "message")
        content = _value(message, "content")
        return {
            "model": _value(response, "model") or requested_model,
            "message": {"content": content},
        }


def _first_choice(response: Any) -> Any:
    choices = _value(response, "choices")
    if not isinstance(choices, list) or not choices:
        return None
    return choices[0]


def _value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


@dataclass
class ChatInferenceProviderRegistry:
    providers: dict[str, ChatInferenceProvider]

    def get(self, provider_id: str) -> ChatInferenceProvider:
        try:
            return self.providers[provider_id]
        except KeyError as error:
            raise ValueError(f"Unknown chat inference provider: {provider_id}") from error


def configured_provider_id() -> str:
    return os.getenv("ATLAS_CHAT_PROVIDER_ID", "ollama").strip() or "ollama"


def default_provider_registry(
    *,
    timeout: float,
    keep_alive: str,
    provider_id: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> ChatInferenceProviderRegistry:
    selected_provider_id = (provider_id or configured_provider_id()).strip() or "ollama"
    providers: dict[str, ChatInferenceProvider] = {
        "ollama": OllamaChatInferenceProvider(timeout=timeout, keep_alive=keep_alive)
    }
    if selected_provider_id != "ollama":
        providers[selected_provider_id] = OpenAICompatibleChatInferenceProvider(
            provider_id=selected_provider_id,
            base_url=base_url if base_url is not None else os.getenv("ATLAS_OPENAI_BASE_URL", ""),
            api_key=api_key if api_key is not None else os.getenv("ATLAS_OPENAI_API_KEY", ""),
            default_model=model if model is not None else os.getenv("ATLAS_OPENAI_MODEL", ""),
        )
    return ChatInferenceProviderRegistry(providers)
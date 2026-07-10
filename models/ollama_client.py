"""Ollama client wrapper for Atlas."""

from collections.abc import Mapping
from typing import Protocol, cast


class _OllamaListClient(Protocol):
    """Protocol for the Ollama client."""

    def list(self) -> object:
        ...


class _OllamaModule(Protocol):
    """Protocol for the official ollama module."""

    def Client(self) -> _OllamaListClient:
        ...


class OllamaClient:
    """Wrapper around the official Ollama client."""

    def __init__(self) -> None:
        self._client = self._load_client()

    def list_models(self) -> list[str]:
        response = self._client.list()
        models = self._extract_models(response)

        result: list[str] = []

        for model in models:
            name = self._extract_model_name(model)
            if name:
                result.append(name)

        return result

    def _load_client(self) -> _OllamaListClient:
        import ollama

        return cast(_OllamaModule, ollama).Client()

    def _extract_models(self, response: object) -> list[object]:
        if isinstance(response, Mapping):
            return response.get("models", [])

        return getattr(response, "models", [])

    def _extract_model_name(self, model: object) -> str | None:
        if isinstance(model, Mapping):
            return model.get("model") or model.get("name")

        return getattr(model, "model", None) or getattr(model, "name", None)
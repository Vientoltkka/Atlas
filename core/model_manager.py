"""Model selection logic for Atlas."""

from collections.abc import Mapping
from typing import Protocol

from models.ollama_client import OllamaClient


class _ModelSource(Protocol):
    """Anything capable of listing installed models."""

    def list_models(self) -> list[str]:
        """Return the installed model names."""


class ModelManager:
    """Choose the best model for each task."""

    _TASK_MODELS: Mapping[str, str] = {
        "coding": "qwen3.6:latest",
        "reasoning": "qwen3.6:latest",
        "chat": "glm-5.2-local:latest",
        "vision": "gemma4:latest",
        "project": "glm-5.2-local:latest",
    }

    def __init__(self, client: _ModelSource | None = None) -> None:
        self._client = client if client is not None else OllamaClient()

    def list_models(self) -> list[str]:
        return self._client.list_models()

    def choose_model(self, task: str) -> str:
        models = self.list_models()

        if not models:
            raise RuntimeError("No hay modelos instalados.")

        preferred = self._TASK_MODELS.get(task)

        if preferred and preferred in models:
            return preferred

        return models[0]
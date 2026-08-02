"""Model selection logic for Atlas."""

from collections.abc import Mapping
from typing import Protocol

from core.execution_resources import (
    ExecutionResourceCatalog,
    ResourceCandidate,
    ResourceHealthStatus,
    ResourceType,
)
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
        "chat": "glm4:9b",
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

    def list_model_candidates(self) -> tuple[ResourceCandidate, ...]:
        """Return local model candidates without querying provider metadata."""
        return tuple(
            ResourceCandidate(
                resource_id=_model_resource_id(model),
                resource_type=ResourceType.MODEL,
                provider_id="ollama",
                capabilities=(),
                estimated_cost=None,
                estimated_latency=None,
                local=True,
                available=True,
                health_status=ResourceHealthStatus.AVAILABLE,
            )
            for model in self.list_models()
        )

    def resource_catalog(self) -> ExecutionResourceCatalog:
        """Return a resource catalog view over currently listed models."""
        return ExecutionResourceCatalog(self.list_model_candidates())


def _model_resource_id(model: object) -> str:
    if isinstance(model, str):
        return model
    if isinstance(model, Mapping):
        for key in ("name", "model", "id"):
            value = model.get(key)
            if isinstance(value, str) and value.strip():
                return value
    raise ValueError("Model entries must be strings or mappings with a model name.")

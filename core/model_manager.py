"""Model inventory and backward-compatible selection logic for Atlas."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
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


def _normalized_values(values: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Model metadata values must be non-empty strings.")
        item = value.strip()
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Provider-neutral metadata for one logical Atlas model."""

    logical_id: str
    provider_id: str
    model_name: str
    capabilities: tuple[str, ...] = ()
    available: bool = True
    relative_cost: float | None = None
    relative_latency: float | None = None
    local: bool = True
    priority: int = 0
    fallback_logical_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("logical_id", "provider_id", "model_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "capabilities", _normalized_values(self.capabilities))
        object.__setattr__(
            self,
            "fallback_logical_ids",
            _normalized_values(self.fallback_logical_ids),
        )
        if type(self.available) is not bool or type(self.local) is not bool:
            raise TypeError("available and local must be bool values.")
        if type(self.priority) is not int:
            raise TypeError("priority must be an int.")
        for name in ("relative_cost", "relative_latency"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                raise ValueError(f"{name} must be non-negative when provided.")

class ModelManager:
    """Expose the model inventory and preserve the current task selection."""


    _TASK_LOGICAL_MODELS: Mapping[str, str] = {
        "coding": "coding-local",
        "reasoning": "coding-local",
        "chat": "chat-local",
        "vision": "vision-local",
        "project": "project-local",
    }

    _DEFAULT_DESCRIPTORS: tuple[ModelDescriptor, ...] = (
        ModelDescriptor(
            logical_id="coding-local",
            provider_id="ollama",
            model_name="qwen3.6:latest",
            capabilities=("general_chat", "coding", "reasoning", "local"),
            priority=100,
            fallback_logical_ids=("chat-local",),
        ),
        ModelDescriptor(
            logical_id="chat-local",
            provider_id="ollama",
            model_name="glm4:9b",
            capabilities=("general_chat", "fast_response", "local"),
            priority=90,
        ),
        ModelDescriptor(
            logical_id="vision-local",
            provider_id="ollama",
            model_name="gemma4:latest",
            capabilities=("general_chat", "vision", "local"),
            priority=80,
            fallback_logical_ids=("chat-local",),
        ),
        ModelDescriptor(
            logical_id="project-local",
            provider_id="ollama",
            model_name="glm-5.2-local:latest",
            capabilities=("general_chat", "coding", "reasoning", "local"),
            priority=95,
            fallback_logical_ids=("coding-local", "chat-local"),
        ),
    )

    def __init__(
        self,
        client: _ModelSource | None = None,
        descriptors: Iterable[ModelDescriptor] = (),
    ) -> None:
        self._client = client if client is not None else OllamaClient()
        self._descriptors = {
            descriptor.logical_id: descriptor
            for descriptor in self._DEFAULT_DESCRIPTORS
        }
        for descriptor in descriptors:
            self.register_model(descriptor)

    def list_models(self) -> list[str]:
        return self._client.list_models()

    def choose_model(self, task: str) -> str:
        models = self.list_models()

        if not models:
            raise RuntimeError("No hay modelos instalados.")

        logical_id = self._TASK_LOGICAL_MODELS.get(task)
        descriptor = self._descriptors.get(logical_id or "")
        preferred = descriptor.model_name if descriptor and descriptor.available else None

        if preferred and preferred in models:
            return preferred

        return models[0]

    def register_model(self, descriptor: ModelDescriptor) -> None:
        """Register model metadata without activating automatic selection."""
        if not isinstance(descriptor, ModelDescriptor):
            raise TypeError("descriptor must be a ModelDescriptor.")
        if descriptor.logical_id in self._descriptors:
            raise ValueError(
                f"Model logical id '{descriptor.logical_id}' is already registered."
            )
        self._descriptors[descriptor.logical_id] = descriptor

    def resolve_model(self, identifier: str) -> ModelDescriptor | None:
        """Resolve a logical id or provider model name with live availability."""
        if not isinstance(identifier, str) or not identifier.strip():
            return None
        identifier = identifier.strip()
        installed = [
            _model_resource_id(model)
            for model in self.list_models()
        ]
        descriptor = self._descriptors.get(identifier)
        if descriptor is None:
            matches = (
                item
                for item in self._descriptors.values()
                if item.model_name == identifier
            )
            descriptor = next(
                iter(sorted(matches, key=lambda item: (-item.priority, item.logical_id))),
                None,
            )
        if descriptor is None and identifier in installed:
            descriptor = self._discovered_descriptor(identifier)
        if descriptor is None:
            return None
        return replace(
            descriptor,
            available=descriptor.available and descriptor.model_name in installed,
        )

    def list_model_descriptors(
        self,
        *,
        available_only: bool = False,
    ) -> tuple[ModelDescriptor, ...]:
        """Return registered and provider-discovered models with live availability."""
        installed = [
            _model_resource_id(model)
            for model in self.list_models()
        ]
        descriptors = [
            replace(
                descriptor,
                available=descriptor.available and descriptor.model_name in installed,
            )
            for descriptor in self._descriptors.values()
        ]
        registered_names = {item.model_name for item in descriptors}
        descriptors.extend(
            self._discovered_descriptor(model)
            for model in installed
            if model not in registered_names
        )
        if available_only:
            descriptors = [item for item in descriptors if item.available]
        return tuple(
            sorted(descriptors, key=lambda item: (-item.priority, item.logical_id))
        )

    def list_model_candidates(self) -> tuple[ResourceCandidate, ...]:
        """Return candidates backed by the shared execution-resource contract."""
        descriptors_by_name: dict[str, ModelDescriptor] = {}
        for descriptor in self.list_model_descriptors(available_only=True):
            current = descriptors_by_name.get(descriptor.model_name)
            if current is None or descriptor.priority > current.priority:
                descriptors_by_name[descriptor.model_name] = descriptor
        candidates: list[ResourceCandidate] = []
        for model in self.list_models():
            model_name = _model_resource_id(model)
            descriptor = descriptors_by_name.get(
                model_name,
                self._discovered_descriptor(model_name),
            )
            candidates.append(
                ResourceCandidate(
                    resource_id=descriptor.model_name,
                    resource_type=ResourceType.MODEL,
                    provider_id=descriptor.provider_id,
                    capabilities=descriptor.capabilities,
                    estimated_cost=descriptor.relative_cost,
                    estimated_latency=descriptor.relative_latency,
                    local=descriptor.local,
                    available=descriptor.available,
                    health_status=(
                        ResourceHealthStatus.AVAILABLE
                        if descriptor.available
                        else ResourceHealthStatus.UNAVAILABLE
                    ),
                )
            )
        return tuple(candidates)

    def resource_catalog(self) -> ExecutionResourceCatalog:
        """Return a resource catalog view over currently listed models."""
        return ExecutionResourceCatalog(self.list_model_candidates())

    @staticmethod
    def _discovered_descriptor(model_name: str) -> ModelDescriptor:
        return ModelDescriptor(
            logical_id=model_name,
            provider_id="ollama",
            model_name=model_name,
            capabilities=(),
            available=True,
            local=True,
        )


def _model_resource_id(model: object) -> str:
    if isinstance(model, str):
        return model
    if isinstance(model, Mapping):
        for key in ("name", "model", "id"):
            value = model.get(key)
            if isinstance(value, str) and value.strip():
                return value
    raise ValueError("Model entries must be strings or mappings with a model name.")

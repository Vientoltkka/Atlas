"""Model inventory and backward-compatible selection logic for Atlas."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import math
from typing import Protocol

from core.execution_resources import (
    ExecutionResourceCatalog,
    ExecutionResourceOptimizer,
    ExecutionResourcePolicy,
    ExecutionResourceRequirements,
    NoCompatibleResourceError,
    OptimizationGoal,
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
    context_window: int | None = None
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
        if self.priority < 0:
            raise ValueError("priority must be non-negative.")
        for name in ("relative_cost", "relative_latency"):
            value = getattr(self, name)
            if (
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                )
            ):
                raise ValueError(
                    f"{name} must be finite and non-negative when provided."
                )
        if self.context_window is not None and (
            type(self.context_window) is not int or self.context_window <= 0
        ):
            raise ValueError("context_window must be a positive int when provided.")

@dataclass(frozen=True, slots=True)
class ModelSelectionRequest:
    """Deterministic requirements for selecting one registered model."""

    task: str
    prefer_local: bool | None = None
    maximum_relative_cost: float | None = None
    maximum_relative_latency: float | None = None
    preferred_model_id: str | None = None
    preferred_provider_id: str | None = None
    allow_fallback: bool = False
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string.")
        object.__setattr__(self, "task", self.task.strip().lower())
        object.__setattr__(
            self,
            "required_capabilities",
            _normalized_values(self.required_capabilities),
        )
        if self.prefer_local is not None and type(self.prefer_local) is not bool:
            raise TypeError("prefer_local must be a bool or None.")
        if type(self.allow_fallback) is not bool:
            raise TypeError("allow_fallback must be a bool.")
        for name in ("maximum_relative_cost", "maximum_relative_latency"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                raise ValueError(f"{name} must be non-negative when provided.")
        for name in ("preferred_model_id", "preferred_provider_id"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a non-empty string when provided.")
                object.__setattr__(self, name, value.strip())


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    """Controlled outcome of deterministic model selection."""

    success: bool
    logical_model_id: str | None
    physical_model_name: str | None
    provider_id: str | None
    reason: str
    is_fallback: bool
    descriptor: ModelDescriptor | None
    error_code: str | None = None

class ModelManager:
    """Expose the model inventory and preserve the current task selection."""


    _TASK_LOGICAL_MODELS: Mapping[str, str] = {
        "coding": "coding-local",
        "reasoning": "coding-local",
        "chat": "chat-local",
        "vision": "vision-local",
        "project": "project-local",
    }

    _TASK_CAPABILITIES: Mapping[str, str] = {
        "chat": "chat",
        "coding": "coding",
        "project": "project",
        "vision": "vision",
        "reasoning": "reasoning",
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
            capabilities=("general_chat", "chat", "fast_response", "local"),
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
            capabilities=(
                "general_chat",
                "project",
                "coding",
                "reasoning",
                "local",
            ),
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

    def choose_model(
        self,
        task: str,
        *,
        selection_result: ModelSelectionResult | None = None,
    ) -> str:
        """Compatibility facade preferring deterministic capability selection."""
        if selection_result is not None and not isinstance(
            selection_result, ModelSelectionResult
        ):
            raise TypeError("selection_result must be a ModelSelectionResult or None.")
        selection = (
            selection_result
            if selection_result is not None
            else self.select_model(ModelSelectionRequest(task=task))
        )
        if selection.success and selection.physical_model_name is not None:
            return selection.physical_model_name

        return self._choose_model_legacy(task)

    def _choose_model_legacy(self, task: str) -> str:
        """Preserve the pre-18.3 first-available compatibility behavior."""
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

    def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResult:
        """Select one model by capability through the shared resource optimizer."""
        if not isinstance(request, ModelSelectionRequest):
            raise TypeError("request must be a ModelSelectionRequest.")

        preferred = (
            self.resolve_model(request.preferred_model_id)
            if request.preferred_model_id is not None
            else None
        )
        if request.preferred_model_id is not None and preferred is None:
            return self._selection_failure()

        catalog = self.resource_catalog()
        selection_catalogs = [catalog]
        if preferred is not None:
            selection_catalogs = [
                ExecutionResourceCatalog(
                    tuple(
                        candidate
                        for candidate in catalog.list_candidates()
                        if candidate.resource_id == preferred.model_name
                    )
                )
            ]
            if request.allow_fallback:
                fallback_model_names = self._fallback_model_names(preferred)
                selection_catalogs.append(
                    ExecutionResourceCatalog(
                        tuple(
                            candidate
                            for candidate in catalog.list_candidates()
                            if candidate.resource_id in fallback_model_names
                        )
                    )
                )

        requirements = ExecutionResourceRequirements(
            required_capabilities=self._required_capabilities(request),
            maximum_estimated_cost=request.maximum_relative_cost,
            maximum_latency_seconds=request.maximum_relative_latency,
            preferred_model_ids=(preferred.model_name,) if preferred is not None else (),
            preferred_provider_ids=(request.preferred_provider_id,)
            if request.preferred_provider_id is not None
            else (),
        )
        decision = None
        for selection_catalog in selection_catalogs:
            preferred_catalogs: list[ExecutionResourceCatalog] = []
            if request.preferred_provider_id is not None:
                preferred_catalogs.append(
                    ExecutionResourceCatalog(
                        tuple(
                            candidate
                            for candidate in selection_catalog.list_candidates()
                            if candidate.provider_id == request.preferred_provider_id
                        )
                    )
                )
            if request.prefer_local is True:
                preferred_catalogs.append(
                    ExecutionResourceCatalog(
                        tuple(
                            candidate
                            for candidate in selection_catalog.list_candidates()
                            if candidate.local
                        )
                    )
                )
            preferred_catalogs.append(selection_catalog)

            for candidate_catalog in preferred_catalogs:
                try:
                    decision = ExecutionResourceOptimizer(
                        ExecutionResourcePolicy(
                            enabled=True,
                            optimization_goal=OptimizationGoal.BALANCED,
                        )
                    ).select(
                        step_id="model_selection",
                        requirements=requirements,
                        catalog=candidate_catalog,
                    )
                    break
                except NoCompatibleResourceError:
                    continue
            if decision is not None:
                break
        if decision is None:
            return self._selection_failure()

        descriptor = self.resolve_model(decision.selected_resource_id or "")
        if descriptor is None:
            return self._selection_failure(
                error_code="SELECTED_MODEL_NOT_RESOLVABLE"
            )
        is_fallback = (
            preferred is not None
            and descriptor.logical_id != preferred.logical_id
        )
        return ModelSelectionResult(
            success=True,
            logical_model_id=descriptor.logical_id,
            physical_model_name=descriptor.model_name,
            provider_id=descriptor.provider_id,
            reason=(
                "Selected declared fallback model."
                if is_fallback
                else f"Selected compatible model ({decision.reason.value})."
            ),
            is_fallback=is_fallback,
            descriptor=descriptor,
        )

    def _required_capabilities(
        self,
        request: ModelSelectionRequest,
    ) -> tuple[str, ...]:
        """Combine the task capability with explicit requirements in stable order."""
        task_capability = self._TASK_CAPABILITIES.get(request.task, request.task)
        return _normalized_values((task_capability, *request.required_capabilities))

    def _fallback_model_names(
        self,
        preferred: ModelDescriptor,
    ) -> set[str]:
        """Return the finite transitive set of declared fallback model names."""
        model_names: set[str] = set()
        visited = {preferred.logical_id}
        pending = list(preferred.fallback_logical_ids)
        position = 0

        while position < len(pending):
            logical_id = pending[position]
            position += 1
            if logical_id in visited:
                continue
            visited.add(logical_id)

            descriptor = self._descriptors.get(logical_id)
            if descriptor is None:
                continue
            model_names.add(descriptor.model_name)
            pending.extend(descriptor.fallback_logical_ids)

        return model_names

    def select_fallback(
        self,
        request: ModelSelectionRequest,
        *,
        initial_model_id: str,
        attempted_model_ids: Iterable[str],
    ) -> ModelSelectionResult:
        """Select the next compatible model from one declared fallback chain."""
        if not isinstance(request, ModelSelectionRequest):
            raise TypeError("request must be a ModelSelectionRequest.")
        if not request.allow_fallback:
            return self._selection_failure(error_code="FALLBACK_NOT_ALLOWED")

        initial = self.resolve_model(initial_model_id)
        if initial is None:
            return self._selection_failure(error_code="INITIAL_MODEL_NOT_RESOLVABLE")

        attempted = set(_normalized_values(attempted_model_ids))
        for descriptor in self._fallback_descriptors(initial):
            if (
                descriptor.logical_id in attempted
                or descriptor.model_name in attempted
            ):
                continue
            selection = self.select_model(
                replace(
                    request,
                    preferred_model_id=descriptor.logical_id,
                    allow_fallback=False,
                )
            )
            if selection.success:
                return replace(
                    selection,
                    reason="Selected next declared fallback after inference failure.",
                    is_fallback=True,
                )

        return self._selection_failure(error_code="FALLBACK_CHAIN_EXHAUSTED")

    def _fallback_descriptors(
        self,
        initial: ModelDescriptor,
    ) -> tuple[ModelDescriptor, ...]:
        """Return declared transitive fallbacks once, preserving declaration order."""
        descriptors: list[ModelDescriptor] = []
        visited = {initial.logical_id}
        pending = list(initial.fallback_logical_ids)
        position = 0

        while position < len(pending):
            logical_id = pending[position]
            position += 1
            if logical_id in visited:
                continue
            visited.add(logical_id)
            descriptor = self._descriptors.get(logical_id)
            if descriptor is None:
                continue
            descriptors.append(descriptor)
            pending.extend(descriptor.fallback_logical_ids)

        return tuple(descriptors)

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
                    quality_tier=descriptor.priority,
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


    @staticmethod
    def _selection_failure(
        *,
        error_code: str = "NO_COMPATIBLE_MODEL",
    ) -> ModelSelectionResult:
        return ModelSelectionResult(
            success=False,
            logical_model_id=None,
            physical_model_name=None,
            provider_id=None,
            reason="No compatible registered model is available.",
            is_fallback=False,
            descriptor=None,
            error_code=error_code,
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

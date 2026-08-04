from __future__ import annotations

import json

import pytest

from core.hybrid_execution_planner import PromptClientStructuredPlanProvider
from core.model_health import ModelHealthResult
from core.model_inference import (
    InferenceFallbackExhaustedError,
    InferenceStreamInterruptedError,
    ModelInferenceRunner,
)
from core.model_manager import (
    ModelDescriptor,
    ModelManager,
    ModelSelectionRequest,
    ModelSelectionResult,
)
from core.model_selection_policy import ModelSelectionPolicy
from core.orchestrator import AtlasOrchestrator
from core.planner import Plan
from core.router import Router
from models.prompt_client import InferenceBackendError


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


class RecordingModelManager(ModelManager):
    def __init__(
        self,
        models: list[str],
        descriptors: tuple[ModelDescriptor, ...] = (),
    ) -> None:
        super().__init__(StaticModelSource(models), descriptors)
        self.requests: list[ModelSelectionRequest] = []
        self.results: list[ModelSelectionResult] = []

    def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResult:
        self.requests.append(request)
        result = super().select_model(request)
        self.results.append(result)
        return result


class RecordingPromptClient:
    def __init__(
        self,
        *,
        response: str = "ok",
        failing_models: tuple[str, ...] = (),
    ) -> None:
        self.response = response
        self.failing_models = set(failing_models)
        self.models: list[str] = []

    def ask(self, model: str, messages: list[dict[str, str]]) -> str:
        self.models.append(model)
        if model in self.failing_models:
            raise InferenceBackendError(model, "simulated inference failure")
        return self.response

    def ask_messages(self, model: str, messages: list[dict[str, str]]) -> str:
        return self.ask(model, messages)


class PromptBackedAgent:
    def __init__(self, prompt_client: RecordingPromptClient) -> None:
        self._prompt_client = prompt_client

    def run(self, model: str, messages: list[dict[str, str]]) -> str:
        return self._prompt_client.ask(model, messages)


class FixedPlanner:
    def __init__(self, task: str) -> None:
        self._task = task

    def create_plan(self, prompt: str) -> Plan:
        return Plan(task=self._task, objective=prompt)


class Memory:
    def __init__(self) -> None:
        self._messages: list[dict[str, str]] = []

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})

    def history(self) -> list[dict[str, str]]:
        return list(self._messages)


class Registry:
    def __init__(self, task: str, agent: PromptBackedAgent) -> None:
        self._task = task
        self._agent = agent

    def get(self, task: str):
        return self._agent if task == self._task else None


class WriteFile:
    def execute(self, path: str, content: str) -> str:
        return "ok"


class RecordingHealthChecker:
    def __init__(self, unhealthy: tuple[str, ...] = ()) -> None:
        self._unhealthy = set(unhealthy)
        self.checked: list[str] = []

    def check(
        self,
        *,
        logical_model_id: str,
        physical_model_name: str,
        provider_id: str,
    ) -> ModelHealthResult:
        self.checked.append(logical_model_id)
        healthy = logical_model_id not in self._unhealthy
        return ModelHealthResult(
            logical_model_id=logical_model_id,
            physical_model_name=physical_model_name,
            provider_id=provider_id,
            healthy=healthy,
            reason=None if healthy else "simulated unhealthy model",
        )


def _descriptor(
    logical_id: str,
    physical_name: str,
    capability: str,
    *,
    provider: str = "ollama",
    local: bool = True,
    priority: int = 0,
    fallbacks: tuple[str, ...] = (),
) -> ModelDescriptor:
    return ModelDescriptor(
        logical_id=logical_id,
        provider_id=provider,
        model_name=physical_name,
        capabilities=(capability,),
        local=local,
        priority=priority,
        fallback_logical_ids=fallbacks,
    )


def _manager(*descriptors: ModelDescriptor) -> RecordingModelManager:
    return RecordingModelManager(
        [descriptor.model_name for descriptor in descriptors],
        descriptors,
    )


def _orchestrator(
    task: str,
    manager: ModelManager,
    prompt_client: RecordingPromptClient,
    *,
    policy: ModelSelectionPolicy | None = None,
) -> AtlasOrchestrator:
    return AtlasOrchestrator(
        planner=FixedPlanner(task),
        router=Router(),
        model_manager=manager,
        memory=Memory(),
        registry=Registry(task, PromptBackedAgent(prompt_client)),
        write_file=WriteFile(),
        model_selection_policy=policy,
    )


@pytest.mark.parametrize(
    ("task", "logical_id", "physical_name"),
    (
        ("chat", "chat-local", "glm4:9b"),
        ("coding", "coding-local", "qwen3.6:latest"),
        ("project", "project-local", "glm-5.2-local:latest"),
    ),
)
def test_real_routes_select_once_and_deliver_final_physical_model_to_prompt_client(
    task: str,
    logical_id: str,
    physical_name: str,
) -> None:
    manager = RecordingModelManager(
        ["glm4:9b", "qwen3.6:latest", "glm-5.2-local:latest"]
    )
    prompt_client = RecordingPromptClient(response=f"{task} response")
    orchestrator = _orchestrator(task, manager, prompt_client)

    first = orchestrator.process_prompt("primera", confirm=lambda _prompt: "")
    second = orchestrator.process_prompt("segunda", confirm=lambda _prompt: "")

    assert (first, second) == (f"{task} response", f"{task} response")
    assert manager.requests == [
        ModelSelectionRequest(task=task, allow_fallback=True),
        ModelSelectionRequest(task=task, allow_fallback=True),
    ]
    assert prompt_client.models == [physical_name, physical_name]
    assert [result.logical_model_id for result in manager.results] == [
        logical_id,
        logical_id,
    ]
    for result in manager.results:
        assert result.physical_model_name == physical_name
        assert result.provider_id == "ollama"
        assert result.is_fallback is False
        assert result.descriptor is not None
        assert task in result.descriptor.capabilities


def test_structured_reasoning_uses_common_selector_policy_and_physical_model() -> None:
    manager = RecordingModelManager(["glm-5.2-local:latest"])
    prompt_client = RecordingPromptClient(response=json.dumps({"steps": []}))
    policy = ModelSelectionPolicy(
        preferred_provider="ollama",
        prefer_local=True,
        allow_fallback=False,
    )
    provider = PromptClientStructuredPlanProvider(
        prompt_client,
        model_name="project-local",
        model_manager=manager,
        model_selection_policy=policy,
    )

    result = provider.generate_plan("analiza el proyecto", json.dumps({"tools": []}))

    assert result.success is True
    assert result.model_name == "glm-5.2-local:latest"
    assert prompt_client.models == ["glm-5.2-local:latest"]
    assert manager.requests == [
        ModelSelectionRequest(
            task="reasoning",
            prefer_local=True,
            preferred_model_id="project-local",
            preferred_provider_id="ollama",
            allow_fallback=False,
        )
    ]
    selection = manager.results[0]
    assert selection.logical_model_id == "project-local"
    assert selection.physical_model_name == "glm-5.2-local:latest"
    assert selection.provider_id == "ollama"
    assert selection.is_fallback is False
    assert selection.descriptor is not None
    assert "reasoning" in selection.descriptor.capabilities


def test_runtime_policy_prefers_local_and_requested_provider_with_safe_degradation() -> None:
    local_manager = _manager(
        _descriptor(
            "remote-chat",
            "remote:latest",
            "chat",
            provider="remote-provider",
            local=False,
            priority=100,
        ),
        _descriptor(
            "local-chat",
            "local:latest",
            "chat",
            provider="ollama",
            local=True,
            priority=1,
        ),
    )
    local_client = RecordingPromptClient()

    _orchestrator(
        "chat",
        local_manager,
        local_client,
        policy=ModelSelectionPolicy(prefer_local=True),
    ).process_prompt("local", confirm=lambda _prompt: "")

    assert local_manager.requests[0].prefer_local is True
    assert local_manager.results[0].logical_model_id == "local-chat"
    assert local_client.models == ["local:latest"]

    provider_manager = _manager(
        _descriptor(
            "provider-a",
            "provider-a:latest",
            "chat",
            provider="provider-a",
        ),
        _descriptor(
            "provider-b",
            "provider-b:latest",
            "chat",
            provider="provider-b",
        ),
    )
    preferred = provider_manager.select_model(
        ModelSelectionPolicy(preferred_provider="provider-b").create_request(
            task="chat"
        )
    )
    degraded = _manager(
        _descriptor(
            "only-provider-a",
            "only-provider-a:latest",
            "chat",
            provider="provider-a",
        )
    ).select_model(
        ModelSelectionPolicy(preferred_provider="missing-provider").create_request(
            task="chat"
        )
    )

    assert preferred.logical_model_id == "provider-b"
    assert preferred.provider_id == "provider-b"
    assert degraded.success is True
    assert degraded.logical_model_id == "only-provider-a"


def test_fallback_disabled_attempts_only_primary_and_returns_structured_error() -> None:
    manager = _manager(
        _descriptor(
            "primary",
            "primary:latest",
            "chat",
            fallbacks=("fallback",),
        ),
        _descriptor("fallback", "fallback:latest", "chat"),
    )
    prompt_client = RecordingPromptClient(failing_models=("primary:latest",))
    runner = ModelInferenceRunner(manager)

    with pytest.raises(InferenceFallbackExhaustedError) as captured:
        runner.run(
            ModelSelectionPolicy(allow_fallback=False).create_request(
                task="chat",
                preferred_model_id="primary",
            ),
            lambda model: prompt_client.ask(model, []),
        )

    assert prompt_client.models == ["primary:latest"]
    assert captured.value.allow_fallback is False
    assert captured.value.attempted_logical_model_ids == ("primary",)


def test_unhealthy_primary_uses_declared_healthy_fallback_before_inference() -> None:
    manager = _manager(
        _descriptor(
            "primary",
            "primary:latest",
            "chat",
            fallbacks=("fallback",),
        ),
        _descriptor("fallback", "fallback:latest", "chat"),
    )
    health = RecordingHealthChecker(unhealthy=("primary",))
    prompt_client = RecordingPromptClient(response="healthy fallback")
    runner = ModelInferenceRunner(manager, health_checker=health)

    response = runner.run(
        ModelSelectionPolicy().create_request(
            task="chat",
            preferred_model_id="primary",
        ),
        lambda model: prompt_client.ask(model, []),
    )

    assert response == "healthy fallback"
    assert health.checked == ["primary", "fallback"]
    assert prompt_client.models == ["fallback:latest"]
    assert runner.last_result is not None
    assert runner.last_result.initial_logical_model_id == "primary"
    assert runner.last_result.final_logical_model_id == "fallback"
    assert runner.last_result.final_physical_model_name == "fallback:latest"
    assert runner.last_result.used_fallback is True


def test_primary_inference_failure_uses_declared_fallback_and_final_physical_model() -> None:
    manager = _manager(
        _descriptor(
            "primary",
            "primary:latest",
            "chat",
            fallbacks=("fallback",),
        ),
        _descriptor("fallback", "fallback:latest", "chat"),
    )
    prompt_client = RecordingPromptClient(
        response="runtime fallback",
        failing_models=("primary:latest",),
    )
    runner = ModelInferenceRunner(manager)

    response = runner.run(
        ModelSelectionPolicy().create_request(
            task="chat",
            preferred_model_id="primary",
        ),
        lambda model: prompt_client.ask(model, []),
    )

    assert response == "runtime fallback"
    assert prompt_client.models == ["primary:latest", "fallback:latest"]
    assert runner.last_result is not None
    assert runner.last_result.initial_logical_model_id == "primary"
    assert runner.last_result.final_logical_model_id == "fallback"
    assert runner.last_result.final_physical_model_name == "fallback:latest"
    assert runner.last_result.used_fallback is True
    assert manager.resolve_model("fallback").provider_id == "ollama"


def test_streaming_partial_output_never_starts_or_mixes_a_fallback_model() -> None:
    manager = _manager(
        _descriptor(
            "primary",
            "primary:latest",
            "chat",
            fallbacks=("fallback",),
        ),
        _descriptor("fallback", "fallback:latest", "chat"),
    )
    attempted_models: list[str] = []

    def stream(model: str):
        attempted_models.append(model)
        if model == "primary:latest":
            yield "partial"
            raise InferenceBackendError(model, "failed after visible output")
        yield "must not be emitted"

    iterator = ModelInferenceRunner(manager).stream(
        ModelSelectionPolicy().create_request(
            task="chat",
            preferred_model_id="primary",
        ),
        stream,
    )

    assert next(iterator) == "partial"
    with pytest.raises(InferenceStreamInterruptedError):
        next(iterator)
    assert attempted_models == ["primary:latest"]


def test_cyclic_fallback_chain_attempts_each_model_once_and_terminates() -> None:
    manager = _manager(
        _descriptor("cycle-a", "cycle-a:latest", "chat", fallbacks=("cycle-b",)),
        _descriptor("cycle-b", "cycle-b:latest", "chat", fallbacks=("cycle-a",)),
    )
    prompt_client = RecordingPromptClient(
        failing_models=("cycle-a:latest", "cycle-b:latest")
    )

    with pytest.raises(InferenceFallbackExhaustedError) as captured:
        ModelInferenceRunner(manager).run(
            ModelSelectionPolicy().create_request(
                task="chat",
                preferred_model_id="cycle-a",
            ),
            lambda model: prompt_client.ask(model, []),
        )

    assert prompt_client.models == ["cycle-a:latest", "cycle-b:latest"]
    assert captured.value.attempted_logical_model_ids == ("cycle-a", "cycle-b")


def test_no_compatible_model_returns_existing_structured_selection_error() -> None:
    manager = RecordingModelManager([])
    request = ModelSelectionPolicy().create_request(task="unsupported-capability")

    first = manager.select_model(request)
    second = manager.select_model(request)

    assert first == second
    assert first.success is False
    assert first.error_code == "NO_COMPATIBLE_MODEL"
    assert first.logical_model_id is None
    assert first.physical_model_name is None
    assert first.provider_id is None
    assert first.is_fallback is False

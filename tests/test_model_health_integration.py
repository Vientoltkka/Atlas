from __future__ import annotations

import json

import ollama

from core.hybrid_execution_planner import PromptClientStructuredPlanProvider
from core.model_health import (
    ModelHealthErrorCode,
    ModelHealthResult,
    OllamaModelHealthChecker,
)
from core.model_manager import ModelDescriptor, ModelManager
from models.chat_inference import ChatInferenceProviderRegistry
from models import prompt_client as prompt_client_module
from models.prompt_client import PromptClient


class CapturingOllamaClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "pong"}}


def test_prompt_client_health_check_is_minimal_and_non_streaming(monkeypatch) -> None:
    backend = CapturingOllamaClient()
    monkeypatch.setattr(
        prompt_client_module.ollama,
        "Client",
        lambda **_kwargs: backend,
    )

    PromptClient().check_model_health("primary:latest")

    assert backend.calls == [
        {
            "model": "primary:latest",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_predict": 1},
        }
    ]


def test_prompt_client_health_check_accepts_valid_whitespace_content(monkeypatch) -> None:
    backend = CapturingOllamaClient()
    backend.chat = lambda **_kwargs: {"message": {"content": "\n"}}
    monkeypatch.setattr(
        prompt_client_module.ollama,
        "Client",
        lambda **_kwargs: backend,
    )

    PromptClient().check_model_health("glm4:9b")


def test_prompt_client_uses_the_selected_provider_for_health_and_inference(
    monkeypatch,
) -> None:
    class Provider:
        def __init__(self, provider_id: str) -> None:
            self.provider_id = provider_id
            self.health_models: list[str] = []
            self.chat_models: list[str] = []

        def health(self, *, model: str) -> dict[str, object]:
            self.health_models.append(model)
            return {"message": {"content": "pong"}}

        def chat(self, *, model: str, messages, stream: bool):
            self.chat_models.append(model)
            return iter([{"message": {"content": "respuesta local"}}])

        def capabilities(self) -> frozenset[str]:
            return frozenset({"chat", "health"})

    gemini = Provider("gemini")
    ollama = Provider("ollama")
    monkeypatch.setattr(
        prompt_client_module,
        "default_provider_registry",
        lambda **_kwargs: ChatInferenceProviderRegistry(
            {"gemini": gemini, "ollama": ollama}
        ),
    )
    monkeypatch.setattr(prompt_client_module, "configured_provider_id", lambda: "gemini")

    client = PromptClient()
    client.check_model_health("glm4:9b", provider_id="ollama")
    response = client.ask_messages(
        model="glm4:9b",
        messages=[{"role": "user", "content": "hola"}],
        provider_id="ollama",
    )

    assert response == "respuesta local"
    assert ollama.health_models == ["glm4:9b"]
    assert ollama.chat_models == ["glm4:9b"]
    assert gemini.health_models == []
    assert gemini.chat_models == []


def test_real_prompt_client_health_checker_rejects_unavailable_model(monkeypatch) -> None:
    class MissingModelOllamaClient:
        def chat(self, **_kwargs):
            raise ollama.ResponseError("model not found", 404)

    monkeypatch.setattr(
        prompt_client_module.ollama,
        "Client",
        lambda **_kwargs: MissingModelOllamaClient(),
    )

    result = OllamaModelHealthChecker(PromptClient()).check(
        logical_model_id="missing",
        physical_model_name="missing:latest",
        provider_id="ollama",
    )

    assert result.healthy is False
    assert result.error_code is ModelHealthErrorCode.MODEL_UNAVAILABLE


class StaticModelSource:
    def list_models(self) -> list[str]:
        return ["reasoning-primary:latest"]


class StructuredPromptClient:
    def __init__(self) -> None:
        self.models: list[str] = []

    def ask_messages(self, model: str, messages: list[dict[str, str]]) -> str:
        self.models.append(model)
        return json.dumps({"steps": []})


class RecordingHealthChecker:
    def __init__(self) -> None:
        self.models: list[str] = []

    def check(self, *, logical_model_id: str, physical_model_name: str, provider_id: str) -> ModelHealthResult:
        self.models.append(physical_model_name)
        return ModelHealthResult(
            logical_model_id=logical_model_id,
            physical_model_name=physical_model_name,
            provider_id=provider_id,
            healthy=True,
        )


def test_structured_planning_uses_injected_health_checker() -> None:
    manager = ModelManager(
        StaticModelSource(),
        (
            ModelDescriptor(
                logical_id="reasoning-primary",
                provider_id="ollama",
                model_name="reasoning-primary:latest",
                capabilities=("reasoning",),
            ),
        ),
    )
    prompt_client = StructuredPromptClient()
    health_checker = RecordingHealthChecker()
    provider = PromptClientStructuredPlanProvider(
        prompt_client,
        model_name="reasoning-primary",
        model_manager=manager,
        health_checker=health_checker,
    )

    result = provider.generate_plan("lee el proyecto", json.dumps({"tools": []}))

    assert result.success is True
    assert health_checker.models == ["reasoning-primary:latest"]
    assert prompt_client.models == ["reasoning-primary:latest"]

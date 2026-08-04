from __future__ import annotations

import json

from core.hybrid_execution_planner import PromptClientStructuredPlanProvider
from core.model_health import ModelHealthResult
from core.model_manager import ModelDescriptor, ModelManager
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

from __future__ import annotations

import json

from core.hybrid_execution_planner import PromptClientStructuredPlanProvider
from core.model_manager import ModelDescriptor, ModelManager
from models.prompt_client import InferenceBackendError


class StaticModelSource:
    def list_models(self) -> list[str]:
        return ["reasoning-primary:latest", "reasoning-fallback:latest"]


class FailingThenSuccessfulPromptClient:
    def __init__(self) -> None:
        self.models: list[str] = []

    def ask_messages(self, model: str, messages: list[dict[str, str]]) -> str:
        self.models.append(model)
        if model == "reasoning-primary:latest":
            raise InferenceBackendError(model, "simulated backend failure")
        return json.dumps({"steps": []})


def test_structured_planning_returns_authorized_fallback_response() -> None:
    manager = ModelManager(
        StaticModelSource(),
        (
            ModelDescriptor(
                logical_id="reasoning-primary",
                provider_id="ollama",
                model_name="reasoning-primary:latest",
                capabilities=("reasoning",),
                fallback_logical_ids=("reasoning-fallback",),
            ),
            ModelDescriptor(
                logical_id="reasoning-fallback",
                provider_id="ollama",
                model_name="reasoning-fallback:latest",
                capabilities=("reasoning",),
            ),
        ),
    )
    prompt_client = FailingThenSuccessfulPromptClient()
    provider = PromptClientStructuredPlanProvider(
        prompt_client,
        model_name="reasoning-primary",
        model_manager=manager,
    )

    result = provider.generate_plan("lee el proyecto", json.dumps({"tools": []}))

    assert result.success is True
    assert result.model_name == "reasoning-fallback:latest"
    assert prompt_client.models == [
        "reasoning-primary:latest",
        "reasoning-fallback:latest",
    ]

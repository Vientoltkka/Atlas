"""Focal tests: specialist agents must keep the selected provider_id."""

from __future__ import annotations

from agents.base_agent import AgentResponse
from agents.medical_agent import MedicalAgent
from agents.nutrition_agent import NutritionAgent
from core.model_manager import ModelDescriptor, ModelManager
from core.orchestrator import AtlasOrchestrator
from core.planner import Plan
from core.router import Router
from models.chat_inference import ChatInferenceError
from models.prompt_client import InferenceBackendError

_ABSENT = object()


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


class ProviderSpyPromptClient:
    def __init__(self, response: str = "respuesta") -> None:
        self._response = response
        self.calls: list[tuple[str, object]] = []

    def ask(self, *, model: str, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append((model, kwargs.get("provider_id", _ABSENT)))
        return self._response


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
    def __init__(self, name: str, agent) -> None:
        self._name = name
        self._agent = agent

    def get(self, name: str):
        return self._agent if name == self._name else None


class RecordingSpecialistAgent:
    def __init__(self, failing_model: str | None = None) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._failing_model = failing_model

    def run(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        provider_id: str | None = None,
    ) -> str:
        self.calls.append((model, provider_id))
        if model == self._failing_model:
            raise InferenceBackendError(model, "fallo simulado del backend") from ChatInferenceError(
                "gemini",
                model,
                "fallo simulado del backend",
            )
        return "ok"


class FixedPlanner:
    def create_plan(self, prompt: str) -> Plan:
        return Plan(task="chat", objective=prompt)


class WriteFile:
    def execute(self, path: str, content: str) -> str:
        return "ok"


def _orchestrator(agent) -> AtlasOrchestrator:
    return AtlasOrchestrator(
        planner=FixedPlanner(),
        router=Router(),
        model_manager=_manager_with_provider_chain(),
        memory=Memory(),
        registry=Registry("chat", agent),
        write_file=WriteFile(),
    )


def _manager_with_provider_chain() -> ModelManager:
    return ModelManager(
        StaticModelSource(["local-chat:latest"]),
        (
            ModelDescriptor(
                logical_id="remote-chat",
                provider_id="gemini",
                model_name="remote-chat-model",
                capabilities=("chat",),
                local=False,
                priority=1000,
                fallback_logical_ids=("local-chat",),
            ),
            ModelDescriptor(
                logical_id="local-chat",
                provider_id="ollama",
                model_name="local-chat:latest",
                capabilities=("chat",),
                priority=999,
            ),
        ),
    )


def test_nutrition_agent_forwards_selected_provider_to_prompt_client() -> None:
    client = ProviderSpyPromptClient()
    agent = NutritionAgent(client)  # type: ignore[arg-type]

    response = agent.run(
        "local-chat:latest",
        [{"role": "user", "content": "plan nutricional semanal"}],
        provider_id="ollama",
    )

    assert response == "respuesta"
    assert client.calls == [("local-chat:latest", "ollama")]


def test_nutrition_agent_without_provider_keeps_default_routing() -> None:
    client = ProviderSpyPromptClient()
    agent = NutritionAgent(client)  # type: ignore[arg-type]

    agent.run("local-chat:latest", [{"role": "user", "content": "plan nutricional semanal"}])

    assert client.calls == [("local-chat:latest", _ABSENT)]


def test_medical_agent_without_provider_keeps_default_routing() -> None:
    client = ProviderSpyPromptClient()
    agent = MedicalAgent(client)  # type: ignore[arg-type]

    agent.run("local-chat:latest", [{"role": "user", "content": "orientacion general"}])

    assert client.calls == [("local-chat:latest", _ABSENT)]


def test_specialist_primary_attempt_uses_selected_provider() -> None:
    agent = RecordingSpecialistAgent()

    response = _orchestrator(agent).process_prompt(
        "hola",
        confirm=lambda _prompt: "",
    )

    assert response == "ok"
    assert agent.calls == [("remote-chat-model", "gemini")]


def test_specialist_fallback_attempt_uses_fallback_provider() -> None:
    agent = RecordingSpecialistAgent(failing_model="remote-chat-model")

    response = _orchestrator(agent).process_prompt(
        "hola",
        confirm=lambda _prompt: "",
    )

    assert response == "ok"
    assert agent.calls == [
        ("remote-chat-model", "gemini"),
        ("local-chat:latest", "ollama"),
    ]


def test_nutrition_local_calculation_fallback_does_not_require_provider() -> None:
    agent = NutritionAgent(ProviderSpyPromptClient())  # type: ignore[arg-type]
    messages = [
        {
            "role": "user",
            "content": (
                "calcula mis calorias para ganar masa: 75 kg, 1.75 m, 30 anos, "
                "hombre, entreno crossfit 5 dias"
            ),
        }
    ]

    fallback = agent.local_calculation_fallback(messages)

    assert isinstance(fallback, AgentResponse)
    assert fallback.requires_follow_up is False
    assert "Calorías" in fallback.text

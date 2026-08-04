from __future__ import annotations

import pytest

from agents.chat_agent import ChatAgent
from core.model_manager import (
    ModelDescriptor,
    ModelManager,
    ModelSelectionRequest,
    ModelSelectionResult,
)
from models.prompt_client import InferenceBackendError
from core.orchestrator import AtlasOrchestrator
from core.planner import Plan
from core.router import Router


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


class RecordingModelManager(ModelManager):
    def __init__(self, models: list[str]) -> None:
        super().__init__(StaticModelSource(models))
        self.selection_requests: list[ModelSelectionRequest] = []
        self.selection_results: list[ModelSelectionResult] = []

    def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResult:
        self.selection_requests.append(request)
        result = super().select_model(request)
        self.selection_results.append(result)
        return result


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
    def __init__(self, name: str, agent) -> None:
        self._name = name
        self._agent = agent

    def get(self, name: str):
        return self._agent if name == self._name else None


class RecordingAgent:
    def __init__(self) -> None:
        self.models: list[str] = []

    def run(self, model: str, messages: list[dict[str, str]]) -> str:
        self.models.append(model)
        return "ok"


class RecordingPromptClient:
    def __init__(self) -> None:
        self.models: list[str] = []

    def ask(self, model: str, messages: list[dict[str, str]]) -> str:
        self.models.append(model)
        return "respuesta"


class WriteFile:
    def execute(self, path: str, content: str) -> str:
        return "ok"


def _orchestrator(
    task: str,
    manager: ModelManager,
    agent,
) -> AtlasOrchestrator:
    return AtlasOrchestrator(
        planner=FixedPlanner(task),
        router=Router(),
        model_manager=manager,
        memory=Memory(),
        registry=Registry(task, agent),
        write_file=WriteFile(),
    )


@pytest.mark.parametrize(
    ("task", "logical_id", "physical_name"),
    (
        ("chat", "chat-local", "glm4:9b"),
        ("coding", "coding-local", "qwen3.6:latest"),
        ("project", "project-local", "glm-5.2-local:latest"),
    ),
)
def test_real_routes_use_selector_result_once_before_agent_execution(
    task: str,
    logical_id: str,
    physical_name: str,
) -> None:
    manager = RecordingModelManager(
        ["glm4:9b", "qwen3.6:latest", "glm-5.2-local:latest"]
    )
    agent = RecordingAgent()

    response = _orchestrator(task, manager, agent).process_prompt(
        f"solicitud {task}",
        confirm=lambda _prompt: "",
    )

    assert response == "ok"
    assert manager.selection_requests == [
        ModelSelectionRequest(task=task, allow_fallback=True)
    ]
    assert manager.selection_results[0].logical_model_id == logical_id
    assert manager.selection_results[0].provider_id == "ollama"
    assert manager.selection_results[0].is_fallback is False
    assert agent.models == [physical_name]


def test_selected_physical_name_reaches_real_chat_agent_prompt_client() -> None:
    manager = RecordingModelManager(["glm4:9b"])
    prompt_client = RecordingPromptClient()
    agent = ChatAgent(prompt_client)

    response = _orchestrator("chat", manager, agent).process_prompt(
        "hola",
        confirm=lambda _prompt: "",
    )

    assert response == "respuesta"
    assert prompt_client.models == ["glm4:9b"]
    assert len(manager.selection_requests) == 1


def test_no_compatible_model_uses_existing_choose_model_degradation() -> None:
    manager = RecordingModelManager(["legacy-only:latest"])
    agent = RecordingAgent()

    response = _orchestrator("chat", manager, agent).process_prompt(
        "hola",
        confirm=lambda _prompt: "",
    )

    assert response == "ok"
    assert manager.selection_results[0].success is False
    assert manager.selection_results[0].error_code == "NO_COMPATIBLE_MODEL"
    assert agent.models == ["legacy-only:latest"]


def test_same_real_route_and_catalog_remain_deterministic_without_double_selection() -> None:
    manager = RecordingModelManager(["glm4:9b"])
    agent = RecordingAgent()
    orchestrator = _orchestrator("chat", manager, agent)

    orchestrator.process_prompt("uno", confirm=lambda _prompt: "")
    orchestrator.process_prompt("dos", confirm=lambda _prompt: "")

    assert agent.models == ["glm4:9b", "glm4:9b"]
    assert len(manager.selection_requests) == 2
    assert [result.logical_model_id for result in manager.selection_results] == [
        "chat-local",
        "chat-local",
    ]


def test_inference_failure_reaches_authorized_fallback_response() -> None:
    manager = ModelManager(
        StaticModelSource(["primary:latest", "fallback:latest"]),
        (
            ModelDescriptor(
                logical_id="runtime-primary",
                provider_id="ollama",
                model_name="primary:latest",
                capabilities=("chat",),
                priority=1000,
                fallback_logical_ids=("runtime-fallback",),
            ),
            ModelDescriptor(
                logical_id="runtime-fallback",
                provider_id="ollama",
                model_name="fallback:latest",
                capabilities=("chat",),
                priority=999,
            ),
        ),
    )

    class FailingThenSuccessfulAgent:
        def __init__(self) -> None:
            self.models: list[str] = []

        def run(self, model: str, messages: list[dict[str, str]]) -> str:
            self.models.append(model)
            if model == "primary:latest":
                raise InferenceBackendError(model, "simulated backend failure")
            return "respuesta fallback"

    agent = FailingThenSuccessfulAgent()

    response = _orchestrator("chat", manager, agent).process_prompt(
        "hola",
        confirm=lambda _prompt: "",
    )

    assert response == "respuesta fallback"
    assert agent.models == ["primary:latest", "fallback:latest"]

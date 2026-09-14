"""Focal tests for the Model/Provider V2 pairing contract.

Covers: model_id + provider_id conservation, cross-provider fallback,
per-attempt provider observability, capability requirements through the
existing policy, unhealthy-provider skips, structural errors, and the
agent/bootstrap paths that must not lose the provider_id.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agents.base_agent import AgentResponse, BaseAgent
from agents.registry import AgentRegistry
from agents.training_agent import TrainingAgent
from bootstrap.bootstrap import _build_async_goal_model_transformer
from core.model_inference import (
    InferenceFallbackExhaustedError,
    ModelHealthCheckError,
    ModelInferenceRunner,
)
from core.model_health import ModelHealthResult
from core.model_manager import ModelDescriptor, ModelManager, ModelSelectionRequest
from core.model_selection_policy import ModelSelectionPolicy
from core.operational_request_router import RequestRoute, RouteDecision
from core.operational_route_executor import (
    AgentDelegationRouteHandler,
    RouteExecutionStatus,
)
from core.request_gateway import RequestGateway
from models.chat_inference import ChatInferenceError
from models.prompt_client import InferenceBackendError
from use_cases.correction_interaction import CorrectionInteractionUseCase

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
ABSENT = object()


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


def _descriptor(
    logical_id: str,
    provider_id: str,
    model_name: str,
    *,
    capability: str = "chat",
    local: bool = True,
    priority: int = 100,
    fallbacks: tuple[str, ...] = (),
) -> ModelDescriptor:
    return ModelDescriptor(
        logical_id=logical_id,
        provider_id=provider_id,
        model_name=model_name,
        capabilities=(capability,),
        local=local,
        priority=priority,
        fallback_logical_ids=fallbacks,
    )


class _StaticHealthChecker:
    """Health checker failing only the configured providers."""

    def __init__(self, unhealthy_providers: set[str]) -> None:
        self._unhealthy = set(unhealthy_providers)

    def check(
        self,
        *,
        logical_model_id: str,
        physical_model_name: str,
        provider_id: str,
    ) -> ModelHealthResult:
        healthy = provider_id not in self._unhealthy
        return ModelHealthResult(
            logical_model_id=logical_model_id,
            physical_model_name=physical_model_name,
            provider_id=provider_id,
            healthy=healthy,
            reason=None if healthy else "provider saturado en prueba",
        )


def _cross_provider_manager() -> ModelManager:
    return ModelManager(
        StaticModelSource(["remote-chat-model", "local-chat:latest"]),
        (
            _descriptor(
                "remote-chat",
                "gemini",
                "remote-chat-model",
                local=False,
                priority=1000,
                fallbacks=("local-chat",),
            ),
            _descriptor("local-chat", "ollama", "local-chat:latest", priority=999),
        ),
    )


def test_selection_result_keeps_model_and_provider_pair() -> None:
    selection = _cross_provider_manager().select_model(
        ModelSelectionRequest(task="chat", preferred_model_id="remote-chat")
    )

    assert selection.success
    assert selection.logical_model_id == "remote-chat"
    assert selection.physical_model_name == "remote-chat-model"
    assert selection.provider_id == "gemini"


def test_selection_by_capabilities_requires_the_declared_capability() -> None:
    manager = ModelManager(
        StaticModelSource(["plain:latest", "tooler:latest"]),
        (
            ModelDescriptor(
                logical_id="plain",
                provider_id="ollama",
                model_name="plain:latest",
                capabilities=("chat",),
                local=True,
                priority=1000,
            ),
            ModelDescriptor(
                logical_id="tooler",
                provider_id="ollama",
                model_name="tooler:latest",
                capabilities=("chat", "tool_calling"),
                local=True,
                priority=1,
            ),
        ),
    )

    policy = ModelSelectionPolicy(required_capabilities=("tool_calling",))
    selection = manager.select_model(policy.create_request(task="chat"))

    assert selection.success
    assert (selection.logical_model_id, selection.provider_id) == ("tooler", "ollama")


def test_selection_without_required_capability_fails_structurally() -> None:
    manager = ModelManager(
        StaticModelSource(["plain:latest"]),
        (_descriptor("plain", "ollama", "plain:latest"),),
    )

    policy = ModelSelectionPolicy(required_capabilities=("tool_calling",))
    selection = manager.select_model(policy.create_request(task="chat"))

    assert not selection.success
    assert selection.logical_model_id is None
    assert selection.provider_id is None


def test_runner_records_provider_pair_for_every_attempt() -> None:
    runner = ModelInferenceRunner(_cross_provider_manager())
    seen: list[tuple[str, str | None]] = []

    def infer(model: str, provider_id: str | None) -> str:
        seen.append((model, provider_id))
        if model == "remote-chat-model":
            raise InferenceBackendError(model, "quota exhausted") from ChatInferenceError(
                "gemini", model, "quota exhausted"
            )
        return "ok"

    result = runner.run(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="remote-chat",
            allow_fallback=True,
        ),
        infer,
    )

    assert result == "ok"
    assert seen == [
        ("remote-chat-model", "gemini"),
        ("local-chat:latest", "ollama"),
    ]
    assert runner.last_result is not None
    assert runner.last_result.initial_logical_model_id == "remote-chat"
    assert runner.last_result.initial_provider_id == "gemini"
    assert runner.last_result.final_logical_model_id == "local-chat"
    assert runner.last_result.final_provider_id == "ollama"
    assert runner.last_result.attempted_provider_ids == ("gemini", "ollama")


def test_exhausted_fallback_records_attempted_providers() -> None:
    manager = ModelManager(
        StaticModelSource(["only:latest"]),
        (_descriptor("only", "gemini", "only:latest", local=False),),
    )
    runner = ModelInferenceRunner(manager)

    def failing_infer(model: str, _provider: str | None) -> str:
        raise InferenceBackendError(model, "failed") from ChatInferenceError(
            "gemini", model, "failed"
        )

    with pytest.raises(InferenceFallbackExhaustedError) as captured:
        runner.run(
            ModelSelectionRequest(
                task="chat",
                preferred_model_id="only",
                allow_fallback=True,
            ),
            failing_infer,
        )

    assert captured.value.attempted_logical_model_ids == ("only",)
    assert captured.value.attempted_provider_ids == ("gemini",)


def test_health_error_records_attempted_providers() -> None:
    manager = ModelManager(
        StaticModelSource(["solo:latest"]),
        (_descriptor("solo", "gemini", "solo:latest", local=False),),
    )
    runner = ModelInferenceRunner(
        manager,
        health_checker=_StaticHealthChecker({"gemini"}),
    )

    with pytest.raises(ModelHealthCheckError) as captured:
        runner.run(
            ModelSelectionRequest(
                task="chat",
                preferred_model_id="solo",
                allow_fallback=False,
            ),
            lambda model, _provider: "no debe ejecutarse",
        )

    assert captured.value.attempted_logical_model_ids == ("solo",)
    assert captured.value.attempted_provider_ids == ("gemini",)


def test_unhealthy_provider_is_skipped_and_fallback_keeps_its_provider() -> None:
    runner = ModelInferenceRunner(
        _cross_provider_manager(),
        health_checker=_StaticHealthChecker({"gemini"}),
    )
    seen: list[tuple[str, str | None]] = []

    def infer(model: str, provider_id: str | None) -> str:
        seen.append((model, provider_id))
        return "ok"

    result = runner.run(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="remote-chat",
            allow_fallback=True,
        ),
        infer,
    )

    assert result == "ok"
    assert seen == [("local-chat:latest", "ollama")]
    assert runner.last_result is not None
    assert runner.last_result.used_fallback is True
    assert runner.last_result.final_provider_id == "ollama"
    assert runner.last_result.fallback_reason == "provider saturado en prueba"


def test_structural_gemini_error_is_never_hidden_by_fallback() -> None:
    runner = ModelInferenceRunner(_cross_provider_manager())
    attempts: list[str] = []

    with pytest.raises(InferenceBackendError, match="configuración inválida"):
        runner.run(
            ModelSelectionRequest(
                task="chat",
                preferred_model_id="remote-chat",
                allow_fallback=True,
            ),
            lambda model, _provider: attempts.append(model)
            or (_ for _ in ()).throw(
                InferenceBackendError(model, "configuración inválida")
            ),
        )

    assert attempts == ["remote-chat-model"]


class _RecordingAgent(BaseAgent):
    def __init__(self, name: str, answer: str) -> None:
        self._name = name
        self.answer = answer
        self.calls: list[tuple[str, str | None]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"recording agent {self._name}"

    def run(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        provider_id: str | None = None,
    ) -> str | AgentResponse:
        self.calls.append((model, provider_id))
        return self.answer


def test_agent_delegation_route_propagates_provider_id() -> None:
    agent = _RecordingAgent("coding", "respuesta")
    registry = AgentRegistry()
    registry.register(agent)
    handler = AgentDelegationRouteHandler(
        registry,
        model_selector=lambda agent_name: "remote-chat-model",
        provider_resolver=lambda model_name: "gemini",
    )
    request = RequestGateway(
        clock=lambda: NOW,
        id_generator=lambda: "request-1",
    ).from_text("revisa el codigo")

    result = handler.execute(
        request,
        _decision(target_agent_name="coding"),
    )

    assert result.status is RouteExecutionStatus.COMPLETED
    assert agent.calls == [("remote-chat-model", "gemini")]


def _decision(**kwargs) -> RouteDecision:
    return RouteDecision(
        request_id="request-1",
        route=RequestRoute.AGENT_DELEGATION,
        confidence=1.0,
        reason="test",
        matched_rules=("test.agent_delegation",),
        created_at=NOW,
        **kwargs,
    )


def test_async_goal_transformer_propagates_provider_id() -> None:
    agent = _RecordingAgent("chat", "transformado")
    model_manager = SimpleNamespace(
        choose_model=lambda _task: "remote-chat-model",
        resolve_model=lambda _name: SimpleNamespace(provider_id="gemini"),
    )

    transform = _build_async_goal_model_transformer(
        SimpleNamespace(get=lambda _name: agent),
        model_manager,
    )

    assert transform("resume esto") == "transformado"
    assert agent.calls == [("remote-chat-model", "gemini")]


class _ProviderSpyClient:
    def __init__(self, fail_models: set[str]) -> None:
        self.calls: list[dict[str, object]] = []
        self._fail_models = fail_models

    def ask(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        provider_id: str | None = None,
    ) -> str:
        self.calls.append({"model": model, "provider_id": provider_id})
        if model in self._fail_models:
            raise InferenceBackendError(model, "fallo simulado") from ChatInferenceError(
                "gemini", model, "fallo simulado"
            )
        return "plan"


class _NoModels:
    def list_models(self) -> list[str]:
        return []


def _training_manager() -> ModelManager:
    return ModelManager(
        StaticModelSource(["glm4:9b"]),
        (
            _descriptor(
                "chat-gemini",
                "gemini",
                "gemini-3.6-flash",
                local=False,
                priority=200,
                fallbacks=("chat-local",),
            ),
        ),
    )


def test_training_agent_falls_back_through_declared_chain_with_provider() -> None:
    client = _ProviderSpyClient({"gemini-3.6-flash"})
    agent = TrainingAgent(client, _training_manager())  # type: ignore[arg-type]

    response = agent.run(
        "glm4:9b",
        [{"role": "user", "content": "Sesión de CrossFit de 60 minutos."}],
    )

    assert response == "plan"
    assert [(call["model"], call["provider_id"]) for call in client.calls] == [
        ("gemini-3.6-flash", "gemini"),
        ("glm4:9b", "ollama"),
    ]


def test_training_agent_without_primary_uses_registry_model_with_provider() -> None:
    client = _ProviderSpyClient(set())
    agent = TrainingAgent(
        client,
        ModelManager(
            StaticModelSource(["gpt-chat"]),
            (
                _descriptor(
                    "chat-openai",
                    "openai",
                    "gpt-chat-model",
                    local=False,
                    priority=150,
                ),
            ),
        ),
    )  # type: ignore[arg-type]

    response = agent.run(
        "glm4:9b",
        [{"role": "user", "content": "Sesión de CrossFit de 60 minutos."}],
    )

    assert response == "plan"
    assert [(call["model"], call["provider_id"]) for call in client.calls] == [
        ("gpt-chat-model", "openai"),
    ]


def test_training_agent_without_any_registry_model_uses_legacy_fallback() -> None:
    client = _ProviderSpyClient(set())
    agent = TrainingAgent(client, ModelManager(client=_NoModels()))  # type: ignore[arg-type]

    response = agent.run(
        "glm4:9b",
        [{"role": "user", "content": "Sesión de CrossFit de 60 minutos."}],
    )

    assert response == "plan"
    assert [(call["model"], call["provider_id"]) for call in client.calls] == [
        ("qwen3.6:latest", "ollama"),
    ]


def test_training_agent_uses_hardcoded_fallback_when_chain_exhausted() -> None:
    client = _ProviderSpyClient({"gemini-3.6-flash"})
    agent = TrainingAgent(
        client,
        ModelManager(
            _NoModels(),
            (
                _descriptor(
                    "chat-gemini",
                    "gemini",
                    "gemini-3.6-flash",
                    local=False,
                    priority=200,
                ),
            ),
        ),
    )  # type: ignore[arg-type]

    response = agent.run(
        "glm4:9b",
        [{"role": "user", "content": "Sesión de CrossFit de 60 minutos."}],
    )

    assert response == "plan"
    assert [(call["model"], call["provider_id"]) for call in client.calls] == [
        ("gemini-3.6-flash", "gemini"),
        ("qwen3.6:latest", "ollama"),
    ]


class _ProviderAwareAskClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def ask(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        provider_id: str | None = None,
    ) -> str:
        self.calls.append({"model": model, "provider_id": provider_id})
        return (
            "PROBLEM:\nx\n\nRISK:\nlow\n\nPROPOSED_FILE:\n```python\nclass Router:\n"
            "    pass\n```"
        )


class _ReadFileFake:
    def execute(self, path: str) -> str:
        return "class Router:\n    pass\n"


class _WriteFileFake:
    def execute(self, path: str, content: str, **_kwargs) -> str:
        return "ok"


class _QueryArchitectureFake:
    def impact_of(self, _path: str) -> SimpleNamespace:
        return SimpleNamespace(affected_files=[])

    def dependencies_of(self, _path: str) -> SimpleNamespace:
        return SimpleNamespace(dependencies=[])


def test_correction_interaction_forwards_provider_id(tmp_path) -> None:
    (tmp_path / "router.py").write_text("class Router:\n    pass\n", encoding="utf-8")
    client = _ProviderAwareAskClient()
    use_case = CorrectionInteractionUseCase(
        _ReadFileFake(),  # type: ignore[arg-type]
        _WriteFileFake(),
        _QueryArchitectureFake(),  # type: ignore[arg-type]
        client,  # type: ignore[arg-type]
    )

    use_case.execute(
        "corrige router.py",
        tmp_path,
        choose_model=lambda _task: "qwen3.6:latest",
        confirm=lambda _prompt: "n",
        resolve_provider=lambda _model: "ollama",
    )

    assert client.calls[0]["provider_id"] == "ollama"


def test_correction_interaction_without_resolver_keeps_legacy_call(tmp_path) -> None:
    (tmp_path / "router.py").write_text("class Router:\n    pass\n", encoding="utf-8")
    client = _ProviderAwareAskClient()
    use_case = CorrectionInteractionUseCase(
        _ReadFileFake(),  # type: ignore[arg-type]
        _WriteFileFake(),
        _QueryArchitectureFake(),  # type: ignore[arg-type]
        client,  # type: ignore[arg-type]
    )

    use_case.execute(
        "corrige router.py",
        tmp_path,
        choose_model=lambda _task: "qwen3.6:latest",
        confirm=lambda _prompt: "n",
    )

    assert client.calls[0]["provider_id"] is None

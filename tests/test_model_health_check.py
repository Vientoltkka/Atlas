from __future__ import annotations

import httpx
import ollama
import pytest

from core.model_health import (
    ModelHealthErrorCode,
    ModelHealthResult,
    OllamaModelHealthChecker,
)
from core.model_inference import ModelHealthCheckError, ModelInferenceRunner
from core.model_manager import ModelDescriptor, ModelManager, ModelSelectionRequest
from models.prompt_client import InferenceBackendError


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


class RecordingHealthChecker:
    def __init__(self, outcomes: dict[str, ModelHealthResult]) -> None:
        self._outcomes = outcomes
        self.checked: list[str] = []

    def check(self, *, logical_model_id: str, physical_model_name: str, provider_id: str) -> ModelHealthResult:
        self.checked.append(logical_model_id)
        return self._outcomes.get(
            logical_model_id,
            ModelHealthResult(
                logical_model_id=logical_model_id,
                physical_model_name=physical_model_name,
                provider_id=provider_id,
                healthy=True,
            ),
        )


def _descriptor(logical_id: str, *, fallbacks: tuple[str, ...] = ()) -> ModelDescriptor:
    return ModelDescriptor(
        logical_id=logical_id,
        provider_id="ollama",
        model_name=f"{logical_id}:latest",
        capabilities=("chat",),
        fallback_logical_ids=fallbacks,
    )


def _manager(*descriptors: ModelDescriptor) -> ModelManager:
    return ModelManager(
        StaticModelSource([item.model_name for item in descriptors]),
        descriptors,
    )


def _health(logical_id: str, healthy: bool, error_code: ModelHealthErrorCode | None = None) -> ModelHealthResult:
    return ModelHealthResult(
        logical_model_id=logical_id,
        physical_model_name=f"{logical_id}:latest",
        provider_id="ollama",
        healthy=healthy,
        error_code=error_code,
        reason=None if healthy else "health failed",
    )


def _request(*, allow_fallback: bool = True) -> ModelSelectionRequest:
    return ModelSelectionRequest(
        task="chat",
        preferred_model_id="primary",
        allow_fallback=allow_fallback,
    )


def _gemini_chat_manager() -> ModelManager:
    return ModelManager(
        StaticModelSource(["glm4:9b"]),
        (
            ModelDescriptor(
                logical_id="chat-gemini",
                provider_id="gemini",
                model_name="gemini-3.6-flash",
                capabilities=("chat",),
                local=False,
                priority=200,
                fallback_logical_ids=("chat-local",),
            ),
        ),
    )


def _gemini_chat_request() -> ModelSelectionRequest:
    return ModelSelectionRequest(
        task="chat",
        preferred_model_id="chat-gemini",
        allow_fallback=True,
    )


def test_healthy_inventory_model_is_used() -> None:
    manager = _manager(_descriptor("primary"))
    health = RecordingHealthChecker({"primary": _health("primary", True)})
    inferred: list[str] = []

    result = ModelInferenceRunner(manager, health_checker=health).run(
        _request(), lambda model: inferred.append(model) or "ok"
    )

    assert result == "ok"
    assert health.checked == ["primary"]
    assert inferred == ["primary:latest"]


def test_unhealthy_gemini_uses_local_chat_fallback() -> None:
    health = RecordingHealthChecker(
        {
            "chat-gemini": _health(
                "chat-gemini",
                False,
                ModelHealthErrorCode.UNKNOWN_BACKEND_ERROR,
            ),
            "chat-local": _health("chat-local", True),
        }
    )
    inferred: list[str] = []

    result = ModelInferenceRunner(_gemini_chat_manager(), health_checker=health).run(
        _gemini_chat_request(),
        lambda model: inferred.append(model) or "respuesta local",
    )

    assert result == "respuesta local"
    assert health.checked == ["chat-gemini", "chat-local"]
    assert inferred == ["glm4:9b"]


def test_healthy_gemini_does_not_use_local_chat_fallback() -> None:
    health = RecordingHealthChecker({"chat-gemini": _health("chat-gemini", True)})
    inferred: list[str] = []

    result = ModelInferenceRunner(_gemini_chat_manager(), health_checker=health).run(
        _gemini_chat_request(),
        lambda model: inferred.append(model) or "respuesta Gemini",
    )

    assert result == "respuesta Gemini"
    assert health.checked == ["chat-gemini"]
    assert inferred == ["gemini-3.6-flash"]


def test_unhealthy_gemini_and_local_chat_preserve_final_health_error() -> None:
    health = RecordingHealthChecker(
        {
            "chat-gemini": _health("chat-gemini", False),
            "chat-local": _health("chat-local", False),
        }
    )
    inferred: list[str] = []

    with pytest.raises(ModelHealthCheckError) as captured:
        ModelInferenceRunner(_gemini_chat_manager(), health_checker=health).run(
            _gemini_chat_request(),
            lambda model: inferred.append(model) or "unused",
        )

    assert health.checked == ["chat-gemini", "chat-local"]
    assert inferred == []
    assert captured.value.attempted_logical_model_ids == ("chat-gemini", "chat-local")
    assert captured.value.last_result.logical_model_id == "chat-local"


def test_unhealthy_inventory_model_is_not_used_without_fallback() -> None:
    manager = _manager(_descriptor("primary", fallbacks=("fallback",)), _descriptor("fallback"))
    health = RecordingHealthChecker({"primary": _health("primary", False, ModelHealthErrorCode.BACKEND_UNAVAILABLE)})
    inferred: list[str] = []

    with pytest.raises(ModelHealthCheckError) as captured:
        ModelInferenceRunner(manager, health_checker=health).run(
            _request(allow_fallback=False), lambda model: inferred.append(model)
        )

    assert inferred == []
    assert health.checked == ["primary"]
    assert captured.value.last_result.error_code is ModelHealthErrorCode.BACKEND_UNAVAILABLE


def test_unhealthy_primary_uses_next_declared_healthy_fallback() -> None:
    manager = _manager(_descriptor("primary", fallbacks=("fallback",)), _descriptor("fallback"))
    health = RecordingHealthChecker({"primary": _health("primary", False), "fallback": _health("fallback", True)})
    inferred: list[str] = []

    result = ModelInferenceRunner(manager, health_checker=health).run(
        _request(), lambda model: inferred.append(model) or "fallback response"
    )

    assert result == "fallback response"
    assert health.checked == ["primary", "fallback"]
    assert inferred == ["fallback:latest"]


def test_unhealthy_fallback_chain_advances_once_and_ignores_irrelevant_models() -> None:
    manager = _manager(
        _descriptor("primary", fallbacks=("fallback-1",)),
        _descriptor("fallback-1", fallbacks=("fallback-2", "primary")),
        _descriptor("fallback-2"),
        _descriptor("irrelevant"),
    )
    health = RecordingHealthChecker({
        "primary": _health("primary", False),
        "fallback-1": _health("fallback-1", False),
        "fallback-2": _health("fallback-2", True),
    })
    inferred: list[str] = []

    result = ModelInferenceRunner(manager, health_checker=health).run(
        _request(), lambda model: inferred.append(model) or "ok"
    )

    assert result == "ok"
    assert health.checked == ["primary", "fallback-1", "fallback-2"]
    assert inferred == ["fallback-2:latest"]


def test_health_timeout_is_a_controlled_failure() -> None:
    manager = _manager(_descriptor("primary"))
    health = RecordingHealthChecker({"primary": _health("primary", False, ModelHealthErrorCode.TIMEOUT)})

    with pytest.raises(ModelHealthCheckError) as captured:
        ModelInferenceRunner(manager, health_checker=health).run(_request(), lambda _model: "unused")

    assert captured.value.last_result.error_code is ModelHealthErrorCode.TIMEOUT


def test_runtime_fallback_remains_active_after_healthy_probe() -> None:
    manager = _manager(_descriptor("primary", fallbacks=("fallback",)), _descriptor("fallback"))
    health = RecordingHealthChecker({"primary": _health("primary", True), "fallback": _health("fallback", True)})
    inferred: list[str] = []

    def infer(model: str) -> str:
        inferred.append(model)
        if model == "primary:latest":
            raise InferenceBackendError(model, "runtime failed")
        return "ok"

    assert ModelInferenceRunner(manager, health_checker=health).run(_request(), infer) == "ok"
    assert health.checked == ["primary", "fallback"]
    assert inferred == ["primary:latest", "fallback:latest"]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("slow"), ModelHealthErrorCode.TIMEOUT),
        (httpx.ReadTimeout("slow"), ModelHealthErrorCode.TIMEOUT),
        (ConnectionError("offline"), ModelHealthErrorCode.BACKEND_UNAVAILABLE),
        (ollama.ResponseError("model not found", 404), ModelHealthErrorCode.MODEL_UNAVAILABLE),
    ],
)
def test_ollama_health_checker_classifies_backend_failures(error: Exception, expected: ModelHealthErrorCode) -> None:
    class FailingPromptClient:
        def check_model_health(
            self,
            _model: str,
            *,
            provider_id: str | None = None,
        ) -> None:
            raise InferenceBackendError("primary:latest", str(error)) from error

    result = OllamaModelHealthChecker(FailingPromptClient()).check(
        logical_model_id="primary",
        physical_model_name="primary:latest",
        provider_id="ollama",
    )

    assert result.healthy is False
    assert result.error_code is expected

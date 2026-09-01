from __future__ import annotations

import pytest

from core.model_inference import (
    InferenceFallbackExhaustedError,
    InferenceStreamInterruptedError,
    ModelInferenceRunner,
)
from core.model_manager import ModelDescriptor, ModelManager, ModelSelectionRequest
from models.chat_inference import ChatInferenceError
from models.prompt_client import InferenceBackendError


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


def _descriptor(
    logical_id: str,
    model_name: str,
    capability: str,
    *,
    available: bool = True,
    fallbacks: tuple[str, ...] = (),
) -> ModelDescriptor:
    return ModelDescriptor(
        logical_id=logical_id,
        provider_id="ollama",
        model_name=model_name,
        capabilities=(capability,),
        available=available,
        fallback_logical_ids=fallbacks,
    )


def _manager(*descriptors: ModelDescriptor) -> ModelManager:
    installed = [item.model_name for item in descriptors]
    return ModelManager(StaticModelSource(installed), descriptors)


def test_primary_failure_uses_exactly_one_authorized_fallback() -> None:
    manager = _manager(
        _descriptor("primary", "primary:latest", "chat", fallbacks=("fallback-1",)),
        _descriptor("fallback-1", "fallback-1:latest", "chat"),
    )
    runner = ModelInferenceRunner(manager)
    attempts: list[str] = []

    def infer(model: str) -> str:
        attempts.append(model)
        if model == "primary:latest":
            raise InferenceBackendError(model, "backend failed")
        return "fallback response"

    response = runner.run(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="primary",
            allow_fallback=True,
        ),
        infer,
    )

    assert response == "fallback response"
    assert attempts == ["primary:latest", "fallback-1:latest"]
    assert runner.last_result is not None
    assert runner.last_result.initial_logical_model_id == "primary"
    assert runner.last_result.final_logical_model_id == "fallback-1"
    assert runner.last_result.used_fallback is True
    assert runner.last_result.attempt_count == 2


def test_fallback_is_never_attempted_when_not_authorized() -> None:
    manager = _manager(
        _descriptor("primary", "primary:latest", "chat", fallbacks=("fallback-1",)),
        _descriptor("fallback-1", "fallback-1:latest", "chat"),
    )
    runner = ModelInferenceRunner(manager)
    attempts: list[str] = []

    with pytest.raises(InferenceFallbackExhaustedError) as captured:
        runner.run(
            ModelSelectionRequest(
                task="chat",
                preferred_model_id="primary",
                allow_fallback=False,
            ),
            lambda model: attempts.append(model)
            or (_ for _ in ()).throw(InferenceBackendError(model, "failed")),
        )

    assert attempts == ["primary:latest"]
    assert captured.value.attempted_logical_model_ids == ("primary",)
    assert captured.value.allow_fallback is False


def test_declared_transitive_chain_is_attempted_once_in_order() -> None:
    manager = _manager(
        _descriptor("primary", "primary:latest", "chat", fallbacks=("fallback-1",)),
        _descriptor(
            "fallback-1",
            "fallback-1:latest",
            "chat",
            fallbacks=("fallback-2",),
        ),
        _descriptor("fallback-2", "fallback-2:latest", "chat"),
    )
    runner = ModelInferenceRunner(manager)
    attempts: list[str] = []

    def infer(model: str) -> str:
        attempts.append(model)
        if model != "fallback-2:latest":
            raise InferenceBackendError(model, "failed")
        return "ok"

    assert runner.run(
        ModelSelectionRequest(
            task="chat", preferred_model_id="primary", allow_fallback=True
        ),
        infer,
    ) == "ok"
    assert attempts == ["primary:latest", "fallback-1:latest", "fallback-2:latest"]


def test_exhausted_chain_raises_structured_error_without_global_model() -> None:
    manager = _manager(
        _descriptor("primary", "primary:latest", "chat", fallbacks=("fallback-1",)),
        _descriptor("fallback-1", "fallback-1:latest", "chat"),
        _descriptor("unrelated", "unrelated:latest", "chat"),
    )
    runner = ModelInferenceRunner(manager)
    attempts: list[str] = []

    with pytest.raises(InferenceFallbackExhaustedError) as captured:
        runner.run(
            ModelSelectionRequest(
                task="chat", preferred_model_id="primary", allow_fallback=True
            ),
            lambda model: attempts.append(model)
            or (_ for _ in ()).throw(InferenceBackendError(model, "failed")),
        )

    assert attempts == ["primary:latest", "fallback-1:latest"]
    assert captured.value.attempted_logical_model_ids == ("primary", "fallback-1")
    assert "2 attempt" in str(captured.value)


def test_incompatible_unknown_and_unavailable_fallbacks_are_not_executed() -> None:
    primary = _descriptor(
        "primary",
        "primary:latest",
        "reasoning",
        fallbacks=("chat-only", "missing", "unavailable", "valid"),
    )
    chat_only = _descriptor("chat-only", "chat-only:latest", "chat")
    unavailable = _descriptor(
        "unavailable", "unavailable:latest", "reasoning", available=False
    )
    valid = _descriptor("valid", "valid:latest", "reasoning")
    manager = _manager(primary, chat_only, unavailable, valid)
    runner = ModelInferenceRunner(manager)
    attempts: list[str] = []

    def infer(model: str) -> str:
        attempts.append(model)
        if model == "primary:latest":
            raise InferenceBackendError(model, "failed")
        return "ok"

    assert runner.run(
        ModelSelectionRequest(
            task="reasoning", preferred_model_id="primary", allow_fallback=True
        ),
        infer,
    ) == "ok"
    assert attempts == ["primary:latest", "valid:latest"]


def test_streaming_falls_back_only_before_any_fragment_is_observable() -> None:
    manager = _manager(
        _descriptor("primary", "primary:latest", "chat", fallbacks=("fallback-1",)),
        _descriptor("fallback-1", "fallback-1:latest", "chat"),
    )
    runner = ModelInferenceRunner(manager)
    attempts: list[str] = []

    def stream(model: str):
        attempts.append(model)
        if model == "primary:latest":
            raise InferenceBackendError(model, "failed before output")
        yield "fallback "
        yield "response"

    assert list(
        runner.stream(
            ModelSelectionRequest(
                task="chat", preferred_model_id="primary", allow_fallback=True
            ),
            stream,
        )
    ) == ["fallback ", "response"]
    assert attempts == ["primary:latest", "fallback-1:latest"]


def test_streaming_never_mixes_fallback_after_partial_output() -> None:
    manager = _manager(
        _descriptor("primary", "primary:latest", "chat", fallbacks=("fallback-1",)),
        _descriptor("fallback-1", "fallback-1:latest", "chat"),
    )
    runner = ModelInferenceRunner(manager)
    attempts: list[str] = []

    def stream(model: str):
        attempts.append(model)
        if model == "primary:latest":
            yield "partial"
            raise InferenceBackendError(model, "failed after output")
        yield "must not run"

    iterator = runner.stream(
        ModelSelectionRequest(
            task="chat", preferred_model_id="primary", allow_fallback=True
        ),
        stream,
    )
    assert next(iterator) == "partial"
    with pytest.raises(InferenceStreamInterruptedError):
        next(iterator)
    assert attempts == ["primary:latest"]


def test_programming_errors_are_not_classified_as_inference_failures() -> None:
    manager = _manager(_descriptor("primary", "primary:latest", "chat"))
    runner = ModelInferenceRunner(manager)

    with pytest.raises(ValueError, match="programming error"):
        runner.run(
            ModelSelectionRequest(task="chat", preferred_model_id="primary"),
            lambda _model: (_ for _ in ()).throw(ValueError("programming error")),
        )


def test_gemini_content_error_does_not_use_local_fallback() -> None:
    manager = ModelManager(
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
    attempts: list[str] = []

    with pytest.raises(InferenceBackendError, match="empty response"):
        ModelInferenceRunner(manager).run(
            ModelSelectionRequest(
                task="chat",
                preferred_model_id="chat-gemini",
                allow_fallback=True,
            ),
            lambda model: attempts.append(model)
            or (_ for _ in ()).throw(
                InferenceBackendError(model, "empty response")
            ),
        )

    assert attempts == ["gemini-3.6-flash"]


def test_gemini_provider_error_uses_local_fallback() -> None:
    manager = ModelManager(
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
    attempts: list[str] = []

    def infer(model: str) -> str:
        attempts.append(model)
        if model == "gemini-3.6-flash":
            error = ChatInferenceError("gemini", model, "quota exhausted")
            raise InferenceBackendError(model, error.reason) from error
        return "respuesta local"

    assert ModelInferenceRunner(manager).run(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="chat-gemini",
            allow_fallback=True,
        ),
        infer,
    ) == "respuesta local"
    assert attempts == ["gemini-3.6-flash", "glm4:9b"]


def test_same_physical_model_is_not_retried_through_an_alias() -> None:
    manager = _manager(
        ModelDescriptor(
            logical_id="primary",
            provider_id="ollama",
            model_name="shared:latest",
            capabilities=("chat",),
            priority=100,
            fallback_logical_ids=("alias", "valid"),
        ),
        ModelDescriptor(
            logical_id="alias",
            provider_id="ollama",
            model_name="shared:latest",
            capabilities=("chat",),
            priority=1,
        ),
        ModelDescriptor(
            logical_id="valid",
            provider_id="ollama",
            model_name="valid:latest",
            capabilities=("chat",),
        ),
    )
    runner = ModelInferenceRunner(manager)
    attempts: list[str] = []

    def infer(model: str) -> str:
        attempts.append(model)
        if model == "shared:latest":
            raise InferenceBackendError(model, "failed")
        return "ok"

    result = runner.run(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="primary",
            allow_fallback=True,
        ),
        infer,
    )

    assert result == "ok"
    assert attempts == ["shared:latest", "valid:latest"]

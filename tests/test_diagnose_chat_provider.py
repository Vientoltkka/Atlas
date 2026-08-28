from __future__ import annotations

import json

import pytest

from models.chat_inference import ChatInferenceError
from scripts import diagnose_chat_provider
from scripts.diagnose_chat_provider import diagnose_from_environment, diagnose_provider


class FakeProvider:
    def __init__(self, *, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.models: list[str] = []

    def health(self, *, model: str):
        self.models.append(model)
        if self.error is not None:
            raise self.error
        return self.response


class FakeRegistry:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider
        self.requested_provider_id = ""

    def get(self, provider_id: str) -> FakeProvider:
        self.requested_provider_id = provider_id
        return self.provider


@pytest.mark.parametrize(
    ("provider_id", "model"),
    [("gemini", "gemini-2.5-flash"), ("remote", "compatible-model")],
)
def test_diagnosis_reports_minimal_provider_health(provider_id: str, model: str) -> None:
    provider = FakeProvider(response={"message": {"content": "ok"}})
    ticks = iter((1.0, 1.025))

    result = diagnose_provider(
        provider,
        provider_id=provider_id,
        model=model,
        clock=lambda: next(ticks),
    )

    assert result == {
        "provider": provider_id,
        "model": model,
        "health": "healthy",
        "latency_ms": 25,
        "response": "health check completed",
        "error": None,
    }
    assert provider.models == [model]


def test_diagnosis_never_echoes_sdk_exception_secrets() -> None:
    secrets = ("api-key-value", "token-value", "bearer-value", "password-value", "credential-value")
    provider = FakeProvider(
        error=ChatInferenceError(
            "gemini",
            "gemini-test",
            "api_key=api-key-value token=token-value Authorization: Bearer bearer-value "
            "password=password-value credentials=credential-value",
        )
    )
    ticks = iter((1.0, 1.001))

    result = diagnose_provider(
        provider,
        provider_id="gemini",
        model="gemini-test",
        clock=lambda: next(ticks),
    )

    assert result["health"] == "unhealthy"
    assert result["error"] == {
        "code": "CHAT_INFERENCE_ERROR",
        "message": "Provider health check failed. Verify provider configuration and connectivity.",
    }
    assert all(secret not in json.dumps(result) for secret in secrets)


def test_diagnosis_never_echoes_configuration_exception_secrets() -> None:
    secrets = ("api-key-value", "bearer-value", "password-value")

    def failing_registry(**_kwargs):
        raise ValueError(
            "api_key=api-key-value Authorization: Bearer bearer-value password=password-value"
        )

    result = diagnose_from_environment(
        registry_builder=failing_registry,
        environment=lambda _name, default: default,
    )

    assert result["error"] == {
        "code": "PROVIDER_CONFIGURATION_ERROR",
        "message": "Provider health check failed. Verify provider configuration and connectivity.",
    }
    assert all(secret not in json.dumps(result) for secret in secrets)


def test_main_writes_sanitized_json_without_stderr(monkeypatch, capsys) -> None:
    secret = "sdk-bearer-secret"
    provider = FakeProvider(
        error=ChatInferenceError("remote", "compatible-model", f"Authorization: Bearer {secret}")
    )
    monkeypatch.setattr(
        diagnose_chat_provider,
        "diagnose_from_environment",
        lambda: diagnose_provider(provider, provider_id="remote", model="compatible-model"),
    )

    assert diagnose_chat_provider.main() == 1

    captured = capsys.readouterr()
    assert captured.err == ""
    assert secret not in captured.out
    assert json.loads(captured.out)["error"]["message"] == (
        "Provider health check failed. Verify provider configuration and connectivity."
    )


def test_diagnosis_uses_gemini_environment_without_passing_credentials(monkeypatch) -> None:
    provider = FakeProvider(response={"message": {"content": "gemini ok"}})
    registry = FakeRegistry(provider)
    calls: list[dict] = []

    def build_registry(**kwargs):
        calls.append(kwargs)
        return registry

    monkeypatch.setenv("ATLAS_CHAT_PROVIDER_ID", "gemini")
    environment = {
        "ATLAS_GEMINI_MODEL": "gemini-2.5-flash",
        "ATLAS_GEMINI_API_KEY": "must-not-be-forwarded",
        "ATLAS_OLLAMA_TIMEOUT": "30",
        "ATLAS_OLLAMA_KEEP_ALIVE": "5m",
    }

    result = diagnose_from_environment(
        registry_builder=build_registry,
        environment=lambda name, default: environment.get(name, default),
    )

    assert result["provider"] == "gemini"
    assert result["model"] == "gemini-2.5-flash"
    assert calls == [{"timeout": 30.0, "keep_alive": "5m", "provider_id": "gemini"}]
    assert "must-not-be-forwarded" not in str(calls)


def test_diagnosis_uses_openai_compatible_environment(monkeypatch) -> None:
    provider = FakeProvider(response={"message": {"content": "compatible ok"}})
    registry = FakeRegistry(provider)

    monkeypatch.setenv("ATLAS_CHAT_PROVIDER_ID", "remote")
    result = diagnose_from_environment(
        registry_builder=lambda **_kwargs: registry,
        environment=lambda name, default: {"ATLAS_OPENAI_MODEL": "compatible-model"}.get(name, default),
    )

    assert result["provider"] == "remote"
    assert result["model"] == "compatible-model"
    assert registry.requested_provider_id == "remote"

from __future__ import annotations

import pytest

from bootstrap.bootstrap import Bootstrap
from core.model_manager import ModelDescriptor, ModelManager, ModelSelectionRequest
from core.model_selection_policy import ModelSelectionPolicy


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


@pytest.mark.parametrize(
    ("task", "physical_name"),
    (
        ("chat", "glm4:9b"),
        ("coding", "qwen3.6:latest"),
        ("project", "glm-5.2-local:latest"),
    ),
)
def test_default_policy_preserves_current_runtime_selection(
    task: str,
    physical_name: str,
) -> None:
    policy = ModelSelectionPolicy()
    manager = ModelManager(
        StaticModelSource(
            ["glm4:9b", "qwen3.6:latest", "glm-5.2-local:latest"]
        )
    )

    request = policy.create_request(task=task)
    result = manager.select_model(request)

    assert request == ModelSelectionRequest(task=task, allow_fallback=True)
    assert result.physical_model_name == physical_name


def test_policy_transports_preferences_without_selecting_a_model() -> None:
    policy = ModelSelectionPolicy(
        preferred_provider="ollama",
        prefer_local=True,
        max_cost=2.5,
        max_latency=1.25,
        allow_fallback=False,
    )

    request = policy.create_request(
        task="reasoning",
        preferred_model_id="project-local",
    )

    assert request == ModelSelectionRequest(
        task="reasoning",
        preferred_provider_id="ollama",
        prefer_local=True,
        maximum_relative_cost=2.5,
        maximum_relative_latency=1.25,
        preferred_model_id="project-local",
        allow_fallback=False,
    )


def test_prefer_local_policy_is_consumed_by_existing_selector() -> None:
    manager = ModelManager(
        StaticModelSource(["remote:latest", "local:latest"]),
        (
            ModelDescriptor(
                logical_id="remote-chat",
                provider_id="remote-provider",
                model_name="remote:latest",
                capabilities=("chat",),
                local=False,
                priority=1000,
            ),
            ModelDescriptor(
                logical_id="local-chat",
                provider_id="ollama",
                model_name="local:latest",
                capabilities=("chat",),
                local=True,
                priority=1,
            ),
        ),
    )

    result = manager.select_model(
        ModelSelectionPolicy(prefer_local=True).create_request(task="chat")
    )

    assert result.logical_model_id == "local-chat"


def test_policy_requests_are_deterministic_and_isolated() -> None:
    first_policy = ModelSelectionPolicy(preferred_provider="ollama")
    second_policy = ModelSelectionPolicy()

    first = first_policy.create_request(task="chat")
    repeated = first_policy.create_request(task="chat")
    second = second_policy.create_request(task="chat")

    assert first == repeated
    assert first is not repeated
    assert first.preferred_provider_id == "ollama"
    assert second.preferred_provider_id is None


def test_bootstrap_reads_explicit_model_selection_policy(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_MODEL_PREFERRED_PROVIDER", " ollama ")
    monkeypatch.setenv("ATLAS_MODEL_PREFER_LOCAL", "yes")
    monkeypatch.setenv("ATLAS_MODEL_MAX_COST", "2.5")
    monkeypatch.setenv("ATLAS_MODEL_MAX_LATENCY", "1.25")
    monkeypatch.setenv("ATLAS_MODEL_ALLOW_FALLBACK", "off")

    policy = Bootstrap.build_model_selection_policy()

    assert policy == ModelSelectionPolicy(
        preferred_provider="ollama",
        prefer_local=True,
        max_cost=2.5,
        max_latency=1.25,
        allow_fallback=False,
    )


def test_bootstrap_absent_model_policy_values_preserve_defaults(monkeypatch) -> None:
    for name in (
        "ATLAS_MODEL_PREFERRED_PROVIDER",
        "ATLAS_MODEL_PREFER_LOCAL",
        "ATLAS_MODEL_MAX_COST",
        "ATLAS_MODEL_MAX_LATENCY",
        "ATLAS_MODEL_ALLOW_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)

    assert Bootstrap.build_model_selection_policy() == ModelSelectionPolicy()


@pytest.mark.parametrize("value", ["invalid", "-1", "nan", "inf"])
def test_invalid_optional_numeric_policy_values_degrade_to_none(
    monkeypatch,
    capsys,
    value: str,
) -> None:
    monkeypatch.setenv("ATLAS_MODEL_MAX_COST", value)

    policy = Bootstrap.build_model_selection_policy()

    assert policy.max_cost is None
    assert "ATLAS_MODEL_MAX_COST" in capsys.readouterr().err


def test_invalid_boolean_policy_values_preserve_safe_defaults(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("ATLAS_MODEL_PREFER_LOCAL", "sometimes")
    monkeypatch.setenv("ATLAS_MODEL_ALLOW_FALLBACK", "sometimes")

    policy = Bootstrap.build_model_selection_policy()

    assert policy.prefer_local is None
    assert policy.allow_fallback is True
    warnings = capsys.readouterr().err
    assert "ATLAS_MODEL_PREFER_LOCAL" in warnings
    assert "ATLAS_MODEL_ALLOW_FALLBACK" in warnings

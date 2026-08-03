from core.model_manager import (
    ModelDescriptor,
    ModelManager,
    ModelSelectionRequest,
)


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


def _manager(*descriptors: ModelDescriptor) -> ModelManager:
    return ModelManager(
        StaticModelSource([item.model_name for item in descriptors]),
        descriptors,
    )


def _descriptor(
    logical_id: str,
    model_name: str,
    capability: str,
    *,
    provider_id: str = "ollama",
    available: bool = True,
    local: bool = True,
    priority: int = 10,
    cost: float | None = 1.0,
    latency: float | None = 1.0,
    fallbacks: tuple[str, ...] = (),
) -> ModelDescriptor:
    return ModelDescriptor(
        logical_id=logical_id,
        provider_id=provider_id,
        model_name=model_name,
        capabilities=(capability,),
        available=available,
        relative_cost=cost,
        relative_latency=latency,
        local=local,
        priority=priority,
        fallback_logical_ids=fallbacks,
    )


def test_chat_selection_returns_expected_logical_and_physical_model() -> None:
    manager = ModelManager(StaticModelSource(["glm4:9b"]))

    result = manager.select_model(ModelSelectionRequest(task="chat"))

    assert result.success is True
    assert result.logical_model_id == "chat-local"
    assert result.physical_model_name == "glm4:9b"
    assert result.provider_id == "ollama"
    assert result.descriptor is not None
    assert result.descriptor.logical_id == "chat-local"


def test_coding_selection_requires_coding_capability() -> None:
    manager = ModelManager(StaticModelSource(["qwen3.6:latest", "glm4:9b"]))

    result = manager.select_model(ModelSelectionRequest(task="coding"))

    assert result.success is True
    assert result.descriptor is not None
    assert "coding" in result.descriptor.capabilities


def test_vision_selection_never_chooses_model_without_vision_capability() -> None:
    manager = ModelManager(StaticModelSource(["glm4:9b", "gemma4:latest"]))

    result = manager.select_model(ModelSelectionRequest(task="vision"))

    assert result.success is True
    assert result.logical_model_id == "vision-local"
    assert result.descriptor is not None
    assert "vision" in result.descriptor.capabilities


def test_unavailable_model_is_excluded() -> None:
    unavailable = _descriptor(
        "unavailable-chat",
        "unavailable:latest",
        "chat",
        available=False,
        priority=100,
    )
    available = _descriptor("available-chat", "available:latest", "chat")
    manager = _manager(unavailable, available)

    result = manager.select_model(ModelSelectionRequest(task="chat"))

    assert result.success is True
    assert result.logical_model_id == "available-chat"


def test_local_preference_favors_compatible_local_model() -> None:
    remote = _descriptor(
        "remote-chat",
        "remote:latest",
        "chat",
        local=False,
        priority=100,
    )
    local = _descriptor(
        "local-chat",
        "local:latest",
        "chat",
        local=True,
        priority=1,
    )
    manager = _manager(remote, local)

    result = manager.select_model(
        ModelSelectionRequest(task="chat", prefer_local=True)
    )

    assert result.success is True
    assert result.logical_model_id == "local-chat"


def test_maximum_relative_cost_excludes_expensive_candidate() -> None:
    expensive = _descriptor(
        "expensive-chat",
        "expensive:latest",
        "chat",
        priority=100,
        cost=5.0,
    )
    cheap = _descriptor(
        "cheap-chat",
        "cheap:latest",
        "chat",
        priority=1,
        cost=1.0,
    )
    manager = _manager(expensive, cheap)

    result = manager.select_model(
        ModelSelectionRequest(task="chat", maximum_relative_cost=2.0)
    )

    assert result.success is True
    assert result.logical_model_id == "cheap-chat"


def test_maximum_relative_latency_excludes_slow_candidate() -> None:
    slow = _descriptor(
        "slow-chat",
        "slow:latest",
        "chat",
        priority=100,
        latency=5.0,
    )
    fast = _descriptor(
        "fast-chat",
        "fast:latest",
        "chat",
        priority=1,
        latency=1.0,
    )
    manager = _manager(slow, fast)

    result = manager.select_model(
        ModelSelectionRequest(task="chat", maximum_relative_latency=2.0)
    )

    assert result.success is True
    assert result.logical_model_id == "fast-chat"


def test_equal_candidates_use_stable_reproducible_tie_break() -> None:
    beta = _descriptor("beta", "beta:latest", "chat")
    alpha = _descriptor("alpha", "alpha:latest", "chat")
    manager = _manager(beta, alpha)
    request = ModelSelectionRequest(task="chat")

    first = manager.select_model(request)
    second = manager.select_model(request)

    assert first.logical_model_id == "alpha"
    assert second.logical_model_id == "alpha"
    assert first.reason == second.reason


def test_no_compatible_candidate_returns_controlled_failure() -> None:
    manager = _manager(_descriptor("chat-only", "chat:latest", "chat"))

    result = manager.select_model(ModelSelectionRequest(task="vision"))

    assert result.success is False
    assert result.logical_model_id is None
    assert result.descriptor is None
    assert result.error_code == "NO_COMPATIBLE_MODEL"
    assert result.reason == "No compatible registered model is available."


def test_declarative_fallback_is_used_only_when_allowed() -> None:
    primary = _descriptor(
        "primary-chat",
        "primary:latest",
        "chat",
        available=False,
        fallbacks=("fallback-chat",),
    )
    fallback = _descriptor("fallback-chat", "fallback:latest", "chat")
    manager = _manager(primary, fallback)

    denied = manager.select_model(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="primary-chat",
            allow_fallback=False,
        )
    )
    allowed = manager.select_model(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="primary-chat",
            allow_fallback=True,
        )
    )

    assert denied.success is False
    assert allowed.success is True
    assert allowed.logical_model_id == "fallback-chat"
    assert allowed.is_fallback is True


def test_available_primary_wins_before_authorized_higher_priority_fallback() -> None:
    primary = _descriptor(
        "primary-chat",
        "primary:latest",
        "chat",
        priority=1,
        fallbacks=("fallback-chat",),
    )
    fallback = _descriptor(
        "fallback-chat",
        "fallback:latest",
        "chat",
        priority=100,
    )
    manager = _manager(primary, fallback)

    result = manager.select_model(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="primary-chat",
            allow_fallback=True,
        )
    )

    assert result.success is True
    assert result.logical_model_id == "primary-chat"
    assert result.is_fallback is False


def test_fallback_chain_reaches_second_available_model() -> None:
    primary = _descriptor(
        "primary-chat",
        "primary:latest",
        "chat",
        fallbacks=("fallback-one",),
    )
    fallback_one = _descriptor(
        "fallback-one",
        "fallback-one:latest",
        "chat",
        fallbacks=("fallback-two",),
    )
    fallback_two = _descriptor("fallback-two", "fallback-two:latest", "chat")
    manager = ModelManager(
        StaticModelSource(["fallback-two:latest"]),
        (primary, fallback_one, fallback_two),
    )

    result = manager.select_model(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="primary-chat",
            allow_fallback=True,
        )
    )

    assert result.success is True
    assert result.logical_model_id == "fallback-two"
    assert result.physical_model_name == "fallback-two:latest"
    assert result.is_fallback is True


def test_exhausted_fallback_chain_returns_structured_failure() -> None:
    primary = _descriptor(
        "primary-chat",
        "primary:latest",
        "chat",
        available=False,
        fallbacks=("fallback-one",),
    )
    fallback_one = _descriptor(
        "fallback-one",
        "fallback-one:latest",
        "chat",
        available=False,
        fallbacks=("fallback-two",),
    )
    fallback_two = _descriptor(
        "fallback-two",
        "fallback-two:latest",
        "chat",
        available=False,
    )
    manager = _manager(primary, fallback_one, fallback_two)

    result = manager.select_model(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="primary-chat",
            allow_fallback=True,
        )
    )

    assert result.success is False
    assert result.error_code == "NO_COMPATIBLE_MODEL"


def test_fallback_without_required_capability_is_rejected() -> None:
    primary = _descriptor(
        "primary-reasoning",
        "primary:latest",
        "reasoning",
        available=False,
        fallbacks=("chat-only",),
    )
    chat_only = _descriptor("chat-only", "chat-only:latest", "chat")
    manager = _manager(primary, chat_only)

    result = manager.select_model(
        ModelSelectionRequest(
            task="reasoning",
            preferred_model_id="primary-reasoning",
            allow_fallback=True,
        )
    )

    assert result.success is False
    assert result.error_code == "NO_COMPATIBLE_MODEL"


def test_cyclic_fallback_metadata_terminates_with_structured_failure() -> None:
    primary = _descriptor(
        "cycle-a",
        "cycle-a:latest",
        "chat",
        available=False,
        fallbacks=("cycle-b",),
    )
    fallback = _descriptor(
        "cycle-b",
        "cycle-b:latest",
        "chat",
        available=False,
        fallbacks=("cycle-a",),
    )
    manager = _manager(primary, fallback)

    result = manager.select_model(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="cycle-a",
            allow_fallback=True,
        )
    )

    assert result.success is False
    assert result.error_code == "NO_COMPATIBLE_MODEL"


def test_fallback_selection_is_deterministic_for_same_chain() -> None:
    primary = _descriptor(
        "primary-chat",
        "primary:latest",
        "chat",
        available=False,
        fallbacks=("fallback-low", "fallback-high"),
    )
    fallback_low = _descriptor(
        "fallback-low",
        "fallback-low:latest",
        "chat",
        priority=1,
    )
    fallback_high = _descriptor(
        "fallback-high",
        "fallback-high:latest",
        "chat",
        priority=100,
    )
    manager = _manager(primary, fallback_low, fallback_high)
    request = ModelSelectionRequest(
        task="chat",
        preferred_model_id="primary-chat",
        allow_fallback=True,
    )

    first = manager.select_model(request)
    second = manager.select_model(request)

    assert first.logical_model_id == "fallback-high"
    assert second.logical_model_id == "fallback-high"
    assert first.reason == second.reason


def test_nonexistent_declared_fallback_is_never_selected() -> None:
    primary = _descriptor(
        "primary-chat",
        "primary:latest",
        "chat",
        available=False,
        fallbacks=("missing-fallback",),
    )
    manager = _manager(primary)

    result = manager.select_model(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="primary-chat",
            allow_fallback=True,
        )
    )

    assert result.success is False
    assert result.logical_model_id is None
    assert result.physical_model_name is None


def test_preferred_provider_favors_compatible_provider() -> None:
    first = _descriptor(
        "provider-a-chat",
        "provider-a:latest",
        "chat",
        provider_id="provider-a",
    )
    preferred = _descriptor(
        "provider-b-chat",
        "provider-b:latest",
        "chat",
        provider_id="provider-b",
    )
    manager = _manager(first, preferred)

    result = manager.select_model(
        ModelSelectionRequest(
            task="chat",
            preferred_provider_id="provider-b",
        )
    )

    assert result.success is True
    assert result.logical_model_id == "provider-b-chat"
    assert result.provider_id == "provider-b"


def test_valid_preferred_model_is_selected_over_higher_priority_candidate() -> None:
    preferred = _descriptor(
        "preferred-chat",
        "preferred:latest",
        "chat",
        priority=1,
    )
    higher_priority = _descriptor(
        "higher-priority-chat",
        "higher:latest",
        "chat",
        priority=100,
    )
    manager = _manager(preferred, higher_priority)

    result = manager.select_model(
        ModelSelectionRequest(
            task="chat",
            preferred_model_id="preferred-chat",
        )
    )

    assert result.success is True
    assert result.logical_model_id == "preferred-chat"
    assert result.physical_model_name == "preferred:latest"


def test_unknown_provider_preference_preserves_available_ollama_selection() -> None:
    manager = _manager(
        _descriptor(
            "ollama-chat",
            "ollama:latest",
            "chat",
            provider_id="ollama",
        )
    )

    result = manager.select_model(
        ModelSelectionRequest(
            task="chat",
            preferred_provider_id="missing-provider",
        )
    )

    assert result.success is True
    assert result.logical_model_id == "ollama-chat"
    assert result.provider_id == "ollama"


def test_cost_and_latency_limits_do_not_invent_missing_metadata() -> None:
    unknown = _descriptor(
        "unknown-metadata-chat",
        "unknown:latest",
        "chat",
        priority=100,
        cost=None,
        latency=None,
    )
    known = _descriptor(
        "known-metadata-chat",
        "known:latest",
        "chat",
        priority=1,
        cost=1.0,
        latency=1.0,
    )
    manager = _manager(unknown, known)

    result = manager.select_model(
        ModelSelectionRequest(
            task="chat",
            maximum_relative_cost=2.0,
            maximum_relative_latency=2.0,
        )
    )

    assert result.success is True
    assert result.logical_model_id == "known-metadata-chat"


def test_choose_model_historical_compatibility_remains_unchanged() -> None:
    manager = ModelManager(
        StaticModelSource(
            [
                "glm4:9b",
                "qwen3.6:latest",
                "glm-5.2-local:latest",
                "gemma4:latest",
            ]
        )
    )

    assert manager.choose_model("chat") == "glm4:9b"
    assert manager.choose_model("coding") == "qwen3.6:latest"
    assert manager.choose_model("project") == "glm-5.2-local:latest"
    assert manager.choose_model("vision") == "gemma4:latest"

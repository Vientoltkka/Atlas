from core.model_manager import ModelDescriptor, ModelManager, ModelSelectionRequest


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


def _descriptor(
    logical_id: str,
    provider_id: str,
    model_name: str,
    capabilities: tuple[str, ...],
    *,
    available: bool = True,
    cost: float = 1.0,
    latency: float = 1.0,
    priority: int = 10,
    fallbacks: tuple[str, ...] = (),
) -> ModelDescriptor:
    return ModelDescriptor(
        logical_id=logical_id,
        provider_id=provider_id,
        model_name=model_name,
        capabilities=capabilities,
        available=available,
        relative_cost=cost,
        relative_latency=latency,
        priority=priority,
        fallback_logical_ids=fallbacks,
        local=False,
    )


def _manager(*descriptors: ModelDescriptor) -> ModelManager:
    return ModelManager(
        StaticModelSource([item.model_name for item in descriptors]),
        descriptors,
    )


def test_automatic_selection_requires_every_requested_capability() -> None:
    partial = _descriptor(
        "fictional-chat",
        "fictional-a",
        "fictional-chat-v1",
        ("chat",),
        priority=100,
    )
    compatible = _descriptor(
        "fictional-vision-chat",
        "fictional-b",
        "fictional-vision-chat-v1",
        ("chat", "vision"),
    )
    manager = _manager(partial, compatible)

    result = manager.select_model(
        ModelSelectionRequest(task="chat", required_capabilities=("vision",))
    )

    assert result.success is True
    assert result.logical_model_id == "fictional-vision-chat"
    assert result.provider_id == "fictional-b"


def test_automatic_selection_excludes_declared_or_live_unavailable_models() -> None:
    disabled = _descriptor(
        "fictional-disabled",
        "fictional-a",
        "fictional-disabled-v1",
        ("chat",),
        available=False,
        priority=100,
    )
    missing = _descriptor(
        "fictional-missing",
        "fictional-b",
        "fictional-missing-v1",
        ("chat",),
        priority=90,
    )
    available = _descriptor(
        "fictional-available",
        "fictional-c",
        "fictional-available-v1",
        ("chat",),
        priority=1,
    )
    manager = ModelManager(
        StaticModelSource([disabled.model_name, available.model_name]),
        (disabled, missing, available),
    )

    result = manager.select_model(ModelSelectionRequest(task="chat"))

    assert result.success is True
    assert result.logical_model_id == "fictional-available"


def test_automatic_selection_uses_cost_and_latency_after_compatibility() -> None:
    costly_slow = _descriptor(
        "fictional-costly-slow",
        "fictional-a",
        "fictional-costly-slow-v1",
        ("chat",),
        cost=8.0,
        latency=8.0,
    )
    efficient = _descriptor(
        "fictional-efficient",
        "fictional-b",
        "fictional-efficient-v1",
        ("chat",),
        cost=0.2,
        latency=0.1,
    )
    manager = _manager(costly_slow, efficient)

    result = manager.select_model(ModelSelectionRequest(task="chat"))

    assert result.success is True
    assert result.logical_model_id == "fictional-efficient"


def test_configured_provider_preference_wins_when_compatible() -> None:
    default = _descriptor(
        "fictional-default",
        "fictional-a",
        "fictional-default-v1",
        ("chat",),
        priority=100,
    )
    configured = _descriptor(
        "fictional-configured",
        "fictional-b",
        "fictional-configured-v1",
        ("chat",),
        priority=1,
    )
    manager = _manager(default, configured)

    result = manager.select_model(
        ModelSelectionRequest(task="chat", preferred_provider_id="fictional-b")
    )

    assert result.success is True
    assert result.logical_model_id == "fictional-configured"


def test_declared_fallback_chain_is_deterministic_across_fictitious_providers() -> None:
    primary = _descriptor(
        "fictional-primary",
        "fictional-a",
        "fictional-primary-v1",
        ("chat",),
        fallbacks=("fictional-first", "fictional-second"),
    )
    first = _descriptor(
        "fictional-first",
        "fictional-b",
        "fictional-first-v1",
        ("chat",),
        priority=1,
    )
    second = _descriptor(
        "fictional-second",
        "fictional-c",
        "fictional-second-v1",
        ("chat",),
        priority=100,
    )
    manager = _manager(primary, first, second)
    request = ModelSelectionRequest(
        task="chat",
        preferred_model_id="fictional-primary",
        allow_fallback=True,
    )
    result = manager.select_fallback(
        request,
        initial_model_id="fictional-primary",
        attempted_model_ids=("fictional-primary",),
    )

    assert result.success is True
    assert result.logical_model_id == "fictional-first"
    assert result.is_fallback is True


def test_automatic_selection_reports_no_candidate_when_all_are_invalid() -> None:
    incompatible = _descriptor(
        "fictional-reasoner", "fictional-a", "fictional-reasoner-v1", ("reasoning",)
    )
    unavailable = _descriptor(
        "fictional-unavailable", "fictional-b", "fictional-unavailable-v1", ("chat",), available=False
    )
    result = _manager(incompatible, unavailable).select_model(
        ModelSelectionRequest(task="chat")
    )

    assert result.success is False
    assert result.error_code == "NO_COMPATIBLE_MODEL"
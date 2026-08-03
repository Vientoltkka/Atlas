from core.model_manager import ModelDescriptor, ModelManager


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


def test_registered_model_exposes_identity_provider_capabilities_and_availability() -> None:
    manager = ModelManager(StaticModelSource(["atlas-chat:latest"]))
    manager.register_model(
        ModelDescriptor(
            logical_id="fast-chat",
            provider_id="ollama",
            model_name="atlas-chat:latest",
            capabilities=("general_chat", "fast_response", "local"),
            relative_cost=1.0,
            relative_latency=1.0,
            local=True,
            priority=10,
        )
    )

    descriptor = manager.resolve_model("fast-chat")

    assert descriptor is not None
    assert descriptor.logical_id == "fast-chat"
    assert descriptor.provider_id == "ollama"
    assert descriptor.model_name == "atlas-chat:latest"
    assert descriptor.capabilities == ("general_chat", "fast_response", "local")
    assert descriptor.available is True
    assert manager.list_model_descriptors(available_only=True) == (descriptor,)


def test_registered_model_is_unavailable_when_backend_does_not_list_it() -> None:
    manager = ModelManager(StaticModelSource([]))
    manager.register_model(
        ModelDescriptor(
            logical_id="coding-missing",
            provider_id="ollama",
            model_name="missing:latest",
            capabilities=("coding", "local"),
        )
    )

    descriptor = manager.resolve_model("coding-missing")

    assert descriptor is not None
    assert descriptor.available is False
    assert manager.list_model_descriptors(available_only=True) == ()


def test_current_ollama_chat_model_remains_resolvable_and_selectable() -> None:
    manager = ModelManager(StaticModelSource(["glm4:9b", "other:latest"]))

    descriptor = manager.resolve_model("chat-local")

    assert descriptor is not None
    assert descriptor.model_name == "glm4:9b"
    assert descriptor.provider_id == "ollama"
    assert descriptor.available is True
    assert "general_chat" in descriptor.capabilities
    assert manager.choose_model("chat") == "glm4:9b"


def test_resource_catalog_reuses_registered_model_metadata() -> None:
    manager = ModelManager(StaticModelSource(["atlas-code:latest"]))
    manager.register_model(
        ModelDescriptor(
            logical_id="coding-test",
            provider_id="ollama",
            model_name="atlas-code:latest",
            capabilities=("coding", "reasoning", "local"),
            relative_cost=2.0,
            relative_latency=3.0,
            local=True,
            priority=20,
            fallback_logical_ids=("chat-local",),
        )
    )

    candidate = manager.resource_catalog().list_candidates()[0]

    assert candidate.resource_id == "atlas-code:latest"
    assert candidate.provider_id == "ollama"
    assert candidate.capabilities == ("coding", "reasoning", "local")
    assert candidate.estimated_cost == 2.0
    assert candidate.estimated_latency == 3.0
    assert candidate.local is True
    assert candidate.available is True

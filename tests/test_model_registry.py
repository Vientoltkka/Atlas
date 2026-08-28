from __future__ import annotations

import json

from core.model_manager import ModelManager, ModelSelectionRequest
from core import model_registry
from core.model_registry import load_model_descriptors_from_environment


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


def test_environment_registry_registers_and_selects_fictitious_provider_models(monkeypatch) -> None:
    monkeypatch.setenv(
        "ATLAS_MODEL_DESCRIPTORS",
        json.dumps({"models": [
            {
                "logical_id": "fictional-chat",
                "provider_id": "fictional-a",
                "model_id": "fictional-chat-v1",
                "capabilities": ["chat"],
                "context_window": 32768,
                "relative_cost": 1.5,
                "relative_latency": 0.4,
                "available": True,
                "local": False,
                "priority": 200,
            },
            {
                "provider_id": "fictional-b",
                "model_id": "fictional-reasoner-v2",
                "capabilities": ["reasoning"],
                "available": True,
                "priority": 190,
            },
        ]}),
    )
    descriptors = load_model_descriptors_from_environment()
    manager = ModelManager(
        StaticModelSource(["fictional-chat-v1", "fictional-reasoner-v2"]), descriptors
    )

    selection = manager.select_model(
        ModelSelectionRequest(task="chat", preferred_provider_id="fictional-a")
    )
    reasoner = manager.resolve_model("fictional-b:fictional-reasoner-v2")

    assert selection.success is True
    assert selection.provider_id == "fictional-a"
    assert selection.physical_model_name == "fictional-chat-v1"
    assert selection.descriptor is not None
    assert selection.descriptor.context_window == 32768
    assert reasoner is not None
    assert reasoner.provider_id == "fictional-b"


def test_file_registry_discards_invalid_entries_and_defaults_remain_available(monkeypatch) -> None:
    payload = json.dumps([
        {
            "logical_id": "configured-chat",
            "provider_id": "fictional",
            "model_id": "configured:1",
            "capabilities": ["chat"],
            "available": True,
        },
        {"provider_id": "broken", "model_id": "", "capabilities": ["chat"]},
        {"provider_id": "broken", "model_id": "bad", "available": "yes"},
    ])
    monkeypatch.setenv("ATLAS_MODEL_REGISTRY_CONFIG_PATH", "models.json")
    monkeypatch.setattr(model_registry.Path, "read_text", lambda _path, **_kwargs: payload)
    descriptors = load_model_descriptors_from_environment()
    manager = ModelManager(StaticModelSource(["configured:1", "glm4:9b"]), descriptors)

    assert [item.logical_id for item in descriptors] == ["configured-chat"]
    assert manager.resolve_model("configured-chat") is not None
    assert manager.choose_model("chat") == "glm4:9b"

def test_invalid_registry_payload_falls_back_to_default_catalog(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_MODEL_DESCRIPTORS", "not-json")
    manager = ModelManager(
        StaticModelSource(["glm4:9b"]), load_model_descriptors_from_environment()
    )

    assert manager.choose_model("chat") == "glm4:9b"

from __future__ import annotations

from core import atlas as atlas_module


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def process_prompt(self, prompt: str, confirm=None) -> str:
        self.prompts.append(prompt)
        return f"normal:{prompt}"


class _FakeBootstrap:
    orchestrator = _FakeOrchestrator()

    @staticmethod
    def build():
        return _FakeBootstrap.orchestrator


def test_operational_inventory_command_short_circuits_orchestrator(tmp_path, monkeypatch):
    monkeypatch.setattr(atlas_module, "Bootstrap", _FakeBootstrap)
    monkeypatch.setattr(atlas_module.Path, "resolve", lambda self: tmp_path / "core" / "atlas.py")
    _FakeBootstrap.orchestrator = _FakeOrchestrator()
    app = atlas_module.Atlas()

    response = app.process_prompt("Tengo 2 kg de arroz")

    assert response == "Inventario actualizado: arroz = 2 kg."
    assert _FakeBootstrap.orchestrator.prompts == []
    assert app._nutrition_state_store.get_food("arroz").quantity == 2


def test_unhandled_prompt_keeps_normal_orchestrator_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(atlas_module, "Bootstrap", _FakeBootstrap)
    monkeypatch.setattr(atlas_module.Path, "resolve", lambda self: tmp_path / "core" / "atlas.py")
    _FakeBootstrap.orchestrator = _FakeOrchestrator()
    app = atlas_module.Atlas()

    response = app.process_prompt("Explícame qué debería comer antes de entrenar")

    assert response == "normal:Explícame qué debería comer antes de entrenar"
    assert _FakeBootstrap.orchestrator.prompts == ["Explícame qué debería comer antes de entrenar"]

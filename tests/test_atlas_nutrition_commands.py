from __future__ import annotations

from datetime import datetime

from core import atlas as atlas_module
from core.daily_coach_profile import DailyCoachProfile


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


def _build_app(tmp_path, monkeypatch):
    monkeypatch.setattr(atlas_module, "Bootstrap", _FakeBootstrap)
    monkeypatch.setattr(atlas_module.Path, "resolve", lambda self: tmp_path / "core" / "atlas.py")
    _FakeBootstrap.orchestrator = _FakeOrchestrator()
    return atlas_module.Atlas()


def test_operational_inventory_command_short_circuits_orchestrator(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    response = app.process_prompt("Tengo 2 kg de arroz")
    assert response == "Inventario actualizado: arroz = 2 kg."
    assert _FakeBootstrap.orchestrator.prompts == []
    assert app._nutrition_state_store.get_food("arroz").quantity == 2


def test_non_nutrition_prompt_keeps_normal_orchestrator_flow(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    response = app.process_prompt("Explícame qué es la fotosíntesis")
    assert response == "normal:Explícame qué es la fotosíntesis"
    assert _FakeBootstrap.orchestrator.prompts == ["Explícame qué es la fotosíntesis"]


def test_daily_nutrition_request_receives_profile_inventory_and_today_schedule(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    today = datetime.now().astimezone().date()
    app._daily_coach_profile_store.save(DailyCoachProfile(
        sex="hombre", age=48, height_cm=180, weight_kg=73.8,
        target_weight_kg=80, objective="ganar masa muscular",
        priorities=("rendimiento", "salud"),
    ))
    app._nutrition_state_store.set_food("arroz", "arroz", 2, "kg")
    app._nutrition_state_store.set_food("huevos", "huevos", 12, "unit")
    app._nutrition_state_store.set_work_schedule(today, [("07:00", "15:00")])
    app._nutrition_state_store.set_training_schedule(today, [("18:00", "19:00")])

    app.process_prompt("Prepárame la alimentación de hoy con lo que tengo en casa")

    routed = _FakeBootstrap.orchestrator.prompts[-1]
    assert "edad: 48 años" in routed
    assert "altura: 180 cm" in routed
    assert "peso actual: 73.8 kg" in routed
    assert "peso objetivo: 80 kg" in routed
    assert "objetivo: ganar masa muscular" in routed
    assert f"Fecha objetivo: {today.isoformat()}" in routed
    assert "- arroz: 2 kg" in routed
    assert "- huevos: 12 unit" in routed
    assert "- 07:00-15:00" in routed
    assert "- 18:00-19:00" in routed


def test_text_repl_routes_daily_nutrition_through_application_context(tmp_path, monkeypatch, capsys):
    app = _build_app(tmp_path, monkeypatch)
    today = datetime.now().astimezone().date()
    app._nutrition_state_store.set_food("arroz", "arroz", 2, "kg")
    app._nutrition_state_store.set_work_schedule(today, [("07:00", "15:00")])
    inputs = iter(("Prepárame lo que debo comer hoy utilizando únicamente los alimentos que tengo en casa.", "salir"))
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    app.start()

    assert len(_FakeBootstrap.orchestrator.prompts) == 1
    routed = _FakeBootstrap.orchestrator.prompts[0]
    assert "[CONTEXTO OPERATIVO AUTORITATIVO DE NUTRICIÓN" in routed
    assert f"Fecha objetivo: {today.isoformat()}" in routed
    assert "- arroz: 2 kg" in routed
    assert "- 07:00-15:00" in routed
    assert "[PETICIÓN ACTUAL DEL USUARIO]" in routed
    assert "Prepárame lo que debo comer hoy" in routed
    assert "Hasta pronto." in capsys.readouterr().out


def test_tomorrow_nutrition_request_uses_tomorrow_schedule(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    today = datetime.now().astimezone().date()
    tomorrow = today.fromordinal(today.toordinal() + 1)
    app._nutrition_state_store.set_work_schedule(tomorrow, [("07:00", "15:00")])
    app.process_prompt("Prepárame las comidas de mañana")
    routed = _FakeBootstrap.orchestrator.prompts[-1]
    assert f"Fecha objetivo: {tomorrow.isoformat()}" in routed
    assert "- 07:00-15:00" in routed


def test_daily_context_render_is_read_only(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)
    state_path = app._nutrition_state_store._path
    profile_path = app._daily_coach_profile_store._path
    assert not state_path.exists()
    assert not profile_path.exists()
    app.process_prompt("Qué debería comer hoy")
    assert not state_path.exists()
    assert not profile_path.exists()

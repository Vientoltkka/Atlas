from __future__ import annotations

from core.daily_coach_profile import DailyCoachProfileStore
from core.daily_coach_profile_commands import DailyCoachProfileCommandHandler


def test_registers_full_spanish_profile(tmp_path):
    store = DailyCoachProfileStore(tmp_path / "profile.json")
    handler = DailyCoachProfileCommandHandler(store)

    result = handler.handle(
        "Mi perfil: hombre, 48 años, 180 cm, 73.8 kg. "
        "Objetivo: llegar a 80 kg priorizando masa muscular, rendimiento, salud y longevidad."
    )

    assert result.handled
    profile = store.load()
    assert profile is not None
    assert profile.sex == "hombre"
    assert profile.age == 48
    assert profile.height_cm == 180
    assert profile.weight_kg == 73.8
    assert profile.target_weight_kg == 80
    assert "llegar a 80 kg" in profile.objective.casefold()


def test_accepts_decimal_comma(tmp_path):
    store = DailyCoachProfileStore(tmp_path / "profile.json")
    result = DailyCoachProfileCommandHandler(store).handle(
        "Mi perfil: varón, 48 años, 180 cm, 73,8 kg. Objetivo: peso objetivo 80 kg."
    )
    assert result.handled
    assert store.load().weight_kg == 73.8
    assert store.load().target_weight_kg == 80


def test_incomplete_profile_is_handled_without_write(tmp_path):
    path = tmp_path / "profile.json"
    result = DailyCoachProfileCommandHandler(DailyCoachProfileStore(path)).handle(
        "Mi perfil: hombre, 48 años"
    )
    assert result.handled
    assert "incompleto" in result.message.casefold()
    assert not path.exists()


def test_unrelated_prompt_is_not_handled(tmp_path):
    result = DailyCoachProfileCommandHandler(
        DailyCoachProfileStore(tmp_path / "profile.json")
    ).handle("Prepárame la alimentación de hoy")
    assert not result.handled

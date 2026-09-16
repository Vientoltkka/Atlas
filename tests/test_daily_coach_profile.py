from __future__ import annotations

import json

import pytest

from core.daily_coach_profile import (
    DailyCoachProfile,
    DailyCoachProfileStore,
    InvalidDailyCoachProfileError,
    render_daily_coach_profile,
)


def test_profile_persists_and_reloads(tmp_path):
    path = tmp_path / "daily_coach_profile.json"
    store = DailyCoachProfileStore(path)
    profile = DailyCoachProfile(
        sex="hombre", age=48, height_cm=180, weight_kg=73.8,
        target_weight_kg=80, objective="ganar masa muscular",
        priorities=("rendimiento", "salud", "longevidad"),
    )
    store.save(profile)

    assert DailyCoachProfileStore(path).load() == profile


def test_profile_payload_is_versioned(tmp_path):
    path = tmp_path / "daily_coach_profile.json"
    store = DailyCoachProfileStore(path)
    store.save(DailyCoachProfile(sex="male", age=48, height_cm=180, weight_kg=73.8))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert set(payload) == {"schema_version", "profile"}


def test_invalid_profile_is_rejected():
    with pytest.raises(InvalidDailyCoachProfileError):
        DailyCoachProfile(sex="hombre", age=48, height_cm=180, weight_kg=-1)


def test_render_profile_contains_planning_fields():
    rendered = render_daily_coach_profile(
        DailyCoachProfile(
            sex="hombre", age=48, height_cm=180, weight_kg=73.8,
            target_weight_kg=80, objective="masa muscular",
            priorities=("rendimiento", "salud"),
        )
    )

    assert "edad: 48 años" in rendered
    assert "altura: 180 cm" in rendered
    assert "peso actual: 73.8 kg" in rendered
    assert "peso objetivo: 80 kg" in rendered
    assert "objetivo: masa muscular" in rendered
    assert "prioridades: rendimiento, salud" in rendered


def test_missing_profile_renders_explicitly():
    assert render_daily_coach_profile(None) == "Perfil Daily Coach: no registrado."

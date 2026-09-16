from __future__ import annotations

from datetime import date

from core.daily_nutrition_state import DailyMealRecord, DailyNutritionStore
from core.nutrition_context import NutritionContextProvider


def test_empty_context_is_explicit(tmp_path):
    store = DailyNutritionStore(tmp_path / "nutrition-state.json")
    provider = NutritionContextProvider(store)

    rendered = provider.render(date(2026, 9, 16))

    assert "Fecha objetivo: 2026-09-16" in rendered
    assert "- sin alimentos disponibles registrados" in rendered
    assert rendered.count("- no registrado") == 2
    assert "Comidas ya consumidas:" in rendered
    assert "- ninguna registrada" in rendered


def test_context_contains_only_available_inventory(tmp_path):
    store = DailyNutritionStore(tmp_path / "nutrition-state.json")
    store.set_food("rice", "Arroz", 2, "kg")
    store.set_food("chicken", "Pollo", 500, "g")
    store.mark_food_exhausted("chicken")
    provider = NutritionContextProvider(store)

    rendered = provider.render(date(2026, 9, 16))

    assert "- Arroz: 2 kg" in rendered
    assert "Pollo" not in rendered


def test_context_contains_work_training_and_notes_for_requested_day(tmp_path):
    store = DailyNutritionStore(tmp_path / "nutrition-state.json")
    requested_day = date(2026, 9, 16)
    other_day = date(2026, 9, 17)
    store.set_work_schedule(requested_day, (("07:00", "15:00"),))
    store.set_training_schedule(requested_day, (("18:00", "19:00"),))
    requested_state = store.get_day(requested_day)
    store.save_day(
        requested_state.__class__(
            day=requested_state.day,
            work_schedule=requested_state.work_schedule,
            training_schedule=requested_state.training_schedule,
            planned_meals=requested_state.planned_meals,
            consumed_meals=requested_state.consumed_meals,
            notes=("Priorizar inventario disponible",),
            updated_at=requested_state.updated_at,
        )
    )
    store.set_work_schedule(other_day, (("10:00", "18:00"),))
    provider = NutritionContextProvider(store)

    rendered = provider.render(requested_day)

    assert "- 07:00-15:00" in rendered
    assert "- 18:00-19:00" in rendered
    assert "- Priorizar inventario disponible" in rendered
    assert "10:00-18:00" not in rendered


def test_context_contains_consumed_meals_for_requested_day(tmp_path):
    store = DailyNutritionStore(tmp_path / "nutrition-state.json")
    requested_day = date(2026, 9, 16)
    other_day = date(2026, 9, 17)
    store.set_consumed_meals(requested_day, (
        DailyMealRecord(
            meal_id="consumed-1",
            label="desayuno",
            scheduled_time=None,
            foods=("3 unit huevos", "150 g arroz"),
        ),
    ))
    store.set_consumed_meals(other_day, (
        DailyMealRecord(
            meal_id="consumed-2",
            label="cena",
            scheduled_time=None,
            foods=("200 g pollo",),
        ),
    ))
    provider = NutritionContextProvider(store)

    rendered = provider.render(requested_day)

    assert "Comidas ya consumidas:" in rendered
    assert "- desayuno: 3 unit huevos; 150 g arroz" in rendered
    assert "200 g pollo" not in rendered


def test_render_does_not_mutate_store_or_persist_empty_day(tmp_path):
    path = tmp_path / "nutrition-state.json"
    store = DailyNutritionStore(path)
    store.set_food("eggs", "Huevos", 12, "unit")
    before = path.read_text(encoding="utf-8")
    provider = NutritionContextProvider(store)

    first = provider.render(date(2026, 9, 16))
    second = provider.render(date(2026, 9, 16))

    assert first == second
    assert path.read_text(encoding="utf-8") == before
    reloaded = DailyNutritionStore(path)
    assert reloaded.get_food("eggs") is not None
    assert reloaded.get_day(date(2026, 9, 16)).work_schedule == ()

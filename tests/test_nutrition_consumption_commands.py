from __future__ import annotations

from datetime import date

from core.daily_nutrition_state import DailyNutritionStore
from core.nutrition_consumption_commands import NutritionConsumptionCommandHandler


TODAY = date(2026, 9, 16)


def test_records_consumption_and_deducts_inventory(tmp_path):
    store = DailyNutritionStore(tmp_path / "state.json")
    store.set_food("huevos", "huevos", 12, "unit")
    store.set_food("arroz", "arroz", 2000, "g")

    result = NutritionConsumptionCommandHandler(store).handle(
        "He desayunado 3 unidades de huevos y 150 g de arroz", today=TODAY
    )

    assert result.handled
    assert store.get_food("huevos").quantity == 9
    assert store.get_food("arroz").quantity == 1850
    meals = store.get_day(TODAY).consumed_meals
    assert len(meals) == 1
    assert meals[0].label == "desayuno"
    assert meals[0].foods == ("3 unit huevos", "150 g arroz")


def test_rejects_insufficient_inventory_without_partial_mutation(tmp_path):
    store = DailyNutritionStore(tmp_path / "state.json")
    store.set_food("huevos", "huevos", 2, "unit")
    store.set_food("arroz", "arroz", 200, "g")

    result = NutritionConsumptionCommandHandler(store).handle(
        "He comido 3 unidades de huevos y 100 g de arroz", today=TODAY
    )

    assert result.handled
    assert "no puedo descontar" in result.message.casefold()
    assert store.get_food("huevos").quantity == 2
    assert store.get_food("arroz").quantity == 200
    assert store.get_day(TODAY).consumed_meals == ()


def test_unknown_food_does_not_mutate_state(tmp_path):
    store = DailyNutritionStore(tmp_path / "state.json")
    result = NutritionConsumptionCommandHandler(store).handle(
        "He cenado 200 g de pollo", today=TODAY
    )
    assert result.handled
    assert store.get_day(TODAY).consumed_meals == ()


def test_non_consumption_prompt_is_not_handled(tmp_path):
    store = DailyNutritionStore(tmp_path / "state.json")
    result = NutritionConsumptionCommandHandler(store).handle(
        "Prepárame el desayuno", today=TODAY
    )
    assert not result.handled

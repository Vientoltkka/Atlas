from __future__ import annotations

from dataclasses import replace
from datetime import date
import json

import pytest

from core.daily_nutrition_state import (
    CorruptedNutritionStateError,
    DailyMealRecord,
    DailyNutritionState,
    DailyNutritionStore,
    InvalidNutritionStateError,
)


def test_inventory_persists_and_reloads(tmp_path):
    path = tmp_path / "nutrition-state.json"
    store = DailyNutritionStore(path)

    store.set_food("chicken", "Chicken breast", 1200, "g")
    store.adjust_food("chicken", -200)

    reloaded = DailyNutritionStore(path)
    item = reloaded.get_food("chicken")

    assert item is not None
    assert item.quantity == 1000
    assert item.unit == "g"


def test_inventory_cannot_become_negative(tmp_path):
    store = DailyNutritionStore(tmp_path / "nutrition-state.json")
    store.set_food("rice", "Rice", 100, "g")

    with pytest.raises(InvalidNutritionStateError):
        store.adjust_food("rice", -101)


def test_mark_food_exhausted_preserves_item(tmp_path):
    store = DailyNutritionStore(tmp_path / "nutrition-state.json")
    store.set_food("eggs", "Eggs", 12, "unit")

    exhausted = store.mark_food_exhausted("eggs")

    assert exhausted.quantity == 0
    assert store.get_food("eggs").quantity == 0


def test_daily_state_persists_schedules_and_meals(tmp_path):
    path = tmp_path / "nutrition-state.json"
    store = DailyNutritionStore(path)
    day = date(2026, 9, 15)
    meal = DailyMealRecord(
        meal_id="post-workout",
        label="Post workout",
        scheduled_time="19:30",
        foods=("rice", "chicken"),
    )

    state = DailyNutritionState(
        day=day,
        work_schedule=(("08:00", "16:00"),),
        training_schedule=(("18:00", "19:00"),),
        planned_meals=(meal,),
        notes=("Prioritize available inventory",),
    )
    store.save_day(state)

    reloaded = DailyNutritionStore(path).get_day(day)

    assert reloaded.work_schedule == (("08:00", "16:00"),)
    assert reloaded.training_schedule == (("18:00", "19:00"),)
    assert reloaded.planned_meals[0].meal_id == "post-workout"
    assert reloaded.notes == ("Prioritize available inventory",)


def test_day_updates_do_not_overwrite_other_fields(tmp_path):
    store = DailyNutritionStore(tmp_path / "nutrition-state.json")
    day = date(2026, 9, 15)
    store.set_work_schedule(day, (("09:00", "17:00"),))
    store.set_training_schedule(day, (("18:00", "19:00"),))

    state = store.get_day(day)

    assert state.work_schedule == (("09:00", "17:00"),)
    assert state.training_schedule == (("18:00", "19:00"),)


def test_consumed_meals_are_distinct_from_planned_meals(tmp_path):
    store = DailyNutritionStore(tmp_path / "nutrition-state.json")
    day = date(2026, 9, 15)
    planned = DailyMealRecord(meal_id="lunch", label="Lunch", scheduled_time="14:00")
    consumed = replace(planned, foods=("rice", "eggs"))

    store.set_planned_meals(day, (planned,))
    store.set_consumed_meals(day, (consumed,))

    state = store.get_day(day)
    assert state.planned_meals[0].foods == ()
    assert state.consumed_meals[0].foods == ("rice", "eggs")


def test_invalid_clock_time_is_rejected(tmp_path):
    store = DailyNutritionStore(tmp_path / "nutrition-state.json")

    with pytest.raises(InvalidNutritionStateError):
        store.set_work_schedule(date(2026, 9, 15), (("25:00", "17:00"),))


def test_corrupt_payload_is_rejected(tmp_path):
    path = tmp_path / "nutrition-state.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(CorruptedNutritionStateError):
        DailyNutritionStore(path)

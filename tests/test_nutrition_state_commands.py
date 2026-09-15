from __future__ import annotations

from datetime import date

from core.daily_nutrition_state import DailyNutritionStore
from core.nutrition_state_commands import NutritionStateCommandHandler


TODAY = date(2026, 9, 16)


def _handler(tmp_path):
    store = DailyNutritionStore(tmp_path / "nutrition-state.json")
    return store, NutritionStateCommandHandler(store)


def test_tengo_sets_inventory(tmp_path):
    store, handler = _handler(tmp_path)

    result = handler.handle("Tengo 2 kg de arroz basmati", today=TODAY)

    assert result.handled
    item = store.get_food("arroz-basmati")
    assert item is not None
    assert item.name == "arroz basmati"
    assert item.quantity == 2
    assert item.unit == "kg"


def test_me_quedan_replaces_known_quantity(tmp_path):
    store, handler = _handler(tmp_path)
    handler.handle("Tengo 8 unidades de yogur", today=TODAY)

    result = handler.handle("Me quedan 2 unidades de yogur", today=TODAY)

    assert result.handled
    assert store.get_food("yogur").quantity == 2


def test_purchase_sets_current_quantity(tmp_path):
    store, handler = _handler(tmp_path)

    result = handler.handle("He comprado 500 g de pollo", today=TODAY)

    assert result.handled
    assert store.get_food("pollo").quantity == 500


def test_exhausted_known_food_sets_zero(tmp_path):
    store, handler = _handler(tmp_path)
    handler.handle("Tengo 500 g de pollo", today=TODAY)

    result = handler.handle("Se acabó el pollo", today=TODAY)

    assert result.handled
    assert store.get_food("pollo").quantity == 0


def test_unknown_exhausted_food_is_not_invented(tmp_path):
    store, handler = _handler(tmp_path)

    result = handler.handle("Se acabó el salmón", today=TODAY)

    assert not result.handled
    assert store.list_inventory() == ()


def test_tomorrow_work_schedule_is_persisted(tmp_path):
    store, handler = _handler(tmp_path)

    result = handler.handle("Mañana trabajo de 7 a 15", today=TODAY)

    assert result.handled
    assert store.get_day(date(2026, 9, 17)).work_schedule == (("07:00", "15:00"),)
    assert store.get_day(TODAY).work_schedule == ()


def test_training_schedule_is_persisted_for_today(tmp_path):
    store, handler = _handler(tmp_path)

    result = handler.handle("Hoy entreno de 18:00 a 19:30", today=TODAY)

    assert result.handled
    assert store.get_day(TODAY).training_schedule == (("18:00", "19:30"),)


def test_ambiguous_or_unrelated_text_does_not_mutate_state(tmp_path):
    store, handler = _handler(tmp_path)

    result = handler.handle("Creo que mañana quizá entrene por la tarde", today=TODAY)

    assert not result.handled
    assert store.list_inventory() == ()
    assert store.get_day(date(2026, 9, 17)).training_schedule == ()

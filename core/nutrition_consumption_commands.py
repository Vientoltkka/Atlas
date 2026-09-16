"""Explicit meal-consumption commands for the Daily Coach."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
import unicodedata

from core.daily_nutrition_state import DailyMealRecord, DailyNutritionStore, InvalidNutritionStateError


@dataclass(frozen=True, slots=True)
class NutritionConsumptionCommandResult:
    handled: bool
    message: str = ""


class NutritionConsumptionCommandHandler:
    """Record only explicit consumption and deduct matching inventory safely."""

    def __init__(self, store: DailyNutritionStore) -> None:
        self._store = store

    def handle(self, text: str, *, today: date) -> NutritionConsumptionCommandResult:
        raw = " ".join(str(text).strip().split())
        folded = _fold(raw)
        match = re.fullmatch(
            r"(?:he (?:desayunado|comido|cenado)|he tomado) (.+)", folded,
        )
        if match is None:
            return NutritionConsumptionCommandResult(False)

        foods = _parse_foods(match.group(1))
        if not foods:
            return NutritionConsumptionCommandResult(False)

        # Validate the whole command before mutating anything.
        resolved = []
        for quantity, unit, folded_name in foods:
            food_id = _slug(folded_name)
            item = self._store.get_food(food_id)
            if item is None or item.unit != unit or item.quantity < quantity:
                return NutritionConsumptionCommandResult(
                    True,
                    f"No puedo descontar {quantity:g} {unit} de {folded_name}: inventario insuficiente o unidad distinta.",
                )
            resolved.append((item, quantity))

        for item, quantity in resolved:
            self._store.adjust_food(item.food_id, -quantity)

        state = self._store.get_day(today)
        meal = DailyMealRecord(
            meal_id=f"consumed-{len(state.consumed_meals) + 1}",
            label=_meal_label(folded),
            foods=tuple(f"{quantity:g} {item.unit} {item.name}" for item, quantity in resolved),
        )
        self._store.set_consumed_meals(today, (*state.consumed_meals, meal))
        detail = ", ".join(meal.foods)
        return NutritionConsumptionCommandResult(True, f"Consumo registrado: {detail}. Inventario actualizado.")


def _parse_foods(value: str) -> tuple[tuple[float, str, str], ...]:
    parts = re.split(r"\s+y\s+|\s*,\s*", value)
    parsed = []
    for part in parts:
        match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*(kg|g|l|ml|unidades?|uds?)\s+(?:de\s+)?(.+)", part.strip())
        if match is None:
            return ()
        parsed.append((float(match.group(1).replace(",", ".")), _unit(match.group(2)), match.group(3).strip()))
    return tuple(parsed)


def _meal_label(text: str) -> str:
    if text.startswith("he desayunado"):
        return "desayuno"
    if text.startswith("he cenado"):
        return "cena"
    if text.startswith("he comido"):
        return "comida"
    return "consumo"


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _fold(value)).strip("-")
    if not slug:
        raise InvalidNutritionStateError("Food name cannot produce a valid id.")
    return slug


def _unit(value: str) -> str:
    return "unit" if value in {"unidad", "unidades", "ud", "uds"} else value

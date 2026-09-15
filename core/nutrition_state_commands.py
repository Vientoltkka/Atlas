"""Conservative natural-language commands for nutrition operational state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re
import unicodedata

from core.daily_nutrition_state import DailyNutritionStore, InvalidNutritionStateError


@dataclass(frozen=True, slots=True)
class NutritionStateCommandResult:
    handled: bool
    message: str = ""


class NutritionStateCommandHandler:
    """Apply only explicit, unambiguous inventory/schedule statements."""

    def __init__(self, store: DailyNutritionStore) -> None:
        self._store = store

    def handle(self, text: str, *, today: date) -> NutritionStateCommandResult:
        raw = " ".join(str(text).strip().split())
        normalized = _fold(raw)
        if not normalized:
            return NutritionStateCommandResult(False)

        result = self._handle_exhausted(raw, normalized)
        if result.handled:
            return result
        result = self._handle_inventory(raw, normalized)
        if result.handled:
            return result
        result = self._handle_schedule(normalized, today=today)
        if result.handled:
            return result
        return NutritionStateCommandResult(False)

    def _handle_exhausted(self, raw: str, normalized: str) -> NutritionStateCommandResult:
        match = re.fullmatch(r"(?:se (?:acabo|termino)|no (?:me )?queda) (?:el |la |los |las )?(.+)", normalized)
        if match is None:
            return NutritionStateCommandResult(False)
        food_name = _extract_original_tail(raw, match.group(1))
        food_id = _slug(food_name)
        if self._store.get_food(food_id) is None:
            return NutritionStateCommandResult(False)
        item = self._store.mark_food_exhausted(food_id)
        return NutritionStateCommandResult(True, f"Inventario actualizado: {item.name} = 0 {item.unit}.")

    def _handle_inventory(self, raw: str, normalized: str) -> NutritionStateCommandResult:
        match = re.fullmatch(
            r"(?:tengo|me quedan|he comprado|compre) (\d+(?:[.,]\d+)?) (kg|g|l|ml|unidades?|uds?) de (.+)",
            normalized,
        )
        if match is None:
            return NutritionStateCommandResult(False)
        quantity = float(match.group(1).replace(",", "."))
        unit = _unit(match.group(2))
        folded_name = match.group(3).strip()
        food_name = _extract_original_tail(raw, folded_name)
        item = self._store.set_food(_slug(food_name), food_name, quantity, unit)
        return NutritionStateCommandResult(True, f"Inventario actualizado: {item.name} = {item.quantity:g} {item.unit}.")

    def _handle_schedule(self, normalized: str, *, today: date) -> NutritionStateCommandResult:
        match = re.fullmatch(
            r"(?:(hoy|manana) )?(trabajo|entreno|entrenamiento) (?:de |desde )?(\d{1,2}(?::\d{2})?) (?:a|hasta) (\d{1,2}(?::\d{2})?)",
            normalized,
        )
        if match is None:
            return NutritionStateCommandResult(False)
        target = today + timedelta(days=1) if match.group(1) == "manana" else today
        start = _clock(match.group(3))
        end = _clock(match.group(4))
        try:
            if match.group(2) == "trabajo":
                self._store.set_work_schedule(target, ((start, end),))
                label = "Trabajo"
            else:
                self._store.set_training_schedule(target, ((start, end),))
                label = "Entrenamiento"
        except InvalidNutritionStateError:
            return NutritionStateCommandResult(False)
        return NutritionStateCommandResult(True, f"{label} registrado para {target.isoformat()}: {start}-{end}.")


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _slug(value: str) -> str:
    folded = _fold(value)
    slug = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    if not slug:
        raise InvalidNutritionStateError("Food name cannot produce a valid id.")
    return slug


def _unit(value: str) -> str:
    return "unit" if value in {"unidad", "unidades", "ud", "uds"} else value


def _clock(value: str) -> str:
    if ":" not in value:
        value = f"{value}:00"
    hour, minute = value.split(":", 1)
    return f"{int(hour):02d}:{int(minute):02d}"


def _extract_original_tail(raw: str, folded_tail: str) -> str:
    words = folded_tail.split()
    if not words:
        return folded_tail
    count = len(words)
    original_words = raw.split()
    return " ".join(original_words[-count:]).strip(" .")

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
        original = str(text).strip()
        if not original:
            return NutritionStateCommandResult(False)

        parts = _split_explicit_commands(original)
        if len(parts) > 1:
            results: list[NutritionStateCommandResult] = []
            for part in parts:
                result = self._handle_single(part, today=today)
                if not result.handled:
                    return NutritionStateCommandResult(False)
                results.append(result)
            return NutritionStateCommandResult(
                True,
                "\n".join(result.message for result in results if result.message),
            )

        return self._handle_single(original, today=today)

    def _handle_single(self, raw: str, *, today: date) -> NutritionStateCommandResult:
        normalized_raw = " ".join(raw.strip().split())
        normalized = _fold(normalized_raw)
        result = self._handle_exhausted(normalized_raw, normalized)
        if result.handled:
            return result
        result = self._handle_inventory(normalized_raw, normalized)
        if result.handled:
            return result
        return self._handle_schedule(normalized, today=today)

    def _handle_exhausted(self, raw: str, normalized: str) -> NutritionStateCommandResult:
        match = re.fullmatch(r"(?:se (?:acabo|termino)|no (?:me )?queda) (?:el |la |los |las )?(.+)", normalized)
        if match is None:
            return NutritionStateCommandResult(False)
        food_name = _original_group(raw, normalized, match.group(1))
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
        food_name = _original_group(raw, normalized, match.group(3))
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


def _split_explicit_commands(raw: str) -> tuple[str, ...]:
    """Split only where the next text clearly starts a supported command."""
    folded = _fold(raw)
    command_start = (
        r"(?:tengo|me quedan|he comprado|compre|se (?:acabo|termino)|"
        r"no (?:me )?queda|(?:hoy|manana) (?:trabajo|entreno|entrenamiento)|"
        r"trabajo|entreno|entrenamiento)\b"
    )
    separators = list(
        re.finditer(
            rf"(?:\s*[\n;]+\s*|\s*,\s*|\s+y\s+)(?={command_start})",
            folded,
            flags=re.IGNORECASE,
        )
    )
    if not separators:
        return (raw.strip(" ."),)

    parts: list[str] = []
    start = 0
    for separator in separators:
        parts.append(raw[start:separator.start()].strip(" ."))
        start = separator.end()
    parts.append(raw[start:].strip(" ."))
    return tuple(part for part in parts if part)


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


def _original_group(raw: str, folded_raw: str, folded_group: str) -> str:
    """Recover the matched group from this already isolated command only."""
    group_start = folded_raw.rfind(folded_group)
    if group_start < 0:
        return folded_group.strip()
    prefix = folded_raw[:group_start]
    raw_words = raw.split()
    prefix_words = prefix.split()
    group_words = folded_group.split()
    start = len(prefix_words)
    end = start + len(group_words)
    candidate = " ".join(raw_words[start:end]).strip(" .")
    return candidate or folded_group.strip()

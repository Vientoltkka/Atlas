"""Persistent operational state for daily nutrition planning and food inventory."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
_ALLOWED_UNITS = frozenset({"g", "kg", "ml", "l", "unit"})
_MAX_FILE_BYTES = 1_000_000


class DailyNutritionStateError(RuntimeError):
    """Base error for nutrition-state persistence and mutation."""


class InvalidNutritionStateError(ValueError):
    """Raised when nutrition-state data is invalid."""


class CorruptedNutritionStateError(DailyNutritionStateError):
    """Raised when persisted nutrition-state data is malformed."""


class NutritionStateWriteError(DailyNutritionStateError):
    """Raised when nutrition state cannot be written atomically."""


@dataclass(frozen=True, slots=True)
class FoodInventoryItem:
    food_id: str
    name: str
    quantity: float
    unit: str
    updated_at: datetime

    def __post_init__(self) -> None:
        food_id = _normalize_food_id(self.food_id)
        name = " ".join(self.name.split()).strip()
        unit = self.unit.strip().casefold()
        if not name:
            raise InvalidNutritionStateError("Food name cannot be empty.")
        if unit not in _ALLOWED_UNITS:
            raise InvalidNutritionStateError(f"Unsupported inventory unit: {unit}.")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, (int, float)):
            raise InvalidNutritionStateError("Food quantity must be numeric.")
        quantity = float(self.quantity)
        if quantity < 0:
            raise InvalidNutritionStateError("Food quantity cannot be negative.")
        _require_aware(self.updated_at, "updated_at")
        object.__setattr__(self, "food_id", food_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "unit", unit)


@dataclass(frozen=True, slots=True)
class DailyMealRecord:
    meal_id: str
    label: str
    scheduled_time: str | None = None
    foods: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        meal_id = self.meal_id.strip()
        label = " ".join(self.label.split()).strip()
        if not meal_id or not label:
            raise InvalidNutritionStateError("Meal id and label are required.")
        if self.scheduled_time is not None:
            _validate_clock_time(self.scheduled_time)
        object.__setattr__(self, "meal_id", meal_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "foods", tuple(" ".join(item.split()).strip() for item in self.foods if str(item).strip()))
        object.__setattr__(self, "notes", self.notes.strip())


@dataclass(frozen=True, slots=True)
class DailyNutritionState:
    day: date
    work_schedule: tuple[tuple[str, str], ...] = ()
    training_schedule: tuple[tuple[str, str], ...] = ()
    planned_meals: tuple[DailyMealRecord, ...] = ()
    consumed_meals: tuple[DailyMealRecord, ...] = ()
    notes: tuple[str, ...] = ()
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _require_aware(self.updated_at, "updated_at")
        object.__setattr__(self, "work_schedule", _normalize_schedule(self.work_schedule))
        object.__setattr__(self, "training_schedule", _normalize_schedule(self.training_schedule))
        object.__setattr__(self, "planned_meals", tuple(self.planned_meals))
        object.__setattr__(self, "consumed_meals", tuple(self.consumed_meals))
        object.__setattr__(self, "notes", tuple(str(note).strip() for note in self.notes if str(note).strip()))


class DailyNutritionStore:
    """Thread-safe, atomic JSON store for food inventory and dated nutrition state."""

    def __init__(self, path: Path, *, max_file_bytes: int = _MAX_FILE_BYTES) -> None:
        self._path = path
        self._max_file_bytes = max_file_bytes
        self._lock = RLock()
        self._inventory: dict[str, FoodInventoryItem] = {}
        self._days: dict[str, DailyNutritionState] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def list_inventory(self) -> tuple[FoodInventoryItem, ...]:
        with self._lock:
            return tuple(sorted(self._inventory.values(), key=lambda item: (item.name.casefold(), item.food_id)))

    def get_food(self, food_id: str) -> FoodInventoryItem | None:
        with self._lock:
            return self._inventory.get(_normalize_food_id(food_id))

    def set_food(self, food_id: str, name: str, quantity: float, unit: str) -> FoodInventoryItem:
        with self._lock:
            item = FoodInventoryItem(
                food_id=food_id,
                name=name,
                quantity=quantity,
                unit=unit,
                updated_at=_utcnow(),
            )
            self._inventory[item.food_id] = item
            self._persist_locked()
            return item

    def adjust_food(self, food_id: str, delta: float) -> FoodInventoryItem:
        with self._lock:
            key = _normalize_food_id(food_id)
            current = self._inventory.get(key)
            if current is None:
                raise InvalidNutritionStateError(f"Unknown food_id: {key}.")
            if isinstance(delta, bool) or not isinstance(delta, (int, float)):
                raise InvalidNutritionStateError("Food delta must be numeric.")
            quantity = current.quantity + float(delta)
            if quantity < 0:
                raise InvalidNutritionStateError("Food quantity cannot become negative.")
            updated = replace(current, quantity=quantity, updated_at=_utcnow())
            self._inventory[key] = updated
            self._persist_locked()
            return updated

    def mark_food_exhausted(self, food_id: str) -> FoodInventoryItem:
        with self._lock:
            key = _normalize_food_id(food_id)
            current = self._inventory.get(key)
            if current is None:
                raise InvalidNutritionStateError(f"Unknown food_id: {key}.")
            updated = replace(current, quantity=0.0, updated_at=_utcnow())
            self._inventory[key] = updated
            self._persist_locked()
            return updated

    def get_day(self, day: date) -> DailyNutritionState:
        with self._lock:
            key = day.isoformat()
            return self._days.get(key, DailyNutritionState(day=day))

    def save_day(self, state: DailyNutritionState) -> DailyNutritionState:
        if not isinstance(state, DailyNutritionState):
            raise InvalidNutritionStateError("state must be DailyNutritionState.")
        with self._lock:
            updated = replace(state, updated_at=_utcnow())
            self._days[state.day.isoformat()] = updated
            self._persist_locked()
            return updated

    def set_work_schedule(self, day: date, schedule: Sequence[tuple[str, str]]) -> DailyNutritionState:
        current = self.get_day(day)
        return self.save_day(replace(current, work_schedule=tuple(schedule)))

    def set_training_schedule(self, day: date, schedule: Sequence[tuple[str, str]]) -> DailyNutritionState:
        current = self.get_day(day)
        return self.save_day(replace(current, training_schedule=tuple(schedule)))

    def set_planned_meals(self, day: date, meals: Sequence[DailyMealRecord]) -> DailyNutritionState:
        current = self.get_day(day)
        return self.save_day(replace(current, planned_meals=tuple(meals)))

    def set_consumed_meals(self, day: date, meals: Sequence[DailyMealRecord]) -> DailyNutritionState:
        current = self.get_day(day)
        return self.save_day(replace(current, consumed_meals=tuple(meals)))

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            if self._path.stat().st_size > self._max_file_bytes:
                raise CorruptedNutritionStateError("Nutrition state file is too large.")
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CorruptedNutritionStateError("Nutrition state file is not valid JSON.") from error
        except OSError as error:
            raise DailyNutritionStateError("Could not load nutrition state.") from error
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "inventory", "days"}:
            raise CorruptedNutritionStateError("Nutrition state payload has an invalid structure.")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise CorruptedNutritionStateError("Unsupported nutrition state schema version.")
        try:
            inventory = {
                item.food_id: item
                for item in (_food_from_payload(raw) for raw in payload["inventory"])
            }
            days = {
                state.day.isoformat(): state
                for state in (_day_from_payload(raw) for raw in payload["days"])
            }
        except (KeyError, TypeError, ValueError) as error:
            raise CorruptedNutritionStateError("Nutrition state contains invalid data.") from error
        self._inventory = inventory
        self._days = days

    def _persist_locked(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "inventory": [_food_to_payload(item) for item in self.list_inventory()],
            "days": [_day_to_payload(self._days[key]) for key in sorted(self._days)],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > self._max_file_bytes:
            raise CorruptedNutritionStateError("Nutrition state exceeds the maximum size.")
        temp_path: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._path)
        except OSError as error:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise NutritionStateWriteError("Could not save nutrition state atomically.") from error


def _normalize_food_id(value: str) -> str:
    if not isinstance(value, str):
        raise InvalidNutritionStateError("food_id must be a string.")
    normalized = "-".join(value.strip().casefold().split())
    if not normalized or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in normalized):
        raise InvalidNutritionStateError("food_id must contain only letters, digits, '-' or '_'.")
    return normalized


def _normalize_schedule(schedule: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    normalized = []
    for start, end in schedule:
        _validate_clock_time(start)
        _validate_clock_time(end)
        normalized.append((start, end))
    return tuple(normalized)


def _validate_clock_time(value: str) -> None:
    try:
        datetime.strptime(value, "%H:%M")
    except ValueError as error:
        raise InvalidNutritionStateError(f"Invalid clock time: {value}.") from error


def _require_aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise InvalidNutritionStateError(f"{name} must be timezone-aware.")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _food_to_payload(item: FoodInventoryItem) -> dict[str, Any]:
    return {
        "food_id": item.food_id,
        "name": item.name,
        "quantity": item.quantity,
        "unit": item.unit,
        "updated_at": item.updated_at.isoformat(),
    }


def _food_from_payload(payload: Mapping[str, Any]) -> FoodInventoryItem:
    return FoodInventoryItem(
        food_id=payload["food_id"],
        name=payload["name"],
        quantity=payload["quantity"],
        unit=payload["unit"],
        updated_at=datetime.fromisoformat(payload["updated_at"]),
    )


def _meal_to_payload(meal: DailyMealRecord) -> dict[str, Any]:
    return {
        "meal_id": meal.meal_id,
        "label": meal.label,
        "scheduled_time": meal.scheduled_time,
        "foods": list(meal.foods),
        "notes": meal.notes,
    }


def _meal_from_payload(payload: Mapping[str, Any]) -> DailyMealRecord:
    return DailyMealRecord(
        meal_id=payload["meal_id"],
        label=payload["label"],
        scheduled_time=payload.get("scheduled_time"),
        foods=tuple(payload.get("foods", ())),
        notes=payload.get("notes", ""),
    )


def _day_to_payload(state: DailyNutritionState) -> dict[str, Any]:
    return {
        "day": state.day.isoformat(),
        "work_schedule": [list(item) for item in state.work_schedule],
        "training_schedule": [list(item) for item in state.training_schedule],
        "planned_meals": [_meal_to_payload(meal) for meal in state.planned_meals],
        "consumed_meals": [_meal_to_payload(meal) for meal in state.consumed_meals],
        "notes": list(state.notes),
        "updated_at": state.updated_at.isoformat(),
    }


def _day_from_payload(payload: Mapping[str, Any]) -> DailyNutritionState:
    return DailyNutritionState(
        day=date.fromisoformat(payload["day"]),
        work_schedule=tuple(tuple(item) for item in payload.get("work_schedule", ())),
        training_schedule=tuple(tuple(item) for item in payload.get("training_schedule", ())),
        planned_meals=tuple(_meal_from_payload(item) for item in payload.get("planned_meals", ())),
        consumed_meals=tuple(_meal_from_payload(item) for item in payload.get("consumed_meals", ())),
        notes=tuple(payload.get("notes", ())),
        updated_at=datetime.fromisoformat(payload["updated_at"]),
    )

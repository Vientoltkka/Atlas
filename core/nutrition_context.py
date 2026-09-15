"""Read-only operational context adapter for daily nutrition planning."""

from __future__ import annotations

from datetime import date

from core.daily_nutrition_state import DailyNutritionStore


class NutritionContextProvider:
    """Render bounded inventory and daily state without mutating persistence."""

    def __init__(self, store: DailyNutritionStore) -> None:
        self._store = store

    def render(self, day: date) -> str:
        inventory = tuple(item for item in self._store.list_inventory() if item.quantity > 0)
        state = self._store.get_day(day)

        lines = [f"Fecha objetivo: {day.isoformat()}", "Inventario disponible:"]
        if inventory:
            lines.extend(f"- {item.name}: {item.quantity:g} {item.unit}" for item in inventory)
        else:
            lines.append("- sin alimentos disponibles registrados")

        lines.append("Horario de trabajo:")
        if state.work_schedule:
            lines.extend(f"- {start}-{end}" for start, end in state.work_schedule)
        else:
            lines.append("- no registrado")

        lines.append("Horario de entrenamiento:")
        if state.training_schedule:
            lines.extend(f"- {start}-{end}" for start, end in state.training_schedule)
        else:
            lines.append("- no registrado")

        if state.notes:
            lines.append("Notas operativas:")
            lines.extend(f"- {note}" for note in state.notes)

        return "\n".join(lines)

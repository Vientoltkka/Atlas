"""Read-only operational context adapter for daily nutrition planning."""

from __future__ import annotations

from datetime import date, datetime
from typing import Callable

from core.daily_nutrition_state import DailyNutritionStore


class NutritionContextProvider:
    """Render bounded inventory and daily state without mutating persistence."""

    def __init__(
        self,
        store: DailyNutritionStore,
        *,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._now_provider = now_provider or (lambda: datetime.now().astimezone())

    def render(self, day: date) -> str:
        inventory = tuple(
            item
            for item in self._store.list_inventory()
            if item.quantity > 0
        )
        state = self._store.get_day(day)

        now = self._now_provider()

        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError(
                "now_provider must return a timezone-aware datetime."
            )

        local_now = now.astimezone()
        is_today = day == local_now.date()
        current_clock = local_now.strftime("%H:%M")

        lines = [
            f"Fecha objetivo: {day.isoformat()}",
        ]

        if is_today:
            lines.append(
                f"Hora local actual: {current_clock}"
            )
        else:
            lines.append(
                "Hora local actual: no aplicable; "
                "la fecha objetivo no es hoy"
            )

        lines.extend(
            [
                "Reglas del contexto operativo:",
                (
                    "- El inventario listado es exhaustivo para peticiones "
                    "basadas en alimentos disponibles/en casa; no uses, "
                    "recomiendes ni supongas alimentos no registrados como "
                    "disponibles."
                ),
                (
                    "- Si el inventario no permite cumplir el objetivo "
                    "nutricional, indícalo y señala qué falta por categoría "
                    "o nutriente, sin fingir que alimentos concretos no "
                    "registrados están disponibles."
                ),
                (
                    "- Las comidas listadas como ya consumidas son hechos "
                    "autoritativos del día: no las vuelvas a planificar ni "
                    "afirmes que no se han consumido."
                ),
                (
                    "- No infieras si una cantidad consumida estaba cruda "
                    "o cocida cuando el registro no lo especifica; si afecta "
                    "al cálculo preciso, pide aclaración o marca la "
                    "estimación como imprecisa."
                ),
                (
                    "- Para hoy, no planifiques comidas a una hora anterior "
                    "a la hora local actual."
                ),
                (
                    "- Una comida planificada cuya hora ya pasó sin registro "
                    "de consumo está pasada/omitida; nunca la trates como "
                    "consumida."
                ),
                "Inventario disponible:",
            ]
        )

        if inventory:
            lines.extend(
                f"- {item.name}: {item.quantity:g} {item.unit}"
                for item in inventory
            )
        else:
            lines.append(
                "- sin alimentos disponibles registrados"
            )

        lines.append("Horario de trabajo:")

        if state.work_schedule:
            lines.extend(
                f"- {start}-{end}"
                for start, end in state.work_schedule
            )
        else:
            lines.append("- no registrado")

        lines.append("Horario de entrenamiento:")

        if state.training_schedule:
            lines.extend(
                f"- {start}-{end}"
                for start, end in state.training_schedule
            )
        else:
            lines.append("- no registrado")

        consumed_ids = {
            meal.meal_id
            for meal in state.consumed_meals
        }

        lines.append(
            "Estado temporal de comidas planificadas:"
        )

        if state.planned_meals:
            for meal in state.planned_meals:
                if meal.meal_id in consumed_ids:
                    status = "consumida"

                elif (
                    is_today
                    and meal.scheduled_time is not None
                    and meal.scheduled_time < current_clock
                ):
                    status = "pasada/omitida"

                else:
                    status = "pendiente"

                when = (
                    f" ({meal.scheduled_time})"
                    if meal.scheduled_time
                    else ""
                )

                lines.append(
                    f"- {meal.label}{when}: {status}"
                )

        else:
            lines.append("- ninguna planificada")

        lines.append("Comidas ya consumidas:")

        if state.consumed_meals:
            for meal in state.consumed_meals:
                foods = (
                    "; ".join(meal.foods)
                    if meal.foods
                    else "sin alimentos detallados"
                )

                when = (
                    f" ({meal.scheduled_time})"
                    if meal.scheduled_time
                    else ""
                )

                lines.append(
                    f"- {meal.label}{when}: {foods}"
                )

        else:
            lines.append("- ninguna registrada")

        if state.notes:
            lines.append("Notas operativas:")

            lines.extend(
                f"- {note}"
                for note in state.notes
            )

        return "\n".join(lines)

"""Atlas main application."""

from datetime import datetime, timedelta
from pathlib import Path
import unicodedata

from bootstrap.bootstrap import Bootstrap
from core.daily_coach_profile import DailyCoachProfileStore, render_daily_coach_profile
from core.daily_coach_profile_commands import DailyCoachProfileCommandHandler
from core.daily_nutrition_state import DailyNutritionStore
from core.nutrition_consumption_commands import NutritionConsumptionCommandHandler
from core.nutrition_context import NutritionContextProvider
from core.nutrition_state_commands import NutritionStateCommandHandler


class Atlas:
    """Main Atlas application."""

    def __init__(self) -> None:
        self._orchestrator = Bootstrap.build()
        data_dir = Path(__file__).resolve().parents[1] / "data"
        self._nutrition_state_store = DailyNutritionStore(data_dir / "nutrition_state.json")
        self._nutrition_state_commands = NutritionStateCommandHandler(self._nutrition_state_store)
        self._nutrition_consumption_commands = NutritionConsumptionCommandHandler(self._nutrition_state_store)
        self._nutrition_context = NutritionContextProvider(self._nutrition_state_store)
        self._daily_coach_profile_store = DailyCoachProfileStore(data_dir / "daily_coach_profile.json")
        self._daily_coach_profile_commands = DailyCoachProfileCommandHandler(self._daily_coach_profile_store)

    def start(self) -> None:
        self._orchestrator.start()

    def start_voice(self, state_listener=None, status_sink=None, typed_input=None) -> None:
        self._orchestrator.start_voice(
            state_listener=state_listener, status_sink=status_sink, typed_input=typed_input,
        )

    def process_prompt(self, prompt: str) -> str:
        today = datetime.now().astimezone().date()
        profile_command = self._daily_coach_profile_commands.handle(prompt)
        if profile_command.handled:
            return profile_command.message
        consumption_command = self._nutrition_consumption_commands.handle(prompt, today=today)
        if consumption_command.handled:
            return consumption_command.message
        command = self._nutrition_state_commands.handle(prompt, today=today)
        if command.handled:
            return command.message

        routed_prompt = prompt
        if _requests_daily_nutrition_context(prompt):
            target_day = today + timedelta(days=1) if _mentions_tomorrow(prompt) else today
            context = self._nutrition_context.render(target_day)
            profile_context = render_daily_coach_profile(self._daily_coach_profile_store.load())
            routed_prompt = (
                "[CONTEXTO OPERATIVO AUTORITATIVO DE NUTRICIÓN — SOLO LECTURA]\n"
                "Los datos siguientes ya son conocidos por Atlas y son hechos operativos "
                "para esta petición. Úsalos antes de pedir información al usuario.\n"
                "No vuelvas a pedir ningún dato que ya figure aquí.\n"
                "Las comidas registradas como consumidas ya ocurrieron: no las vuelvas a "
                "planificar ni afirmes que no se consumieron.\n"
                "El inventario con stock positivo es el inventario disponible real. Si la "
                "petición exige usar únicamente lo disponible/en casa, no incluyas como "
                "parte del plan alimentos que no estén en ese inventario.\n"
                "Si falta un dato imprescindible que realmente no figure en este contexto, "
                "pide solo esa aclaración.\n\n"
                f"{profile_context}\n{context}\n"
                "[FIN DEL CONTEXTO OPERATIVO AUTORITATIVO]\n\n"
                "[PETICIÓN ACTUAL DEL USUARIO]\n"
                f"{prompt}"
            )
        return self._orchestrator.process_prompt(routed_prompt, confirm=lambda _prompt: "")

    def add_supervision_state_listener(self, listener) -> None:
        self._orchestrator.add_supervision_state_listener(listener)

    def start_assistant(self) -> None:
        self._orchestrator.start_assistant()

    def list_microphones(self) -> str:
        return self._orchestrator.list_microphones()

    def close(self) -> None:
        close = getattr(self._orchestrator, "close", None)
        if callable(close):
            close()


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _mentions_tomorrow(prompt: str) -> bool:
    return "manana" in _fold(prompt)


def _requests_daily_nutrition_context(prompt: str) -> bool:
    folded = _fold(prompt)
    markers = (
        "alimentacion", "nutricion", "dieta", "comida", "comer", "desayuno",
        "almuerzo", "merienda", "cena", "calorias", "macro", "pre entren", "post entren",
    )
    if any(marker in folded for marker in markers):
        return True

    # Natural household-inventory questions belong to Daily Coach even when
    # the user does not explicitly say nutrition, food or diet.
    inventory_queries = (
        "que me queda en casa",
        "que tengo en casa",
        "que alimentos tengo",
        "que comida tengo",
        "que me queda de comida",
        "que queda en el inventario",
        "que tengo en el inventario",
    )
    return any(query in folded for query in inventory_queries)

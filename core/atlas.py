"""Atlas main application."""

from datetime import datetime, timedelta
from pathlib import Path
import unicodedata

from bootstrap.bootstrap import Bootstrap
from core.daily_nutrition_state import DailyNutritionStore
from core.nutrition_context import NutritionContextProvider
from core.nutrition_state_commands import NutritionStateCommandHandler


class Atlas:
    """Main Atlas application."""

    def __init__(self) -> None:
        """Initialize Atlas."""

        self._orchestrator = Bootstrap.build()
        state_path = Path(__file__).resolve().parents[1] / "data" / "nutrition_state.json"
        self._nutrition_state_store = DailyNutritionStore(state_path)
        self._nutrition_state_commands = NutritionStateCommandHandler(self._nutrition_state_store)
        self._nutrition_context = NutritionContextProvider(self._nutrition_state_store)

    def start(self) -> None:
        """Start Atlas."""

        self._orchestrator.start()

    def start_voice(
        self,
        state_listener=None,
        status_sink=None,
        typed_input=None,
    ) -> None:
        """Start Atlas in manual voice mode."""

        self._orchestrator.start_voice(
            state_listener=state_listener,
            status_sink=status_sink,
            typed_input=typed_input,
        )

    def process_prompt(self, prompt: str) -> str:
        """Process one textual Orbe request without blocking the UI thread."""
        today = datetime.now().astimezone().date()
        command = self._nutrition_state_commands.handle(prompt, today=today)
        if command.handled:
            return command.message

        routed_prompt = prompt
        if _requests_daily_nutrition_context(prompt):
            target_day = today + timedelta(days=1) if _mentions_tomorrow(prompt) else today
            context = self._nutrition_context.render(target_day)
            routed_prompt = (
                f"{prompt}\n\n"
                "[CONTEXTO OPERATIVO DE NUTRICIÓN — SOLO LECTURA]\n"
                f"{context}\n"
                "[FIN DEL CONTEXTO OPERATIVO]"
            )
        return self._orchestrator.process_prompt(routed_prompt, confirm=lambda _prompt: "")

    def add_supervision_state_listener(self, listener) -> None:
        """Expose supervised execution transitions to an optional UI observer."""
        self._orchestrator.add_supervision_state_listener(listener)

    def start_assistant(self) -> None:
        """Start Atlas in permanent assistant mode."""

        self._orchestrator.start_assistant()

    def list_microphones(self) -> str:
        """Return available input microphones."""

        return self._orchestrator.list_microphones()

    def close(self) -> None:
        """Release runtime resources owned by the orchestrator, when supported."""

        close = getattr(self._orchestrator, "close", None)
        if callable(close):
            close()


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _mentions_tomorrow(prompt: str) -> bool:
    return "manana" in _fold(prompt)


def _requests_daily_nutrition_context(prompt: str) -> bool:
    """Keep operational context bounded to explicit nutrition/meal requests."""
    folded = _fold(prompt)
    markers = (
        "alimentacion",
        "nutricion",
        "dieta",
        "comida",
        "comer",
        "desayuno",
        "almuerzo",
        "merienda",
        "cena",
        "calorias",
        "macro",
        "pre entren",
        "post entren",
    )
    return any(marker in folded for marker in markers)

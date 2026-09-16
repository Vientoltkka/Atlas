"""Atlas main application."""

from datetime import datetime
from pathlib import Path

from bootstrap.bootstrap import Bootstrap
from core.daily_nutrition_state import DailyNutritionStore
from core.nutrition_state_commands import NutritionStateCommandHandler


class Atlas:
    """Main Atlas application."""

    def __init__(self) -> None:
        """Initialize Atlas."""

        self._orchestrator = Bootstrap.build()
        state_path = Path(__file__).resolve().parents[1] / "data" / "nutrition_state.json"
        self._nutrition_state_store = DailyNutritionStore(state_path)
        self._nutrition_state_commands = NutritionStateCommandHandler(self._nutrition_state_store)

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
        command = self._nutrition_state_commands.handle(
            prompt,
            today=datetime.now().astimezone().date(),
        )
        if command.handled:
            return command.message
        return self._orchestrator.process_prompt(prompt, confirm=lambda _prompt: "")

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

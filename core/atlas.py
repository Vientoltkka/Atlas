"""Atlas main application."""

from bootstrap.bootstrap import Bootstrap


class Atlas:
    """Main Atlas application."""

    def __init__(self) -> None:
        """Initialize Atlas."""

        self._orchestrator = Bootstrap.build()

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

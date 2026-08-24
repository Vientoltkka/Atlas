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
        typed_input=None,
    ) -> None:
        """Start Atlas in manual voice mode."""

        self._orchestrator.start_voice(
            state_listener=state_listener,
            typed_input=typed_input,
        )

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

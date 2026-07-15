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

    def start_voice(self) -> None:
        """Start Atlas in manual voice mode."""

        self._orchestrator.start_voice()

    def list_microphones(self) -> str:
        """Return available input microphones."""

        return self._orchestrator.list_microphones()

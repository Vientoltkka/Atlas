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
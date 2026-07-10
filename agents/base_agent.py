"""Base class for every Atlas agent."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base interface for every Atlas agent."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human readable description."""
        ...

    @abstractmethod
    def run(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Execute the agent."""
        ...
"""Base class for every Atlas agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Visible agent output with optional continuation state."""

    text: str
    requires_follow_up: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("AgentResponse text must be a non-empty string.")
        if not isinstance(self.requires_follow_up, bool):
            raise ValueError("AgentResponse requires_follow_up must be a bool.")


def ask_prompt_client(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    provider_id: str | None,
) -> str:
    """Forward the selected provider only when the selection chain provided one."""
    if provider_id is None:
        return client.ask(model=model, messages=messages)
    return client.ask(model=model, messages=messages, provider_id=provider_id)


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
        *,
        provider_id: str | None = None,
    ) -> str | AgentResponse:
        """Execute the agent."""
        ...

"""Chat Agent."""

from __future__ import annotations

from collections.abc import Iterator

from agents.base_agent import BaseAgent
from models.prompt_client import PromptClient


class ChatAgent(BaseAgent):
    """Default conversational agent."""

    def __init__(self, prompt_client: PromptClient):
        self._client = prompt_client

    @property
    def name(self) -> str:
        return "chat"

    @property
    def description(self) -> str:
        return "General conversation"

    def run(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:

        return self._client.ask(
            model=model,
            messages=messages,
        )

    def stream(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        """Yield one conversational response through PromptClient streaming."""
        return self._client.stream_messages(
            model=model,
            messages=messages,
        )

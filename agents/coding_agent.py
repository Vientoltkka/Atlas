"""Coding Agent."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.prompt_client import PromptClient


class CodingAgent(BaseAgent):
    """Agent specialized in programming tasks."""

    def __init__(
        self,
        prompt_client: PromptClient,
    ) -> None:

        self._client = prompt_client

    @property
    def name(self) -> str:
        return "coding"

    @property
    def description(self) -> str:
        return "Programming assistant."

    def run(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Execute a coding request."""

        return self._client.ask(
            model=model,
            messages=messages,
        )
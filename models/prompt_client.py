"""Prompt client for Atlas."""

from __future__ import annotations

from typing import Any

import ollama


class PromptClient:
    """Client used to communicate with Ollama."""

    def __init__(self) -> None:
        self._client = ollama.Client()

    def ask(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Send a conversation to the selected model."""

        response = self._client.chat(
            model=model,
            messages=messages,
        )

        return self._extract_content(response)

    def _extract_content(self, response: Any) -> str:
        """Extract text from the Ollama response."""

        if isinstance(response, dict):
            return response["message"]["content"]

        return response.message.content
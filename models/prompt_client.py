"""Prompt client for Atlas."""

from __future__ import annotations

from collections.abc import Iterator
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
        return self.ask_messages(model=model, messages=messages)

    def ask_messages(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Send exactly the provided messages to the selected model."""
        response = self._client.chat(
            model=model,
            messages=messages,
        )

        return self._extract_content(response)

    def stream_messages(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        """Stream exactly the provided messages and yield content fragments."""
        stream = self._client.chat(
            model=model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            content = self._extract_stream_content(chunk)
            if content:
                yield content

    def _extract_content(self, response: Any) -> str:
        """Extract text from the Ollama response."""

        if isinstance(response, dict):
            return response["message"]["content"]

        return response.message.content

    def _extract_stream_content(self, chunk: Any) -> str | None:
        """Extract one streamed content fragment from an Ollama chunk."""
        if isinstance(chunk, dict):
            message = chunk.get("message")
            if message is None:
                return None
            if not isinstance(message, dict):
                raise ValueError("Malformed Ollama stream chunk.")
            content = message.get("content")
        else:
            message = getattr(chunk, "message", None)
            if message is None:
                return None
            content = getattr(message, "content", None)

        if content is None:
            return None
        if not isinstance(content, str):
            raise ValueError("Malformed Ollama stream chunk.")
        return content

"""Conversation memory."""

from __future__ import annotations


class ConversationMemory:
    def __init__(self) -> None:
        self._messages: list[dict[str, str]] = []

    def add_user(self, text: str) -> None:
        self._messages.append(
            {
                "role": "user",
                "content": text,
            }
        )

    def add_assistant(self, text: str) -> None:
        self._messages.append(
            {
                "role": "assistant",
                "content": text,
            }
        )

    def history(self) -> list[dict[str, str]]:
        return self._messages.copy()
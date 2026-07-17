"""Simple task router for Atlas."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from core.planner import Plan


class Router:
    """Determine which agent should execute the plan."""

    _TASK_ROUTES: dict[str, str] = {
        "chat": "chat",
        "coding": "coding",
        "project": "project",
        "research": "chat",
    }

    def route(self, plan: Plan) -> Literal["chat", "coding", "project"]:
        """Return the agent that must execute the plan.

        Defaults to 'chat' if the task type is not explicitly mapped.
        """
        if self._is_project_file_lookup(plan.objective):
            return "project"

        return self._TASK_ROUTES.get(plan.task, "chat")

    def route_voice_command(
        self,
        prompt: str,
    ) -> Literal[
        "voice_time",
        "voice_date",
        "voice_datetime",
        "voice_open_notepad",
        "voice_open_vscode",
    ] | None:
        """Return a supported voice tool route or None for model fallback."""
        text = self._normalize(prompt)

        if not text:
            return None

        asks_time = self._asks_current_time(text)
        asks_date = self._asks_current_date(text)
        tokens = self._tokens(text)

        if asks_time and asks_date:
            return "voice_datetime"

        if asks_time:
            return "voice_time"

        if asks_date:
            return "voice_date"

        if tokens in (["bloc", "de", "notas"], ["notepad"]):
            return "voice_open_notepad"

        if tokens in (["vs", "code"], ["vscode"], ["visual", "studio", "code"]):
            return "voice_open_vscode"

        if self._asks_open(text) and self._mentions_notepad(text):
            return "voice_open_notepad"

        if self._asks_open(text) and self._mentions_vscode(text):
            return "voice_open_vscode"

        return None

    def _is_project_file_lookup(
        self,
        objective: str,
    ) -> bool:
        """Detect direct Python file analysis requests."""
        text = objective.strip().lower()

        if not text:
            return False

        lookup_words = (
            "analiza",
            "analizar",
            "archivo",
            "clase",
            "funcion",
            "funcion",
            "localiza",
            "encuentra",
            "ruta",
            "path",
        )

        return ".py" in text and any(word in text for word in lookup_words)

    def _asks_current_time(
        self,
        text: str,
    ) -> bool:
        """Return whether the prompt asks for the current time."""
        tokens = self._tokens(text)

        if tokens == ["hora"] or tokens == ["hora", "es"]:
            return True

        if tokens in (
            ["que", "hora"],
            ["que", "hora", "es"],
            ["que", "hora", "es", "hoy"],
            ["dime", "la", "hora"],
        ):
            return True

        if any(
            phrase in text
            for phrase in (
                "dime la hora",
                "hora actual",
            )
        ):
            return True

        return (
            "hora" in tokens
            and any(marker in tokens for marker in ("actual", "ahora"))
            and (
                text.startswith("que ")
                or "dime" in tokens
                or "actual" in tokens
                or "mismo" in tokens
            )
        )

    def _asks_current_date(
        self,
        text: str,
    ) -> bool:
        """Return whether the prompt asks for the current date."""
        tokens = self._tokens(text)

        if tokens == ["fecha"] or tokens == ["fecha", "hoy"]:
            return True

        if tokens in (
            ["que", "fecha", "es", "hoy"],
            ["que", "dia", "es", "hoy"],
            ["dime", "la", "fecha"],
        ):
            return True

        return any(
            phrase in text
            for phrase in (
                "que dia es",
                "cual es la fecha",
                "fecha actual",
                "fecha y hora",
                "que fecha es",
            )
        )

    def _asks_open(
        self,
        text: str,
    ) -> bool:
        """Return whether the prompt asks to open an application."""
        return any(
            text.startswith(prefix)
            for prefix in (
                "abre ",
                "abrir ",
                "abreme ",
                "abre el ",
                "abre la ",
                "inicia ",
                "lanza ",
                "open ",
            )
        )

    def _mentions_notepad(
        self,
        text: str,
    ) -> bool:
        """Return whether the prompt targets Notepad."""
        tokens = self._tokens(text)
        return (
            "bloc de notas" in text
            or "notepad" in tokens
            or ("bloc" in tokens and "notas" in tokens)
        )

    def _mentions_vscode(
        self,
        text: str,
    ) -> bool:
        """Return whether the prompt targets Visual Studio Code."""
        tokens = self._tokens(text)
        return any(
            phrase in text
            for phrase in (
                "visual studio code",
                "vs code",
                "vscode",
            )
        ) or ("visual" in tokens and "studio" in tokens and "code" in tokens)

    def _normalize(
        self,
        text: str,
    ) -> str:
        """Normalize accents, punctuation and whitespace for voice matching."""
        normalized = unicodedata.normalize("NFKD", text.strip().lower())
        without_accents = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        without_punctuation = re.sub(r"[^\w\s]", " ", without_accents)
        return " ".join(without_punctuation.split())

    def _tokens(
        self,
        text: str,
    ) -> list[str]:
        """Return normalized tokens for intent matching."""
        return text.split()

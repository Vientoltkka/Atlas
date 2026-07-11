"""Coding Agent."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.prompt_client import PromptClient
from use_cases.read_file import ReadFileUseCase


class CodingAgent(BaseAgent):
    """Agent specialized in programming tasks."""

    SYSTEM_PROMPT = """
Eres Atlas Coding Agent.

Eres un ingeniero de software senior.

Tu trabajo es:

- Analizar código.
- Corregir errores.
- Refactorizar.
- Mejorar el código.
- Mantener Clean Architecture.
- Mantener SOLID.
- Nunca inventar APIs.
- Devuelve siempre el archivo completo cuando propongas cambios.
"""

    def __init__(
        self,
        prompt_client: PromptClient,
        read_file: ReadFileUseCase,
    ) -> None:

        self._client = prompt_client
        self._read_file = read_file

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

        if not messages:
            return "No hay mensajes."

        prompt = messages[-1]["content"].strip()
        lower = prompt.lower()

        # -----------------------------------------
        # Leer un archivo
        # -----------------------------------------

        if lower.startswith("lee "):

            path = prompt[4:].strip()

            try:
                return self._read_file.execute(path)
            except Exception as exc:
                return f"Error leyendo '{path}': {exc}"

        # -----------------------------------------
        # Corregir un archivo
        # -----------------------------------------

        if lower.startswith("corrige "):

            path = prompt[8:].strip()

            try:
                content = self._read_file.execute(path)
            except Exception as exc:
                return f"Error leyendo '{path}': {exc}"

            conversation = [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"""
Corrige el siguiente archivo Python.

Ruta:
{path}

Devuelve únicamente el archivo completo corregido.

Código:

{content}
""",
                },
            ]

            return self._client.ask(
                model=model,
                messages=conversation,
            )

        # -----------------------------------------
        # Conversación normal
        # -----------------------------------------

        conversation = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            }
        ]

        conversation.extend(messages)

        return self._client.ask(
            model=model,
            messages=conversation,
        )
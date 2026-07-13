"""Planner for Atlas."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata


@dataclass
class Plan:
    task: str
    objective: str


class Planner:
    """Creates an execution plan from the user's request."""

    def create_plan(self, prompt: str) -> Plan:

        text = prompt.lower()
        normalized_text = self._normalize_text(prompt)

        if self._is_architecture_query(normalized_text):
            return Plan(
                task="project",
                objective=prompt,
            )

        # ---------------------------------
        # Project analysis
        # ---------------------------------

        if any(word in text for word in (
            "analiza este proyecto",
            "analiza el proyecto",
            "proyecto",
            "repositorio",
            "arquitectura",
            "estructura",
        )):
            return Plan(
                task="project",
                objective=prompt,
            )

        # ---------------------------------
        # File operations
        # ---------------------------------

        if any(word in text for word in (
            "lee ",
            "abrir ",
            "abre ",
            "mostrar ",
            "muestra ",
            "corrige ",
            "modifica ",
            "editar ",
            "edita ",
        )):
            return Plan(
                task="coding",
                objective=prompt,
            )

        # ---------------------------------
        # Coding
        # ---------------------------------

        if any(word in text for word in (
            "programa",
            "python",
            "código",
            "codigo",
            "script",
            "función",
            "funcion",
        )):
            return Plan(
                task="coding",
                objective=prompt,
            )

        # ---------------------------------
        # Research
        # ---------------------------------

        if any(word in text for word in (
            "investiga",
            "buscar",
            "busca",
            "resume",
        )):
            return Plan(
                task="research",
                objective=prompt,
            )

        # ---------------------------------
        # Default
        # ---------------------------------

        return Plan(
            task="chat",
            objective=prompt,
        )

    def _is_architecture_query(
        self,
        text: str,
    ) -> bool:
        """Detect deterministic architecture graph queries."""
        return (
            "quien usa" in text
            or "modulos dependen de" in text
            or "clases importa" in text
            or (
                "archivos" in text
                and "afectados" in text
                and "modifico" in text
            )
        )

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        """Normalize accents and punctuation for query detection."""
        normalized = unicodedata.normalize("NFKD", text.lower())
        without_accents = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )

        return without_accents.translate(
            str.maketrans("¿?¡!", "    ")
        )

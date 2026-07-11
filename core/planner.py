"""Planner for Atlas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Plan:
    task: str
    objective: str


class Planner:
    """Creates an execution plan from the user's request."""

    def create_plan(self, prompt: str) -> Plan:

        text = prompt.lower()

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
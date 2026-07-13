"""Simple task router for Atlas."""

from __future__ import annotations

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
            "función",
            "localiza",
            "encuentra",
            "ruta",
            "path",
        )

        return ".py" in text and any(word in text for word in lookup_words)

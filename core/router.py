"""Simple task router for Atlas."""

from __future__ import annotations


class Router:
    """Determine which agent should execute the plan."""

    def route(self, plan) -> str:
        """Return the agent name for a plan."""

        # Si el Planner ya ha determinado la tarea, usamos esa información.
        task = getattr(plan, "task", None)

        if task:
            return task

        # Compatibilidad temporal por si todavía llega un string.
        if isinstance(plan, str):
            text = plan.lower()

            coding_words = (
                "python",
                "programa",
                "función",
                "codigo",
                "code",
                "bug",
                "script",
                "javascript",
            )

            reasoning_words = (
                "calcula",
                "demuestra",
                "razona",
                "matemática",
                "problema",
            )

            vision_words = (
                "imagen",
                "foto",
                "describe",
                "analiza esta imagen",
            )

            if any(word in text for word in coding_words):
                return "coding"

            if any(word in text for word in reasoning_words):
                return "reasoning"

            if any(word in text for word in vision_words):
                return "vision"

        return "chat"
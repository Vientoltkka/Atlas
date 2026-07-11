"""Simple task router for Atlas."""

from __future__ import annotations

from core.planner import Plan


class Router:
    """Determine which agent should execute the plan."""

    def route(
        self,
        plan: Plan,
    ) -> str:
        """Return the agent that must execute the plan."""

        routes = {
            "chat": "chat",
            "coding": "coding",
            "project": "project",
            "research": "chat",
        }

        return routes.get(
            plan.task,
            "chat",
        )
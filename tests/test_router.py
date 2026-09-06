"""Focused tests for the task router."""

from core.planner import Plan
from core.router import Router



def test_task_routing_ignores_letter_case() -> None:
    router = Router()

    assert router.route(Plan(task="Coding", objective="x")) == "coding"
    assert router.route(Plan(task="Research", objective="x")) == "chat"

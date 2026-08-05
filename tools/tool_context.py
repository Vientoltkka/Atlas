"""Shared context for Atlas tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from core.skill_execution_context import SkillExecutionContext



@dataclass(slots=True)
class ToolContext:
    """Shared execution context for every tool."""

    parameters: dict[str, Any] = field(default_factory=dict)

    step_id: str | None = None

    plan_signature: str | None = None

    previous_results: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    execution_context: SkillExecutionContext | None = None

    def __post_init__(self) -> None:
        self.parameters = dict(self.parameters)
        self.previous_results = dict(self.previous_results)
        self.metadata = dict(self.metadata)
        if self.execution_context is not None and not isinstance(
            self.execution_context,
            SkillExecutionContext,
        ):
            raise TypeError("execution_context must be SkillExecutionContext or None.")

    @property
    def arguments(self) -> dict[str, Any]:
        """Return the tool arguments transported in this context."""
        return self.parameters

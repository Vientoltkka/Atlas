"""Tool Executor."""

from __future__ import annotations

from collections.abc import Mapping

from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class ToolExecutor:
    """Executes registered Atlas tools."""

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:

        self._registry = registry

    def execute(
        self,
        tool_name: str,
        context: ToolContext | None = None,
        *,
        arguments: Mapping[str, object] | None = None,
    ):
        """Execute a registered tool."""

        tool = self._registry.get(tool_name)
        active_context = context or ToolContext(
            parameters=dict(arguments or {}),
        )

        return tool.execute(active_context)

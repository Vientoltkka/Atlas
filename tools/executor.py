"""Tool Executor."""

from __future__ import annotations

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
        context: ToolContext,
    ):
        """Execute a registered tool."""

        tool = self._registry.get(tool_name)

        if tool is None:
            raise RuntimeError(
                f"Tool '{tool_name}' is not registered."
            )

        return tool.execute(context)
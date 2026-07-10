"""Tool Registry."""

from __future__ import annotations

from tools.base_tool import BaseTool


class ToolRegistry:
    """Stores every available Atlas tool."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""

        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Return a tool."""

        return self._tools.get(name)

    def exists(self, name: str) -> bool:
        """Check if a tool exists."""

        return name in self._tools

    def list(self) -> list[str]:
        """Return registered tools."""

        return sorted(self._tools.keys())
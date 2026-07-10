"""List Directory Tool."""

from __future__ import annotations

from pathlib import Path

from tools.base_tool import BaseTool
from tools.tool_context import ToolContext


class ListDirectoryTool(BaseTool):
    """List directory contents."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and directories."

    def execute(
        self,
        context: ToolContext,
    ) -> list[str]:

        path = context.parameters.get("path", ".")

        directory = Path(path)

        if not directory.exists():
            raise FileNotFoundError(path)

        if not directory.is_dir():
            raise NotADirectoryError(path)

        return sorted(
            item.name
            for item in directory.iterdir()
        )
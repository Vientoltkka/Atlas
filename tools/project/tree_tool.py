"""Project Tree Tool."""

from __future__ import annotations

from pathlib import Path

from tools.base_tool import BaseTool
from tools.tool_context import ToolContext


class TreeTool(BaseTool):
    """Return every Python file in a project."""

    @property
    def name(self) -> str:
        return "project_tree"

    @property
    def description(self) -> str:
        return "Return all Python files inside a project."

    def execute(
        self,
        context: ToolContext,
    ) -> list[str]:

        root = context.parameters.get("path", ".")

        root_path = Path(root)

        if not root_path.exists():
            raise FileNotFoundError(root)

        ignored = {
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
        }

        files: list[str] = []

        for file in root_path.rglob("*.py"):

            if any(part in ignored for part in file.parts):
                continue

            files.append(
                str(file.relative_to(root_path))
            )

        return sorted(files)
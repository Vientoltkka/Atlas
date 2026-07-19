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

    def semantic_metadata(self) -> dict[str, object]:
        """Return semantic metadata for catalog generation."""
        return {
            "capabilities": ["list_directory"],
            "supported_intents": ["list files in a directory"],
            "input_description": "Accepts an optional local directory path.",
            "output_description": "Sorted list of directory entry names.",
            "risk_level": "low",
            "preconditions": ["path must exist", "path must point to a directory"],
            "limitations": ["does not read file contents", "does not recurse into subdirectories"],
            "negative_examples": ["summarize the contents of every file", "create a directory"],
            "compatible_tools": ["read_file"],
            "tags": ["filesystem", "directory"],
            "positive_examples": ["lista los archivos del escritorio"],
            "category": "filesystem",
        }

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

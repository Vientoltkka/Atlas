"""Read File Tool."""

from __future__ import annotations

from services.file_service import FileService
from tools.base_tool import BaseTool
from tools.tool_context import ToolContext


class ReadFileTool(BaseTool):
    """Read a UTF-8 text file."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read a UTF-8 text file."

    def semantic_metadata(self) -> dict[str, object]:
        """Return semantic metadata for catalog generation."""
        return {
            "capabilities": ["read_file"],
            "supported_intents": ["read a local file"],
            "input_description": "Requires a local text file path.",
            "output_description": "UTF-8 file content as text.",
            "risk_level": "low",
            "preconditions": ["path must exist", "path must point to a file"],
            "limitations": ["does not interpret file contents", "does not read remote paths"],
            "negative_examples": ["explain what a file is", "write new file content"],
            "compatible_tools": ["write_file"],
            "tags": ["filesystem", "read"],
            "positive_examples": ["lee el archivo notas.txt"],
            "category": "filesystem",
        }

    def execute(
        self,
        context: ToolContext,
    ) -> str:

        path = context.parameters.get("path")

        if not path:
            raise ValueError("Missing parameter 'path'.")

        content = FileService.read(path)

        limit = context.parameters.get("limit")

        if limit is None:
            return content

        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("Parameter 'limit' must be a positive integer.")

        return "\n".join(content.splitlines()[:limit])

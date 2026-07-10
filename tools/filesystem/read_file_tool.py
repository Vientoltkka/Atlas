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

    def execute(
        self,
        context: ToolContext,
    ) -> str:

        path = context.parameters.get("path")

        if not path:
            raise ValueError("Missing parameter 'path'.")

        return FileService.read(path)
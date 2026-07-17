"""Write File Tool."""

from __future__ import annotations

from services.file_service import FileService
from tools.base_tool import BaseTool
from tools.tool_context import ToolContext


class WriteFileTool(BaseTool):
    """Write a UTF-8 text file."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write a UTF-8 text file."

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(
        self,
        context: ToolContext,
    ) -> str:

        path = context.parameters.get("path")
        content = context.parameters.get("content")

        if not path:
            raise ValueError("Missing parameter 'path'.")

        if content is None:
            raise ValueError("Missing parameter 'content'.")

        FileService.write(
            path=path,
            content=content,
        )

        return f"Archivo '{path}' actualizado correctamente."

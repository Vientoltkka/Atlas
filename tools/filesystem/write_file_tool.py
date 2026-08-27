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


    @property
    def required_permissions(self) -> tuple[str, ...]:
        return ("filesystem.write",)
    def semantic_metadata(self) -> dict[str, object]:
        """Return semantic metadata for catalog generation."""
        return {
            "capabilities": ["write_file"],
            "supported_intents": ["create or update a text file"],
            "input_description": "Requires a target path and text content.",
            "output_description": "Human-readable write confirmation message.",
            "risk_level": "medium",
            "risk_reasons": ["can create or overwrite local files"],
            "requires_confirmation": True,
            "preconditions": ["path must be provided", "content must be provided", "user confirmation is required"],
            "limitations": ["does not merge file contents", "does not recover overwritten content"],
            "negative_examples": ["explain how writing files works", "read a file without modifying it"],
            "compatible_tools": ["read_file"],
            "tags": ["filesystem", "write"],
            "positive_examples": ["escribe hola en notas.txt"],
            "category": "filesystem",
        }

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

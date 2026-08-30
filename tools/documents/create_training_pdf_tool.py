"""Tool for creating a PDF from generated training content."""

from __future__ import annotations

from tools.base_tool import BaseTool
from tools.tool_context import ToolContext
from use_cases.create_training_pdf import CreateTrainingPdfUseCase


class CreateTrainingPdfTool(BaseTool):
    """Create and open a PDF training document after explicit confirmation."""

    def __init__(self, use_case: CreateTrainingPdfUseCase) -> None:
        self._use_case = use_case

    @property
    def name(self) -> str:
        return "training.create_pdf"

    @property
    def description(self) -> str:
        return "Create and open a PDF from generated training content."

    @property
    def requires_confirmation(self) -> bool:
        return True

    @property
    def required_permissions(self) -> tuple[str, ...]:
        return ("filesystem.write",)

    def execute(self, context: ToolContext) -> str:
        content = context.parameters.get("content")
        output_dir = context.parameters.get("output_dir", "artifacts/documents")
        return self._use_case.execute(content, output_dir)

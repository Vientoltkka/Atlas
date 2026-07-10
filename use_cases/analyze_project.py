"""Analyze project use case."""

from __future__ import annotations

from tools.executor import ToolExecutor
from tools.tool_context import ToolContext


class AnalyzeProjectUseCase:
    """Obtain all Python files from a project."""

    def __init__(
        self,
        executor: ToolExecutor,
    ) -> None:
        self._executor = executor

    def execute(
        self,
        root: str = ".",
    ) -> list[str]:

        context = ToolContext(
            parameters={
                "path": root,
            }
        )

        return self._executor.execute(
            "project_tree",
            context,
        )
"""Write file use case."""

from __future__ import annotations

from tools.executor import ToolExecutor
from tools.tool_context import ToolContext


class WriteFileUseCase:
    """Write a file using Atlas tools."""

    def __init__(
        self,
        executor: ToolExecutor,
    ) -> None:
        self._executor = executor

    def execute(
        self,
        path: str,
        content: str,
    ) -> str:

        context = ToolContext(
            parameters={
                "path": path,
                "content": content,
            }
        )

        return self._executor.execute(
            "write_file",
            context,
        )
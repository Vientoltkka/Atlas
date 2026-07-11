"""Read file use case."""

from __future__ import annotations

from tools.executor import ToolExecutor
from tools.tool_context import ToolContext


class ReadFileUseCase:
    """Read a file using Atlas tools."""

    def __init__(
        self,
        executor: ToolExecutor,
    ) -> None:
        self._executor = executor

    def execute(
        self,
        path: str,
    ) -> str:

        context = ToolContext(
            parameters={
                "path": path,
            }
        )

        return self._executor.execute(
            "read_file",
            context,
        )
"""Read file use case."""

from tools.executor import ToolExecutor
from tools.tool_context import ToolContext


class ReadFileUseCase:
    """High level file reading."""

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
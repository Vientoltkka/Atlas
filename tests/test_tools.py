from pathlib import Path

from tools.filesystem.read_file_tool import ReadFileTool
from tools.tool_context import ToolContext


def test_read_file_tool(tmp_path: Path):

    file = tmp_path / "demo.txt"

    file.write_text("Hola Atlas", encoding="utf-8")

    tool = ReadFileTool()

    context = ToolContext(
        parameters={
            "path": str(file)
        }
    )

    result = tool.execute(context)

    assert result == "Hola Atlas"
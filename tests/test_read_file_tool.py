"""Focused tests for the bounded read variant of ReadFileTool."""

import pytest

from tools.filesystem.read_file_tool import ReadFileTool
from tools.tool_context import ToolContext


def _tool(tmp_path, content="linea1\nlinea2\nlinea3\n"):
    path = tmp_path / "notas.txt"
    path.write_text(content, encoding="utf-8")
    return ReadFileTool(), str(path)


def test_read_without_limit_returns_whole_file(tmp_path) -> None:
    tool, path = _tool(tmp_path)

    assert tool.execute(ToolContext(parameters={"path": path})) == "linea1\nlinea2\nlinea3\n"


def test_read_with_limit_returns_first_lines(tmp_path) -> None:
    tool, path = _tool(tmp_path)

    assert tool.execute(ToolContext(parameters={"path": path, "limit": 2})) == "linea1\nlinea2"


@pytest.mark.parametrize("limit", [0, -1, "2", True])
def test_read_rejects_invalid_limit(tmp_path, limit) -> None:
    tool, path = _tool(tmp_path)

    with pytest.raises(ValueError):
        tool.execute(ToolContext(parameters={"path": path, "limit": limit}))

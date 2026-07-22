from pathlib import Path
from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from core.step_output_reference import StepOutputReference
from tools.base_tool import BaseTool
from tools.executor import ToolExecutor
from tools.filesystem.read_file_tool import ReadFileTool
from tools.registry import (
    ToolAlreadyRegisteredError,
    ToolNotRegisteredError,
    ToolRegistry,
)
from tools.tool_context import ToolContext


class FakeTool(BaseTool):
    def __init__(
        self,
        name: str = "fake.tool",
        description: str = "Fake tool.",
        result: Any = "ok",
    ) -> None:
        self._name = name
        self._description = description
        self._result = result

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(
        self,
        context: ToolContext,
    ) -> Any:
        return self._result


class CapturingTool(FakeTool):
    def __init__(self) -> None:
        super().__init__("capture.tool")
        self.context: ToolContext | None = None

    def execute(
        self,
        context: ToolContext,
    ) -> Any:
        self.context = context
        return context.parameters


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


def test_tool_registry_registers_and_gets_tool() -> None:
    registry = ToolRegistry()
    tool = FakeTool()

    registry.register(tool)

    assert registry.get("fake.tool") is tool


def test_tool_registry_checks_existence() -> None:
    registry = ToolRegistry()

    assert registry.exists("fake.tool") is False

    registry.register(FakeTool())

    assert registry.exists("fake.tool") is True


def test_tool_registry_lists_registered_tool_names() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool("z.tool"))
    registry.register(FakeTool("a.tool"))

    assert registry.list() == ("a.tool", "z.tool")


def test_tool_registry_exposes_descriptors() -> None:
    registry = ToolRegistry()
    tool = FakeTool(description="Tool description.")
    registry.register(tool)

    descriptor = registry.descriptor("fake.tool")

    assert descriptor.name == "fake.tool"
    assert descriptor.description == "Tool description."
    assert descriptor.tool is tool
    assert registry.descriptors() == (descriptor,)


def test_tool_registry_rejects_duplicate_identifiers() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool("fake.tool"))

    with pytest.raises(ToolAlreadyRegisteredError, match="already registered"):
        registry.register(FakeTool("fake.tool"))


def test_tool_registry_raises_clear_error_for_missing_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotRegisteredError, match="Tool 'missing' is not registered"):
        registry.get("missing")


def test_tool_registry_internal_collection_cannot_be_modified_from_outside() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool())
    names = registry.list()
    tools = registry.tools

    names += ("other.tool",)

    assert registry.list() == ("fake.tool",)

    with pytest.raises(TypeError):
        tools["other.tool"] = FakeTool("other.tool")  # type: ignore[index]

    assert registry.exists("other.tool") is False


def test_tool_executor_uses_registry_missing_tool_error() -> None:
    executor = ToolExecutor(ToolRegistry())

    with pytest.raises(ToolNotRegisteredError, match="Tool 'missing' is not registered"):
        executor.execute("missing", ToolContext())


def test_tool_executor_accepts_arguments_without_explicit_context() -> None:
    registry = ToolRegistry()
    tool = CapturingTool()
    registry.register(tool)

    result = ToolExecutor(registry).execute(
        "capture.tool",
        arguments={"query": "is:unread"},
    )

    assert result == {"query": "is:unread"}
    assert tool.context is not None
    assert tool.context.parameters == {"query": "is:unread"}


def test_tool_executor_rejects_unresolved_step_output_reference() -> None:
    registry = ToolRegistry()
    tool = CapturingTool()
    registry.register(tool)

    with pytest.raises(ValueError, match="unresolved StepOutputReference"):
        ToolExecutor(registry).execute(
            "capture.tool",
            arguments={"query": StepOutputReference("read")},
        )

    assert tool.context is None


def test_bootstrap_tool_registry_lists_real_registered_tools() -> None:
    registry = Bootstrap.build_tool_registry()
    names = registry.list()

    assert "read_file" in names
    assert "write_file" in names
    assert "list_directory" in names
    assert "project_tree" in names
    assert "desktop.open_application" in names
    assert "desktop.open_file" in names
    assert "desktop.copy_clipboard_text" in names
    assert "desktop.close_window" in names
    assert len(names) == len(set(names))
    assert all(registry.descriptor(name).description for name in names)

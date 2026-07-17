from __future__ import annotations

from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from tools.base_tool import BaseTool
from tools.intent_selector import (
    ToolIntent,
    ToolIntentAlreadyRegisteredError,
    ToolIntentNotSupportedError,
    ToolIntentRegistry,
    ToolSelector,
)
from tools.registry import ToolNotRegisteredError, ToolRegistry
from tools.tool_context import ToolContext


class FakeTool(BaseTool):
    def __init__(
        self,
        name: str,
        description: str = "Fake tool.",
    ) -> None:
        self._name = name
        self._description = description
        self.executed = False

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
        self.executed = True
        raise AssertionError("selection must not execute tools")


def _selector(
    action: str = "file.read",
    tool_name: str = "read_file",
) -> tuple[ToolSelector, FakeTool, ToolIntentRegistry]:
    tool_registry = ToolRegistry()
    tool = FakeTool(tool_name)
    tool_registry.register(tool)
    intent_registry = ToolIntentRegistry()
    intent_registry.register(action, tool_name)

    return ToolSelector(tool_registry, intent_registry), tool, intent_registry


def test_intent_registry_registers_intent_to_tool_mapping() -> None:
    registry = ToolIntentRegistry()

    registry.register("file.read", "read_file")

    assert registry.resolve("file.read") == "read_file"


@pytest.mark.parametrize(
    ("action", "tool_name"),
    [
        ("file.read", "read_file"),
        ("file.write", "write_file"),
        ("directory.list", "list_directory"),
        ("project.tree", "project_tree"),
        ("desktop.application.open", "desktop.open_application"),
        ("desktop.file.open", "desktop.open_file"),
        ("desktop.text.type", "desktop.type_text"),
        ("desktop.hotkey.press", "desktop.press_hotkey"),
        ("desktop.windows.list", "desktop.list_windows"),
    ],
)
def test_bootstrap_selector_resolves_initial_mappings(
    action: str,
    tool_name: str,
) -> None:
    selector = Bootstrap.build_tool_selector()

    selection = selector.select(
        ToolIntent(
            action=action,
            arguments={"example": action},
        )
    )

    assert selection.intent.action == action
    assert selection.tool_name == tool_name
    assert selection.descriptor.name == tool_name
    assert dict(selection.arguments) == {"example": action}
    assert selection.executed is False


def test_selector_preserves_arguments_without_interpreting_them() -> None:
    selector, _tool, _registry = _selector()
    arguments = {
        "path": "README.md",
        "dangerous_text": "__import__('os').system('echo no')",
    }

    selection = selector.select(ToolIntent("file.read", arguments))

    assert dict(selection.arguments) == arguments


def test_selector_does_not_execute_selected_tool() -> None:
    selector, tool, _registry = _selector()

    selection = selector.select(ToolIntent("file.read", {"path": "README.md"}))

    assert selection.tool_name == "read_file"
    assert selection.executed is False
    assert tool.executed is False


def test_selector_checks_supported_intent() -> None:
    selector, _tool, _registry = _selector()

    assert selector.supports("file.read") is True
    assert selector.supports("file.delete") is False


def test_selector_lists_supported_intents() -> None:
    intent_registry = ToolIntentRegistry()
    intent_registry.register("z.intent", "z_tool")
    intent_registry.register("a.intent", "a_tool")

    assert intent_registry.list() == ("a.intent", "z.intent")


def test_intent_registry_rejects_duplicate_intents() -> None:
    registry = ToolIntentRegistry()
    registry.register("file.read", "read_file")

    with pytest.raises(ToolIntentAlreadyRegisteredError, match="already registered"):
        registry.register("file.read", "other_tool")


def test_selector_raises_clear_error_for_unknown_intent() -> None:
    selector, _tool, _registry = _selector()

    with pytest.raises(
        ToolIntentNotSupportedError,
        match="Tool intent 'file.delete' is not supported",
    ):
        selector.select(ToolIntent("file.delete"))


def test_selector_raises_clear_error_for_missing_registered_tool() -> None:
    intent_registry = ToolIntentRegistry()
    intent_registry.register("file.read", "read_file")
    selector = ToolSelector(
        ToolRegistry(),
        intent_registry,
    )

    with pytest.raises(
        ToolNotRegisteredError,
        match="Tool intent 'file.read' maps to missing tool 'read_file'",
    ):
        selector.select(ToolIntent("file.read"))


def test_intent_registry_internal_mappings_are_encapsulated() -> None:
    registry = ToolIntentRegistry()
    registry.register("file.read", "read_file")
    mappings = registry.mappings
    listed = registry.list()

    listed += ("file.write",)

    assert registry.list() == ("file.read",)

    with pytest.raises(TypeError):
        mappings["file.write"] = "write_file"  # type: ignore[index]

    assert registry.supports("file.write") is False


def test_bootstrap_selector_uses_real_tool_registry_without_execution() -> None:
    selector = Bootstrap.build_tool_selector()

    selection = selector.select(
        ToolIntent(
            "file.read",
            {"path": "README.md"},
        )
    )

    assert selection.tool_name == "read_file"
    assert selection.descriptor.description
    assert dict(selection.arguments) == {"path": "README.md"}
    assert selection.executed is False

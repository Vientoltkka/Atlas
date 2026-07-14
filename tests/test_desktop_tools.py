from pathlib import Path

import pytest

from tools.desktop.desktop_tools import (
    ActivateWindowTool,
    OpenApplicationTool,
    OpenFileTool,
    OpenFolderTool,
    PressHotkeyTool,
    SaveFileTool,
    TypeTextTool,
)
from tools.tool_context import ToolContext


class FakeDesktopController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.windows: set[str] = {"Visual Studio Code"}

    def open_application(self, application: str) -> None:
        self.calls.append(("open_application", application))

    def open_folder(self, path: Path) -> None:
        self.calls.append(("open_folder", path))

    def open_file(
        self,
        path: Path,
        application: str | None = None,
    ) -> None:
        self.calls.append(("open_file", (path, application)))

    def window_exists(self, title: str) -> bool:
        return title in self.windows

    def activate_window(self, title: str) -> None:
        self.calls.append(("activate_window", title))

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", text))

    def press_hotkey(self, keys: list[str]) -> None:
        self.calls.append(("press_hotkey", keys))


def test_open_application_tool() -> None:
    controller = FakeDesktopController()
    tool = OpenApplicationTool(controller)

    result = tool.execute(
        ToolContext(parameters={"application": "Visual Studio Code"})
    )

    assert result == "Visual Studio Code abierto."
    assert controller.calls == [
        ("open_application", "Visual Studio Code")
    ]


def test_open_folder_tool_requires_existing_folder(tmp_path: Path) -> None:
    controller = FakeDesktopController()
    tool = OpenFolderTool(controller)

    result = tool.execute(ToolContext(parameters={"path": str(tmp_path)}))

    assert result == f"Carpeta abierta: {tmp_path}"
    assert controller.calls == [("open_folder", tmp_path)]


def test_open_folder_tool_errors_when_missing(tmp_path: Path) -> None:
    tool = OpenFolderTool(FakeDesktopController())

    with pytest.raises(FileNotFoundError):
        tool.execute(
            ToolContext(parameters={"path": str(tmp_path / "missing")})
        )


def test_open_file_tool_requires_existing_file(tmp_path: Path) -> None:
    file = tmp_path / "router.py"
    file.write_text("print('demo')", encoding="utf-8")
    controller = FakeDesktopController()
    tool = OpenFileTool(controller)

    result = tool.execute(
        ToolContext(
            parameters={
                "path": str(file),
                "application": "Visual Studio Code",
            }
        )
    )

    assert result == f"Archivo abierto en Visual Studio Code: {file}"
    assert controller.calls == [
        ("open_file", (file, "Visual Studio Code"))
    ]


def test_type_text_requires_existing_window() -> None:
    controller = FakeDesktopController()
    controller.windows.clear()
    tool = TypeTextTool(controller)

    with pytest.raises(RuntimeError):
        tool.execute(
            ToolContext(
                parameters={
                    "window_title": "Visual Studio Code",
                    "text": "print('Hola')",
                }
            )
        )


def test_type_text_activates_window_before_writing() -> None:
    controller = FakeDesktopController()
    tool = TypeTextTool(controller)

    result = tool.execute(
        ToolContext(
            parameters={
                "window_title": "Visual Studio Code",
                "text": "print('Hola')",
            }
        )
    )

    assert result == "Texto escrito."
    assert controller.calls == [
        ("activate_window", "Visual Studio Code"),
        ("type_text", "print('Hola')"),
    ]


def test_save_file_sends_ctrl_s_to_existing_window() -> None:
    controller = FakeDesktopController()
    tool = SaveFileTool(controller)

    result = tool.execute(
        ToolContext(parameters={"window_title": "Visual Studio Code"})
    )

    assert result == "Archivo guardado."
    assert controller.calls == [
        ("activate_window", "Visual Studio Code"),
        ("press_hotkey", ["ctrl", "s"]),
    ]


def test_press_hotkey_sends_keys_to_existing_window() -> None:
    controller = FakeDesktopController()
    tool = PressHotkeyTool(controller)

    result = tool.execute(
        ToolContext(
            parameters={
                "window_title": "Visual Studio Code",
                "keys": ["ctrl", "s"],
            }
        )
    )

    assert result == "Atajo enviado."
    assert controller.calls == [
        ("activate_window", "Visual Studio Code"),
        ("press_hotkey", ["ctrl", "s"]),
    ]


def test_activate_window_tool() -> None:
    controller = FakeDesktopController()
    tool = ActivateWindowTool(controller)

    result = tool.execute(
        ToolContext(parameters={"window_title": "Visual Studio Code"})
    )

    assert result == "Ventana activada: Visual Studio Code"
    assert controller.calls == [
        ("activate_window", "Visual Studio Code")
    ]


from pathlib import Path

import pytest

from tools.desktop.desktop_tools import (
    ActivateWindowTool,
    BringWindowToFrontTool,
    CaptureScreenshotTool,
    ClearClipboardTool,
    ClipboardHasTextTool,
    CloseApplicationTool,
    CloseWindowTool,
    CopyClipboardTextTool,
    DoubleClickTool,
    GetCursorPositionTool,
    GetForegroundWindowTool,
    GetProcessTool,
    GetScreenSizeTool,
    GetWindowRectTool,
    LeftClickTool,
    IsProcessRunningTool,
    ListProcessesTool,
    ListWindowsTool,
    MaximizeWindowTool,
    MinimizeWindowTool,
    MoveCursorTool,
    MoveResizeWindowTool,
    MoveWindowTool,
    OpenApplicationTool,
    OpenFileTool,
    OpenFolderTool,
    PasteClipboardTool,
    PressHotkeyTool,
    ReadClipboardTextTool,
    ResizeWindowTool,
    RestoreWindowTool,
    RightClickTool,
    SaveFileTool,
    ScrollVerticalTool,
    TerminateProcessTool,
    TypeTextTool,
)
from tools.desktop.windows_controller import ProcessInfo, WindowsDesktopController
from tools.tool_context import ToolContext


class FakeDesktopController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.windows: set[str] = {"Visual Studio Code"}
        self.screen_size = (1000, 800)
        self.cursor_position = (12, 34)
        self.fail_capture = False
        self.virtual_desktop = (0, 0, 1920, 1080)
        self.clipboard_text: str | None = "Hola Atlas"
        self.next_pid = 500
        self.processes: list[dict[str, object]] = []
        self.existing_pids: set[int] = set()
        self.window_matches = [
            {
                "handle": 10,
                "title": "Visual Studio Code - Atlas",
                "rect": (20, 30, 820, 630),
                "visible": True,
            },
            {
                "handle": 11,
                "title": "Visual Studio Code - Hidden",
                "rect": (0, 0, 100, 100),
                "visible": False,
            },
        ]

    def open_application(self, application: str) -> int:
        self.calls.append(("open_application", application))
        return self.next_pid

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

    def copy_clipboard_text(self, text: str) -> int:
        self.calls.append(("copy_clipboard_text", text))
        self.clipboard_text = text
        return len(text)

    def read_clipboard_text(self) -> str | None:
        self.calls.append(("read_clipboard_text", None))
        return self.clipboard_text

    def clear_clipboard(self) -> None:
        self.calls.append(("clear_clipboard", None))
        self.clipboard_text = None

    def clipboard_has_text(self) -> bool:
        self.calls.append(("clipboard_has_text", None))
        return self.clipboard_text is not None

    def press_hotkey(self, keys: list[str]) -> None:
        self.calls.append(("press_hotkey", keys))

    def get_screen_size(self) -> tuple[int, int]:
        self.calls.append(("get_screen_size", None))
        return self.screen_size

    def get_virtual_desktop_rect(self) -> tuple[int, int, int, int]:
        self.calls.append(("get_virtual_desktop_rect", None))
        return self.virtual_desktop

    def get_cursor_position(self) -> tuple[int, int]:
        self.calls.append(("get_cursor_position", None))
        return self.cursor_position

    def move_cursor(self, x: int, y: int) -> None:
        self.calls.append(("move_cursor", (x, y)))

    def left_click(self, x: int, y: int) -> None:
        self.calls.append(("left_click", (x, y)))

    def double_click(self, x: int, y: int) -> None:
        self.calls.append(("double_click", (x, y)))

    def right_click(self, x: int, y: int) -> None:
        self.calls.append(("right_click", (x, y)))

    def scroll_vertical(self, amount: int) -> None:
        self.calls.append(("scroll_vertical", amount))

    def capture_screen(self, path: Path) -> None:
        self.calls.append(("capture_screen", path))

        if self.fail_capture:
            raise RuntimeError("capture failed")

        path.write_bytes(b"\x89PNG\r\n\x1a\n")

    def list_windows(self, title: str) -> list[dict[str, object]]:
        self.calls.append(("list_windows", title))
        return [
            {
                "handle": window["handle"],
                "title": window["title"],
                "rect": window["rect"],
            }
            for window in self.window_matches
            if window["visible"] and title.lower() in str(window["title"]).lower()
        ]

    def get_window_rect(self, handle: int) -> tuple[int, int, int, int]:
        self.calls.append(("get_window_rect", handle))

        for window in self.window_matches:
            if window["handle"] == handle:
                return window["rect"]

        raise RuntimeError("missing")

    def get_foreground_window(self) -> dict[str, object]:
        self.calls.append(("get_foreground_window", None))
        return {
            "handle": 10,
            "title": "Visual Studio Code - Atlas",
            "rect": (20, 30, 820, 630),
        }

    def bring_window_to_front(self, handle: int) -> None:
        self.calls.append(("bring_window_to_front", handle))

    def maximize_window(self, handle: int) -> None:
        self.calls.append(("maximize_window", handle))

    def minimize_window(self, handle: int) -> None:
        self.calls.append(("minimize_window", handle))

    def restore_window(self, handle: int) -> None:
        self.calls.append(("restore_window", handle))

    def move_window(self, handle: int, x: int, y: int) -> None:
        self.calls.append(("move_window", (handle, x, y)))

    def resize_window(self, handle: int, width: int, height: int) -> None:
        self.calls.append(("resize_window", (handle, width, height)))

    def move_resize_window(
        self,
        handle: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        self.calls.append(("move_resize_window", (handle, x, y, width, height)))

    def close_window(self, handle: int) -> None:
        self.calls.append(("close_window", handle))

    def list_processes(self, query: str):
        self.calls.append(("list_processes", query))
        normalized = query.lower()
        aliases = {
            "visual studio code": "code",
            "vs code": "code",
            "vscode": "code",
        }
        normalized = aliases.get(normalized, normalized)
        matches = [
            process
            for process in self.processes
            if normalized in str(process.name).lower()
            or normalized in str(process.name).lower().replace(".exe", "")
        ]
        return sorted(matches, key=lambda process: (process.name.lower(), process.pid))

    def process_exists(self, pid: int) -> bool:
        self.calls.append(("process_exists", pid))
        return pid in self.existing_pids

    def get_process(self, pid: int):
        self.calls.append(("get_process", pid))

        for process in self.processes:
            if process.pid == pid:
                return process

        return None

    def close_process_windows(self, pid: int) -> int:
        self.calls.append(("close_process_windows", pid))
        return 1

    def terminate_process(self, pid: int) -> None:
        self.calls.append(("terminate_process", pid))
        self.existing_pids.discard(pid)


def test_open_application_tool() -> None:
    controller = FakeDesktopController()
    tool = OpenApplicationTool(controller)

    result = tool.execute(
        ToolContext(parameters={"application": "Visual Studio Code"})
    )

    assert result == "Visual Studio Code abierto. PID: 500"
    assert controller.calls == [
        ("list_processes", "Visual Studio Code"),
        ("open_application", "Visual Studio Code")
    ]


def test_open_application_tool_reports_existing_process() -> None:
    controller = FakeDesktopController()
    controller.processes = [
        ProcessInfo(10, "Code.exe", None, ("Atlas",), True),
    ]
    tool = OpenApplicationTool(controller)

    result = tool.execute(
        ToolContext(parameters={"application": "Visual Studio Code"})
    )

    assert result == "Visual Studio Code ya estaba abierto. PID: 10"
    assert controller.calls == [("list_processes", "Visual Studio Code")]


def test_list_processes_tool_returns_plain_structures() -> None:
    controller = FakeDesktopController()
    controller.processes = [
        ProcessInfo(20, "chrome.exe", None, ("Chrome",), True),
        ProcessInfo(10, "chrome.exe", None, (), True),
    ]
    tool = ListProcessesTool(controller)

    result = tool.execute(ToolContext(parameters={"query": "chrome"}))

    assert result == [
        {
            "pid": 10,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        },
        {
            "pid": 20,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": ("Chrome",),
            "is_running": True,
        },
    ]


def test_is_process_running_tool() -> None:
    controller = FakeDesktopController()
    tool = IsProcessRunningTool(controller)

    assert tool.execute(ToolContext(parameters={"query": "chrome"})) is False

    controller.processes = [
        ProcessInfo(20, "chrome.exe", None, (), True),
    ]

    assert tool.execute(ToolContext(parameters={"query": "chrome"})) is True


def test_get_process_tool_returns_process_by_pid() -> None:
    controller = FakeDesktopController()
    controller.processes = [
        ProcessInfo(20, "chrome.exe", None, ("Chrome",), True),
    ]
    tool = GetProcessTool(controller)

    result = tool.execute(ToolContext(parameters={"pid": 20}))

    assert result == {
        "pid": 20,
        "name": "chrome.exe",
        "executable_path": None,
        "window_titles": ("Chrome",),
        "is_running": True,
    }


def test_close_application_tool_sends_normal_close() -> None:
    controller = FakeDesktopController()
    tool = CloseApplicationTool(controller)

    result = tool.execute(ToolContext(parameters={"pid": 20}))

    assert result == "Solicitud de cierre enviada a 1 ventana(s)."
    assert controller.calls == [("close_process_windows", 20)]


def test_terminate_process_tool_rejects_pid_zero_and_four() -> None:
    tool = TerminateProcessTool(FakeDesktopController())

    with pytest.raises(ValueError):
        tool.execute(ToolContext(parameters={"pid": 0}))

    with pytest.raises(ValueError):
        tool.execute(ToolContext(parameters={"pid": 4}))


def test_terminate_process_tool_blocks_protected_process() -> None:
    controller = FakeDesktopController()
    controller.processes = [
        ProcessInfo(88, "lsass.exe", None, (), True),
    ]
    tool = TerminateProcessTool(controller)

    with pytest.raises(ValueError):
        tool.execute(ToolContext(parameters={"pid": 88}))


def test_terminate_process_tool_terminates_after_lookup_and_verification() -> None:
    controller = FakeDesktopController()
    controller.processes = [
        ProcessInfo(99, "example.exe", None, (), True),
    ]
    controller.existing_pids = {99}
    tool = TerminateProcessTool(controller)

    result = tool.execute(ToolContext(parameters={"pid": 99}))

    assert result == "Proceso terminado: example.exe - PID 99"
    assert controller.calls == [
        ("get_process", 99),
        ("close_process_windows", 99),
        ("terminate_process", 99),
        ("process_exists", 99),
    ]


def test_terminate_process_tool_reports_still_running() -> None:
    class StickyController(FakeDesktopController):
        def terminate_process(self, pid: int) -> None:
            self.calls.append(("terminate_process", pid))

    controller = StickyController()
    controller.processes = [
        ProcessInfo(99, "example.exe", None, (), True),
    ]
    controller.existing_pids = {99}
    tool = TerminateProcessTool(controller)

    with pytest.raises(RuntimeError):
        tool.execute(ToolContext(parameters={"pid": 99}))


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


def test_copy_clipboard_text_tool_writes_unicode_text() -> None:
    controller = FakeDesktopController()
    tool = CopyClipboardTextTool(controller)

    result = tool.execute(ToolContext(parameters={"text": "Hola Ã¡Ã±\nAtlas"}))

    assert result == 15
    assert controller.clipboard_text == "Hola Ã¡Ã±\nAtlas"
    assert controller.calls == [("copy_clipboard_text", "Hola Ã¡Ã±\nAtlas")]


def test_copy_clipboard_text_tool_rejects_non_str() -> None:
    tool = CopyClipboardTextTool(FakeDesktopController())

    with pytest.raises(TypeError):
        tool.execute(ToolContext(parameters={"text": 123}))


def test_read_clipboard_text_tool_returns_text_or_none() -> None:
    controller = FakeDesktopController()
    tool = ReadClipboardTextTool(controller)

    assert tool.execute(ToolContext()) == "Hola Atlas"
    controller.clipboard_text = None
    assert tool.execute(ToolContext()) is None


def test_clear_clipboard_tool_clears_text() -> None:
    controller = FakeDesktopController()
    tool = ClearClipboardTool(controller)

    result = tool.execute(ToolContext())

    assert result == "Portapapeles vaciado."
    assert controller.clipboard_text is None
    assert controller.calls == [("clear_clipboard", None)]


def test_clipboard_has_text_tool() -> None:
    controller = FakeDesktopController()
    tool = ClipboardHasTextTool(controller)

    assert tool.execute(ToolContext()) is True
    controller.clipboard_text = None
    assert tool.execute(ToolContext()) is False


def test_paste_clipboard_tool_uses_ctrl_v_not_type_text() -> None:
    controller = FakeDesktopController()
    tool = PasteClipboardTool(controller)

    result = tool.execute(
        ToolContext(parameters={"window_title": "Visual Studio Code"})
    )

    assert result == "Contenido pegado."
    assert controller.calls == [
        ("clipboard_has_text", None),
        ("activate_window", "Visual Studio Code"),
        ("press_hotkey", ["ctrl", "v"]),
    ]
    assert all(call[0] != "type_text" for call in controller.calls)


def test_paste_clipboard_tool_rejects_clipboard_without_text() -> None:
    controller = FakeDesktopController()
    controller.clipboard_text = None
    tool = PasteClipboardTool(controller)

    with pytest.raises(RuntimeError):
        tool.execute(ToolContext(parameters={"window_title": "Visual Studio Code"}))

    assert controller.calls == [("clipboard_has_text", None)]


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


def test_get_screen_size_tool() -> None:
    controller = FakeDesktopController()
    tool = GetScreenSizeTool(controller)

    result = tool.execute(ToolContext())

    assert result == (1000, 800)


def test_get_cursor_position_tool() -> None:
    controller = FakeDesktopController()
    tool = GetCursorPositionTool(controller)

    result = tool.execute(ToolContext())

    assert result == (12, 34)


def test_move_cursor_tool_accepts_valid_coordinates() -> None:
    controller = FakeDesktopController()
    tool = MoveCursorTool(controller)

    result = tool.execute(ToolContext(parameters={"x": 500, "y": 300}))

    assert result == "Cursor movido a (500, 300)."
    assert controller.calls == [
        ("get_screen_size", None),
        ("move_cursor", (500, 300)),
    ]


def test_move_cursor_tool_rejects_negative_coordinates() -> None:
    controller = FakeDesktopController()
    tool = MoveCursorTool(controller)

    with pytest.raises(ValueError):
        tool.execute(ToolContext(parameters={"x": -1, "y": 300}))

    assert controller.calls == []


def test_move_cursor_tool_rejects_out_of_screen_coordinates() -> None:
    controller = FakeDesktopController()
    tool = MoveCursorTool(controller)

    with pytest.raises(ValueError):
        tool.execute(ToolContext(parameters={"x": 99999, "y": 99999}))

    assert controller.calls == [("get_screen_size", None)]


def test_move_cursor_tool_rejects_non_numeric_coordinates() -> None:
    controller = FakeDesktopController()
    tool = MoveCursorTool(controller)

    with pytest.raises(ValueError):
        tool.execute(ToolContext(parameters={"x": "abc", "y": 300}))

    assert controller.calls == []


def test_left_click_tool_uses_valid_coordinates() -> None:
    controller = FakeDesktopController()
    tool = LeftClickTool(controller)

    result = tool.execute(ToolContext(parameters={"x": 500, "y": 300}))

    assert result == "Clic realizado en (500, 300)."
    assert controller.calls == [
        ("get_screen_size", None),
        ("left_click", (500, 300)),
    ]


def test_double_click_tool_uses_valid_coordinates() -> None:
    controller = FakeDesktopController()
    tool = DoubleClickTool(controller)

    result = tool.execute(ToolContext(parameters={"x": 500, "y": 300}))

    assert result == "Doble clic realizado en (500, 300)."
    assert controller.calls == [
        ("get_screen_size", None),
        ("double_click", (500, 300)),
    ]


def test_right_click_tool_uses_valid_coordinates() -> None:
    controller = FakeDesktopController()
    tool = RightClickTool(controller)

    result = tool.execute(ToolContext(parameters={"x": 500, "y": 300}))

    assert result == "Clic derecho realizado en (500, 300)."
    assert controller.calls == [
        ("get_screen_size", None),
        ("right_click", (500, 300)),
    ]


def test_scroll_vertical_tool_down() -> None:
    controller = FakeDesktopController()
    tool = ScrollVerticalTool(controller)

    result = tool.execute(ToolContext(parameters={"direction": "down"}))

    assert result == "Scroll hacia abajo."
    assert controller.calls == [("scroll_vertical", -120)]


def test_scroll_vertical_tool_up() -> None:
    controller = FakeDesktopController()
    tool = ScrollVerticalTool(controller)

    result = tool.execute(ToolContext(parameters={"direction": "up"}))

    assert result == "Scroll hacia arriba."
    assert controller.calls == [("scroll_vertical", 120)]


def test_capture_screenshot_tool_creates_folder_and_png(
    tmp_path: Path,
) -> None:
    controller = FakeDesktopController()
    tool = CaptureScreenshotTool(controller)
    output_dir = tmp_path / "artifacts" / "screenshots"

    result = tool.execute(
        ToolContext(parameters={"output_dir": str(output_dir)})
    )

    path = Path(result)
    assert path.suffix == ".png"
    assert path.exists()
    assert output_dir.exists()
    assert path.read_bytes().startswith(b"\x89PNG")


def test_capture_screenshot_tool_does_not_overwrite_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDateTime:
        @classmethod
        def now(cls):
            class FixedNow:
                def strftime(self, _: str) -> str:
                    return "20260714_101530"

            return FixedNow()

    monkeypatch.setattr(
        "tools.desktop.desktop_tools.datetime",
        FixedDateTime,
    )
    controller = FakeDesktopController()
    tool = CaptureScreenshotTool(controller)
    output_dir = tmp_path / "screenshots"
    output_dir.mkdir()
    existing = output_dir / "screenshot_20260714_101530.png"
    existing.write_bytes(b"existing")

    result = tool.execute(
        ToolContext(parameters={"output_dir": str(output_dir)})
    )

    assert Path(result).name == "screenshot_20260714_101530_1.png"
    assert existing.read_bytes() == b"existing"


def test_capture_screenshot_tool_reports_save_failure(
    tmp_path: Path,
) -> None:
    controller = FakeDesktopController()
    controller.fail_capture = True
    tool = CaptureScreenshotTool(controller)

    with pytest.raises(RuntimeError):
        tool.execute(
            ToolContext(parameters={"output_dir": str(tmp_path)})
        )


def test_list_windows_tool_lists_visible_matches() -> None:
    controller = FakeDesktopController()
    tool = ListWindowsTool(controller)

    result = tool.execute(ToolContext(parameters={"title": "code"}))

    assert result == [
        {
            "handle": 10,
            "title": "Visual Studio Code - Atlas",
            "rect": (20, 30, 820, 630),
        }
    ]


def test_list_windows_tool_ignores_non_matching_or_hidden_windows() -> None:
    controller = FakeDesktopController()
    tool = ListWindowsTool(controller)

    result = tool.execute(ToolContext(parameters={"title": "Hidden"}))

    assert result == []


def test_get_window_rect_tool() -> None:
    controller = FakeDesktopController()
    tool = GetWindowRectTool(controller)

    result = tool.execute(ToolContext(parameters={"handle": 10}))

    assert result == (20, 30, 820, 630)


def test_get_foreground_window_tool() -> None:
    controller = FakeDesktopController()
    tool = GetForegroundWindowTool(controller)

    result = tool.execute(ToolContext())

    assert result == {
        "handle": 10,
        "title": "Visual Studio Code - Atlas",
        "rect": (20, 30, 820, 630),
    }


def test_bring_window_to_front_tool() -> None:
    controller = FakeDesktopController()
    tool = BringWindowToFrontTool(controller)

    result = tool.execute(ToolContext(parameters={"handle": 10}))

    assert result == "Ventana activada."
    assert controller.calls == [("bring_window_to_front", 10)]


def test_maximize_window_tool() -> None:
    controller = FakeDesktopController()
    tool = MaximizeWindowTool(controller)

    result = tool.execute(ToolContext(parameters={"handle": 10}))

    assert result == "Ventana maximizada."
    assert controller.calls == [("maximize_window", 10)]


def test_minimize_window_tool() -> None:
    controller = FakeDesktopController()
    tool = MinimizeWindowTool(controller)

    result = tool.execute(ToolContext(parameters={"handle": 10}))

    assert result == "Ventana minimizada."
    assert controller.calls == [("minimize_window", 10)]


def test_restore_window_tool() -> None:
    controller = FakeDesktopController()
    tool = RestoreWindowTool(controller)

    result = tool.execute(ToolContext(parameters={"handle": 10}))

    assert result == "Ventana restaurada."
    assert controller.calls == [("restore_window", 10)]


def test_move_window_tool() -> None:
    controller = FakeDesktopController()
    tool = MoveWindowTool(controller)

    result = tool.execute(
        ToolContext(parameters={"handle": 10, "x": 100, "y": 100})
    )

    assert result == "Ventana movida a (100, 100)."
    assert controller.calls == [
        ("get_window_rect", 10),
        ("get_virtual_desktop_rect", None),
        ("move_window", (10, 100, 100)),
    ]


def test_move_window_tool_rejects_invalid_coordinates() -> None:
    controller = FakeDesktopController()
    tool = MoveWindowTool(controller)

    with pytest.raises(ValueError):
        tool.execute(
            ToolContext(parameters={"handle": 10, "x": "abc", "y": 100})
        )


def test_move_window_tool_rejects_position_outside_virtual_desktop() -> None:
    controller = FakeDesktopController()
    tool = MoveWindowTool(controller)

    with pytest.raises(ValueError):
        tool.execute(
            ToolContext(parameters={"handle": 10, "x": 99999, "y": 99999})
        )


def test_resize_window_tool() -> None:
    controller = FakeDesktopController()
    tool = ResizeWindowTool(controller)

    result = tool.execute(
        ToolContext(parameters={"handle": 10, "width": 1200, "height": 800})
    )

    assert result == "Ventana redimensionada a 1200 x 800."
    assert controller.calls == [("resize_window", (10, 1200, 800))]


def test_resize_window_tool_rejects_zero_dimension() -> None:
    controller = FakeDesktopController()
    tool = ResizeWindowTool(controller)

    with pytest.raises(ValueError):
        tool.execute(
            ToolContext(parameters={"handle": 10, "width": 0, "height": 800})
        )


def test_resize_window_tool_rejects_negative_dimension() -> None:
    controller = FakeDesktopController()
    tool = ResizeWindowTool(controller)

    with pytest.raises(ValueError):
        tool.execute(
            ToolContext(parameters={"handle": 10, "width": -1, "height": 800})
        )


def test_move_resize_window_tool() -> None:
    controller = FakeDesktopController()
    tool = MoveResizeWindowTool(controller)

    result = tool.execute(
        ToolContext(
            parameters={
                "handle": 10,
                "x": 100,
                "y": 100,
                "width": 1200,
                "height": 800,
            }
        )
    )

    assert result == (
        "Ventana movida a (100, 100) "
        "y redimensionada a 1200 x 800."
    )
    assert controller.calls == [
        ("get_virtual_desktop_rect", None),
        ("move_resize_window", (10, 100, 100, 1200, 800)),
    ]


def test_close_window_tool_requests_close_without_killing_process() -> None:
    controller = FakeDesktopController()
    tool = CloseWindowTool(controller)

    result = tool.execute(ToolContext(parameters={"handle": 10}))

    assert result == "Solicitud de cierre enviada."
    assert controller.calls == [("close_window", 10)]


class FakeUser32Clipboard:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.open_ok = True
        self.close_ok = True
        self.empty_ok = True
        self.set_ok = True
        self.has_text = True
        self.read_handle = 700

    def OpenClipboard(self, handle):
        self.calls.append(("OpenClipboard", handle))
        return self.open_ok

    def CloseClipboard(self):
        self.calls.append(("CloseClipboard", None))
        return self.close_ok

    def EmptyClipboard(self):
        self.calls.append(("EmptyClipboard", None))
        return self.empty_ok

    def SetClipboardData(self, clipboard_format, handle):
        self.calls.append(("SetClipboardData", (clipboard_format, handle)))
        return handle if self.set_ok else 0

    def IsClipboardFormatAvailable(self, clipboard_format):
        self.calls.append(("IsClipboardFormatAvailable", clipboard_format))
        return self.has_text

    def GetClipboardData(self, clipboard_format):
        self.calls.append(("GetClipboardData", clipboard_format))
        return self.read_handle


class FakeKernel32Clipboard:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.alloc_ok = True
        self.lock_ok = True
        self.handle = 900
        self.pointer = 1200

    def GlobalAlloc(self, flags, size):
        self.calls.append(("GlobalAlloc", (flags, size)))
        return self.handle if self.alloc_ok else 0

    def GlobalLock(self, handle):
        self.calls.append(("GlobalLock", handle))
        return self.pointer if self.lock_ok else 0

    def GlobalUnlock(self, handle):
        self.calls.append(("GlobalUnlock", handle))
        return 1

    def GlobalFree(self, handle):
        self.calls.append(("GlobalFree", handle))
        return 0


def _fake_native_clipboard_controller() -> tuple[
    WindowsDesktopController,
    FakeUser32Clipboard,
    FakeKernel32Clipboard,
]:
    controller = WindowsDesktopController(max_clipboard_text_chars=20)
    user32 = FakeUser32Clipboard()
    kernel32 = FakeKernel32Clipboard()
    controller._USER32 = user32
    controller._KERNEL32 = kernel32

    return controller, user32, kernel32


def test_windows_controller_copy_writes_unicode_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, user32, kernel32 = _fake_native_clipboard_controller()
    copied: list[tuple[int, bytes, int]] = []

    def fake_memmove(pointer, data, size):
        copied.append((pointer, bytes(data), size))
        return pointer

    monkeypatch.setattr(
        "tools.desktop.windows_controller.ctypes.memmove",
        fake_memmove,
    )

    result = controller.copy_clipboard_text("Hola\nAtlas")

    assert result == 10
    assert copied == [
        (1200, "Hola\nAtlas".encode("utf-16-le") + b"\x00\x00", 22)
    ]
    assert user32.calls == [
        ("OpenClipboard", None),
        ("EmptyClipboard", None),
        ("SetClipboardData", (13, 900)),
        ("CloseClipboard", None),
    ]
    assert ("GlobalFree", 900) not in kernel32.calls


def test_windows_controller_rejects_empty_clipboard_copy() -> None:
    controller, user32, _ = _fake_native_clipboard_controller()

    with pytest.raises(ValueError):
        controller.copy_clipboard_text("")

    assert user32.calls == []


def test_windows_controller_rejects_too_large_clipboard_copy() -> None:
    controller, user32, _ = _fake_native_clipboard_controller()

    with pytest.raises(ValueError):
        controller.copy_clipboard_text("x" * 21)

    assert user32.calls == []


def test_windows_controller_closes_clipboard_after_copy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, user32, kernel32 = _fake_native_clipboard_controller()
    user32.set_ok = False
    monkeypatch.setattr(
        "tools.desktop.windows_controller.ctypes.memmove",
        lambda pointer, data, size: pointer,
    )

    with pytest.raises(RuntimeError, match="No se pudo escribir"):
        controller.copy_clipboard_text("Hola")

    assert ("CloseClipboard", None) in user32.calls
    assert ("GlobalFree", 900) in kernel32.calls


def test_windows_controller_handles_open_clipboard_failure() -> None:
    controller, user32, _ = _fake_native_clipboard_controller()
    user32.open_ok = False

    with pytest.raises(RuntimeError, match="No se pudo abrir"):
        controller.copy_clipboard_text("Hola")

    assert user32.calls == [("OpenClipboard", None)]


def test_windows_controller_handles_global_alloc_failure() -> None:
    controller, user32, kernel32 = _fake_native_clipboard_controller()
    kernel32.alloc_ok = False

    with pytest.raises(RuntimeError, match="reservar memoria"):
        controller.copy_clipboard_text("Hola")

    assert ("EmptyClipboard", None) not in user32.calls
    assert ("CloseClipboard", None) in user32.calls


def test_windows_controller_handles_global_lock_failure() -> None:
    controller, user32, kernel32 = _fake_native_clipboard_controller()
    kernel32.lock_ok = False

    with pytest.raises(RuntimeError, match="bloquear memoria"):
        controller.copy_clipboard_text("Hola")

    assert ("EmptyClipboard", None) not in user32.calls
    assert ("GlobalFree", 900) in kernel32.calls
    assert ("CloseClipboard", None) in user32.calls


def test_windows_controller_read_returns_none_without_unicode_text() -> None:
    controller, user32, _ = _fake_native_clipboard_controller()
    user32.has_text = False

    result = controller.read_clipboard_text()

    assert result is None
    assert user32.calls == [
        ("OpenClipboard", None),
        ("IsClipboardFormatAvailable", 13),
        ("CloseClipboard", None),
    ]


def test_windows_controller_read_unicode_text_and_unlocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, user32, kernel32 = _fake_native_clipboard_controller()
    monkeypatch.setattr(
        "tools.desktop.windows_controller.ctypes.wstring_at",
        lambda pointer: "Hola\nAtlas",
    )

    result = controller.read_clipboard_text()

    assert result == "Hola\nAtlas"
    assert user32.calls == [
        ("OpenClipboard", None),
        ("IsClipboardFormatAvailable", 13),
        ("GetClipboardData", 13),
        ("CloseClipboard", None),
    ]
    assert kernel32.calls == [
        ("GlobalLock", 700),
        ("GlobalUnlock", 700),
    ]


def test_windows_controller_clear_clipboard_closes() -> None:
    controller, user32, _ = _fake_native_clipboard_controller()

    controller.clear_clipboard()

    assert user32.calls == [
        ("OpenClipboard", None),
        ("EmptyClipboard", None),
        ("CloseClipboard", None),
    ]


def test_windows_controller_clipboard_has_text_uses_unicode_format() -> None:
    controller, user32, _ = _fake_native_clipboard_controller()

    assert controller.clipboard_has_text() is True

    assert user32.calls == [("IsClipboardFormatAvailable", 13)]


def test_windows_controller_open_application_uses_popen_without_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "demo.exe"
    executable.write_text("", encoding="utf-8")
    calls: list[dict[str, object]] = []

    class FakePopen:
        pid = 1234

        def __init__(self, args, stdout, stderr, shell=False):
            calls.append(
                {
                    "args": args,
                    "stdout": stdout,
                    "stderr": stderr,
                    "shell": shell,
                }
            )

    monkeypatch.setattr(
        "tools.desktop.windows_controller.subprocess.Popen",
        FakePopen,
    )
    controller = WindowsDesktopController()

    result = controller.open_application(str(executable))

    assert result == 1234
    assert calls[0]["args"] == [str(executable)]
    assert calls[0]["shell"] is False


def test_windows_controller_rejects_arbitrary_application_command() -> None:
    controller = WindowsDesktopController()

    with pytest.raises(ValueError):
        controller.open_application("cmd /c dir")


def test_windows_controller_list_processes_uses_tasklist_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class Completed:
        returncode = 0
        stdout = (
            '"Image Name","PID","Session Name","Session#","Mem Usage",'
            '"Status","User Name","CPU Time","Window Title"\n'
            '"Code.exe","20","Console","1","10 K","Running",'
            '"User","0:00:00","Atlas - Visual Studio Code"\n'
            '"Code.exe","10","Console","1","10 K","Running",'
            '"User","0:00:00","N/A"\n'
        )
        stderr = ""

    def fake_run(args, capture_output, text, errors=None, shell=False):
        calls.append(
            {
                "args": args,
                "capture_output": capture_output,
                "text": text,
                "errors": errors,
                "shell": shell,
            }
        )
        return Completed()

    monkeypatch.setattr(
        "tools.desktop.windows_controller.subprocess.run",
        fake_run,
    )
    controller = WindowsDesktopController()

    result = controller.list_processes("Visual Studio Code")

    assert [process.pid for process in result] == [10, 20]
    assert result[1].window_titles == ("Atlas - Visual Studio Code",)
    assert calls[0]["args"] == ["tasklist", "/FO", "CSV", "/V"]
    assert calls[0]["shell"] is False


def test_windows_controller_list_processes_falls_back_when_verbose_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class Denied:
        returncode = 1
        stdout = ""
        stderr = "Error: Acceso denegado"

    class Completed:
        returncode = 0
        stdout = (
            '"Image Name","PID","Session Name","Session#","Mem Usage"\n'
            '"Code.exe","20","Console","1","10 K"\n'
        )
        stderr = ""

    def fake_run(args, capture_output, text, errors=None, shell=False):
        calls.append(args)
        return Denied() if len(calls) == 1 else Completed()

    monkeypatch.setattr(
        "tools.desktop.windows_controller.subprocess.run",
        fake_run,
    )
    controller = WindowsDesktopController()

    result = controller.list_processes("Code")

    assert [process.pid for process in result] == [20]
    assert calls == [
        ["tasklist", "/FO", "CSV", "/V"],
        ["tasklist", "/FO", "CSV"],
    ]


def test_windows_controller_terminate_process_uses_taskkill_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, capture_output, text, shell=False):
        calls.append(
            {
                "args": args,
                "capture_output": capture_output,
                "text": text,
                "shell": shell,
            }
        )
        return Completed()

    monkeypatch.setattr(
        "tools.desktop.windows_controller.subprocess.run",
        fake_run,
    )
    controller = WindowsDesktopController()

    controller.terminate_process(1234)

    assert calls[0]["args"] == ["taskkill", "/PID", "1234", "/F"]
    assert calls[0]["shell"] is False

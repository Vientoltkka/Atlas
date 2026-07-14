"""Desktop system tools."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

from tools.base_tool import BaseTool
from tools.desktop.windows_controller import (
    DesktopController,
    ProcessInfo,
    WindowsDesktopController,
)
from tools.tool_context import ToolContext


class DesktopTool(BaseTool):
    """Base class for desktop tools."""

    def __init__(
        self,
        controller: DesktopController | None = None,
    ) -> None:
        self._controller = controller or WindowsDesktopController()

    def _window_title(self, context: ToolContext) -> str:
        """Return a required target window title."""
        title = context.parameters.get("window_title")

        if not title:
            raise ValueError("Missing parameter 'window_title'.")

        if not self._controller.window_exists(str(title)):
            raise RuntimeError(f"No existe la ventana destino '{title}'.")

        return str(title)

    def _coordinates(self, context: ToolContext) -> tuple[int, int]:
        """Return validated absolute screen coordinates."""
        x = context.parameters.get("x")
        y = context.parameters.get("y")

        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError("Las coordenadas deben ser numericas.")

        if x < 0 or y < 0:
            raise ValueError("Las coordenadas no pueden ser negativas.")

        width, height = self._controller.get_screen_size()

        if x >= width or y >= height:
            raise ValueError(
                f"Coordenadas fuera de pantalla: ({x}, {y})."
            )

        return x, y

    def _window_handle(self, context: ToolContext) -> int:
        """Return a required window handle."""
        handle = context.parameters.get("handle")

        if not isinstance(handle, int) or handle <= 0:
            raise ValueError("Missing or invalid parameter 'handle'.")

        return handle

    def _window_rect(
        self,
        context: ToolContext,
    ) -> tuple[int, int, int, int]:
        """Return validated window geometry."""
        x = context.parameters.get("x")
        y = context.parameters.get("y")
        width = context.parameters.get("width")
        height = context.parameters.get("height")

        if not all(isinstance(value, int) for value in (x, y, width, height)):
            raise ValueError("La geometria de ventana debe ser numerica.")

        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(width, int)
        assert isinstance(height, int)

        if width <= 0 or height <= 0:
            raise ValueError("El ancho y el alto deben ser mayores que cero.")

        if width > 100000 or height > 100000:
            raise ValueError("Dimensiones de ventana absurdas.")

        self._validate_virtual_desktop_intersection(x, y, width, height)

        return x, y, width, height

    def _window_size(
        self,
        context: ToolContext,
    ) -> tuple[int, int]:
        """Return validated window size."""
        width = context.parameters.get("width")
        height = context.parameters.get("height")

        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("Las dimensiones deben ser numericas.")

        if width <= 0 or height <= 0:
            raise ValueError("El ancho y el alto deben ser mayores que cero.")

        if width > 100000 or height > 100000:
            raise ValueError("Dimensiones de ventana absurdas.")

        return width, height

    def _validate_virtual_desktop_intersection(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Reject rectangles completely outside the virtual desktop."""
        left, top, right, bottom = self._controller.get_virtual_desktop_rect()

        if x + width <= left or y + height <= top or x >= right or y >= bottom:
            raise ValueError("La posicion queda fuera del escritorio.")

    def _process_query(self, context: ToolContext) -> str:
        """Return a required process query."""
        query = context.parameters.get("query")

        if not query:
            raise ValueError("Missing parameter 'query'.")

        return str(query).strip()

    def _pid(self, context: ToolContext) -> int:
        """Return a validated process ID."""
        pid = context.parameters.get("pid")

        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("PID invalido.")

        return pid

    def _format_process(self, process: ProcessInfo) -> dict[str, object]:
        """Format process information as a plain structure."""
        return {
            "pid": process.pid,
            "name": process.name,
            "executable_path": (
                str(process.executable_path)
                if process.executable_path is not None
                else None
            ),
            "window_titles": process.window_titles,
            "is_running": process.is_running,
        }


class OpenApplicationTool(DesktopTool):
    """Open an installed application."""

    @property
    def name(self) -> str:
        return "desktop.open_application"

    @property
    def description(self) -> str:
        return "Open an installed desktop application."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        application = context.parameters.get("application")

        if not application:
            raise ValueError("Missing parameter 'application'.")

        query = str(application)
        existing = self._controller.list_processes(query)

        if existing:
            pids = ", ".join(str(process.pid) for process in existing)
            return f"{query} ya estaba abierto. PID: {pids}"

        pid = self._controller.open_application(query)

        if pid is None:
            return f"{application} abierto."

        return f"{application} abierto. PID: {pid}"


class ListProcessesTool(DesktopTool):
    """List running processes matching a query."""

    @property
    def name(self) -> str:
        return "desktop.list_processes"

    @property
    def description(self) -> str:
        return "List running processes matching a query."

    def execute(
        self,
        context: ToolContext,
    ) -> list[dict[str, object]]:
        query = self._process_query(context)
        processes = self._controller.list_processes(query)

        return [self._format_process(process) for process in processes]


class IsProcessRunningTool(DesktopTool):
    """Return whether a process matching a query is running."""

    @property
    def name(self) -> str:
        return "desktop.is_process_running"

    @property
    def description(self) -> str:
        return "Return whether a process matching a query is running."

    def execute(
        self,
        context: ToolContext,
    ) -> bool:
        return bool(self._controller.list_processes(self._process_query(context)))


class GetProcessTool(DesktopTool):
    """Return process information by PID."""

    @property
    def name(self) -> str:
        return "desktop.get_process"

    @property
    def description(self) -> str:
        return "Return process information by PID."

    def execute(
        self,
        context: ToolContext,
    ) -> dict[str, object] | None:
        process = self._controller.get_process(self._pid(context))

        if process is None:
            return None

        return self._format_process(process)


class CloseApplicationTool(DesktopTool):
    """Request normal close for one process by PID."""

    @property
    def name(self) -> str:
        return "desktop.close_application"

    @property
    def description(self) -> str:
        return "Request normal close for one process by PID."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        pid = self._pid(context)
        count = self._controller.close_process_windows(pid)

        if count == 0:
            raise RuntimeError("El proceso no tiene ventanas visibles.")

        return f"Solicitud de cierre enviada a {count} ventana(s)."


class TerminateProcessTool(DesktopTool):
    """Terminate one process by PID."""

    _PROTECTED_NAMES = {
        "system",
        "registry",
        "smss.exe",
        "csrss.exe",
        "wininit.exe",
        "services.exe",
        "lsass.exe",
        "winlogon.exe",
        "svchost.exe",
        "explorer.exe",
    }

    @property
    def name(self) -> str:
        return "desktop.terminate_process"

    @property
    def description(self) -> str:
        return "Terminate one process by PID."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        pid = self._pid(context)

        if pid in {0, 4}:
            raise ValueError("Proceso protegido.")

        if pid == os.getpid():
            raise ValueError("No se puede terminar el propio proceso de Atlas.")

        process = self._controller.get_process(pid)

        if process is None:
            raise RuntimeError(f"No existe un proceso con PID {pid}.")

        if process.name.lower() in self._PROTECTED_NAMES:
            raise ValueError("Proceso protegido.")

        self._controller.close_process_windows(pid)
        self._controller.terminate_process(pid)

        if self._controller.process_exists(pid):
            raise RuntimeError("El proceso continua en ejecucion.")

        return f"Proceso terminado: {process.name} - PID {pid}"


class OpenFolderTool(DesktopTool):
    """Open an existing folder."""

    @property
    def name(self) -> str:
        return "desktop.open_folder"

    @property
    def description(self) -> str:
        return "Open an existing folder."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        raw_path = context.parameters.get("path")

        if not raw_path:
            raise ValueError("Missing parameter 'path'.")

        path = Path(str(raw_path))

        if not path.exists():
            raise FileNotFoundError(str(path))

        if not path.is_dir():
            raise NotADirectoryError(str(path))

        self._controller.open_folder(path)

        return f"Carpeta abierta: {path}"


class OpenFileTool(DesktopTool):
    """Open an existing file."""

    @property
    def name(self) -> str:
        return "desktop.open_file"

    @property
    def description(self) -> str:
        return "Open an existing file."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        raw_path = context.parameters.get("path")
        application = context.parameters.get("application")

        if not raw_path:
            raise ValueError("Missing parameter 'path'.")

        path = Path(str(raw_path))

        if not path.exists():
            raise FileNotFoundError(str(path))

        if not path.is_file():
            raise IsADirectoryError(str(path))

        self._controller.open_file(
            path,
            str(application) if application else None,
        )

        if application:
            return f"Archivo abierto en {application}: {path}"

        return f"Archivo abierto: {path}"


class TypeTextTool(DesktopTool):
    """Type text into an existing target window."""

    @property
    def name(self) -> str:
        return "desktop.type_text"

    @property
    def description(self) -> str:
        return "Type text into an existing target window."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        text = context.parameters.get("text")

        if text is None:
            raise ValueError("Missing parameter 'text'.")

        title = self._window_title(context)
        self._controller.activate_window(title)
        self._controller.type_text(str(text))

        return "Texto escrito."


class CopyClipboardTextTool(DesktopTool):
    """Copy Unicode text into the clipboard."""

    @property
    def name(self) -> str:
        return "desktop.copy_clipboard_text"

    @property
    def description(self) -> str:
        return "Copy Unicode text into the clipboard."

    def execute(
        self,
        context: ToolContext,
    ) -> int:
        text = context.parameters.get("text")

        if not isinstance(text, str):
            raise TypeError("El contenido del portapapeles debe ser texto.")

        return self._controller.copy_clipboard_text(text)


class ReadClipboardTextTool(DesktopTool):
    """Read Unicode text from the clipboard."""

    @property
    def name(self) -> str:
        return "desktop.read_clipboard_text"

    @property
    def description(self) -> str:
        return "Read Unicode text from the clipboard."

    def execute(
        self,
        context: ToolContext,
    ) -> str | None:
        return self._controller.read_clipboard_text()


class ClearClipboardTool(DesktopTool):
    """Clear the clipboard."""

    @property
    def name(self) -> str:
        return "desktop.clear_clipboard"

    @property
    def description(self) -> str:
        return "Clear the clipboard."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        self._controller.clear_clipboard()

        return "Portapapeles vaciado."


class ClipboardHasTextTool(DesktopTool):
    """Return whether the clipboard contains Unicode text."""

    @property
    def name(self) -> str:
        return "desktop.clipboard_has_text"

    @property
    def description(self) -> str:
        return "Return whether the clipboard contains Unicode text."

    def execute(
        self,
        context: ToolContext,
    ) -> bool:
        return self._controller.clipboard_has_text()


class PasteClipboardTool(DesktopTool):
    """Paste clipboard content into an existing target window."""

    @property
    def name(self) -> str:
        return "desktop.paste_clipboard"

    @property
    def description(self) -> str:
        return "Paste clipboard content into an existing target window."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        title = self._window_title(context)

        if not self._controller.clipboard_has_text():
            raise RuntimeError("El portapapeles no contiene texto.")

        self._controller.activate_window(title)
        self._controller.press_hotkey(["ctrl", "v"])

        return "Contenido pegado."


class PressHotkeyTool(DesktopTool):
    """Send a keyboard shortcut to an existing target window."""

    @property
    def name(self) -> str:
        return "desktop.press_hotkey"

    @property
    def description(self) -> str:
        return "Send a keyboard shortcut to an existing target window."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        keys = context.parameters.get("keys")

        if not isinstance(keys, list) or not keys:
            raise ValueError("Missing parameter 'keys'.")

        title = self._window_title(context)
        self._controller.activate_window(title)
        self._controller.press_hotkey([str(key) for key in keys])

        return "Atajo enviado."


class SaveFileTool(DesktopTool):
    """Save the active file in an existing target window."""

    @property
    def name(self) -> str:
        return "desktop.save_file"

    @property
    def description(self) -> str:
        return "Save the active file in an existing target window."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        title = self._window_title(context)
        self._controller.activate_window(title)
        self._controller.press_hotkey(["ctrl", "s"])

        return "Archivo guardado."


class ActivateWindowTool(DesktopTool):
    """Activate an existing window."""

    @property
    def name(self) -> str:
        return "desktop.activate_window"

    @property
    def description(self) -> str:
        return "Activate an existing desktop window."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        title = self._window_title(context)
        self._controller.activate_window(title)

        return f"Ventana activada: {title}"


class GetScreenSizeTool(DesktopTool):
    """Return the primary screen size."""

    @property
    def name(self) -> str:
        return "desktop.get_screen_size"

    @property
    def description(self) -> str:
        return "Return the primary screen size."

    def execute(
        self,
        context: ToolContext,
    ) -> tuple[int, int]:
        return self._controller.get_screen_size()


class GetCursorPositionTool(DesktopTool):
    """Return the current cursor position."""

    @property
    def name(self) -> str:
        return "desktop.get_cursor_position"

    @property
    def description(self) -> str:
        return "Return the current cursor position."

    def execute(
        self,
        context: ToolContext,
    ) -> tuple[int, int]:
        return self._controller.get_cursor_position()


class MoveCursorTool(DesktopTool):
    """Move the cursor to absolute coordinates."""

    @property
    def name(self) -> str:
        return "desktop.move_cursor"

    @property
    def description(self) -> str:
        return "Move the cursor to absolute screen coordinates."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        x, y = self._coordinates(context)
        self._controller.move_cursor(x, y)

        return f"Cursor movido a ({x}, {y})."


class LeftClickTool(DesktopTool):
    """Perform a left click at absolute coordinates."""

    @property
    def name(self) -> str:
        return "desktop.left_click"

    @property
    def description(self) -> str:
        return "Perform a left click at absolute screen coordinates."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        x, y = self._coordinates(context)
        self._controller.left_click(x, y)

        return f"Clic realizado en ({x}, {y})."


class DoubleClickTool(DesktopTool):
    """Perform a double click at absolute coordinates."""

    @property
    def name(self) -> str:
        return "desktop.double_click"

    @property
    def description(self) -> str:
        return "Perform a double click at absolute screen coordinates."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        x, y = self._coordinates(context)
        self._controller.double_click(x, y)

        return f"Doble clic realizado en ({x}, {y})."


class RightClickTool(DesktopTool):
    """Perform a right click at absolute coordinates."""

    @property
    def name(self) -> str:
        return "desktop.right_click"

    @property
    def description(self) -> str:
        return "Perform a right click at absolute screen coordinates."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        x, y = self._coordinates(context)
        self._controller.right_click(x, y)

        return f"Clic derecho realizado en ({x}, {y})."


class ScrollVerticalTool(DesktopTool):
    """Scroll vertically."""

    @property
    def name(self) -> str:
        return "desktop.scroll_vertical"

    @property
    def description(self) -> str:
        return "Scroll vertically."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        direction = str(context.parameters.get("direction", "")).lower()

        if direction == "up":
            amount = 120
            label = "arriba"
        elif direction == "down":
            amount = -120
            label = "abajo"
        else:
            raise ValueError("Direccion de scroll no soportada.")

        self._controller.scroll_vertical(amount)

        return f"Scroll hacia {label}."


class CaptureScreenshotTool(DesktopTool):
    """Capture the full screen as PNG."""

    @property
    def name(self) -> str:
        return "desktop.capture_screenshot"

    @property
    def description(self) -> str:
        return "Capture the full screen as PNG."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        output_dir = Path(
            str(
                context.parameters.get(
                    "output_dir",
                    Path("artifacts") / "screenshots",
                )
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        path = self._next_screenshot_path(output_dir)
        self._controller.capture_screen(path)

        return str(path)

    def _next_screenshot_path(
        self,
        output_dir: Path,
    ) -> Path:
        """Return a non-existing screenshot path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = output_dir / f"screenshot_{timestamp}.png"

        if not base.exists():
            return base

        index = 1

        while True:
            path = output_dir / f"screenshot_{timestamp}_{index}.png"

            if not path.exists():
                return path

            index += 1


class ListWindowsTool(DesktopTool):
    """List visible windows matching a title."""

    @property
    def name(self) -> str:
        return "desktop.list_windows"

    @property
    def description(self) -> str:
        return "List visible windows matching a title."

    def execute(
        self,
        context: ToolContext,
    ) -> list[dict[str, object]]:
        title = context.parameters.get("title")

        if not title:
            raise ValueError("Missing parameter 'title'.")

        return self._controller.list_windows(str(title))


class GetWindowRectTool(DesktopTool):
    """Return a window rectangle."""

    @property
    def name(self) -> str:
        return "desktop.get_window_rect"

    @property
    def description(self) -> str:
        return "Return a window rectangle."

    def execute(
        self,
        context: ToolContext,
    ) -> tuple[int, int, int, int]:
        return self._controller.get_window_rect(
            self._window_handle(context),
        )


class GetForegroundWindowTool(DesktopTool):
    """Return the current foreground window."""

    @property
    def name(self) -> str:
        return "desktop.get_foreground_window"

    @property
    def description(self) -> str:
        return "Return the current foreground window."

    def execute(
        self,
        context: ToolContext,
    ) -> dict[str, object]:
        return self._controller.get_foreground_window()


class BringWindowToFrontTool(DesktopTool):
    """Bring a window to the foreground."""

    @property
    def name(self) -> str:
        return "desktop.bring_window_to_front"

    @property
    def description(self) -> str:
        return "Bring a window to the foreground."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        self._controller.bring_window_to_front(
            self._window_handle(context),
        )

        return "Ventana activada."


class MaximizeWindowTool(DesktopTool):
    """Maximize a window."""

    @property
    def name(self) -> str:
        return "desktop.maximize_window"

    @property
    def description(self) -> str:
        return "Maximize a window."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        self._controller.maximize_window(
            self._window_handle(context),
        )

        return "Ventana maximizada."


class MinimizeWindowTool(DesktopTool):
    """Minimize a window."""

    @property
    def name(self) -> str:
        return "desktop.minimize_window"

    @property
    def description(self) -> str:
        return "Minimize a window."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        self._controller.minimize_window(
            self._window_handle(context),
        )

        return "Ventana minimizada."


class RestoreWindowTool(DesktopTool):
    """Restore a window."""

    @property
    def name(self) -> str:
        return "desktop.restore_window"

    @property
    def description(self) -> str:
        return "Restore a window."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        self._controller.restore_window(
            self._window_handle(context),
        )

        return "Ventana restaurada."


class MoveWindowTool(DesktopTool):
    """Move a window preserving size."""

    @property
    def name(self) -> str:
        return "desktop.move_window"

    @property
    def description(self) -> str:
        return "Move a window preserving size."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        handle = self._window_handle(context)
        x = context.parameters.get("x")
        y = context.parameters.get("y")

        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError("Las coordenadas deben ser numericas.")

        left, top, right, bottom = self._controller.get_window_rect(handle)
        width = right - left
        height = bottom - top
        self._validate_virtual_desktop_intersection(x, y, width, height)
        self._controller.move_window(handle, x, y)

        return f"Ventana movida a ({x}, {y})."


class ResizeWindowTool(DesktopTool):
    """Resize a window preserving position."""

    @property
    def name(self) -> str:
        return "desktop.resize_window"

    @property
    def description(self) -> str:
        return "Resize a window preserving position."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        handle = self._window_handle(context)
        width, height = self._window_size(context)
        self._controller.resize_window(handle, width, height)

        return f"Ventana redimensionada a {width} x {height}."


class MoveResizeWindowTool(DesktopTool):
    """Move and resize a window."""

    @property
    def name(self) -> str:
        return "desktop.move_resize_window"

    @property
    def description(self) -> str:
        return "Move and resize a window."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        handle = self._window_handle(context)
        x, y, width, height = self._window_rect(context)
        self._controller.move_resize_window(handle, x, y, width, height)

        return (
            f"Ventana movida a ({x}, {y}) "
            f"y redimensionada a {width} x {height}."
        )


class CloseWindowTool(DesktopTool):
    """Request closing a window."""

    @property
    def name(self) -> str:
        return "desktop.close_window"

    @property
    def description(self) -> str:
        return "Request closing a window."

    def execute(
        self,
        context: ToolContext,
    ) -> str:
        self._controller.close_window(
            self._window_handle(context),
        )

        return "Solicitud de cierre enviada."

"""Desktop system tools."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from tools.base_tool import BaseTool
from tools.desktop.windows_controller import (
    DesktopController,
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

        self._controller.open_application(str(application))

        return f"{application} abierto."


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

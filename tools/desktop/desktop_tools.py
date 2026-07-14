"""Desktop system tools."""

from __future__ import annotations

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


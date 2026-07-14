"""Native Windows desktop controller."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import shutil
import subprocess
from typing import Protocol


class DesktopController(Protocol):
    """Interface for desktop operations."""

    def open_application(self, application: str) -> None:
        """Open an installed application."""

    def open_folder(self, path: Path) -> None:
        """Open an existing folder."""

    def open_file(self, path: Path, application: str | None = None) -> None:
        """Open an existing file."""

    def window_exists(self, title: str) -> bool:
        """Return whether a window matching title exists."""

    def activate_window(self, title: str) -> None:
        """Activate a matching window."""

    def type_text(self, text: str) -> None:
        """Type text into the active window."""

    def press_hotkey(self, keys: list[str]) -> None:
        """Press a keyboard shortcut."""


class WindowsDesktopController:
    """Control Windows desktop using stdlib and Win32 APIs."""

    _USER32 = ctypes.windll.user32
    _KERNEL32 = ctypes.windll.kernel32

    _KERNEL32.GlobalAlloc.restype = wintypes.HGLOBAL
    _KERNEL32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
    _KERNEL32.GlobalLock.restype = ctypes.c_void_p
    _KERNEL32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
    _KERNEL32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
    _USER32.SetClipboardData.restype = wintypes.HANDLE
    _USER32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)

    _KNOWN_APPLICATIONS: dict[str, tuple[str, ...]] = {
        "visual studio code": (
            "code",
            str(
                Path.home()
                / "AppData"
                / "Local"
                / "Programs"
                / "Microsoft VS Code"
                / "Code.exe"
            ),
        ),
        "vscode": (
            "code",
            str(
                Path.home()
                / "AppData"
                / "Local"
                / "Programs"
                / "Microsoft VS Code"
                / "Code.exe"
            ),
        ),
        "vs code": (
            "code",
            str(
                Path.home()
                / "AppData"
                / "Local"
                / "Programs"
                / "Microsoft VS Code"
                / "Code.exe"
            ),
        ),
    }

    _VIRTUAL_KEYS: dict[str, int] = {
        "ctrl": 0x11,
        "control": 0x11,
        "shift": 0x10,
        "alt": 0x12,
        "enter": 0x0D,
        "tab": 0x09,
        "esc": 0x1B,
        "escape": 0x1B,
        "s": 0x53,
        "o": 0x4F,
        "p": 0x50,
        "n": 0x4E,
        "w": 0x57,
    }

    def open_application(self, application: str) -> None:
        """Open an installed application."""
        executable = self._resolve_application(application)
        subprocess.Popen(
            [executable],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def open_folder(self, path: Path) -> None:
        """Open an existing folder."""
        os.startfile(str(path))

    def open_file(self, path: Path, application: str | None = None) -> None:
        """Open an existing file."""
        if application:
            executable = self._resolve_application(application)
            subprocess.Popen(
                [executable, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        os.startfile(str(path))

    def window_exists(self, title: str) -> bool:
        """Return whether a window matching title exists."""
        return self._find_window(title) != 0

    def activate_window(self, title: str) -> None:
        """Activate a matching window."""
        handle = self._find_window(title)

        if handle == 0:
            raise RuntimeError(f"No existe una ventana con titulo '{title}'.")

        self._USER32.ShowWindow(handle, 9)
        self._USER32.SetForegroundWindow(handle)

    def type_text(self, text: str) -> None:
        """Type text into the active window."""
        self._set_clipboard_text(text)
        self.press_hotkey(["ctrl", "v"])

    def press_hotkey(self, keys: list[str]) -> None:
        """Press a keyboard shortcut."""
        codes = [self._key_code(key) for key in keys]

        for code in codes:
            self._USER32.keybd_event(code, 0, 0, 0)

        for code in reversed(codes):
            self._USER32.keybd_event(code, 0, 2, 0)

    def _resolve_application(self, application: str) -> str:
        """Resolve an application name to an executable path."""
        normalized = application.strip().lower()
        candidates = self._KNOWN_APPLICATIONS.get(
            normalized,
            (application,),
        )

        for candidate in candidates:
            found = shutil.which(candidate)

            if found:
                return found

            path = Path(candidate)

            if path.exists():
                return str(path)

        raise FileNotFoundError(
            f"No se encontro la aplicacion '{application}'."
        )

    def _find_window(self, title: str) -> int:
        """Find a top-level window by partial title."""
        target = title.lower()
        found = 0

        enum_windows_proc = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM,
        )

        def callback(
            handle: int,
            _: int,
        ) -> bool:
            nonlocal found

            if not self._USER32.IsWindowVisible(handle):
                return True

            length = self._USER32.GetWindowTextLengthW(handle)

            if length == 0:
                return True

            buffer = ctypes.create_unicode_buffer(length + 1)
            self._USER32.GetWindowTextW(handle, buffer, length + 1)

            if target in buffer.value.lower():
                found = handle
                return False

            return True

        self._USER32.EnumWindows(enum_windows_proc(callback), 0)

        return found

    def _set_clipboard_text(self, text: str) -> None:
        """Put text into the Windows clipboard."""
        if not self._USER32.OpenClipboard(None):
            raise RuntimeError("No se pudo abrir el portapapeles.")

        try:
            self._USER32.EmptyClipboard()
            data = text.encode("utf-16-le") + b"\x00\x00"
            handle = self._KERNEL32.GlobalAlloc(0x0042, len(data))

            if not handle:
                raise RuntimeError("No se pudo reservar memoria.")

            pointer = self._KERNEL32.GlobalLock(handle)

            if pointer is None:
                raise RuntimeError("No se pudo bloquear memoria.")

            ctypes.memmove(pointer, data, len(data))
            self._KERNEL32.GlobalUnlock(handle)

            if not self._USER32.SetClipboardData(13, handle):
                raise RuntimeError("No se pudo escribir en el portapapeles.")
        finally:
            self._USER32.CloseClipboard()

    def _key_code(self, key: str) -> int:
        """Return the Windows virtual-key code for a key."""
        normalized = key.strip().lower()

        if normalized in self._VIRTUAL_KEYS:
            return self._VIRTUAL_KEYS[normalized]

        if len(normalized) == 1 and normalized.isalnum():
            return ord(normalized.upper())

        raise ValueError(f"Tecla no soportada: {key}")

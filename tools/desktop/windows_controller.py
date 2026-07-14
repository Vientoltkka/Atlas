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

    def get_screen_size(self) -> tuple[int, int]:
        """Return the primary screen size."""

    def get_cursor_position(self) -> tuple[int, int]:
        """Return the current cursor position."""

    def move_cursor(self, x: int, y: int) -> None:
        """Move the cursor to absolute coordinates."""

    def left_click(self, x: int, y: int) -> None:
        """Perform a left click at absolute coordinates."""

    def double_click(self, x: int, y: int) -> None:
        """Perform a double left click at absolute coordinates."""

    def right_click(self, x: int, y: int) -> None:
        """Perform a right click at absolute coordinates."""

    def scroll_vertical(self, amount: int) -> None:
        """Scroll vertically."""

    def capture_screen(self, path: Path) -> None:
        """Capture the full screen as PNG."""


class WindowsDesktopController:
    """Control Windows desktop using stdlib and Win32 APIs."""

    _USER32 = ctypes.windll.user32
    _KERNEL32 = ctypes.windll.kernel32
    _GDI32 = ctypes.windll.gdi32
    _GDIPLUS = ctypes.windll.gdiplus
    _OLE32 = ctypes.windll.ole32

    _KERNEL32.GlobalAlloc.restype = wintypes.HGLOBAL
    _KERNEL32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
    _KERNEL32.GlobalLock.restype = ctypes.c_void_p
    _KERNEL32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
    _KERNEL32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
    _USER32.SetClipboardData.restype = wintypes.HANDLE
    _USER32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
    _USER32.GetDC.restype = wintypes.HDC
    _USER32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
    _GDI32.CreateCompatibleDC.restype = wintypes.HDC
    _GDI32.CreateCompatibleDC.argtypes = (wintypes.HDC,)
    _GDI32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    _GDI32.CreateCompatibleBitmap.argtypes = (
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
    )
    _GDI32.SelectObject.restype = wintypes.HGDIOBJ
    _GDI32.SelectObject.argtypes = (wintypes.HDC, wintypes.HGDIOBJ)
    _GDI32.BitBlt.restype = wintypes.BOOL
    _GDI32.BitBlt.argtypes = (
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.DWORD,
    )
    _GDI32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
    _GDI32.DeleteDC.argtypes = (wintypes.HDC,)
    _GDIPLUS.GdiplusStartup.argtypes = (
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    _GDIPLUS.GdipCreateBitmapFromHBITMAP.argtypes = (
        wintypes.HBITMAP,
        wintypes.HPALETTE,
        ctypes.POINTER(ctypes.c_void_p),
    )
    _GDIPLUS.GdipSaveImageToFile.argtypes = (
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    _GDIPLUS.GdipDisposeImage.argtypes = (ctypes.c_void_p,)
    _GDIPLUS.GdiplusShutdown.argtypes = (ctypes.c_ulong,)
    _OLE32.CLSIDFromString.argtypes = (wintypes.LPCWSTR, ctypes.c_void_p)

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

    _LEFT_DOWN = 0x0002
    _LEFT_UP = 0x0004
    _RIGHT_DOWN = 0x0008
    _RIGHT_UP = 0x0010
    _WHEEL = 0x0800
    _WHEEL_DELTA = 120
    _SRCCOPY = 0x00CC0020
    _PNG_ENCODER = "{557CF406-1A04-11D3-9A73-0000F81EF32E}"

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

    def get_screen_size(self) -> tuple[int, int]:
        """Return the primary screen size."""
        return (
            int(self._USER32.GetSystemMetrics(0)),
            int(self._USER32.GetSystemMetrics(1)),
        )

    def get_cursor_position(self) -> tuple[int, int]:
        """Return the current cursor position."""
        point = _Point()

        if not self._USER32.GetCursorPos(ctypes.byref(point)):
            raise RuntimeError("No se pudo obtener la posicion del cursor.")

        return int(point.x), int(point.y)

    def move_cursor(self, x: int, y: int) -> None:
        """Move the cursor to absolute coordinates."""
        if not self._USER32.SetCursorPos(x, y):
            raise RuntimeError("No se pudo mover el cursor.")

    def left_click(self, x: int, y: int) -> None:
        """Perform a left click at absolute coordinates."""
        self.move_cursor(x, y)
        self._mouse_event(self._LEFT_DOWN)
        self._mouse_event(self._LEFT_UP)

    def double_click(self, x: int, y: int) -> None:
        """Perform a double left click at absolute coordinates."""
        self.left_click(x, y)
        self.left_click(x, y)

    def right_click(self, x: int, y: int) -> None:
        """Perform a right click at absolute coordinates."""
        self.move_cursor(x, y)
        self._mouse_event(self._RIGHT_DOWN)
        self._mouse_event(self._RIGHT_UP)

    def scroll_vertical(self, amount: int) -> None:
        """Scroll vertically."""
        self._USER32.mouse_event(self._WHEEL, 0, 0, amount, 0)

    def capture_screen(self, path: Path) -> None:
        """Capture the full screen as PNG."""
        width, height = self.get_screen_size()
        screen_dc = self._USER32.GetDC(None)

        if not screen_dc:
            raise RuntimeError("No se pudo obtener el contexto de pantalla.")

        memory_dc = self._GDI32.CreateCompatibleDC(screen_dc)
        bitmap = self._GDI32.CreateCompatibleBitmap(screen_dc, width, height)
        old_object = None

        try:
            if not memory_dc or not bitmap:
                raise RuntimeError("No se pudo preparar la captura.")

            old_object = self._GDI32.SelectObject(memory_dc, bitmap)

            if not self._GDI32.BitBlt(
                memory_dc,
                0,
                0,
                width,
                height,
                screen_dc,
                0,
                0,
                self._SRCCOPY,
            ):
                raise RuntimeError("No se pudo capturar la pantalla.")

            self._save_bitmap_as_png(bitmap, path)
        finally:
            if old_object:
                self._GDI32.SelectObject(memory_dc, old_object)

            if bitmap:
                self._GDI32.DeleteObject(bitmap)

            if memory_dc:
                self._GDI32.DeleteDC(memory_dc)

            self._USER32.ReleaseDC(None, screen_dc)

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

    def _mouse_event(
        self,
        event: int,
        data: int = 0,
    ) -> None:
        """Send a mouse event."""
        self._USER32.mouse_event(event, 0, 0, data, 0)

    def _save_bitmap_as_png(
        self,
        bitmap: wintypes.HBITMAP,
        path: Path,
    ) -> None:
        """Save a GDI bitmap as PNG using native GDI+."""
        token = ctypes.c_ulong()
        startup_input = _GdiplusStartupInput()
        startup_input.GdiplusVersion = 1

        status = self._GDIPLUS.GdiplusStartup(
            ctypes.byref(token),
            ctypes.byref(startup_input),
            None,
        )

        if status != 0:
            raise RuntimeError("No se pudo iniciar GDI+.")

        image = ctypes.c_void_p()

        try:
            status = self._GDIPLUS.GdipCreateBitmapFromHBITMAP(
                bitmap,
                None,
                ctypes.byref(image),
            )

            if status != 0:
                raise RuntimeError("No se pudo crear la imagen PNG.")

            encoder = _Guid()
            self._OLE32.CLSIDFromString(
                ctypes.c_wchar_p(self._PNG_ENCODER),
                ctypes.byref(encoder),
            )
            status = self._GDIPLUS.GdipSaveImageToFile(
                image,
                str(path),
                ctypes.byref(encoder),
                None,
            )

            if status != 0:
                raise RuntimeError("No se pudo guardar la captura.")
        finally:
            if image:
                self._GDIPLUS.GdipDisposeImage(image)

            self._GDIPLUS.GdiplusShutdown(token)


class _Point(ctypes.Structure):
    """Win32 POINT structure."""

    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


class _Guid(ctypes.Structure):
    """Win32 GUID structure."""

    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _GdiplusStartupInput(ctypes.Structure):
    """GDI+ startup input structure."""

    _fields_ = [
        ("GdiplusVersion", wintypes.UINT),
        ("DebugEventCallback", ctypes.c_void_p),
        ("SuppressBackgroundThread", wintypes.BOOL),
        ("SuppressExternalCodecs", wintypes.BOOL),
    ]

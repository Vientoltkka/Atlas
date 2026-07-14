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

    def get_virtual_desktop_rect(self) -> tuple[int, int, int, int]:
        """Return the virtual desktop rectangle."""

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

    def list_windows(self, title: str) -> list[dict[str, object]]:
        """List visible windows matching title."""

    def get_window_rect(self, handle: int) -> tuple[int, int, int, int]:
        """Return a window rectangle."""

    def get_foreground_window(self) -> dict[str, object]:
        """Return the current foreground window."""

    def bring_window_to_front(self, handle: int) -> None:
        """Bring a window to the foreground."""

    def maximize_window(self, handle: int) -> None:
        """Maximize a window."""

    def minimize_window(self, handle: int) -> None:
        """Minimize a window."""

    def restore_window(self, handle: int) -> None:
        """Restore a window."""

    def move_window(self, handle: int, x: int, y: int) -> None:
        """Move a window preserving its current size."""

    def resize_window(self, handle: int, width: int, height: int) -> None:
        """Resize a window preserving its current position."""

    def move_resize_window(
        self,
        handle: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Move and resize a window."""

    def close_window(self, handle: int) -> None:
        """Request a window close."""


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
    _USER32.IsWindow.argtypes = (wintypes.HWND,)
    _USER32.IsWindow.restype = wintypes.BOOL
    _USER32.IsWindowVisible.argtypes = (wintypes.HWND,)
    _USER32.IsWindowVisible.restype = wintypes.BOOL
    _USER32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
    _USER32.GetWindowTextLengthW.restype = ctypes.c_int
    _USER32.GetWindowTextW.argtypes = (
        wintypes.HWND,
        wintypes.LPWSTR,
        ctypes.c_int,
    )
    _USER32.GetWindowTextW.restype = ctypes.c_int
    _USER32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.c_void_p)
    _USER32.GetWindowRect.restype = wintypes.BOOL
    _USER32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    _USER32.ShowWindow.restype = wintypes.BOOL
    _USER32.SetForegroundWindow.argtypes = (wintypes.HWND,)
    _USER32.SetForegroundWindow.restype = wintypes.BOOL
    _USER32.BringWindowToTop.argtypes = (wintypes.HWND,)
    _USER32.BringWindowToTop.restype = wintypes.BOOL
    _USER32.GetForegroundWindow.restype = wintypes.HWND
    _USER32.GetWindowThreadProcessId.argtypes = (
        wintypes.HWND,
        ctypes.c_void_p,
    )
    _USER32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _USER32.AttachThreadInput.argtypes = (
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.BOOL,
    )
    _USER32.AttachThreadInput.restype = wintypes.BOOL
    _KERNEL32.GetCurrentThreadId.restype = wintypes.DWORD
    _USER32.MoveWindow.argtypes = (
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.BOOL,
    )
    _USER32.MoveWindow.restype = wintypes.BOOL
    _USER32.PostMessageW.argtypes = (
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )
    _USER32.PostMessageW.restype = wintypes.BOOL
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
    _SW_RESTORE = 9
    _SW_MAXIMIZE = 3
    _SW_MINIMIZE = 6
    _WM_CLOSE = 0x0010

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
        return bool(self.list_windows(title))

    def activate_window(self, title: str) -> None:
        """Activate a matching window."""
        handle = self._single_window_handle(title)
        self.bring_window_to_front(handle)

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

    def get_virtual_desktop_rect(self) -> tuple[int, int, int, int]:
        """Return the virtual desktop rectangle."""
        left = int(self._USER32.GetSystemMetrics(76))
        top = int(self._USER32.GetSystemMetrics(77))
        width = int(self._USER32.GetSystemMetrics(78))
        height = int(self._USER32.GetSystemMetrics(79))

        return left, top, left + width, top + height

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

    def list_windows(self, title: str) -> list[dict[str, object]]:
        """List visible windows matching title."""
        target = title.strip().lower()
        windows: list[dict[str, object]] = []

        enum_windows_proc = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM,
        )

        def callback(
            handle: int,
            _: int,
        ) -> bool:
            if not self._USER32.IsWindowVisible(handle):
                return True

            length = self._USER32.GetWindowTextLengthW(handle)

            if length == 0:
                return True

            buffer = ctypes.create_unicode_buffer(length + 1)
            self._USER32.GetWindowTextW(handle, buffer, length + 1)
            window_title = buffer.value

            if target and target not in window_title.lower():
                return True

            windows.append(
                {
                    "handle": int(handle),
                    "title": window_title,
                    "rect": self.get_window_rect(int(handle)),
                    "order": len(windows),
                }
            )

            return True

        self._USER32.EnumWindows(enum_windows_proc(callback), 0)

        return sorted(
            windows,
            key=lambda item: (str(item["title"]).lower(), int(item["handle"])),
        )

    def get_window_rect(self, handle: int) -> tuple[int, int, int, int]:
        """Return a window rectangle."""
        self._ensure_window(handle)
        rect = _Rect()

        if not self._USER32.GetWindowRect(handle, ctypes.byref(rect)):
            raise RuntimeError("No se pudo obtener el rectangulo de ventana.")

        return (
            int(rect.left),
            int(rect.top),
            int(rect.right),
            int(rect.bottom),
        )

    def get_foreground_window(self) -> dict[str, object]:
        """Return the current foreground window."""
        handle = int(self._USER32.GetForegroundWindow())

        if handle <= 0:
            raise RuntimeError("No se pudo obtener la ventana activa.")

        length = self._USER32.GetWindowTextLengthW(handle)
        buffer = ctypes.create_unicode_buffer(length + 1)
        self._USER32.GetWindowTextW(handle, buffer, length + 1)

        return {
            "handle": handle,
            "title": buffer.value,
            "rect": self.get_window_rect(handle),
        }

    def bring_window_to_front(self, handle: int) -> None:
        """Bring a window to the foreground."""
        self._ensure_window(handle)
        self._USER32.ShowWindow(handle, self._SW_RESTORE)

        if not self._USER32.SetForegroundWindow(handle) and not (
            self._attach_and_set_foreground(handle)
        ):
            raise RuntimeError("No se pudo activar la ventana.")

    def maximize_window(self, handle: int) -> None:
        """Maximize a window."""
        self._show_window(handle, self._SW_MAXIMIZE)

    def minimize_window(self, handle: int) -> None:
        """Minimize a window."""
        self._show_window(handle, self._SW_MINIMIZE)

    def restore_window(self, handle: int) -> None:
        """Restore a window."""
        self._show_window(handle, self._SW_RESTORE)

    def move_window(self, handle: int, x: int, y: int) -> None:
        """Move a window preserving its current size."""
        left, top, right, bottom = self.get_window_rect(handle)
        self.move_resize_window(handle, x, y, right - left, bottom - top)

    def resize_window(self, handle: int, width: int, height: int) -> None:
        """Resize a window preserving its current position."""
        left, top, _, _ = self.get_window_rect(handle)
        self.move_resize_window(handle, left, top, width, height)

    def move_resize_window(
        self,
        handle: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Move and resize a window."""
        self._ensure_window(handle)

        if not self._USER32.MoveWindow(handle, x, y, width, height, True):
            raise RuntimeError("No se pudo mover o redimensionar la ventana.")

    def close_window(self, handle: int) -> None:
        """Request a window close."""
        self._ensure_window(handle)

        if not self._USER32.PostMessageW(handle, self._WM_CLOSE, 0, 0):
            raise RuntimeError("No se pudo enviar la solicitud de cierre.")

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
        windows = self.list_windows(title)

        if not windows:
            return 0

        return int(windows[0]["handle"])

    def _single_window_handle(self, title: str) -> int:
        """Return a single matching visible window handle."""
        windows = self.list_windows(title)

        if not windows:
            raise RuntimeError(f"No existe una ventana con titulo '{title}'.")

        if len(windows) > 1:
            raise RuntimeError(f"Varias ventanas coinciden con '{title}'.")

        return int(windows[0]["handle"])

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

    def _ensure_window(self, handle: int) -> None:
        """Validate a native window handle."""
        if not isinstance(handle, int) or handle <= 0:
            raise ValueError("Handle de ventana invalido.")

        if not self._USER32.IsWindow(handle):
            raise RuntimeError("La ventana ya no existe.")

    def _show_window(
        self,
        handle: int,
        command: int,
    ) -> None:
        """Run ShowWindow on a valid handle."""
        self._ensure_window(handle)
        self._USER32.ShowWindow(handle, command)

    def _attach_and_set_foreground(
        self,
        handle: int,
    ) -> bool:
        """Try foreground activation by temporarily joining input queues."""
        foreground = self._USER32.GetForegroundWindow()
        current_thread = self._KERNEL32.GetCurrentThreadId()
        target_thread = self._USER32.GetWindowThreadProcessId(handle, None)
        foreground_thread = (
            self._USER32.GetWindowThreadProcessId(foreground, None)
            if foreground
            else 0
        )

        attached_target = False
        attached_foreground = False

        try:
            if target_thread and target_thread != current_thread:
                attached_target = bool(
                    self._USER32.AttachThreadInput(
                        current_thread,
                        target_thread,
                        True,
                    )
                )

            if foreground_thread and foreground_thread != current_thread:
                attached_foreground = bool(
                    self._USER32.AttachThreadInput(
                        current_thread,
                        foreground_thread,
                        True,
                    )
                )

            self._USER32.BringWindowToTop(handle)
            return bool(self._USER32.SetForegroundWindow(handle))
        finally:
            if attached_foreground:
                self._USER32.AttachThreadInput(
                    current_thread,
                    foreground_thread,
                    False,
                )

            if attached_target:
                self._USER32.AttachThreadInput(
                    current_thread,
                    target_thread,
                    False,
                )

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


class _Rect(ctypes.Structure):
    """Win32 RECT structure."""

    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
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

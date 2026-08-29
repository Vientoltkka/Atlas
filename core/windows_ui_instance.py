"""Win32 mutex used to keep the Atlas desktop interface single-instance."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os


MUTEX_NAME = "Local\\AtlasDesktopUi"
_ERROR_ALREADY_EXISTS = 183


class WindowsUiInstance:
    """Own a process-wide Windows mutex for one Atlas desktop UI."""

    def __init__(self) -> None:
        self._handle: int | None = None

    def acquire(self) -> bool:
        """Return false when another Atlas chat/UI process already owns it."""
        if os.name != "nt":
            return True
        kernel32 = _kernel32()
        handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateMutexW fallo")
        if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is None or os.name != "nt":
            return
        _kernel32().CloseHandle(self._handle)
        self._handle = None


def _kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32

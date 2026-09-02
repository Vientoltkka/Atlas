"""Minimal native Windows global hotkey listener for the Atlas chat."""
from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import wintypes
from typing import Callable


_HOTKEY_ID = 1
_MOD_CONTROL = 0x0002
_VK_SPACE = 0x20
_WM_HOTKEY = 0x0312
_WM_QUIT = 0x0012


class WindowsGlobalHotkey:
    """Register Ctrl+Space and invoke a callback from a native message loop."""

    def __init__(self, callback: Callable[[], None], logger=None) -> None:
        self._callback = callback
        self._logger = logger or logging.getLogger(__name__)
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._ready = threading.Event()
        self._registered = False

    def start(self) -> bool:
        """Start the listener once; a duplicate registration is avoided."""
        if self._thread is not None and self._thread.is_alive():
            return self._registered
        self._ready.clear()
        self._registered = False
        self._thread = threading.Thread(
            target=self._listen, daemon=True, name="atlas-chat-hotkey"
        )
        self._thread.start()
        self._ready.wait(timeout=1.0)
        if not self._ready.is_set():
            self._logger.warning("El registro de Ctrl+Espacio no respondio en un segundo.")
        return self._registered

    def stop(self) -> None:
        """Stop the native message loop when the application exits."""
        thread_id = self._thread_id
        if thread_id is not None:
            ctypes.windll.user32.PostThreadMessageW(thread_id, _WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _listen(self) -> None:
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, _HOTKEY_ID, _MOD_CONTROL, _VK_SPACE):
            self._logger.warning("No se pudo registrar el atajo global Ctrl+Espacio.")
            self._ready.set()
            return

        self._registered = True
        self._logger.info("Atajo global Ctrl+Espacio registrado")
        self._ready.set()
        message = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                if message.message == _WM_HOTKEY:
                    self._logger.info("Ctrl+Espacio recibido; ejecutando callback")
                    try:
                        self._callback()
                    except Exception:
                        self._logger.exception("Fallo al ejecutar el callback de Ctrl+Espacio")
        finally:
            user32.UnregisterHotKey(None, _HOTKEY_ID)
            self._registered = False
            self._thread_id = None

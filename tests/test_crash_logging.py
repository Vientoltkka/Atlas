"""Crash logging leaves a logged cause for exits outside main's try/except."""

from __future__ import annotations

import logging
import sys
import threading
from types import SimpleNamespace

import pytest

from core.startup import install_crash_logging


@pytest.fixture
def hooks_restored():
    previous_exception_hook = sys.excepthook
    previous_thread_hook = threading.excepthook
    yield
    from core.startup import disable_crash_logging

    disable_crash_logging()
    sys.excepthook = previous_exception_hook
    threading.excepthook = previous_thread_hook


def _log_records(logger: logging.Logger) -> list[logging.LogRecord]:
    handler = logger.handlers[0]
    return getattr(handler, "records", [])


def test_uncaught_exception_hook_logs_to_operational_log(tmp_path, hooks_restored):
    logger = logging.getLogger("atlas.operational.test-crash-main")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _MemoryHandler()
    logger.addHandler(handler)

    install_crash_logging(logger, tmp_path)

    try:
        sys.excepthook(RuntimeError, RuntimeError("explosion en UI"), None)
    finally:
        sys.excepthook = sys.__excepthook__

    messages = [record.getMessage() for record in handler.records]
    assert any(
        "Excepcion no capturada" in message and "explosion en UI" in message
        for message in messages
    )


def test_uncaught_thread_exception_hook_logs_to_operational_log(tmp_path, hooks_restored):
    logger = logging.getLogger("atlas.operational.test-crash-thread")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _MemoryHandler()
    logger.addHandler(handler)

    install_crash_logging(logger, tmp_path)

    try:
        threading.excepthook(
            SimpleNamespace(
                exc_type=RuntimeError,
                exc_value=RuntimeError("pump explosion"),
                exc_traceback=None,
                thread=threading.current_thread(),
            )
        )
    finally:
        threading.excepthook = threading.__excepthook__

    messages = [record.getMessage() for record in handler.records]
    assert any(
        "Excepcion no capturada en hilo" in message and "pump explosion" in message
        for message in messages
    )


def test_disabling_crash_logging_keeps_thread_creation_working(tmp_path, hooks_restored):
    """threading.excepthook must never stay None after disable (Python >= 3.12
    raises RuntimeError on Thread.__init__ otherwise)."""
    logger = logging.getLogger("atlas.operational.test-crash-disable")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(_MemoryHandler())

    install_crash_logging(logger, tmp_path)
    from core.startup import disable_crash_logging

    disable_crash_logging()

    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()


def test_crash_log_file_is_created_for_native_faults(tmp_path) -> None:
    logger = logging.getLogger("atlas.operational.test-crash-file")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    install_crash_logging(logger, tmp_path)

    crash_path = tmp_path / "logs" / "atlas_crash.log"
    try:
        assert crash_path.exists()
        assert f"pid=" in crash_path.read_text(encoding="utf-8")
    finally:
        from core.startup import disable_crash_logging

        disable_crash_logging()


class _MemoryHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

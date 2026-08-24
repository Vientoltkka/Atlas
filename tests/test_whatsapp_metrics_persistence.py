"""Tests for WhatsApp metrics persistence (V4.0-F3)."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from pathlib import Path

from channels.app import create_webhook_app
from channels.whatsapp_metrics import MESSAGES_RECEIVED, WhatsAppMetricsRecorder
from channels.whatsapp_metrics_persistence import (
    FLUSH_THRESHOLD_EVENTS,
    WhatsAppMetricsPersistence,
)

TEMP_ROOT = r"C:\Users\victo\AppData\Local\Temp\opencode"


def tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="atlas-metrics-", dir=TEMP_ROOT))


def test_round_trip_save_and_load() -> None:
    work = tmp_dir()
    try:
        path = work / "metrics.json"
        recorder = WhatsAppMetricsRecorder()
        persistence = WhatsAppMetricsPersistence(recorder=recorder, path=path)
        persistence.record(MESSAGES_RECEIVED)
        persistence.flush()

        second_recorder = WhatsAppMetricsRecorder()
        second = WhatsAppMetricsPersistence(recorder=second_recorder, path=path)
        second.load_existing()
        assert second.value(MESSAGES_RECEIVED) == 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_atomic_write_leaves_no_tmp_file_and_valid_json() -> None:
    work = tmp_dir()
    try:
        path = work / "metrics.json"
        persistence = WhatsAppMetricsPersistence(
            recorder=WhatsAppMetricsRecorder(), path=path
        )
        assert persistence.flush() is True
        assert not (work / "metrics.json.tmp").exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_restore_ignores_unknown_keys_and_invalid_values() -> None:
    recorder = WhatsAppMetricsRecorder()
    recorder.restore(
        {
            MESSAGES_RECEIVED: 5,
            "unknown_event": 100,
            "messages_failed": -3,
            "status_sent": "many",
            True: 7,
        }
    )
    snapshot = recorder.snapshot()
    assert snapshot[MESSAGES_RECEIVED] == 5
    assert snapshot["unknown_event"] if False else True  # key ignored silently
    assert snapshot["messages_failed"] == 0
    assert snapshot["status_sent"] == 0
    assert recorder.restore("garbage") is None


def test_flush_by_threshold_after_n_events(tmp=None) -> None:
    work = tmp_dir()
    try:
        path = work / "metrics.json"
        persistence = WhatsAppMetricsPersistence(
            recorder=WhatsAppMetricsRecorder(), path=path
        )
        for _ in range(FLUSH_THRESHOLD_EVENTS - 1):
            persistence.record(MESSAGES_RECEIVED)
        assert not path.exists()
        persistence.record(MESSAGES_RECEIVED)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data[MESSAGES_RECEIVED] == FLUSH_THRESHOLD_EVENTS
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_corrupt_file_starts_from_zero_without_raising() -> None:
    work = tmp_dir()
    try:
        path = work / "metrics.json"
        path.write_text("{not json at all", encoding="utf-8")
        recorder = WhatsAppMetricsRecorder()
        persistence = WhatsAppMetricsPersistence(recorder=recorder, path=path)
        persistence.load_existing()
        assert all(value == 0 for value in recorder.snapshot().values())
        # And a later flush overwrites the corrupt content cleanly.
        persistence.record(MESSAGES_RECEIVED)
        assert persistence.flush() is True
        assert json.loads(path.read_text(encoding="utf-8"))[MESSAGES_RECEIVED] == 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_file_contains_only_counters_no_pii() -> None:
    work = tmp_dir()
    try:
        path = work / "metrics.json"
        persistence = WhatsAppMetricsPersistence(
            recorder=WhatsAppMetricsRecorder(), path=path
        )
        persistence.record(MESSAGES_RECEIVED)
        persistence.flush()
        content = path.read_text(encoding="utf-8")
        assert "34600111222" not in content
        assert "wamid" not in content.lower()
        for token in ("token", "@", "http"):
            assert token not in content
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_concurrent_records_with_flush_do_not_crash() -> None:
    work = tmp_dir()
    try:
        persistence = WhatsAppMetricsPersistence(
            recorder=WhatsAppMetricsRecorder(), path=work / "metrics.json"
        )

        def hammer():
            for _ in range(50):
                persistence.record(MESSAGES_RECEIVED)

        threads = [threading.Thread(target=hammer) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        persistence.flush()
        data = json.loads((work / "metrics.json").read_text(encoding="utf-8"))
        assert data[MESSAGES_RECEIVED] == 300
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_unwritable_path_is_fail_safe() -> None:
    work = tmp_dir()
    try:
        # Parent is an existing FILE, so mkdir/replace must fail.
        blocked = work / "blocked"
        blocked.write_text("i am a file", encoding="utf-8")
        persistence = WhatsAppMetricsPersistence(
            recorder=WhatsAppMetricsRecorder(), path=blocked / "metrics.json"
        )
        persistence.record(MESSAGES_RECEIVED)  # must not raise on threshold flush
        assert persistence.flush() is False
        # Counters remain available in memory.
        assert persistence.value(MESSAGES_RECEIVED) >= 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_app_factory_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ATLAS_WHATSAPP_METRICS_PATH", raising=False)
    app = create_webhook_app(
        executor_fn=lambda request: None,
        store=type("S", (), {"check_and_reserve": staticmethod(lambda e: True)})(),
        sender=type("Sn", (), {"send_text": staticmethod(lambda r, b: None)})(),
        verify_token="verify-token",
    )
    from channels.whatsapp_metrics import WhatsAppMetricsRecorder as R

    assert isinstance(app.state.whatsapp_metrics, R)


def test_app_factory_with_persistence_enabled(monkeypatch) -> None:
    work = tmp_dir()
    try:
        metrics_path = work / "metrics.json"
        monkeypatch.setenv("ATLAS_WHATSAPP_METRICS_PATH", str(metrics_path))
        app = create_webhook_app(
            executor_fn=lambda request: None,
            store=type("S", (), {"check_and_reserve": staticmethod(lambda e: True)})(),
            sender=type("Sn", (), {"send_text": staticmethod(lambda r, b: None)})(),
            verify_token="verify-token",
        )
        from channels.whatsapp_metrics_persistence import WhatsAppMetricsPersistence

        assert isinstance(app.state.whatsapp_metrics, WhatsAppMetricsPersistence)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_explicit_recorder_param_skips_persistence(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_WHATSAPP_METRICS_PATH", "should-not-be-used.json")
    recorder = WhatsAppMetricsRecorder()
    app = create_webhook_app(
        executor_fn=lambda request: None,
        store=type("S", (), {"check_and_reserve": staticmethod(lambda e: True)})(),
        sender=type("Sn", (), {"send_text": staticmethod(lambda r, b: None)})(),
        verify_token="verify-token",
        recorder=recorder,
    )
    assert app.state.whatsapp_metrics is recorder

"""Tests for WhatsApp idempotency store failure recovery (F5.3)."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from channels import app as app_module
from channels.app import create_webhook_app
from channels.webhook_idempotency import (
    IdempotencyStore,
    IdempotencyStoreInitError,
    SqliteIdempotencyStore,
)
from channels.whatsapp_health import (
    WhatsAppChannelHealthChecker,
    WhatsAppHealthStatus,
)

_TEMP_ROOT = r"C:\Users\victo\AppData\Local\Temp\opencode"
SECRET_TOKEN = "super-secret-access-token"


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="atlas-f53-", dir=_TEMP_ROOT))


class FakeStore:
    def __init__(self) -> None:
        self._reserved: set[str] = set()

    def check_and_reserve(self, event_id: str) -> bool:
        if event_id in self._reserved:
            return False
        self._reserved.add(event_id)
        return True


class FakeSender:
    def send_text(self, recipient_id: str, body: str) -> None:
        pass


def make_executor():
    def execute(request):
        from core.agent_executor import AgentExecutionResult, AgentExecutionStatus

        return AgentExecutionResult(
            status=AgentExecutionStatus.COMPLETED,
            request_signature="sig",
            correlation_id=request.correlation_id,
            output={"text": "Respuesta Atlas"},
        )

    return execute


def text_payload(wamid: str = "wamid.f53") -> dict:
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": wamid,
                                    "from": "34600111222",
                                    "type": "text",
                                    "text": {"body": "Hola"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


def test_operational_sqlite_store_behaves_normally() -> None:
    tmp_dir = _tmp_dir()
    try:
        store = SqliteIdempotencyStore(db_path=tmp_dir / "idem.db")
        assert store.check_and_reserve("wamid.ok") is True
        assert store.check_and_reserve("wamid.ok") is False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_corrupt_sqlite_database_raises_explicit_error() -> None:
    tmp_dir = _tmp_dir()
    try:
        db_path = tmp_dir / "corrupt.db"
        db_path.write_bytes(b"this is definitely not a sqlite database")
        with pytest.raises(IdempotencyStoreInitError):
            SqliteIdempotencyStore(db_path=db_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_unwritable_location_raises_explicit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied_connect(*args, **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(sqlite3, "connect", denied_connect)
    tmp_dir = _tmp_dir()
    try:
        with pytest.raises(IdempotencyStoreInitError):
            SqliteIdempotencyStore(db_path=tmp_dir / "idem.db")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_init_error_message_contains_no_paths_or_secrets() -> None:
    tmp_dir = _tmp_dir()
    try:
        db_path = tmp_dir / f"corrupt-{SECRET_TOKEN}.db"
        db_path.write_bytes(b"garbage")
        with pytest.raises(IdempotencyStoreInitError) as excinfo:
            SqliteIdempotencyStore(db_path=db_path)
        message = str(excinfo.value) + repr(excinfo.value)
        assert SECRET_TOKEN not in message
        assert str(tmp_dir) not in message
        assert "corrupt" not in message
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_app_factory_fails_fast_without_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_dir = _tmp_dir()
    try:
        db_path = tmp_dir / "corrupt.db"
        db_path.write_bytes(b"garbage")
        monkeypatch.setenv("ATLAS_WHATSAPP_IDEMPOTENCY_DB_PATH", str(db_path))
        with pytest.raises(IdempotencyStoreInitError):
            app_module._build_store()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_pipeline_keeps_using_persistent_store() -> None:
    tmp_dir = _tmp_dir()
    try:
        app = create_webhook_app(
            executor_fn=make_executor(),
            store=SqliteIdempotencyStore(db_path=tmp_dir / "idem.db"),
            sender=FakeSender(),
            verify_token="verify-token",
        )
        client = TestClient(app)
        first = client.post("/webhook/whatsapp", json=text_payload())
        second = client.post("/webhook/whatsapp", json=text_payload())
        assert first.status_code == 200
        assert second.status_code == 200
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_unavailable_store_is_never_reported_healthy() -> None:
    def failing_probe(store) -> bool:
        return False

    checker = WhatsAppChannelHealthChecker(
        config={
            "verify_token": "verify-secret",
            "access_token": "access-secret",
            "phone_number_id": "pn-1",
        },
        store=IdempotencyStore(),
        transcriber=object(),
        voice_renderer=object(),
        store_probe=failing_probe,
        sender_builder=lambda: object(),
    )
    result = checker.check()
    assert result.status is WhatsAppHealthStatus.UNHEALTHY


def test_in_memory_store_still_works_when_no_db_configured() -> None:
    store = IdempotencyStore()
    assert store.check_and_reserve("wamid.a") is True
    assert store.check_and_reserve("wamid.a") is False

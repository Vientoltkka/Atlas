"""Tests for WhatsApp channel health diagnostics (F5.2)."""

from __future__ import annotations

import threading
import shutil
import tempfile
from pathlib import Path

from channels.app import create_webhook_app
from channels.webhook_idempotency import IdempotencyStore, SqliteIdempotencyStore
from channels.whatsapp_health import (
    WhatsAppChannelHealthChecker,
    WhatsAppHealthCode,
    WhatsAppHealthStatus,
)

_TEMP_ROOT = r"C:\Users\victo\AppData\Local\Temp\opencode"


FULL_CONFIG = {
    "verify_token": "verify-secret",
    "access_token": "access-secret",
    "phone_number_id": "pn-999888777",
}


class StrictNoNetworkSender:
    """Fails the test if any outbound method is ever invoked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _forbidden(self, name: str):
        def method(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"health check must not call {name}")

        return method

    def __getattr__(self, name: str):
        return self._forbidden(name)


def make_checker(
    config=None,
    store=None,
    transcriber=None,
    voice_renderer=None,
    store_probe=None,
    sender_builder=None,
) -> WhatsAppChannelHealthChecker:
    return WhatsAppChannelHealthChecker(
        config=config if config is not None else FULL_CONFIG,
        store=store if store is not None else IdempotencyStore(),
        transcriber=transcriber,
        voice_renderer=voice_renderer,
        store_probe=store_probe,
        sender_builder=sender_builder or StrictNoNetworkSender,
    )


def test_full_config_is_healthy() -> None:
    result = make_checker(
        transcriber=object(),
        voice_renderer=object(),
    ).check()
    assert result.status is WhatsAppHealthStatus.HEALTHY
    assert all(code == WhatsAppHealthCode.OK.value for code in result.checks.values())


def test_missing_verify_token_is_unhealthy() -> None:
    config = dict(FULL_CONFIG, verify_token="")
    result = make_checker(config=config).check()
    assert result.status is WhatsAppHealthStatus.UNHEALTHY
    assert result.checks["verify_token"] == WhatsAppHealthCode.MISSING_CREDENTIAL.value


def test_missing_access_token_is_unhealthy() -> None:
    config = dict(FULL_CONFIG, access_token="")
    result = make_checker(config=config).check()
    assert result.status is WhatsAppHealthStatus.UNHEALTHY
    assert result.checks["access_token"] == WhatsAppHealthCode.MISSING_CREDENTIAL.value


def test_missing_phone_number_id_is_unhealthy() -> None:
    config = dict(FULL_CONFIG, phone_number_id="")
    result = make_checker(config=config).check()
    assert result.status is WhatsAppHealthStatus.UNHEALTHY
    assert result.checks["phone_number_id"] == WhatsAppHealthCode.MISSING_CREDENTIAL.value


def test_inaccessible_sqlite_store_is_unhealthy() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="atlas-health-", dir=_TEMP_ROOT))
    try:
        store = SqliteIdempotencyStore(db_path=tmp_dir / "idem.db")
        # Point the store at an unreachable location after construction.
        store._path = tmp_dir / "missing-dir" / "idem.db"

        checker = WhatsAppChannelHealthChecker(
            config=FULL_CONFIG,
            store=store,
            transcriber=object(),
            voice_renderer=object(),
            sender_builder=StrictNoNetworkSender,
        )
        result = checker.check()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    assert result.status is WhatsAppHealthStatus.UNHEALTHY
    assert (
        result.checks["idempotency_store"] == WhatsAppHealthCode.STORE_UNAVAILABLE.value
    )


def test_raising_store_probe_is_controlled() -> None:
    def failing_probe(candidate) -> bool:
        raise RuntimeError("storage down")

    result = make_checker(store_probe=failing_probe).check()
    assert result.status is WhatsAppHealthStatus.UNHEALTHY
    assert (
        result.checks["idempotency_store"] == WhatsAppHealthCode.STORE_UNAVAILABLE.value
    )


def test_missing_transcriber_is_degraded() -> None:
    result = make_checker(voice_renderer=object()).check()
    assert result.status is WhatsAppHealthStatus.DEGRADED


def test_missing_voice_renderer_is_degraded() -> None:
    result = make_checker(transcriber=object()).check()
    assert result.status is WhatsAppHealthStatus.DEGRADED


def test_full_config_with_optional_components_is_healthy() -> None:
    class FakeTranscriber:
        def transcribe_media_id(self, media_id: str) -> str:
            return "text"

    class FakeRenderer:
        def render(self, text: str) -> Path:
            return Path("reply.ogg")

    result = make_checker(
        transcriber=FakeTranscriber(),
        voice_renderer=FakeRenderer(),
    ).check()
    assert result.status is WhatsAppHealthStatus.HEALTHY


def test_no_secrets_in_result_or_repr() -> None:
    result = make_checker(transcriber=object()).check()
    serialized = repr(result) + repr(result.status) + repr(dict(result.checks))
    assert FULL_CONFIG["verify_token"] not in serialized
    assert FULL_CONFIG["access_token"] not in serialized


def test_no_phone_numbers_in_result() -> None:
    result = make_checker().check()
    serialized = repr(result) + repr(dict(result.checks))
    assert FULL_CONFIG["phone_number_id"] not in serialized
    assert "pn-" not in serialized


def test_failing_probe_and_builder_are_controlled() -> None:
    def broken_probe(store) -> bool:
        raise OSError("disk gone")

    def broken_builder():
        raise RuntimeError("cannot build")

    result = make_checker(
        store_probe=broken_probe,
        sender_builder=broken_builder,
    ).check()
    assert result.status is WhatsAppHealthStatus.UNHEALTHY
    assert result.checks["sender"] == WhatsAppHealthCode.COMPONENT_UNAVAILABLE.value


def test_checker_is_thread_safe_basic() -> None:
    checker = make_checker(transcriber=object(), voice_renderer=object())
    results: list[WhatsAppHealthStatus] = []
    lock = threading.Lock()

    def run() -> None:
        status = checker.check().status
        with lock:
            results.append(status)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(WhatsAppHealthStatus.HEALTHY) == 8


def test_health_check_performs_no_external_calls() -> None:
    sender = StrictNoNetworkSender()

    def build_sender():
        return sender

    result = make_checker(
        transcriber=object(),
        voice_renderer=object(),
        sender_builder=build_sender,
    ).check()
    assert result.status is WhatsAppHealthStatus.HEALTHY
    assert sender.calls == []


def test_health_checker_injected_from_app_factory() -> None:
    captured: dict[str, object] = {}

    class RecordingChecker:
        def check(self):
            captured["checked"] = True
            from channels.whatsapp_health import WhatsAppChannelHealthResult

            return WhatsAppChannelHealthResult(
                status=WhatsAppHealthStatus.HEALTHY,
                checks={"injected": WhatsAppHealthCode.OK.value},
            )

    app = create_webhook_app(
        executor_fn=lambda request: None,
        store=IdempotencyStore(),
        sender=StrictNoNetworkSender(),
        verify_token="verify-secret",
        health_checker=RecordingChecker(),
    )
    stored = app.state.whatsapp_health
    assert isinstance(stored, RecordingChecker)
    stored.check()
    assert captured["checked"] is True


def test_app_factory_builds_default_checker() -> None:
    from channels.whatsapp_health import WhatsAppChannelHealthChecker

    app = create_webhook_app(
        executor_fn=lambda request: None,
        store=IdempotencyStore(),
        sender=StrictNoNetworkSender(),
        verify_token="verify-secret",
        transcriber=object(),
        voice_renderer=object(),
    )
    assert isinstance(app.state.whatsapp_health, WhatsAppChannelHealthChecker)

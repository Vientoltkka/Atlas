"""Tests for WhatsApp webhook rate limiting (V4.0-F2)."""

from __future__ import annotations

import threading
from typing import Any

from fastapi.testclient import TestClient

from channels.app import create_webhook_app
from channels.webhook_idempotency import IdempotencyStore
from channels.whatsapp_metrics import MESSAGES_RECEIVED, RATE_LIMITED, WhatsAppMetricsRecorder
from channels.whatsapp_rate_limit import WhatsAppRateLimiter
from core.agent_executor import AgentExecutionResult, AgentExecutionStatus

VERIFY_TOKEN = "atlas-verify-token"


class FakeStore:
    def __init__(self) -> None:
        self.reserved: list[str] = []
        self._lock = threading.Lock()

    def check_and_reserve(self, event_id: str) -> bool:
        with self._lock:
            if event_id in self.reserved:
                return False
            self.reserved.append(event_id)
            return True


class CountingExecutor:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, request):
        self.calls.append(request)
        return AgentExecutionResult(
            status=AgentExecutionStatus.COMPLETED,
            request_signature="sig",
            correlation_id=request.correlation_id,
            output={"text": "Respuesta Atlas"},
        )


class FakeSender:
    def send_text(self, recipient_id: str, body: str) -> None:
        pass


def text_payload(wamid: str, sender: str = "34600111222") -> dict:
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": wamid,
                                    "from": sender,
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


class FakeClock:
    """Controllable monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_client(
    *,
    limiter=None,
    store=None,
    recorder=None,
    executor=None,
):
    app = create_webhook_app(
        executor_fn=executor or CountingExecutor(),
        store=store or FakeStore(),
        sender=FakeSender(),
        verify_token=VERIFY_TOKEN,
        recorder=recorder,
        rate_limiter=limiter,
    )
    return TestClient(app)


def test_under_limit_all_messages_processed() -> None:
    executor = CountingExecutor()
    limiter = WhatsAppRateLimiter(limit_per_minute=3)
    client = make_client(limiter=limiter, executor=executor)
    for i in range(3):
        response = client.post("/webhook/whatsapp", json=text_payload(f"wamid.{i}"))
        assert response.status_code == 200
    assert len(executor.calls) == 3


def test_excess_messages_rejected_without_execution_or_reservation() -> None:
    executor = CountingExecutor()
    recorder = WhatsAppMetricsRecorder()
    store = FakeStore()
    limiter = WhatsAppRateLimiter(limit_per_minute=1)
    client = make_client(limiter=limiter, store=store, recorder=recorder, executor=executor)

    first = client.post("/webhook/whatsapp", json=text_payload("wamid.a"))
    second = client.post("/webhook/whatsapp", json=text_payload("wamid.b"))

    assert first.status_code == 200 and second.status_code == 200
    assert len(executor.calls) == 1
    assert len(store.reserved) == 1
    assert recorder.value(RATE_LIMITED) == 1
    assert recorder.value(MESSAGES_RECEIVED) == 1


def test_window_expiry_re_admits_sender() -> None:
    clock = FakeClock()
    limiter = WhatsAppRateLimiter(limit_per_minute=1, clock=clock)
    client = make_client(limiter=limiter)
    assert client.post("/webhook/whatsapp", json=text_payload("wamid.1")).status_code == 200
    blocked = client.post("/webhook/whatsapp", json=text_payload("wamid.2"))
    assert blocked.status_code == 200
    clock.now += 61.0
    accepted = client.post("/webhook/whatsapp", json=text_payload("wamid.3"))
    assert accepted.status_code == 200


def test_limits_are_independent_per_sender() -> None:
    limiter = WhatsAppRateLimiter(limit_per_minute=1)
    client = make_client(limiter=limiter)
    assert client.post("/webhook/whatsapp", json=text_payload("wamid.1")).status_code == 200
    other = client.post(
        "/webhook/whatsapp", json=text_payload("wamid.2", sender="34600999888")
    )
    assert other.status_code == 200


def test_keys_and_repr_contain_no_phone_numbers(monkeypatch=None) -> None:
    captured: dict[str, Any] = {}

    class SpyLimiter(WhatsAppRateLimiter):
        def allow(self, sender_key: str) -> bool:
            captured["key"] = sender_key
            return super().allow(sender_key)

    limiter = SpyLimiter(limit_per_minute=10)
    client = make_client(limiter=limiter)
    client.post("/webhook/whatsapp", json=text_payload("wamid.1"))
    serialized = repr(limiter) + repr(captured.get("key"))
    assert "34600111222" not in serialized
    assert captured.get("key", "").startswith("wa_")


def test_broken_limiter_fails_open() -> None:
    class BrokenLimiter:
        def allow(self, sender_key: str) -> bool:
            raise RuntimeError("limiter down")

    executor = CountingExecutor()
    client = make_client(limiter=BrokenLimiter(), executor=executor)
    response = client.post("/webhook/whatsapp", json=text_payload("wamid.ok"))
    assert response.status_code == 200
    assert len(executor.calls) == 1


def test_disabled_limiter_preserves_current_behavior() -> None:
    for disabled in (0, -5):
        limiter = WhatsAppRateLimiter(limit_per_minute=disabled)
        assert limiter.enabled is False
        executor = CountingExecutor()
        client = make_client(limiter=limiter, executor=executor)
        for i in range(25):
            response = client.post("/webhook/whatsapp", json=text_payload(f"wamid.d{i}"))
            assert response.status_code == 200
        assert len(executor.calls) == 25
    # None means no limiter wired at all.
    executor = CountingExecutor()
    client = make_client(limiter=None, executor=executor)
    for i in range(25):
        assert client.post("/webhook/whatsapp", json=text_payload(f"wamid.n{i}")).status_code == 200
    assert len(executor.calls) == 25


def test_concurrent_flood_admits_exactly_limit() -> None:
    limit = 5
    limiter = WhatsAppRateLimiter(limit_per_minute=limit)
    results: list[bool] = []
    lock = threading.Lock()

    def attempt(index: int) -> None:
        allowed = limiter.allow("wa_abcdef123456")
        with lock:
            results.append(allowed)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(results) == limit


def test_env_configuration_disabled_by_default(monkeypatch) -> None:
    from channels import app as app_module

    monkeypatch.delenv("ATLAS_WHATSAPP_RATE_LIMIT_PER_MINUTE", raising=False)
    limiter = app_module._build_rate_limiter()
    assert isinstance(limiter, WhatsAppRateLimiter)
    assert limiter.enabled is False

    monkeypatch.setenv("ATLAS_WHATSAPP_RATE_LIMIT_PER_MINUTE", "30")
    limiter = app_module._build_rate_limiter()
    assert limiter.enabled is True

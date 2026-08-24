"""Tests for V4.1-F1: HTTP health and metrics endpoints."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi.testclient import TestClient

from channels.app import create_webhook_app
from channels.whatsapp_health import WhatsAppChannelHealthChecker
from channels.whatsapp_metrics import MESSAGES_RECEIVED, WhatsAppMetricsRecorder
from channels.whatsapp_metrics_persistence import WhatsAppMetricsPersistence
from channels.webhook_idempotency import IdempotencyStore

VERIFY_TOKEN = "atlas-verify-token"


class FakeSender:
    def send_text(self, recipient_id: str, body: str) -> None:
        pass


def make_checker(
    *,
    access_token: str = "access-secret",
    transcriber=object(),
    voice_renderer=object(),
) -> WhatsAppChannelHealthChecker:
    return WhatsAppChannelHealthChecker(
        config={
            "verify_token": VERIFY_TOKEN,
            "access_token": access_token,
            "phone_number_id": "pnid-123",
        },
        store=IdempotencyStore(),
        transcriber=transcriber,
        voice_renderer=voice_renderer,
    )


def make_client(**kwargs) -> TestClient:
    params = {
        "executor_fn": lambda request: None,
        "store": IdempotencyStore(),
        "sender": FakeSender(),
        "verify_token": VERIFY_TOKEN,
    }
    params.update(kwargs)
    return TestClient(create_webhook_app(**params))


def text_payload(wamid: str = "wamid.ops1") -> dict:
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
                                    "text": {"body": "Hola Atlas"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_healthy_returns_200_with_checks() -> None:
    client = make_client(health_checker=make_checker())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "HEALTHY"
    for name in ("verify_token", "access_token", "phone_number_id", "idempotency_store", "sender"):
        assert body["checks"][name] == "OK"
    assert response.text == response.text  # json serializable


def test_health_missing_credentials_returns_503_unhealthy() -> None:
    client = make_client(health_checker=make_checker(access_token=""))
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "UNHEALTHY"
    assert body["checks"]["access_token"] == "MISSING_CREDENTIAL"


def test_health_degraded_returns_200_and_reports_state() -> None:
    client = make_client(health_checker=make_checker(transcriber=None, voice_renderer=None))
    response = client.get("/health")
    # Degraded keeps the service operational: 200 with the state in the body.
    assert response.status_code == 200
    assert response.json()["status"] == "DEGRADED"


def test_health_checker_exception_becomes_controlled_503(
    caplog: logging.LogCaptureFixture,
) -> None:
    class BrokenChecker:
        def check(self) -> None:
            raise RuntimeError("boom with access-secret inside")

    client = make_client(health_checker=BrokenChecker())
    with caplog.at_level(logging.WARNING, logger="channels.app"):
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "UNHEALTHY", "checks": {}}
    assert "boom" not in caplog.text
    assert "secret" not in response.text


def test_health_never_exposes_secrets() -> None:
    client = make_client(health_checker=make_checker())
    serialized = repr(client.get("/health").content)
    assert VERIFY_TOKEN not in serialized
    assert "access-secret" not in serialized


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


def test_metrics_valid_token_returns_snapshot() -> None:
    client = make_client()
    response = client.get(
        "/metrics", headers={"Authorization": f"Bearer {VERIFY_TOKEN}"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["messages_received"] == 0
    assert set(body.keys()) >= {MESSAGES_RECEIVED}


def test_metrics_without_token_is_rejected() -> None:
    client = make_client()
    response = client.get("/metrics")
    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}


def test_metrics_wrong_token_is_rejected_without_secret_leak() -> None:
    client = make_client()
    for header in ("Bearer wrong-token", f"bearer {VERIFY_TOKEN}", "Basic abc"):
        response = client.get("/metrics", headers={"Authorization": header})
        assert response.status_code == 401
        assert VERIFY_TOKEN not in response.text


def test_metrics_reflect_real_activity_after_webhook_post() -> None:
    executor_calls: list = []

    def executor(request):
        executor_calls.append(request)

        from core.agent_executor import AgentExecutionResult, AgentExecutionStatus

        return AgentExecutionResult(
            status=AgentExecutionStatus.COMPLETED,
            request_signature="sig",
            correlation_id=request.correlation_id,
            output={"text": "respuesta"},
        )

    client = make_client(executor_fn=executor)
    post_response = client.post("/webhook/whatsapp", json=text_payload())
    assert post_response.status_code == 200
    response = client.get(
        "/metrics", headers={"Authorization": f"Bearer {VERIFY_TOKEN}"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["messages_received"] == 1
    assert len(executor_calls) == 1


def test_metrics_endpoint_reflects_loaded_persistence(tmp_path: Path) -> None:
    recorder = WhatsAppMetricsRecorder()
    recorder.record("status_delivered")
    recorder.record("status_read")

    persistence_path = tmp_path / "metrics.json"
    first = WhatsAppMetricsPersistence(recorder=recorder, path=persistence_path)
    assert first.flush()

    loaded_recorder = WhatsAppMetricsRecorder()
    persistence = WhatsAppMetricsPersistence(recorder=loaded_recorder, path=persistence_path)
    persistence.load_existing()

    client = make_client(recorder=persistence)
    response = client.get(
        "/metrics", headers={"Authorization": f"Bearer {VERIFY_TOKEN}"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status_delivered"] == 1
    assert body["status_read"] == 1


def test_metrics_rejection_never_logs_the_token(caplog: logging.LogCaptureFixture) -> None:
    client = make_client()
    with caplog.at_level(logging.DEBUG):
        client.get("/metrics", headers={"Authorization": "Bearer wrong"})
        client.get("/metrics")
    assert VERIFY_TOKEN not in caplog.text


# ---------------------------------------------------------------------------
# Webhook regressions through the same app
# ---------------------------------------------------------------------------


def test_regression_webhook_verification_still_works() -> None:
    client = make_client()
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub_mode": "subscribe",
            "hub_verify_token": VERIFY_TOKEN,
            "hub_challenge": "challenge123",
        },
    )
    assert response.status_code == 200
    assert response.text == "challenge123"


def test_regression_webhook_post_still_works() -> None:
    client = make_client(health_checker=make_checker())
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200
    health = client.get("/health")
    assert health.status_code == 200

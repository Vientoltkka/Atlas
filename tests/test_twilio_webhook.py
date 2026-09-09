"""Focal tests for the Twilio WhatsApp webhook MVP (text-only).

Twilio SDK REST calls are mocked; no network access happens here.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from channels.app import build_twilio_webhook_app
from channels.twilio_sender import TwilioWhatsAppSender
from channels.webhook_idempotency import IdempotencyStore
from core.agent_executor import AgentExecutionResult, AgentExecutionStatus


AUTH_TOKEN = "atlas-twilio-auth-token"
FROM_NUMBER = "whatsapp:+34600000000"
ALLOWED_SENDER = "whatsapp:+34600111222"
ENDPOINT = "/webhook/twilio/whatsapp"
URL = f"http://testserver{ENDPOINT}"


class FakeStore(IdempotencyStore):
    """Reuse of the real in-memory store contract."""


class FakeSender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def send_text(self, recipient_id: str, body: str) -> None:
        with self._lock:
            self.sent.append((recipient_id, body))


class RecordingMessages:
    def __init__(self, created: list[dict[str, Any]]) -> None:
        self._created = created

    def create(self, **kwargs: Any) -> None:
        self._created.append(kwargs)


class RecordingTwilioClient:
    """Twilio REST client mock: captures messages.create kwargs."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.messages = RecordingMessages(self.created)


def make_executor(results: list[AgentExecutionResult] | None = None, hook: Any = None):
    calls: list[Any] = []

    def execute(request):
        calls.append(request)
        if hook is not None:
            hook(request)
        if results:
            return results.pop(0)
        return AgentExecutionResult(
            status=AgentExecutionStatus.COMPLETED,
            request_signature="sig",
            correlation_id=request.correlation_id,
            output={"text": "Respuesta Atlas"},
        )

    execute.calls = calls  # type: ignore[attr-defined]
    return execute


def form_params(
    sender: str = ALLOWED_SENDER,
    body: str = "Hola Atlas",
    message_sid: str = "SMtwilio0001",
    num_media: str = "0",
) -> dict[str, str]:
    return {
        "From": sender,
        "Body": body,
        "MessageSid": message_sid,
        "NumMedia": num_media,
    }


def signed_post(client: TestClient, params: dict[str, str], token: str = AUTH_TOKEN):
    signature = RequestValidator(token).compute_signature(URL, params)
    return client.post(ENDPOINT, data=params, headers={"X-Twilio-Signature": signature})


def make_client(executor=None, sender=None, store=None, allowed_numbers=None):
    app = build_twilio_webhook_app(
        executor_fn=executor or make_executor(),
        sender=sender or FakeSender(),
        store=store or FakeStore(),
        auth_token=AUTH_TOKEN,
        from_number=FROM_NUMBER,
        allowed_numbers=(
            allowed_numbers
            if allowed_numbers is not None
            else frozenset({ALLOWED_SENDER})
        ),
    )
    return TestClient(app)


def wait_for(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_valid_signature_executes_atlas_and_replies():
    sender = FakeSender()
    executor = make_executor()
    client = make_client(executor=executor, sender=sender)

    response = signed_post(client, form_params())

    assert response.status_code == 200
    assert wait_for(lambda: len(sender.sent) == 1)
    recipient, body = sender.sent[0]
    assert recipient == ALLOWED_SENDER
    assert body == "Respuesta Atlas"
    assert wait_for(lambda: len(executor.calls) == 1)
    assert executor.calls[0].user_input == "Hola Atlas"


def test_invalid_signature_rejected_without_execution():
    sender = FakeSender()
    executor = make_executor()
    client = make_client(executor=executor, sender=sender)

    params = form_params()
    signature = RequestValidator("wrong-token").compute_signature(URL, params)
    response = client.post(ENDPOINT, data=params, headers={"X-Twilio-Signature": signature})

    assert response.status_code == 401
    time.sleep(0.1)
    assert executor.calls == []
    assert sender.sent == []


def test_missing_signature_rejected_without_execution():
    sender = FakeSender()
    executor = make_executor()
    client = make_client(executor=executor, sender=sender)

    response = client.post(ENDPOINT, data=form_params())

    assert response.status_code == 401
    time.sleep(0.1)
    assert executor.calls == []
    assert sender.sent == []


def test_raw_urlencoded_body_without_signature_returns_401_not_400():
    """Regression: a real curl POST (raw urlencoded bytes, no signature) must
    reach signature validation (401), never a premature empty 400 from form
    parsing (e.g. missing python-multipart)."""
    sender = FakeSender()
    executor = make_executor()
    client = make_client(executor=executor, sender=sender)

    raw = (
        "From=whatsapp%3A%2B34600111222&Body=hola"
        "&MessageSid=SMTEST125&NumMedia=0"
    )
    response = client.post(
        ENDPOINT,
        content=raw.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 401
    assert response.content == b""
    time.sleep(0.1)
    assert executor.calls == []
    assert sender.sent == []


def test_unauthorized_number_rejected_without_execution():
    sender = FakeSender()
    executor = make_executor()
    client = make_client(executor=executor, sender=sender)

    intruder = "whatsapp:+34699988877"
    response = signed_post(client, form_params(sender=intruder))

    assert response.status_code == 200
    time.sleep(0.1)
    assert executor.calls == []
    assert sender.sent == []


def test_empty_allowlist_rejects_every_sender():
    sender = FakeSender()
    executor = make_executor()
    client = make_client(executor=executor, sender=sender, allowed_numbers=frozenset())

    response = signed_post(client, form_params())

    assert response.status_code == 200
    time.sleep(0.1)
    assert executor.calls == []
    assert sender.sent == []


def test_duplicate_message_sid_executed_once():
    sender = FakeSender()
    executor = make_executor()
    client = make_client(executor=executor, sender=sender)

    first = signed_post(client, form_params())
    assert first.status_code == 200
    duplicate = signed_post(client, form_params())
    assert duplicate.status_code == 200

    assert wait_for(lambda: len(executor.calls) == 1)
    assert wait_for(lambda: len(sender.sent) == 1)
    time.sleep(0.1)
    assert len(executor.calls) == 1
    assert len(sender.sent) == 1


def test_body_reaches_executor_with_pseudonymized_sender():
    executor = make_executor()
    client = make_client(executor=executor, sender=FakeSender())

    signed_post(client, form_params(body="Estado del sistema"))

    assert wait_for(lambda: len(executor.calls) == 1)
    request = executor.calls[0]
    assert request.user_input == "Estado del sistema"
    assert ALLOWED_SENDER not in request.session_id
    assert "twi_" in request.session_id


def test_webhook_responds_without_blocking_on_slow_executor():
    started = threading.Event()
    release = threading.Event()

    def hook(request):
        started.set()
        release.wait(timeout=10)

    sender = FakeSender()
    executor = make_executor(hook=hook)
    client = make_client(executor=executor, sender=sender)

    begin = time.perf_counter()
    response = signed_post(client, form_params())
    elapsed = time.perf_counter() - begin

    assert response.status_code == 200
    assert elapsed < 2.0, "the webhook must not wait for Atlas execution"
    assert started.wait(timeout=5)
    release.set()
    assert wait_for(lambda: len(sender.sent) == 1)


def test_media_message_gets_courtesy_reply_without_atlas():
    sender = FakeSender()
    executor = make_executor()
    client = make_client(executor=executor, sender=sender)

    response = signed_post(client, form_params(num_media="1"))

    assert response.status_code == 200
    assert wait_for(lambda: len(sender.sent) == 1)
    assert sender.sent[0][1] == "Solo puedo procesar mensajes de texto por ahora."
    time.sleep(0.1)
    assert executor.calls == []


def test_executor_failure_sends_single_controlled_error_reply():
    sender = FakeSender()

    def execute(request):
        raise RuntimeError("atlas exploded")

    client = make_client(executor=execute, sender=sender)

    response = signed_post(client, form_params())

    assert response.status_code == 200
    assert wait_for(lambda: len(sender.sent) == 1)
    assert sender.sent[0][1] == "Se ha producido un error procesando tu mensaje."
    time.sleep(0.1)
    assert len(sender.sent) == 1


def test_twilio_sender_uses_messages_create_with_text_only():
    recording = RecordingTwilioClient()
    sender = TwilioWhatsAppSender(
        account_sid="ACtest",
        auth_token="auth-token",
        from_number="whatsapp:+34600000000",
        client_factory=lambda: recording,
    )

    sender.send_text("whatsapp:+34600111222", "Respuesta final")

    assert recording.created == [
        {
            "to": "whatsapp:+34600111222",
            "from_": "whatsapp:+34600000000",
            "body": "Respuesta final",
        }
    ]


def test_twilio_sender_trial_mode_uses_content_sid_instead_of_body():
    recording = RecordingTwilioClient()
    sender = TwilioWhatsAppSender(
        account_sid="ACtest",
        auth_token="auth-token",
        from_number="whatsapp:+14155238886",
        client_factory=lambda: recording,
        content_sid="HXaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )

    sender.send_text("whatsapp:+34600111222", "Respuesta final")

    assert recording.created == [
        {
            "to": "whatsapp:+34600111222",
            "from_": "whatsapp:+14155238886",
            "content_sid": "HXaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        }
    ]


def test_twilio_sender_trial_mode_passes_content_variables_when_present():
    recording = RecordingTwilioClient()
    sender = TwilioWhatsAppSender(
        account_sid="ACtest",
        auth_token="auth-token",
        from_number="whatsapp:+14155238886",
        client_factory=lambda: recording,
        content_sid="HXaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        content_variables='{"1": "valor"}',
    )

    sender.send_text("whatsapp:+34600111222", "Respuesta final")

    assert recording.created == [
        {
            "to": "whatsapp:+34600111222",
            "from_": "whatsapp:+14155238886",
            "content_sid": "HXaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "content_variables": '{"1": "valor"}',
        }
    ]


def test_twilio_sender_rejects_invalid_construction():
    with pytest.raises(ValueError):
        TwilioWhatsAppSender(account_sid="", auth_token="t", from_number="whatsapp:+1")
    with pytest.raises(ValueError):
        TwilioWhatsAppSender(account_sid="AC", auth_token="", from_number="whatsapp:+1")
    with pytest.raises(ValueError):
        TwilioWhatsAppSender(account_sid="AC", auth_token="t", from_number="")


def test_twilio_sender_default_factory_builds_real_sdk_client():
    """Regression: real twilio 9.x Client signature (no ``timeout=`` kwarg).

    Building the client is offline; only messages.create hits the network.
    """
    import twilio.http.http_client as http_client_module
    from twilio.rest import Client as RealClient

    sender = TwilioWhatsAppSender(
        account_sid="ACtest",
        auth_token="auth-token",
        from_number="whatsapp:+34600000000",
        timeout_seconds=3.0,
    )

    client = sender._get_client()

    assert isinstance(client, RealClient)
    assert isinstance(client.http_client, http_client_module.TwilioHttpClient)
    assert client.http_client.timeout == 3.0
    assert sender._get_client() is client

"""Tests for the Atlas WhatsApp webhook (Phase 2)."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from channels.app import create_webhook_app
from channels.webhook_idempotency import IdempotencyStore
from core.agent_executor import AgentExecutionResult, AgentExecutionStatus


VERIFY_TOKEN = "atlas-verify-token"
APP_SECRET = "atlas-app-secret"


class FakeStore:
    """Thread-safe in-memory idempotency store mirroring IdempotencyStore."""

    def __init__(self) -> None:
        self._reserved: set[str] = set()
        self._lock = threading.Lock()

    def check_and_reserve(self, event_id: str) -> bool:
        with self._lock:
            if event_id in self._reserved:
                return False
            self._reserved.add(event_id)
            return True


class FakeSender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send_text(self, recipient_id: str, body: str) -> None:
        self.sent.append((recipient_id, body))


def make_executor(results: list[AgentExecutionResult] | None = None):
    calls: list[Any] = []

    def execute(request):
        calls.append(request)
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


def text_payload(wamid: str = "wamid.test001", sender: str = "34600111222", body: str = "Hola Atlas") -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "entry1",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "34600000000", "phone_number_id": "pnid"},
                            "contacts": [{"profile": {"name": "Usuario"}, "wa_id": sender}],
                            "messages": [
                                {"id": wamid, "from": sender, "timestamp": "1", "type": "text", "text": {"body": body}}
                            ],
                        },
                    }
                ],
            }
        ],
    }


def _signed_client(client: TestClient) -> TestClient:
    original_post = client.post

    def post(url: str, **kwargs):
        body = kwargs.pop("content", None)
        if body is None and "json" in kwargs:
            body = json.dumps(kwargs.pop("json"), separators=(",", ":")).encode("utf-8")
            kwargs.setdefault("headers", {})["Content-Type"] = "application/json"
        if body is not None:
            if isinstance(body, str):
                body = body.encode("utf-8")
            headers = dict(kwargs.pop("headers", {}))
            headers["X-Hub-Signature-256"] = "sha256=" + hmac.new(APP_SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
            kwargs["headers"] = headers
            kwargs["content"] = body
        return original_post(url, **kwargs)

    client.post = post  # type: ignore[method-assign]
    return client


def make_client(executor=None, store=None, sender=None, verify_token: str = VERIFY_TOKEN, **kwargs):
    return _signed_client(TestClient(
        create_webhook_app(
            executor_fn=executor or make_executor(),
            store=store or FakeStore(),
            sender=sender or FakeSender(),
            verify_token=verify_token,
            app_secret=APP_SECRET,
            **kwargs,
        )
    ))


def test_get_verification_ok() -> None:
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


def test_get_verification_wrong_token() -> None:
    client = make_client()
    response = client.get(
        "/webhook/whatsapp",
        params={"hub_mode": "subscribe", "hub_verify_token": "wrong", "hub_challenge": "x"},
    )
    assert response.status_code == 403


def test_post_text_message_full_flow() -> None:
    executor = make_executor()
    sender = FakeSender()
    client = make_client(executor=executor, sender=sender)
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200
    assert len(executor.calls) == 1
    request = executor.calls[0]
    assert request.user_input == "Hola Atlas"
    assert request.correlation_id.startswith("wa-")
    assert len(request.correlation_id) <= 128
    assert request.session_id.startswith("whatsapp-wa_")
    assert sender.sent == [("34600111222", "Respuesta Atlas")]


def test_post_json_malformed_returns_400() -> None:
    client = make_client()
    response = client.post(
        "/webhook/whatsapp",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400


def test_post_empty_payload_returns_200() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    assert client.post("/webhook/whatsapp", json={}).status_code == 200
    assert client.post("/webhook/whatsapp", json={"foo": "bar"}).status_code == 200
    assert client.post("/webhook/whatsapp", json=[1, 2, 3]).status_code == 200
    assert len(executor.calls) == 0


def test_post_status_ack_ignored() -> None:
    executor = make_executor()
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "statuses": [{"id": "wamid.s1", "status": "delivered"}],
                        }
                    }
                ]
            }
        ]
    }
    response = make_client(executor=executor).post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 0


def test_unsupported_message_type_sends_courtesy() -> None:
    executor = make_executor()
    sender = FakeSender()
    payload = text_payload(wamid="wamid.audio1")
    payload["entry"][0]["changes"][0]["value"]["messages"][0] = {
        "id": "wamid.audio1",
        "from": "34600111222",
        "type": "audio",
        "audio": {"id": "x"},
    }
    response = make_client(executor=executor, sender=sender).post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 0
    assert sender.sent and "texto" in sender.sent[0][1]


def test_duplicate_wamid_executes_once() -> None:
    executor = make_executor()
    store = FakeStore()
    sender = FakeSender()
    client = make_client(executor=executor, store=store, sender=sender)
    first = client.post("/webhook/whatsapp", json=text_payload())
    second = client.post("/webhook/whatsapp", json=text_payload())
    assert first.status_code == 200 and second.status_code == 200
    assert len(executor.calls) == 1
    assert len(sender.sent) == 1


def test_concurrent_duplicates_execute_exactly_once() -> None:
    executor = make_executor()

    class SlowFakeStore(FakeStore):
        """Widens the race window inside check_and_reserve."""

        def check_and_reserve(self, event_id: str) -> bool:
            with self._lock:
                if event_id in self._reserved:
                    return False
                import time

                time.sleep(0.2)
                self._reserved.add(event_id)
                return True

    store = SlowFakeStore()
    sender = FakeSender()
    client = make_client(executor=executor, store=store, sender=sender)
    responses: list[Any] = []
    lock = threading.Lock()

    def post() -> None:
        response = client.post("/webhook/whatsapp", json=text_payload(wamid="wamid.race"))
        with lock:
            responses.append(response)

    threads = [threading.Thread(target=post) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(r.status_code == 200 for r in responses)
    assert len(executor.calls) == 1
    assert len(sender.sent) == 1


def test_executor_failure_returns_200_and_notifies_user() -> None:
    def failing_executor(request):
        raise RuntimeError("boom")

    sender = FakeSender()
    client = make_client(executor=failing_executor, sender=sender)
    response = client.post("/webhook/whatsapp", json=text_payload())
    assert response.status_code == 200
    assert sender.sent and "error" in sender.sent[0][1].lower()


def test_infrastructure_failure_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    from channels import whatsapp_webhook as module

    def broken_extract(payload: Any) -> Any:
        raise ConnectionError("storage down")

    monkeypatch.setattr(module, "_iter_change_values", broken_extract)
    client = _signed_client(TestClient(
        create_webhook_app(
            executor_fn=make_executor(),
            store=FakeStore(),
            sender=FakeSender(),
            verify_token=VERIFY_TOKEN,
            app_secret=APP_SECRET,
        )
    )
    )
    with pytest.raises(ConnectionError):
        client.post("/webhook/whatsapp", json=text_payload())


def test_sender_body_never_contains_access_token() -> None:
    sender = FakeSender()
    client = make_client(sender=sender)
    client.post("/webhook/whatsapp", json=text_payload())
    serialized = repr(sender.sent) + repr(client.app.routes)
    assert "secret-access-token" not in serialized


def test_idempotency_store_restart_allows_reprocessing_documented() -> None:
    store_a = IdempotencyStore()
    assert store_a.check_and_reserve("wamid.x") is True
    assert store_a.check_and_reserve("wamid.x") is False
    store_b = IdempotencyStore()
    assert store_b.check_and_reserve("wamid.x") is True


def test_correlation_id_length_limit() -> None:
    long_wamid = "wamid." + "A" * 300
    executor = make_executor()
    client = make_client(executor=executor)
    client.post("/webhook/whatsapp", json=text_payload(wamid=long_wamid))
    assert executor.calls
    correlation_id = executor.calls[0].correlation_id
    assert correlation_id is not None and len(correlation_id) <= 128


def _image_payload(wamid: str = "wamid.img1", caption: str | None = "Que entrenamiento hago hoy?") -> dict:
    image: dict = {"id": "media123", "mime_type": "image/jpeg", "sha256": "abc"}
    if caption is not None:
        image["caption"] = caption
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"id": wamid, "from": "34600111222", "type": "image", "image": image}
                            ]
                        }
                    }
                ]
            }
        ]
    }


def test_image_with_caption_processed_as_text() -> None:
    executor = make_executor()
    sender = FakeSender()
    client = make_client(executor=executor, sender=sender)
    response = client.post("/webhook/whatsapp", json=_image_payload())
    assert response.status_code == 200
    assert len(executor.calls) == 1
    assert executor.calls[0].user_input == "Que entrenamiento hago hoy?"


def test_image_without_caption_sends_courtesy_and_skips_atlas() -> None:
    executor = make_executor()
    sender = FakeSender()
    client = make_client(executor=executor, sender=sender)
    response = client.post("/webhook/whatsapp", json=_image_payload(caption=None))
    assert response.status_code == 200
    assert len(executor.calls) == 0
    assert sender.sent and "texto" in sender.sent[0][1]


def test_image_duplicate_wamid_executes_once() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    first = client.post("/webhook/whatsapp", json=_image_payload())
    second = client.post("/webhook/whatsapp", json=_image_payload())
    assert first.status_code == 200 and second.status_code == 200
    assert len(executor.calls) == 1


def test_regression_text_audio_document_unchanged() -> None:
    executor = make_executor()
    sender = FakeSender()
    client = make_client(executor=executor, sender=sender)

    text_response = client.post("/webhook/whatsapp", json=text_payload())
    audio_payload = {
        "entry": [{"changes": [{"value": {"messages": [
            {"id": "wamid.aud", "from": "34600", "type": "audio", "audio": {"id": "x"}}
        ]}}]}]
    }
    document_payload = {
        "entry": [{"changes": [{"value": {"messages": [
            {
                "id": "wamid.doc",
                "from": "34600",
                "type": "document",
                "document": {"id": "y", "filename": "informe.pdf"},
            }
        ]}}]}]
    }
    audio_response = client.post("/webhook/whatsapp", json=audio_payload)
    document_response = client.post("/webhook/whatsapp", json=document_payload)

    assert text_response.status_code == 200
    assert audio_response.status_code == 200
    assert document_response.status_code == 200
    # text + normalized document reach Atlas; audio without transcriber does not.
    assert len(executor.calls) == 2
    assert executor.calls[1].user_input == "[documento: informe.pdf]"
    courtesy_count = sum(1 for _, body in sender.sent if "texto" in body)
    assert courtesy_count == 1  # audio keeps the courtesy message


# ---------------------------------------------------------------------------
# V4.0-F4: inbound non-text message coverage
# ---------------------------------------------------------------------------


def _message_payload(wamid: str, mtype: str, block: dict | list) -> dict:
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"id": wamid, "from": "34600111222", "type": mtype, mtype: block}
                            ]
                        }
                    }
                ]
            }
        ]
    }


def test_document_with_filename_and_caption_reaches_executor() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    payload = _message_payload(
        "wamid.doc1", "document",
        {"id": "d1", "filename": "contrato.pdf", "caption": "revisa la clausula 3"},
    )
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 1
    assert executor.calls[0].user_input == "[documento: contrato.pdf] revisa la clausula 3"


def test_document_without_filename_uses_caption() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    payload = _message_payload("wamid.doc2", "document", {"id": "d2", "caption": "resumelo"})
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 1
    assert executor.calls[0].user_input == "[documento] resumelo"


def test_document_without_caption_keeps_filename() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    payload = _message_payload("wamid.doc3", "document", {"id": "d3", "filename": "notas.txt"})
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 1
    assert executor.calls[0].user_input == "[documento: notas.txt]"


def test_document_malformed_sends_courtesy() -> None:
    executor = make_executor()
    sender = FakeSender()
    client = make_client(executor=executor, sender=sender)
    for block in ({"id": "x"}, {}, "no-mapping", None):
        payload = _message_payload(f"wamid.docm-{id(block)}", "document", block)
        response = client.post("/webhook/whatsapp", json=payload)
        assert response.status_code == 200
    assert len(executor.calls) == 0
    assert sender.sent and all("texto" in body for _, body in sender.sent)


def test_location_with_coordinates_reaches_executor() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    payload = _message_payload(
        "wamid.loc1", "location",
        {"latitude": 40.4168, "longitude": -3.7038},
    )
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 1
    assert executor.calls[0].user_input == "[ubicación: lat=40.4168, lng=-3.7038]"


def test_location_with_name_and_address_included() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    payload = _message_payload(
        "wamid.loc2", "location",
        {
            "latitude": 40.4168,
            "longitude": -3.7038,
            "name": "Casa",
            "address": "Calle Mayor 1",
        },
    )
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 1
    assert executor.calls[0].user_input == (
        "[ubicación: lat=40.4168, lng=-3.7038] Casa Calle Mayor 1"
    )


def test_location_invalid_coordinates_sends_courtesy() -> None:
    executor = make_executor()
    sender = FakeSender()
    client = make_client(executor=executor, sender=sender)
    for block in ({}, {"latitude": "40", "longitude": -3.7}, {"latitude": True, "longitude": 0}):
        payload = _message_payload(f"wamid.loci-{id(block)}", "location", block)
        response = client.post("/webhook/whatsapp", json=payload)
        assert response.status_code == 200
    assert len(executor.calls) == 0
    assert sender.sent and all("texto" in body for _, body in sender.sent)


def test_contacts_display_name_reaches_executor() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    payload = _message_payload(
        "wamid.con1", "contacts",
        [{"name": {"formatted_name": "Ana Garcia", "first_name": "Ana"}, "phones": []}],
    )
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 1
    assert executor.calls[0].user_input == "[contacto: Ana Garcia]"


def test_contacts_malformed_sends_courtesy() -> None:
    executor = make_executor()
    sender = FakeSender()
    client = make_client(executor=executor, sender=sender)
    for block in ([], [{"phones": ["123"]}], [{"name": {}}], "bad"):
        payload = _message_payload(f"wamid.conm-{id(block)}", "contacts", block)
        response = client.post("/webhook/whatsapp", json=payload)
        assert response.status_code == 200
    assert len(executor.calls) == 0
    assert sender.sent and all("texto" in body for _, body in sender.sent)


def test_button_reply_title_reaches_executor() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    payload = _message_payload(
        "wamid.btn1", "interactive",
        {"type": "button_reply", "button_reply": {"id": "btn_si", "title": "Confirmar"}},
    )
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 1
    assert executor.calls[0].user_input == "Confirmar"


def test_list_reply_title_reaches_executor() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    payload = _message_payload(
        "wamid.lst1", "interactive",
        {"type": "list_reply", "list_reply": {"id": "row2", "title": "Ver agenda"}},
    )
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 1
    assert executor.calls[0].user_input == "Ver agenda"


def test_interactive_malformed_sends_courtesy() -> None:
    executor = make_executor()
    sender = FakeSender()
    client = make_client(executor=executor, sender=sender)
    for block in ({}, {"button_reply": {}}, {"list_reply": "bad"}, None):
        payload = _message_payload(f"wamid.intm-{id(block)}", "interactive", block)
        response = client.post("/webhook/whatsapp", json=payload)
        assert response.status_code == 200
    assert len(executor.calls) == 0
    assert sender.sent and all("texto" in body for _, body in sender.sent)


def test_truly_unknown_types_keep_courtesy() -> None:
    executor = make_executor()
    sender = FakeSender()
    client = make_client(executor=executor, sender=sender)
    for wamid, mtype in (("wamid.stk", "sticker"), ("wamid.ord", "order"), ("wamid.xyz", "totally_new")):
        payload = _message_payload(wamid, mtype, {})
        response = client.post("/webhook/whatsapp", json=payload)
        assert response.status_code == 200
    assert len(executor.calls) == 0
    assert len(sender.sent) == 3
    assert all("texto" in body for _, body in sender.sent)


def test_normalized_text_respects_max_length() -> None:
    from channels.whatsapp_channel import MAX_WHATSAPP_TEXT_LENGTH

    executor = make_executor()
    client = make_client(executor=executor)
    long_caption = "c" * (MAX_WHATSAPP_TEXT_LENGTH + 500)
    payload = _message_payload(
        "wamid.doclong", "document",
        {"id": "d9", "filename": "a.pdf", "caption": long_caption},
    )
    response = client.post("/webhook/whatsapp", json=payload)
    assert response.status_code == 200
    assert len(executor.calls) == 1
    user_input = executor.calls[0].user_input
    assert len(user_input) <= MAX_WHATSAPP_TEXT_LENGTH


def test_document_duplicate_wamid_executes_once() -> None:
    executor = make_executor()
    client = make_client(executor=executor)
    payload = _message_payload(
        "wamid.docdup", "document", {"id": "dd", "filename": "informe.pdf"}
    )
    first = client.post("/webhook/whatsapp", json=payload)
    second = client.post("/webhook/whatsapp", json=payload)
    assert first.status_code == 200 and second.status_code == 200
    assert len(executor.calls) == 1


def test_rate_limit_applies_to_new_message_types() -> None:
    from channels.whatsapp_rate_limit import WhatsAppRateLimiter

    class OneShotLimiter:
        def __init__(self) -> None:
            self.allowed_first = True

        def allow(self, sender_key: str) -> bool:
            result = self.allowed_first
            self.allowed_first = False
            return result

    executor = make_executor()
    sender = FakeSender()
    limiter = OneShotLimiter()
    client = make_client(executor=executor, store=FakeStore(), sender=sender, rate_limiter=limiter)
    payloads = [
        _message_payload("wamid.rl-doc", "document", {"id": "rld", "filename": "f.pdf"}),
        _message_payload("wamid.rl-loc", "location", {"latitude": 1, "longitude": 2}),
        _message_payload(
            "wamid.rl-btn", "interactive",
            {"type": "button_reply", "button_reply": {"title": "Si"}},
        ),
    ]
    for index, payload in enumerate(payloads):
        response = client.post("/webhook/whatsapp", json=payload)
        assert response.status_code == 200
    # Only the first request passes the limiter; later ones are dropped
    # silently before idempotency and never reach Atlas.
    assert len(executor.calls) == 1


def test_f4_payload_content_never_leaks_into_logs(caplog: pytest.LogCaptureFixture) -> None:
    import logging as logging_module

    executor = make_executor()
    sender = FakeSender()
    client = make_client(executor=executor, sender=sender)
    secret_filename = "contrato-secreto-xyz.pdf"
    contact_name = "Persona-Muy-Privada"
    address = "Calle-Privada-99"
    with caplog.at_level(logging_module.DEBUG, logger="channels.whatsapp_webhook"):
        client.post(
            "/webhook/whatsapp",
            json=_message_payload(
                "wamid.pii1", "document",
                {"id": "pii", "filename": secret_filename, "caption": "hola"},
            ),
        )
        client.post(
            "/webhook/whatsapp",
            json=_message_payload(
                "wamid.pii2", "location",
                {"latitude": 1.5, "longitude": 2.5, "name": "Sitio", "address": address},
            ),
        )
        client.post(
            "/webhook/whatsapp",
            json=_message_payload(
                "wamid.pii3", "contacts",
                [{"name": {"formatted_name": contact_name}}],
            ),
        )
    serialized_logs = caplog.text
    assert secret_filename not in serialized_logs
    assert contact_name not in serialized_logs
    assert address not in serialized_logs

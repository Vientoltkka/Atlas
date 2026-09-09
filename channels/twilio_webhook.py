"""Twilio WhatsApp webhook endpoint for Atlas (MVP text-only).

Flow: form parse -> X-Twilio-Signature validation -> allowlist check ->
MessageSid idempotency reserve (BEFORE responding) -> immediate 200 to
Twilio -> Atlas execution + outbound delivery in a daemon thread.

HTTP policy mirrors channels/whatsapp_webhook.py: functional errors are
absorbed with an empty 200 (Twilio must not retry); only infrastructure
problems return 5xx.

Media messages are out of scope for this MVP: they get the same courtesy
text reply used by the WhatsApp channel and never reach Atlas.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Callable, Iterable

from fastapi import APIRouter, Request, Response
from twilio.request_validator import RequestValidator

from channels.base_channel import InvalidChannelMessageError
from channels.whatsapp_channel import WhatsAppChannel
from core.agent_executor import AgentExecutionRequest, AgentExecutionResult

logger = logging.getLogger(__name__)

COURTESY_MEDIA_REPLY = "Solo puedo procesar mensajes de texto por ahora."
ERROR_REPLY = "Se ha producido un error procesando tu mensaje."
UNPARSABLE_REPLY = "No he podido entender el mensaje."


def build_twilio_webhook_router(
    *,
    channel: WhatsAppChannel,
    executor_fn: Callable[[AgentExecutionRequest], AgentExecutionResult],
    sender: Any,
    auth_token: str,
    store: Any,
    allowed_numbers: Iterable[str] = frozenset(),
) -> APIRouter:
    """Compose the Twilio WhatsApp webhook router with injected dependencies.

    The allowlist is fail-closed: an empty allowlist rejects every sender.
    """
    validator = RequestValidator(auth_token)
    allowed = frozenset(normalize_twilio_number(entry) for entry in allowed_numbers)
    router = APIRouter(prefix="/webhook/twilio")

    @router.post("/whatsapp")
    async def receive_twilio_whatsapp(request: Request) -> Response:
        try:
            form = await request.form()
        except Exception:
            logger.warning("twilio webhook received unreadable form body")
            return Response(status_code=400)
        params = {key: form[key] for key in form.keys() if isinstance(form[key], str)}
        signature = request.headers.get("X-Twilio-Signature", "")
        if not signature or not validator.validate(str(request.url), params, signature):
            logger.warning("twilio webhook signature rejected")
            return Response(status_code=401)

        from_number = params.get("From", "").strip()
        message_sid = params.get("MessageSid", "").strip()
        body = params.get("Body", "").strip()
        num_media = params.get("NumMedia", "0").strip()
        if not from_number or not message_sid:
            logger.warning("twilio webhook message missing From or MessageSid")
            return Response(status_code=400)

        if not _is_allowed(from_number, allowed):
            logger.warning(
                "twilio webhook sender not allowlisted | pseudo=%s",
                _pseudonymize(from_number),
            )
            return Response(status_code=200)

        if not store.check_and_reserve(message_sid):
            logger.info("twilio webhook duplicate message skipped | pseudo=%s", _pseudonymize(from_number))
            return Response(status_code=200)

        if num_media not in ("", "0"):
            _start_background(
                _send_courtesy,
                sender=sender,
                recipient=from_number,
                courtesy=COURTESY_MEDIA_REPLY,
            )
            return Response(status_code=200)

        if not body:
            return Response(status_code=200)

        correlation_id = _correlation_id(message_sid)
        pseudo_sender = _pseudonymize(from_number)
        _start_background(
            _process_message,
            channel=channel,
            executor_fn=executor_fn,
            sender=sender,
            body=body,
            recipient=from_number,
            correlation_id=correlation_id,
            pseudo_sender=pseudo_sender,
        )
        return Response(status_code=200)

    return router


def _is_allowed(from_number: str, allowed: frozenset[str]) -> bool:
    if not allowed:
        return False
    return normalize_twilio_number(from_number) in allowed


def normalize_twilio_number(value: str) -> str:
    """Normalize ``whatsapp:+34...`` / ``+34...`` forms for comparison."""
    normalized = value.strip()
    prefix = "whatsapp:"
    if normalized.lower().startswith(prefix):
        normalized = normalized[len(prefix):]
    return normalized.strip()


def _start_background(target: Callable[..., None], **kwargs: Any) -> None:
    thread = threading.Thread(target=target, kwargs=kwargs, daemon=True)
    thread.start()


def _send_courtesy(*, sender: Any, recipient: str, courtesy: str) -> None:
    try:
        sender.send_text(recipient, courtesy)
    except Exception:
        logger.exception("twilio courtesy delivery failed")


def _process_message(
    *,
    channel: WhatsAppChannel,
    executor_fn: Callable[[AgentExecutionRequest], AgentExecutionResult],
    sender: Any,
    body: str,
    recipient: str,
    correlation_id: str,
    pseudo_sender: str,
) -> None:
    """Background task: run Atlas and deliver the answer. Never raises."""
    try:
        normalized = {
            "id": correlation_id,
            "from": pseudo_sender,
            "text": body,
        }
        request = channel.parse_inbound(normalized)
        result = executor_fn(request)
        outbound = channel.format_outbound(result)
        reply = outbound.get("body")
        if isinstance(reply, str) and reply.strip():
            sender.send_text(recipient, reply)
    except InvalidChannelMessageError as error:
        logger.warning("twilio inbound translation failed | error=%s", error)
        _send_courtesy(sender=sender, recipient=recipient, courtesy=UNPARSABLE_REPLY)
    except Exception:
        logger.exception("twilio background processing failed")
        _send_courtesy(sender=sender, recipient=recipient, courtesy=ERROR_REPLY)


def _correlation_id(message_sid: str) -> str:
    """Deterministic short identifier safe for AgentExecutionRequest (<=128)."""
    return f"tw-{hashlib.sha256(message_sid.encode('utf-8')).hexdigest()[:32]}"


def _pseudonymize(sender_id: str) -> str:
    """Pseudonymized representation used in logs and towards Atlas core."""
    return f"twi_{hashlib.sha256(sender_id.encode('utf-8')).hexdigest()[:12]}"

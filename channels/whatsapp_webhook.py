"""WhatsApp webhook endpoints for Atlas (Phase 2).

Flow: parse -> extract -> idempotency reserve (BEFORE the HTTP 200) ->
immediate 200 to Meta -> Atlas execution + outbound delivery as a
FastAPI background task.

HTTP policy: functional errors return 200 (so Meta does not retry);
only real infrastructure failures return 500.
"""

from __future__ import annotations

import hashlib
import json
import hmac
import logging
import math
from typing import Any, Callable, Mapping

from fastapi import APIRouter, Request, Response

from channels.base_channel import InvalidChannelMessageError
from channels.whatsapp_channel import WhatsAppChannel
from channels.whatsapp_metrics import (
    CHANNEL_ERRORS,
    AUDIO_RECEIVED,
    MESSAGES_DUPLICATED,
    MESSAGES_FAILED,
    MESSAGES_RECEIVED,
    RATE_LIMITED,
    VOICE_REPLIES,
    status_event,
    safe_record,
)
from core.agent_executor import AgentExecutionRequest, AgentExecutionResult, AgentExecutionStatus


from channels.whatsapp_channel import MAX_WHATSAPP_TEXT_LENGTH


logger = logging.getLogger(__name__)

MAX_CORRELATION_ID_LENGTH = 128


def build_webhook_router(
    *,
    channel: WhatsAppChannel,
    executor_fn: Callable[[AgentExecutionRequest], AgentExecutionResult],
    sender: Any,
    verify_token: str,
    app_secret: str,
    store: Any,
    transcriber: Any = None,
    voice_renderer: Any = None,
    recorder: Any = None,
    rate_limiter: Any = None,
) -> APIRouter:
    """Compose the WhatsApp webhook router with injected dependencies."""

    router = APIRouter(prefix="/webhook/whatsapp")

    @router.get("")
    def verify_webhook(hub_mode: str = "", hub_verify_token: str = "", hub_challenge: str = "") -> Response:
        expected = verify_token.encode("utf-8")
        received = (hub_verify_token or "").encode("utf-8")
        if hub_mode == "subscribe" and hmac.compare_digest(expected, received):
            return Response(content=hub_challenge, media_type="text/plain", status_code=200)
        return Response(status_code=403)

    def process_pending() -> None:
        claim = getattr(store, "claim_pending", None)
        if not callable(claim):
            return
        while (job := claim()) is not None:
            event_id, kind, item = job
            try:
                if kind == "message":
                    normalized = {"id": item["correlation_id"], "from": item["pseudo_sender"], "text": _message_text(item["message"])}
                    result = executor_fn(channel.parse_inbound(normalized))
                    body = channel.format_outbound(result).get("body")
                    if isinstance(body, str) and body.strip():
                        _deliver_reply(sender, voice_renderer, item["recipient_id"], body, recorder)
                elif kind == "courtesy":
                    sender.send_text(item["recipient_id"], item["courtesy"])
                else:
                    transcript = transcriber.transcribe_media_id(item["media_id"])
                    normalized = {"id": item["correlation_id"], "from": item["pseudo_sender"], "text": transcript}
                    result = executor_fn(channel.parse_inbound(normalized))
                    body = channel.format_outbound(result).get("body")
                    if isinstance(body, str) and body.strip():
                        _deliver_reply(sender, voice_renderer, item["recipient_id"], body, recorder)
            except Exception:
                safe_record(recorder, MESSAGES_FAILED)
                logger.exception("whatsapp durable job failed")
                store.fail(event_id)
            else:
                store.complete(event_id)

    @router.post("")
    async def receive_webhook(request: Request) -> Response:
        raw_body = await request.body()
        received = request.headers.get("X-Hub-Signature-256", "")
        expected = "sha256=" + hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        if not received or not hmac.compare_digest(expected, received):
            logger.warning("whatsapp webhook signature rejected")
            return Response(status_code=401)
        try:
            payload = json.loads(raw_body)
        except Exception:
            logger.warning("whatsapp webhook received unreadable json body")
            return Response(status_code=400)
        for value in _iter_change_values(payload):
            statuses = value.get("statuses", [])
            if isinstance(statuses, list):
                for status in statuses:
                    if not isinstance(status, Mapping) or not isinstance(status.get("id"), str):
                        continue
                    event_id = f"status:{status['id']}:{status.get('status', '')}"
                    if store.check_and_reserve(event_id):
                        _record_status_ack(status, recorder)
            messages = value.get("messages", [])
            if not isinstance(messages, list):
                continue
            for message in messages:
                if not isinstance(message, Mapping) or not isinstance(message.get("id"), str):
                    continue
                wamid = message["id"].strip()
                if not wamid:
                    continue
                raw_sender = message.get("from")
                if rate_limiter is not None and isinstance(raw_sender, str) and raw_sender.strip():
                    try:
                        if not rate_limiter.allow(_pseudonymize(raw_sender.strip())):
                            safe_record(recorder, RATE_LIMITED)
                            continue
                    except Exception:
                        logger.warning("whatsapp webhook rate limiter failed | fail-open")
                try:
                    recipient_id = _extract_sender_id(message)
                except InvalidChannelMessageError:
                    continue
                audio_id = _audio_media_id(message) if transcriber is not None else None
                kind = "audio" if audio_id is not None else "message" if _is_supported_text_message(message) else "courtesy"
                item = {"message": dict(message), "recipient_id": recipient_id, "correlation_id": _correlation_id(wamid), "pseudo_sender": _pseudonymize(recipient_id), "media_id": audio_id, "courtesy": "Solo puedo procesar mensajes de texto por ahora."}
                enqueue = getattr(store, "enqueue", None)
                accepted = enqueue(wamid, kind, item) if callable(enqueue) else store.check_and_reserve(wamid)
                if not accepted:
                    safe_record(recorder, MESSAGES_DUPLICATED)
                    continue
                safe_record(recorder, MESSAGES_RECEIVED)
                if kind == "audio":
                    safe_record(recorder, AUDIO_RECEIVED)
                if not callable(enqueue):
                    if kind == "message":
                        _process_message(channel=channel, executor_fn=executor_fn, sender=sender, message=message, recipient_id=recipient_id, correlation_id=item["correlation_id"], pseudo_sender=item["pseudo_sender"], voice_renderer=voice_renderer, recorder=recorder)
                    elif kind == "audio":
                        _process_audio_message(channel=channel, executor_fn=executor_fn, sender=sender, transcriber=transcriber, media_id=audio_id, recipient_id=recipient_id, correlation_id=item["correlation_id"], pseudo_sender=item["pseudo_sender"], voice_renderer=voice_renderer, recorder=recorder)
                    else:
                        _send_courtesy_message(channel=channel, executor_fn=None, sender=sender, result=None, recipient_id=recipient_id, courtesy=item["courtesy"])
        process_pending()
        return Response(status_code=200)

    router.whatsapp_recover_pending = process_pending  # type: ignore[attr-defined]
    return router


def _deliver_reply(
    sender: Any,
    voice_renderer: Any,
    recipient_id: str,
    body: str,
    recorder: Any = None,
) -> None:
    """Deliver the reply, preferring audio when a renderer is configured.

    Any failure in the voice path falls back to plain text so that the
    reply from Atlas is never lost.
    """
    if voice_renderer is None:
        sender.send_text(recipient_id, body)
        return
    audio_path = None
    try:
        audio_path = voice_renderer.render(body)
        media_id = sender.upload_media(str(audio_path), "audio/ogg")
        sender.send_audio(recipient_id, media_id)
        safe_record(recorder, VOICE_REPLIES)
        return
    except Exception:
        logger.exception("whatsapp voice reply failed | fallback=text")
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("failed to remove whatsapp voice reply file")
    sender.send_text(recipient_id, body)


def _process_message(
    *,
    channel: WhatsAppChannel,
    executor_fn: Callable[[AgentExecutionRequest], AgentExecutionResult],
    sender: Any,
    message: Mapping[str, Any] | None,
    recipient_id: str,
    correlation_id: str,
    pseudo_sender: str,
    voice_renderer: Any = None,
    recorder: Any = None,
) -> None:
    """Background task: run Atlas and deliver the answer. Never raises."""
    if message is None:
        return
    try:
        normalized = {
            "id": correlation_id,
            "from": pseudo_sender,
            "text": _message_text(message),
        }
        request = channel.parse_inbound(normalized)
        result = executor_fn(request)
        outbound = channel.format_outbound(result)
        body = outbound.get("body")
        if isinstance(body, str) and body.strip():
            _deliver_reply(sender, voice_renderer, recipient_id, body, recorder=recorder)
    except InvalidChannelMessageError as error:
        safe_record(recorder, MESSAGES_FAILED)
        logger.warning("whatsapp inbound translation failed | error=%s", error)
        try:
            sender.send_text(recipient_id, "No he podido entender el mensaje.")
        except Exception:
            logger.exception("whatsapp courtesy delivery failed")
            safe_record(recorder, CHANNEL_ERRORS)
    except Exception:
        # Functional failure: log it; Meta must not be asked to retry.
        safe_record(recorder, MESSAGES_FAILED)
        logger.exception("whatsapp background processing failed")
        try:
            sender.send_text(recipient_id, "Se ha producido un error procesando tu mensaje.")
        except Exception:
            logger.exception("whatsapp error notification delivery failed")
            safe_record(recorder, CHANNEL_ERRORS)


def _process_audio_message(
    *,
    channel: WhatsAppChannel,
    executor_fn: Callable[[AgentExecutionRequest], AgentExecutionResult],
    sender: Any,
    transcriber: Any,
    voice_renderer: Any = None,
    media_id: str,
    recipient_id: str,
    correlation_id: str,
    pseudo_sender: str,
    recorder: Any = None,
) -> None:
    """Background task: download, transcribe and process an audio message."""
    try:
        transcript = transcriber.transcribe_media_id(media_id)
        normalized = {
            "id": correlation_id,
            "from": pseudo_sender,
            "text": transcript,
        }
        request = channel.parse_inbound(normalized)
        result = executor_fn(request)
        outbound = channel.format_outbound(result)
        body = outbound.get("body")
        if isinstance(body, str) and body.strip():
            _deliver_reply(sender, voice_renderer, recipient_id, body, recorder=recorder)
    except Exception:
        # Download or transcription failure: controlled courtesy response.
        safe_record(recorder, MESSAGES_FAILED)
        logger.exception("whatsapp audio processing failed")
        try:
            sender.send_text(recipient_id, "No he podido entender el mensaje.")
        except Exception:
            logger.exception("whatsapp courtesy delivery failed")


def _send_courtesy_message(
    *,
    channel: WhatsAppChannel,
    executor_fn: Any,
    sender: Any,
    result: Any,
    recipient_id: str,
    courtesy: str,
) -> None:
    try:
        sender.send_text(recipient_id, courtesy)
    except Exception:
        logger.exception("whatsapp courtesy delivery failed")


def _record_status_ack(status: Any, recorder: Any) -> None:
    """Record one delivery-status acknowledgement (V4.0-F1).

    Only the status code is observed; message ids, recipients and error
    payloads are never logged or counted. Always safe against a broken
    recorder.
    """
    if not isinstance(status, Mapping):
        logger.debug("whatsapp webhook malformed status acknowledgement")
        return
    code = status.get("status")
    event = status_event(code) if isinstance(code, str) else None
    if event is None:
        logger.debug(
            "whatsapp webhook untracked status acknowledgement | known=false"
        )
        return
    safe_record(recorder, event)
    logger.debug("whatsapp webhook status acknowledged | state=%s", event)


def _iter_change_values(payload: Any):
    if not isinstance(payload, Mapping):
        return
    entries = payload.get("entry")
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("changes"), list):
            continue
        for change in entry["changes"]:
            value = change.get("value") if isinstance(change, Mapping) else None
            if isinstance(value, Mapping):
                yield value


def _extract_change_value(payload: Any) -> Mapping[str, Any] | None:
    return next(_iter_change_values(payload), None)


def _message_text(message: Mapping[str, Any]) -> str:
    """Normalized textual representation of a supported inbound message.

    Unsupported or malformed payloads return an empty string so the
    caller falls back to the existing courtesy path. Payload content
    (filenames, captions, contact names, coordinates) is treated as
    untrusted data: it is labelled and clipped to the channel text
    limit, and never logged.
    """
    message_type = message.get("type")
    if not isinstance(message_type, str):
        return ""
    if message_type == "text":
        text = message.get("text")
        if isinstance(text, Mapping):
            body = text.get("body")
            if isinstance(body, str):
                return body.strip()
        return ""
    if message_type == "image":
        return _image_caption(message)
    if message_type == "document":
        return _document_text(message.get("document"))
    if message_type == "location":
        return _location_text(message.get("location"))
    if message_type == "contacts":
        return _contacts_text(message.get("contacts"))
    if message_type == "interactive":
        return _interactive_text(message.get("interactive"))
    return ""


def _extract_sender_id(message: Mapping[str, Any]) -> str:
    sender_id = message.get("from")
    if not isinstance(sender_id, str):
        raise InvalidChannelMessageError("message is missing a valid 'from' sender.")
    return sender_id.strip()


def _is_supported_text_message(message: Mapping[str, Any]) -> bool:
    return bool(_message_text(message))


def _clean_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _document_text(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    filename = _clean_str(payload.get("filename"))
    caption = _clean_str(payload.get("caption"))
    if filename:
        text = f"[documento: {filename}]"
        if caption:
            text = f"{text} {caption}"
    elif caption:
        text = f"[documento] {caption}"
    else:
        return ""
    return _clip(text)


def _location_text(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    if not _is_coordinate(latitude) or not _is_coordinate(longitude):
        return ""
    text = f"[ubicación: lat={latitude}, lng={longitude}]"
    name = _clean_str(payload.get("name"))
    address = _clean_str(payload.get("address"))
    if name:
        text = f"{text} {name}"
    if address:
        text = f"{text} {address}"
    return _clip(text)


def _contacts_text(payload: Any) -> str:
    if not isinstance(payload, list) or not payload:
        return ""
    first = payload[0]
    if not isinstance(first, Mapping):
        return ""
    name_block = first.get("name")
    if not isinstance(name_block, Mapping):
        return ""
    display_name = (
        _clean_str(name_block.get("formatted_name"))
        or _clean_str(name_block.get("first_name"))
        or _clean_str(name_block.get("last_name"))
    )
    if not display_name:
        return ""
    return _clip(f"[contacto: {display_name}]")


def _interactive_text(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    for reply_key in ("button_reply", "list_reply"):
        block = payload.get(reply_key)
        if not isinstance(block, Mapping):
            continue
        title = _clean_str(block.get("title"))
        if title:
            return _clip(title)
        identifier = _clean_str(block.get("id"))
        if identifier:
            return _clip(identifier)
    return ""


def _is_coordinate(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _clip(text: str) -> str:
    return text[:MAX_WHATSAPP_TEXT_LENGTH]


def _image_caption(message: Mapping[str, Any]) -> str:
    image = message.get("image")
    if not isinstance(image, Mapping):
        return ""
    caption = image.get("caption")
    if not isinstance(caption, str):
        return ""
    caption = caption.strip()
    return caption or ""


def _audio_media_id(message: Mapping[str, Any]) -> str | None:
    """Return the media id of an audio/voice message, or None."""
    if message.get("type") not in ("audio", "voice"):
        return None
    payload = message.get("audio")
    if not isinstance(payload, Mapping):
        return None
    media_id = payload.get("id")
    if not isinstance(media_id, str) or not media_id.strip():
        return None
    return media_id.strip()


def _correlation_id(wamid: str) -> str:
    """Deterministic short identifier safe for AgentExecutionRequest (<=128)."""
    return f"wa-{hashlib.sha256(wamid.encode('utf-8')).hexdigest()[:32]}"


def _pseudonymize(sender_id: str) -> str:
    """Pseudonymized representation propagated towards Atlas core."""
    return f"wa_{hashlib.sha256(sender_id.encode('utf-8')).hexdigest()[:12]}"

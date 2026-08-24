"""WhatsApp webhook endpoints for Atlas (Phase 2).

Flow: parse -> extract -> idempotency reserve (BEFORE the HTTP 200) ->
immediate 200 to Meta -> Atlas execution + outbound delivery as a
FastAPI background task.

HTTP policy: functional errors return 200 (so Meta does not retry);
only real infrastructure failures return 500.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
from typing import Any, Callable, Mapping

from fastapi import APIRouter, BackgroundTasks, Request, Response

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

    @router.post("")
    async def receive_webhook(request: Request, background_tasks: BackgroundTasks) -> Response:
        try:
            payload = await request.json()
        except Exception:
            logger.warning("whatsapp webhook received unreadable json body")
            return Response(status_code=400)

        value = _extract_change_value(payload)
        if value is None:
            # Uninterpretable payloads must NOT trigger Meta retries.
            logger.debug("whatsapp webhook payload without interpretable changes")
            return Response(status_code=200)

        statuses = value.get("statuses")
        if isinstance(statuses, list) and statuses:
            _record_status_ack(statuses[0], recorder)
            return Response(status_code=200)

        messages = value.get("messages")
        message = messages[0] if isinstance(messages, list) and messages and isinstance(messages[0], Mapping) else None
        if message is None:
            logger.debug("whatsapp webhook payload without messages")
            return Response(status_code=200)

        wamid = message.get("id")
        if not isinstance(wamid, str) or not wamid.strip():
            logger.debug("whatsapp webhook message without wamid")
            return Response(status_code=200)
        wamid = wamid.strip()

        # Rate limit by pseudonymous sender BEFORE reserving idempotency so
        # that rejected floods do not consume reservations (V4.0-F2).
        if rate_limiter is not None:
            raw_sender = message.get("from")
            if isinstance(raw_sender, str) and raw_sender.strip():
                sender_key = _pseudonymize(raw_sender.strip())
                try:
                    allowed = rate_limiter.allow(sender_key)
                except Exception:
                    logger.warning("whatsapp webhook rate limiter failed | fail-open")
                    allowed = True
                if not allowed:
                    safe_record(recorder, RATE_LIMITED)
                    logger.debug("whatsapp webhook rate limit exceeded")
                    return Response(status_code=200)

        safe_record(recorder, MESSAGES_RECEIVED)

        # Reserve BEFORE responding so concurrent duplicates cannot both execute.
        if not store.check_and_reserve(wamid):
            logger.debug("whatsapp webhook duplicate wamid ignored")
            safe_record(recorder, MESSAGES_DUPLICATED)
            return Response(status_code=200)

        recipient_id = _extract_sender_id(message)
        correlation_id = _correlation_id(wamid)
        pseudo_sender = _pseudonymize(recipient_id)
        audio_media_id = _audio_media_id(message) if transcriber is not None else None
        supported = _is_supported_text_message(message) or audio_media_id is not None

        if audio_media_id is not None:
            safe_record(recorder, AUDIO_RECEIVED)
            background_tasks.add_task(
                _process_audio_message,
                channel=channel,
                executor_fn=executor_fn,
                sender=sender,
                transcriber=transcriber,
                voice_renderer=voice_renderer,
                media_id=audio_media_id,
                recipient_id=recipient_id,
                correlation_id=correlation_id,
                pseudo_sender=pseudo_sender,
                recorder=recorder,
            )
        elif supported:
            background_tasks.add_task(
                _process_message,
                channel=channel,
                executor_fn=executor_fn,
                sender=sender,
                message=message,
                recipient_id=recipient_id,
                correlation_id=correlation_id,
                pseudo_sender=pseudo_sender,
                voice_renderer=voice_renderer,
                recorder=recorder,
            )
        else:
            background_tasks.add_task(
                _send_courtesy_message,
                channel=channel,
                executor_fn=None,
                sender=sender,
                result=None,
                recipient_id=recipient_id,
                courtesy="Solo puedo procesar mensajes de texto por ahora.",
            )
        return Response(status_code=200)

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


def _extract_change_value(payload: Any) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    entry = payload.get("entry")
    if not isinstance(entry, list) or not entry or not isinstance(entry[0], Mapping):
        return None
    changes = entry[0].get("changes")
    if not isinstance(changes, list) or not changes or not isinstance(changes[0], Mapping):
        return None
    value = changes[0].get("value")
    return value if isinstance(value, Mapping) else None


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

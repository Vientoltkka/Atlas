"""Application factory for the Atlas WhatsApp webhook (Phase 2).

Run with uvicorn workers=1: the in-memory idempotency store is per-process
and NOT safe for multiple workers. Shared persistence belongs to Phase 3.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from fastapi import FastAPI

from channels.base_channel import InvalidChannelMessageError
from channels.webhook_idempotency import IdempotencyStore, SqliteIdempotencyStore
from channels.whatsapp_channel import WhatsAppChannel
from channels.whatsapp_sender import WhatsAppGraphSender
from channels.whatsapp_webhook import build_webhook_router
from core.agent_executor import AgentExecutionRequest, AgentExecutionResult


def _build_store() -> IdempotencyStore | SqliteIdempotencyStore:
    """Development default: in-memory. Set ATLAS_WHATSAPP_IDEMPOTENCY_DB_PATH
    to enable a persistent store shared across processes and workers."""
    db_path = os.environ.get("ATLAS_WHATSAPP_IDEMPOTENCY_DB_PATH", "")
    if db_path:
        return SqliteIdempotencyStore(db_path=db_path)
    return IdempotencyStore()


def build_webhook_app(
    *,
    channel: WhatsAppChannel | None = None,
    executor_fn: Callable[[AgentExecutionRequest], AgentExecutionResult] | None = None,
    store: IdempotencyStore | SqliteIdempotencyStore | None = None,
    sender: Any = None,
    verify_token: str | None = None,
    transcriber: Any = None,
) -> FastAPI:
    """Build the FastAPI app with all webhook dependencies injected."""
    if channel is None:
        channel = WhatsAppChannel()
    if store is None:
        store = _build_store()
    if verify_token is None:
        verify_token = os.environ.get("ATLAS_WHATSAPP_VERIFY_TOKEN", "")
    if not verify_token:
        raise ValueError("verify_token is required (set ATLAS_WHATSAPP_VERIFY_TOKEN).")
    if sender is None:
        access_token = os.environ.get("ATLAS_WHATSAPP_ACCESS_TOKEN", "")
        phone_number_id = os.environ.get("ATLAS_WHATSAPP_PHONE_NUMBER_ID", "")
        max_attempts = int(os.environ.get("ATLAS_WHATSAPP_SEND_MAX_ATTEMPTS", "3"))
        sender = WhatsAppGraphSender(
            access_token=access_token,
            phone_number_id=phone_number_id,
            max_attempts=max_attempts,
        )
    if transcriber is None:
        transcriber = _build_audio_transcriber()
    if executor_fn is None:
        raise ValueError("executor_fn is required.")

    app = FastAPI(title="Atlas WhatsApp Webhook")
    app.include_router(
        build_webhook_router(
            channel=channel,
            executor_fn=executor_fn,
            sender=sender,
            verify_token=verify_token,
            store=store,
            transcriber=transcriber,
        )
    )
    return app


def _build_audio_transcriber() -> Any:
    """Default audio pipeline: secure downloader + local faster-whisper.

    Returns None when WhatsApp credentials are absent so that audio
    messages degrade to the courtesy path instead of failing startup.
    """
    from channels.whatsapp_audio import WhatsAppAudioTranscriber
    from channels.whatsapp_media import WhatsAppMediaDownloader

    access_token = os.environ.get("ATLAS_WHATSAPP_ACCESS_TOKEN", "")
    if not access_token:
        return None
    downloader = WhatsAppMediaDownloader(access_token=access_token)

    class _ProviderHolder:
        provider: Any = None

    holder = _ProviderHolder()

    def _provider():
        if holder.provider is None:
            from use_cases.speech_engine import FasterWhisperSpeechToTextProvider

            holder.provider = FasterWhisperSpeechToTextProvider()
        return holder.provider

    class _LazyTranscriber:
        def transcribe_media_id(self, media_id: str) -> str:
            return WhatsAppAudioTranscriber(
                downloader=downloader,
                provider=_provider(),
            ).transcribe_media_id(media_id)

    return _LazyTranscriber()


# Backwards-compatible alias used by earlier phases.
create_webhook_app = build_webhook_app

"""Application factory for the Atlas WhatsApp webhook (Phase 2).

Run with uvicorn workers=1: the in-memory idempotency store is per-process
and NOT safe for multiple workers. Shared persistence belongs to Phase 3.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from channels.base_channel import InvalidChannelMessageError
from channels.webhook_idempotency import (
    IdempotencyStore,
    IdempotencyStoreInitError,
    SqliteIdempotencyStore,
)
from channels.whatsapp_channel import WhatsAppChannel
from channels.whatsapp_sender import WhatsAppGraphSender
from channels.whatsapp_health import WhatsAppChannelHealthChecker, WhatsAppHealthStatus
from channels.whatsapp_metrics import WhatsAppMetricsRecorder
from channels.whatsapp_metrics_persistence import WhatsAppMetricsPersistence
from channels.whatsapp_rate_limit import WhatsAppRateLimiter
from channels.whatsapp_webhook import build_webhook_router
from core.agent_executor import AgentExecutionRequest, AgentExecutionResult


logger = logging.getLogger(__name__)


def _build_store() -> IdempotencyStore | SqliteIdempotencyStore:
    """Development default: in-memory. Set ATLAS_WHATSAPP_IDEMPOTENCY_DB_PATH
    to enable a persistent store shared across processes and workers.

    A persistent store that cannot be initialized is a hard startup
    failure: silently falling back to the per-process in-memory store
    would weaken the exactly-once guarantee without anyone noticing.
    """
    db_path = os.environ.get("ATLAS_WHATSAPP_IDEMPOTENCY_DB_PATH", "data/whatsapp_webhook.db").strip()
    if not db_path:
        raise IdempotencyStoreInitError("persistent whatsapp idempotency store is required.")
    try:
        return SqliteIdempotencyStore(db_path=db_path)
    except Exception as error:
        logger.error(
            "whatsapp idempotency store init failed | type=%s | env=ATLAS_WHATSAPP_IDEMPOTENCY_DB_PATH",
            type(error).__name__,
        )
        raise IdempotencyStoreInitError(
            "persistent whatsapp idempotency store is unavailable; "
            "refusing to start without idempotency protection "
            "(check ATLAS_WHATSAPP_IDEMPOTENCY_DB_PATH accessibility)."
        ) from error


def build_webhook_app(
    *,
    channel: WhatsAppChannel | None = None,
    executor_fn: Callable[[AgentExecutionRequest], AgentExecutionResult] | None = None,
    store: IdempotencyStore | SqliteIdempotencyStore | None = None,
    sender: Any = None,
    verify_token: str | None = None,
    app_secret: str | None = None,
    transcriber: Any = None,
    voice_renderer: Any = None,
    recorder: WhatsAppMetricsRecorder | None = None,
    health_checker: WhatsAppChannelHealthChecker | None = None,
    rate_limiter: WhatsAppRateLimiter | None = None,
) -> FastAPI:
    """Build the FastAPI app with all webhook dependencies injected."""
    if channel is None:
        channel = WhatsAppChannel()
    if recorder is None:
        recorder = _build_recorder()
    if store is None:
        store = _build_store()
    if verify_token is None:
        verify_token = os.environ.get("ATLAS_WHATSAPP_VERIFY_TOKEN", "")
    if not verify_token:
        raise ValueError("verify_token is required (set ATLAS_WHATSAPP_VERIFY_TOKEN).")
    if app_secret is None:
        app_secret = os.environ.get("ATLAS_WHATSAPP_APP_SECRET", "")
    if not app_secret:
        raise ValueError("app_secret is required (set ATLAS_WHATSAPP_APP_SECRET).")
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
    if voice_renderer is None and os.environ.get(
        "ATLAS_WHATSAPP_VOICE_REPLIES", ""
    ).strip().lower() in ("1", "true", "yes", "on"):
        voice_renderer = _build_voice_renderer()
    if executor_fn is None:
        raise ValueError("executor_fn is required.")
    if health_checker is None:
        health_checker = _build_health_checker(
            verify_token=verify_token,
            store=store,
            sender=sender,
            transcriber=transcriber,
            voice_renderer=voice_renderer,
        )
    recover_interrupted = getattr(store, "recover_interrupted", None)
    if callable(recover_interrupted):
        recover_interrupted()
    if rate_limiter is None:
        rate_limiter = _build_rate_limiter()

    app = FastAPI(title="Atlas WhatsApp Webhook")
    app.state.whatsapp_health = health_checker
    app.state.whatsapp_metrics = recorder
    app.include_router(
        build_webhook_router(
            channel=channel,
            executor_fn=executor_fn,
            sender=sender,
            verify_token=verify_token,
            app_secret=app_secret,
            store=store,
            transcriber=transcriber,
            voice_renderer=voice_renderer,
            recorder=recorder,
            rate_limiter=rate_limiter,
        )
    )

    @app.get("/health", include_in_schema=False)
    def health_endpoint() -> Response:
        """Local readiness probe (V4.1-F1).

        Semantics: HEALTHY -> 200, DEGRADED -> 200 (operational with
        reduced capabilities, reported in the body), UNHEALTHY -> 503.
        The result payload only contains booleans-derived stable codes;
        an unexpected checker failure becomes a controlled 503 without
        internal details.
        """
        try:
            result = health_checker.check()
        except Exception:
            logger.warning("whatsapp health endpoint check failed")
            return JSONResponse(
                status_code=503,
                content={"status": "UNHEALTHY", "checks": {}},
            )
        status_code = 200 if result.status is not WhatsAppHealthStatus.UNHEALTHY else 503
        return JSONResponse(
            status_code=status_code,
            content={"status": result.status.value, "checks": dict(result.checks)},
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint(request: Request) -> Response:
        """Aggregated channel counters, protected by the verify token.

        Requires ``Authorization: Bearer <ATLAS_WHATSAPP_VERIFY_TOKEN>``.
        Rejections never reveal whether the token was wrong or missing
        and never echo any secret.
        """
        expected = f"Bearer {verify_token}".encode("utf-8")
        provided = request.headers.get("Authorization", "").encode("utf-8")
        if not verify_token or not hmac.compare_digest(expected, provided):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return JSONResponse(content=recorder.snapshot())

    return app


def _build_rate_limiter() -> WhatsAppRateLimiter:
    """Anti-flood protection, disabled by default.

    Set ATLAS_WHATSAPP_RATE_LIMIT_PER_MINUTE to a positive integer to
    enable a per-sender sliding-window limit.
    """
    raw = os.environ.get("ATLAS_WHATSAPP_RATE_LIMIT_PER_MINUTE", "0").strip()
    try:
        limit = int(raw)
    except ValueError:
        logger.warning(
            "invalid ATLAS_WHATSAPP_RATE_LIMIT_PER_MINUTE | value ignored | fallback=disabled"
        )
        limit = 0
    return WhatsAppRateLimiter(limit_per_minute=max(limit, 0))


def _build_recorder() -> WhatsAppMetricsRecorder | WhatsAppMetricsPersistence:
    """Metrics recorder, optionally backed by local JSON persistence.

    Set ATLAS_WHATSAPP_METRICS_PATH to persist aggregated counters across
    restarts (atomic writes, threshold-based flush). Without the variable
    the recorder stays purely in-memory, exactly as before V4.0-F3.
    """
    recorder = WhatsAppMetricsRecorder()
    metrics_path = os.environ.get("ATLAS_WHATSAPP_METRICS_PATH", "").strip()
    if not metrics_path:
        return recorder
    persistence = WhatsAppMetricsPersistence(recorder=recorder, path=metrics_path)
    persistence.load_existing()
    return persistence


def _build_health_checker(
    *,
    verify_token: str,
    store: Any,
    sender: Any,
    transcriber: Any,
    voice_renderer: Any,
) -> WhatsAppChannelHealthChecker:
    """Build the channel diagnostic from the same wiring the app uses."""
    access_token = os.environ.get("ATLAS_WHATSAPP_ACCESS_TOKEN", "")
    phone_number_id = os.environ.get("ATLAS_WHATSAPP_PHONE_NUMBER_ID", "")
    config = {
        "verify_token": verify_token,
        "access_token": access_token,
        "phone_number_id": phone_number_id,
    }
    if sender is None:
        sender_builder: Callable[[], Any] = lambda: WhatsAppGraphSender(
            access_token=access_token,
            phone_number_id=phone_number_id,
        )
    else:
        sender_builder = lambda: sender
    return WhatsAppChannelHealthChecker(
        config=config,
        store=store,
        transcriber=transcriber,
        voice_renderer=voice_renderer,
        sender_builder=sender_builder,
    )


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


def _build_voice_renderer() -> Any:
    """Default voice reply renderer backed by local pyttsx3 + PyAV."""
    from channels.whatsapp_voice_reply import WhatsAppVoiceReplyRenderer

    return WhatsAppVoiceReplyRenderer()

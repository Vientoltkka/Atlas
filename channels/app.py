"""Application factory for the Atlas WhatsApp webhook (Phase 2).

Run with uvicorn workers=1: the in-memory idempotency store is per-process
and NOT safe for multiple workers. Shared persistence belongs to Phase 3.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from fastapi import FastAPI

from channels.base_channel import InvalidChannelMessageError
from channels.webhook_idempotency import IdempotencyStore
from channels.whatsapp_channel import WhatsAppChannel
from channels.whatsapp_sender import WhatsAppGraphSender
from channels.whatsapp_webhook import build_webhook_router
from core.agent_executor import AgentExecutionRequest, AgentExecutionResult


def create_webhook_app(
    *,
    channel: WhatsAppChannel | None = None,
    executor_fn: Callable[[AgentExecutionRequest], AgentExecutionResult] | None = None,
    store: IdempotencyStore | None = None,
    sender: Any = None,
    verify_token: str | None = None,
) -> FastAPI:
    """Build the FastAPI app with all webhook dependencies injected."""
    if channel is None:
        channel = WhatsAppChannel()
    if store is None:
        store = IdempotencyStore()
    if verify_token is None:
        verify_token = os.environ.get("ATLAS_WHATSAPP_VERIFY_TOKEN", "")
    if not verify_token:
        raise ValueError("verify_token is required (set ATLAS_WHATSAPP_VERIFY_TOKEN).")
    if sender is None:
        access_token = os.environ.get("ATLAS_WHATSAPP_ACCESS_TOKEN", "")
        phone_number_id = os.environ.get("ATLAS_WHATSAPP_PHONE_NUMBER_ID", "")
        sender = WhatsAppGraphSender(
            access_token=access_token,
            phone_number_id=phone_number_id,
        )
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
        )
    )
    return app

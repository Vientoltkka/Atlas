"""Outbound WhatsApp Graph API client (Atlas Phase 2).

Transport-only component: it delivers the body produced by
``WhatsAppChannel.format_outbound``. It never logs or propagates the
access token, and it is injectable/stubbeable in tests.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol


logger = logging.getLogger(__name__)

GRAPH_API_BASE_URL = "https://graph.facebook.com/v21.0"


class MessageSender(Protocol):
    """Contract for outbound channel senders (stub-friendly)."""

    def send_text(self, recipient_id: str, body: str) -> None:
        ...


class WhatsAppGraphSender:
    """Sends WhatsApp text messages through the Meta Graph API."""

    def __init__(
        self,
        *,
        access_token: str,
        phone_number_id: str,
        base_url: str = GRAPH_API_BASE_URL,
        timeout_seconds: float = 15.0,
        transport: Any = None,
    ) -> None:
        if not access_token or not access_token.strip():
            raise ValueError("access_token must be a non-empty string.")
        if not phone_number_id or not phone_number_id.strip():
            raise ValueError("phone_number_id must be a non-empty string.")
        self._access_token = access_token
        self._phone_number_id = phone_number_id.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    @property
    def phone_number_id(self) -> str:
        return self._phone_number_id

    def send_text(self, recipient_id: str, body: str) -> None:
        """Deliver ``body`` to ``recipient_id``. Raises on transport failure."""
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "text",
            "text": {"body": body},
        }
        status, response_text = self._post(
            f"{self._base_url}/{self._phone_number_id}/messages",
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
            },
            json_payload=payload,
        )
        if status < 200 or status >= 300:
            # Never include the token in the raised message.
            raise RuntimeError(
                f"whatsapp graph api delivery failed with status {status}."
            )
        logger.debug("whatsapp message delivered | status=%s", status)

    def _post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json_payload: Mapping[str, Any],
    ) -> tuple[int, str]:
        import httpx

        if self._transport is not None:
            return self._transport(url, dict(headers), dict(json_payload))
        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(url, headers=dict(headers), json=dict(json_payload))
            return response.status_code, response.text

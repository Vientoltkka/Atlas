"""Outbound WhatsApp Graph API client (Atlas Phase 2).

Transport-only component: it delivers the body produced by
``WhatsAppChannel.format_outbound``. It never logs or propagates the
access token, and it is injectable/stubbeable in tests.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping, Protocol


logger = logging.getLogger(__name__)

GRAPH_API_BASE_URL = "https://graph.facebook.com/v21.0"

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE_SECONDS = 0.5


class PermanentDeliveryError(RuntimeError):
    """Permanent rejection by the Graph API (4xx): must not be retried."""


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
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not access_token or not access_token.strip():
            raise ValueError("access_token must be a non-empty string.")
        if not phone_number_id or not phone_number_id.strip():
            raise ValueError("phone_number_id must be a non-empty string.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self._access_token = access_token
        self._phone_number_id = phone_number_id.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._max_attempts = max_attempts
        self._backoff_base_seconds = backoff_base_seconds
        self._sleeper = sleeper

    @property
    def phone_number_id(self) -> str:
        return self._phone_number_id

    def send_text(self, recipient_id: str, body: str) -> None:
        """Deliver ``body`` to ``recipient_id``. Raises on transport failure.

        Transient failures (network errors, 5xx, 429) are retried with
        exponential backoff. Permanent 4xx failures are raised immediately
        without retrying.
        """
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "text",
            "text": {"body": body},
        }
        url = f"{self._base_url}/{self._phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                status, _response_text = self._post(
                    url,
                    headers=headers,
                    json_payload=payload,
                )
            except Exception as error:
                # Network/timeout errors are transient by definition.
                last_error = error
            else:
                if 200 <= status < 300:
                    logger.debug("whatsapp message delivered | status=%s", status)
                    return
                if status < 500 and status != 429:
                    # Permanent client error (payload/token/other 4xx).
                    raise PermanentDeliveryError(
                        f"whatsapp graph api delivery rejected with status {status}."
                    )
                last_error = RuntimeError(
                    f"whatsapp graph api delivery failed with status {status}."
                )
            if attempt < self._max_attempts:
                self._sleeper(self._backoff_base_seconds * (2 ** (attempt - 1)))
        raise last_error if last_error is not None else RuntimeError(
            "whatsapp graph api delivery failed."
        )

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

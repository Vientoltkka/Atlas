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


def _extract_media_id(response_text: str) -> str | None:
    import json

    try:
        data = json.loads(response_text)
    except ValueError:
        return None
    if isinstance(data, dict):
        media_id = data.get("id")
        if isinstance(media_id, str) and media_id.strip():
            return media_id.strip()
    return None


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
        max_upload_bytes: int = 20 * 1024 * 1024,
        allowed_upload_mime_types: frozenset[str] = frozenset(
            {"audio/ogg", "audio/mpeg", "audio/mp4"}
        ),
        upload_transport: Any = None,
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
        self._max_upload_bytes = max_upload_bytes
        self._allowed_upload_mime_types = allowed_upload_mime_types
        self._upload_transport = upload_transport

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
        self._deliver_payload(payload)

    def upload_media(self, file_path: str, mime_type: str) -> str:
        """Upload an audio file to the Graph API and return its media id.

        Reuses the same retry/backoff policy as message delivery. Raises
        ``PermanentDeliveryError`` for 4xx rejections and never exposes
        the access token or the authenticated URL in errors.
        """
        from pathlib import Path

        path = Path(file_path)
        size_bytes = path.stat().st_size
        if size_bytes > self._max_upload_bytes:
            raise PermanentDeliveryError(
                f"whatsapp media upload exceeds the maximum allowed size."
            )
        if mime_type not in self._allowed_upload_mime_types:
            raise PermanentDeliveryError("whatsapp media type is not allowed.")

        url = f"{self._base_url}/{self._phone_number_id}/media"
        headers = {"Authorization": f"Bearer {self._access_token}"}
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                status, response_text = self._post_multipart(
                    url,
                    headers=headers,
                    file_path=path,
                    mime_type=mime_type,
                )
            except Exception as error:
                last_error = error
            else:
                if 200 <= status < 300:
                    media_id = _extract_media_id(response_text)
                    if not media_id:
                        raise PermanentDeliveryError(
                            "whatsapp media upload returned no media id."
                        )
                    logger.debug("whatsapp media uploaded | bytes=%s", size_bytes)
                    return media_id
                if status < 500 and status != 429:
                    raise PermanentDeliveryError(
                        f"whatsapp media upload rejected with status {status}."
                    )
                last_error = RuntimeError(
                    f"whatsapp media upload failed with status {status}."
                )
            if attempt < self._max_attempts:
                self._sleeper(self._backoff_base_seconds * (2 ** (attempt - 1)))
        raise last_error if last_error is not None else RuntimeError(
            "whatsapp media upload failed."
        )

    def send_audio(self, recipient_id: str, media_id: str) -> None:
        """Deliver an uploaded audio media id as a WhatsApp voice message."""
        if not isinstance(media_id, str) or not media_id.strip():
            raise PermanentDeliveryError("media_id must be a non-empty string.")
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_id,
            "type": "audio",
            "audio": {"id": media_id.strip()},
        }
        self._deliver_payload(payload)

    def _deliver_payload(self, payload: Mapping[str, Any]) -> None:
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
                last_error = error
            else:
                if 200 <= status < 300:
                    logger.debug("whatsapp message delivered | status=%s", status)
                    return
                if status < 500 and status != 429:
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

    def _post_multipart(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        file_path: Path,
        mime_type: str,
    ) -> tuple[int, str]:
        import httpx

        files = {
            "file": (file_path.name, file_path.read_bytes(), mime_type),
        }
        data = {"messaging_product": "whatsapp", "type": mime_type}
        if self._upload_transport is not None:
            return self._upload_transport(url, dict(headers), dict(data), files)
        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(
                url,
                headers=dict(headers),
                data=data,
                files=files,
            )
            return response.status_code, response.text

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

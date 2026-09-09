"""Outbound Twilio WhatsApp client (MVP text-only).

Transport-only component: it delivers the final text produced by the
channel layer through ``client.messages.create``. It never logs or
propagates the Auth Token, and it is injectable/stubbeable in tests.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol


logger = logging.getLogger(__name__)


class MessageSender(Protocol):
    """Contract for outbound channel senders (stub-friendly)."""

    def send_text(self, recipient_id: str, body: str) -> None:
        ...


class TwilioWhatsAppSender:
    """Sends WhatsApp text messages through the Twilio REST API."""

    def __init__(
        self,
        *,
        account_sid: str,
        auth_token: str,
        from_number: str,
        client_factory: Callable[[], Any] | None = None,
        timeout_seconds: float = 15.0,
        content_sid: str | None = None,
        content_variables: str | None = None,
    ) -> None:
        if not account_sid or not account_sid.strip():
            raise ValueError("account_sid must be a non-empty string.")
        if not auth_token or not auth_token.strip():
            raise ValueError("auth_token must be a non-empty string.")
        if not from_number or not from_number.strip():
            raise ValueError("from_number must be a non-empty string.")
        self._account_sid = account_sid.strip()
        self._auth_token = auth_token.strip()
        self._from_number = from_number.strip()
        self._timeout_seconds = timeout_seconds
        self._client_factory = client_factory
        self._client: Any = None
        self._content_sid = (content_sid or "").strip() or None
        self._content_variables = (content_variables or "").strip() or None

    def send_text(self, recipient_id: str, body: str) -> None:
        """Deliver ``body`` to ``recipient_id``. Raises on delivery failure.

        No retry policy is applied in this MVP: delivery failures are
        logged upstream (courtesy/error replies) without retry loops.
        """
        if not isinstance(recipient_id, str) or not recipient_id.strip():
            raise ValueError("recipient_id must be a non-empty string.")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("body must be a non-empty string.")
        client = self._get_client()
        if self._content_sid is not None:
            # Twilio trial sandbox: REST sends are restricted to approved
            # Content templates (error 21654). Content-only mode, no body.
            kwargs: dict[str, Any] = {
                "to": recipient_id.strip(),
                "from_": self._from_number,
                "content_sid": self._content_sid,
            }
            if self._content_variables is not None:
                kwargs["content_variables"] = self._content_variables
            client.messages.create(**kwargs)
            logger.debug("twilio content message submitted | bytes=%s", len(body.encode("utf-8")))
            return
        client.messages.create(
            to=recipient_id.strip(),
            from_=self._from_number,
            body=body,
        )
        logger.debug("twilio text message submitted | bytes=%s", len(body.encode("utf-8")))

    def _get_client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from twilio.http.http_client import TwilioHttpClient
                from twilio.rest import Client

                self._client = Client(
                    self._account_sid,
                    self._auth_token,
                    http_client=TwilioHttpClient(timeout=self._timeout_seconds),
                )
        return self._client

"""Shared contract helpers for signed WhatsApp webhook tests."""

from __future__ import annotations

import hashlib
import hmac
import json
import os

import pytest
from starlette.testclient import TestClient


WHATSAPP_APP_SECRET = "atlas-test-whatsapp-app-secret"


@pytest.fixture(autouse=True)
def signed_whatsapp_webhook_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the mandatory app secret and sign implicit webhook test posts."""
    monkeypatch.setenv("ATLAS_WHATSAPP_APP_SECRET", WHATSAPP_APP_SECRET)
    original_post = TestClient.post

    def post(client: TestClient, url: str, *args, **kwargs):
        if url != "/webhook/whatsapp":
            return original_post(client, url, *args, **kwargs)
        headers = dict(kwargs.get("headers") or {})
        if "X-Hub-Signature-256" in headers:
            return original_post(client, url, *args, **kwargs)
        body = kwargs.pop("content", None)
        if body is None and "json" in kwargs:
            body = json.dumps(kwargs.pop("json"), separators=(",", ":")).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        if body is not None:
            if isinstance(body, str):
                body = body.encode("utf-8")
            headers["X-Hub-Signature-256"] = "sha256=" + hmac.new(
                WHATSAPP_APP_SECRET.encode("utf-8"), body, hashlib.sha256
            ).hexdigest()
            kwargs["content"] = body
            kwargs["headers"] = headers
        return original_post(client, url, *args, **kwargs)

    monkeypatch.setattr(TestClient, "post", post)
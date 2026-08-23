"""Secure WhatsApp media download through the Graph API (Phase 3, Block 4).

Downloads a media resource by id using the authenticated Graph API flow:
1. GET /{media_id} -> metadata with a temporary download URL.
2. GET the URL with the bearer token -> binary content.

The content is stored only in a secure temporary file outside the
repository. Access tokens and authenticated URLs are never logged or
included in raised error messages.
"""

from __future__ import annotations

import logging
import os
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any


logger = logging.getLogger(__name__)

GRAPH_API_BASE_URL = "https://graph.facebook.com/v21.0"

DEFAULT_MAX_MEDIA_BYTES = 20 * 1024 * 1024

ALLOWED_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "audio/ogg",
        "audio/mpeg",
        "audio/mp4",
        "video/mp4",
        "application/pdf",
    }
)


class MediaDownloadError(RuntimeError):
    """Controlled failure while downloading WhatsApp media."""


class InvalidMediaIdError(MediaDownloadError):
    """The provided media id is missing or malformed."""


@dataclass(frozen=True)
class DownloadedMedia:
    """Internal representation of a securely downloaded media resource."""

    path: Path
    media_id_hash: str
    mime_type: str
    size_bytes: int


class WhatsAppMediaDownloader:
    """Downloads WhatsApp media by id into secure temporary storage."""

    def __init__(
        self,
        *,
        access_token: str,
        base_url: str = GRAPH_API_BASE_URL,
        max_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
        allowed_mime_types: frozenset[str] = ALLOWED_MIME_TYPES,
        transport: Any = None,
        temp_root: str | None = None,
    ) -> None:
        if not access_token or not access_token.strip():
            raise ValueError("access_token must be a non-empty string.")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        self._access_token = access_token
        self._base_url = base_url.rstrip("/")
        self._max_bytes = max_bytes
        self._allowed_mime_types = allowed_mime_types
        self._transport = transport
        self._temp_root = temp_root

    def download(self, media_id: str) -> DownloadedMedia:
        """Download ``media_id`` and return its temporary representation."""
        if not isinstance(media_id, str) or not media_id.strip():
            raise InvalidMediaIdError("media_id must be a non-empty string.")
        media_id = media_id.strip()
        metadata_url = f"{self._base_url}/{media_id}"
        status, body_text = self._request(metadata_url)
        if status != 200:
            raise MediaDownloadError(
                f"whatsapp media metadata request failed with status {status}."
            )
        import json

        try:
            metadata = json.loads(body_text)
        except ValueError as error:
            raise MediaDownloadError("whatsapp media metadata is not valid json.") from error
        if not isinstance(metadata, dict):
            raise MediaDownloadError("whatsapp media metadata payload is unexpected.")

        download_url = metadata.get("url")
        if not isinstance(download_url, str) or not download_url.startswith("https://"):
            raise MediaDownloadError("whatsapp media metadata has no usable download url.")
        mime_type = metadata.get("mime_type")
        if not isinstance(mime_type, str) or mime_type not in self._allowed_mime_types:
            raise MediaDownloadError(f"whatsapp media type is not allowed.")
        expected_size = metadata.get("file_size")
        if isinstance(expected_size, int) and expected_size > self._max_bytes:
            raise MediaDownloadError("whatsapp media exceeds the maximum allowed size.")

        content_status, content = self._request(download_url)
        if content_status != 200 or not content:
            raise MediaDownloadError(
                f"whatsapp media download failed with status {content_status}."
            )
        if len(content) > self._max_bytes:
            raise MediaDownloadError("whatsapp media exceeds the maximum allowed size.")

        suffix = _safe_suffix(mime_type)
        file_descriptor, temp_path = tempfile.mkstemp(
            prefix="atlas-media-", suffix=suffix, dir=self._temp_root
        )
        try:
            with closing(os.fdopen(file_descriptor, "wb")) as handle:
                handle.write(content)
        except Exception as error:
            # Ensure the reserved descriptor is released even if fdopen
            # itself failed, otherwise the partial file cannot be removed.
            try:
                os.close(file_descriptor)
            except OSError:
                pass
            _discard(temp_path)
            raise MediaDownloadError(
                "whatsapp media could not be written to temporary storage."
            ) from error
        logger.info(
            "whatsapp media downloaded | bytes=%s | type=%s",
            len(content),
            mime_type,
        )
        return DownloadedMedia(
            path=Path(temp_path),
            media_id_hash=_hash_media_id(media_id),
            mime_type=mime_type,
            size_bytes=len(content),
        )

    def _request(self, url: str) -> tuple[int, str | bytes]:
        headers = {"Authorization": f"Bearer {self._access_token}"}
        if self._transport is not None:
            return self._transport(url, dict(headers))
        import httpx

        with httpx.Client(timeout=30.0, follow_redirects=False) as client:
            response = client.get(url, headers=headers)
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.status_code, response.text
            return response.status_code, response.content


def _hash_media_id(media_id: str) -> str:
    import hashlib

    return hashlib.sha256(media_id.encode("utf-8")).hexdigest()[:16]


def _safe_suffix(mime_type: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "video/mp4": ".mp4",
        "application/pdf": ".pdf",
    }
    return mapping.get(mime_type, ".bin")


def _discard(path: str) -> None:
    from pathlib import Path as _Path

    try:
        _Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("failed to remove partial whatsapp media temp file")

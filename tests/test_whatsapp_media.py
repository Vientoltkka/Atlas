"""Tests for secure WhatsApp media download (Phase 3, Block 4, Function 2)."""

from __future__ import annotations

import json

import pytest

from channels.whatsapp_media import (
    DownloadedMedia,
    InvalidMediaIdError,
    MediaDownloadError,
    WhatsAppMediaDownloader,
)


METADATA_URL = "https://graph.facebook.com/v21.0/media123"
DOWNLOAD_URL = "https://lookaside.fbsbx.com/download/media123"


@pytest.fixture()
def temp_dir(tmp_path_factory):
    import tempfile

    path = tempfile.mkdtemp(prefix="atlas-media-test-")
    yield path


def make_downloader(transport, temp_dir, **kwargs) -> WhatsAppMediaDownloader:
    return WhatsAppMediaDownloader(
        access_token="secret-token",
        transport=transport,
        temp_root=temp_dir,
        **kwargs,
    )


def metadata_transport(metadata_status=200, download_status=200):
    """Two-step fake transport: metadata JSON, then binary content."""
    calls: list[tuple[str, dict]] = []

    def transport(url: str, headers: dict):
        calls.append((url, headers))
        if url == METADATA_URL:
            return metadata_status, json.dumps(
                {
                    "url": DOWNLOAD_URL,
                    "mime_type": "image/jpeg",
                    "file_size": 5,
                }
            )
        return download_status, b"hello"

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def test_valid_media_downloads_to_temp_file(temp_dir) -> None:
    transport = metadata_transport()
    downloader = make_downloader(transport, temp_dir)
    media = downloader.download("media123")
    assert isinstance(media, DownloadedMedia)
    assert media.path.exists()
    assert media.path.read_bytes() == b"hello"
    assert media.path.parent.exists() and "atlas-media-test" not in str(media.path.resolve())[:0]
    assert media.mime_type == "image/jpeg"
    assert media.size_bytes == 5
    assert media.path.suffix == ".jpg"
    media.path.unlink()


def test_authentication_sent_and_token_not_exposed(temp_dir) -> None:
    transport = metadata_transport()
    downloader = make_downloader(transport, temp_dir)
    media = downloader.download("media123")
    for _url, headers in transport.calls:
        assert headers["Authorization"] == "Bearer secret-token"
    # The token never appears in the returned representation.
    serialized = repr(media)
    assert "secret-token" not in serialized
    assert DOWNLOAD_URL not in serialized
    media.path.unlink()


def test_metadata_error_is_controlled(temp_dir) -> None:
    transport = metadata_transport(metadata_status=404)
    downloader = make_downloader(transport, temp_dir)
    with pytest.raises(MediaDownloadError):
        downloader.download("media123")


def test_download_error_is_controlled_without_partial_file(temp_dir) -> None:
    transport = metadata_transport(download_status=500)
    downloader = make_downloader(transport, temp_dir)
    with pytest.raises(MediaDownloadError):
        downloader.download("media123")
    # No partial file was created in the temp root.
    assert len(list(__import__("pathlib").Path(temp_dir).glob("atlas-media-*"))) == 0


def test_invalid_media_id_rejected(temp_dir) -> None:
    downloader = make_downloader(metadata_transport(), temp_dir)
    with pytest.raises(InvalidMediaIdError):
        downloader.download("")
    with pytest.raises(InvalidMediaIdError):
        downloader.download(None)  # type: ignore[arg-type]


def test_oversized_media_rejected(temp_dir) -> None:
    transport = metadata_transport()
    downloader = make_downloader(transport, temp_dir, max_bytes=3)
    with pytest.raises(MediaDownloadError):
        downloader.download("media123")


def test_disallowed_mime_type_rejected(temp_dir) -> None:
    def transport(url: str, headers: dict):
        return 200, json.dumps(
            {"url": DOWNLOAD_URL, "mime_type": "application/zip", "file_size": 5}
        )

    downloader = make_downloader(transport, temp_dir)
    with pytest.raises(MediaDownloadError):
        downloader.download("media123")


def test_non_https_url_rejected(temp_dir) -> None:
    def transport(url: str, headers: dict):
        if url.endswith("/media123"):
            return 200, json.dumps({"url": "http://evil.example/file", "mime_type": "image/jpeg"})
        return 200, b"data"

    downloader = make_downloader(transport, temp_dir)
    with pytest.raises(MediaDownloadError):
        downloader.download("media123")


def test_write_failure_leaves_no_partial_file(temp_dir) -> None:
    """Simulated write failure: controlled error, fd closed, no leftovers."""
    import os
    from pathlib import Path as _Path

    def transport(url: str, headers: dict):
        if url.endswith("/media123"):
            return 200, json.dumps(
                {"url": DOWNLOAD_URL, "mime_type": "image/jpeg", "file_size": 5}
            )
        return 200, b"hello"

    original_fdopen = os.fdopen

    def failing_fdopen(fd, *args, **kwargs):
        raise OSError("disk full")

    os.fdopen = failing_fdopen
    try:
        downloader = WhatsAppMediaDownloader(
            access_token="secret-token",
            transport=transport,
            temp_root=temp_dir,
        )
        with pytest.raises(MediaDownloadError):
            downloader.download("media123")
    finally:
        os.fdopen = original_fdopen

    # No partial file remains in the temp root.
    assert list(_Path(temp_dir).glob("atlas-media-*")) == []

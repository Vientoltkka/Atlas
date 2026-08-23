"""Audio transcription adapter for WhatsApp voice messages (Phase 3, Block 4).

Bridges the secure media downloader (Function 2) with the existing local
faster-whisper speech-to-text provider. Decodes the downloaded audio to
mono float32 PCM at 16 kHz using PyAV (already a faster-whisper dependency)
and feeds it to the existing provider without duplicating any Whisper logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
DEFAULT_MAX_DURATION_SECONDS = 120.0


class AudioTranscriptionError(RuntimeError):
    """Controlled failure while transcribing a WhatsApp audio message."""


class WhatsAppAudioTranscriber:
    """Downloads and transcribes a WhatsApp audio media id into text."""

    def __init__(
        self,
        *,
        downloader,
        provider,
        max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
    ) -> None:
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive.")
        self._downloader = downloader
        self._provider = provider
        self._max_duration_seconds = max_duration_seconds

    def transcribe_media_id(self, media_id: str) -> str:
        """Download ``media_id`` securely and return its transcript."""
        media = self._downloader.download(media_id)
        try:
            return self._transcribe_file(media.path)
        finally:
            _discard_temp(media.path)

    def _transcribe_file(self, path: Path) -> str:
        try:
            samples = self._decode(path)
        except Exception as error:
            raise AudioTranscriptionError(
                "whatsapp audio could not be decoded."
            ) from error
        duration = len(samples) / TARGET_SAMPLE_RATE
        if duration > self._max_duration_seconds:
            raise AudioTranscriptionError(
                "whatsapp audio exceeds the maximum allowed duration."
            )
        if not len(samples):
            raise AudioTranscriptionError("whatsapp audio contains no samples.")

        result = self._provider.transcribe(samples, TARGET_SAMPLE_RATE)
        text = getattr(result, "text", "")
        if not isinstance(text, str) or not text.strip():
            raise AudioTranscriptionError("whatsapp audio produced no transcript.")
        logger.info(
            "whatsapp audio transcribed | seconds=%.1f | chars=%s",
            duration,
            len(text),
        )
        return text.strip()

    def _decode(self, path: Path) -> np.ndarray:
        """Decode any supported container to mono float32 at 16 kHz via PyAV."""
        import av

        chunks: list[np.ndarray] = []
        with av.open(str(path)) as container:
            stream = container.streams.audio
            if not stream:
                raise AudioTranscriptionError("whatsapp audio has no audio stream.")
            resampler = av.AudioResampler(
                format="fltp",
                layout="mono",
                rate=TARGET_SAMPLE_RATE,
            )
            for frame in container.decode(stream[0]):
                for resampled in resampler.resample(frame):
                    array = resampled.to_ndarray()
                    chunks.append(array.reshape(-1))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)


def _discard_temp(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("failed to remove whatsapp audio temp file")

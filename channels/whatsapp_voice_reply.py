"""Voice reply rendering for WhatsApp (Phase 3, Block 4, Function 4).

Renders an Atlas text reply into an OGG/Opus audio file suitable for the
WhatsApp Graph API by reusing pyttsx3 (save_to_file, no local playback)
and PyAV for conversion. Temporary files live outside the repository and
are always cleaned up.
"""

from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger(__name__)

DEFAULT_MAX_REPLY_SECONDS = 120.0
TARGET_SAMPLE_RATE = 16_000


class VoiceReplyError(RuntimeError):
    """Controlled failure while rendering a voice reply."""


class WhatsAppVoiceReplyRenderer:
    """Renders reply text into an OGG/Opus file via pyttsx3 + PyAV."""

    def __init__(
        self,
        *,
        engine_factory=None,
        temp_root: str | None = None,
        max_bytes: int = 20 * 1024 * 1024,
        max_duration_seconds: float = DEFAULT_MAX_REPLY_SECONDS,
        converter=None,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive.")
        self._engine_factory = engine_factory or _default_engine_factory
        self._temp_root = temp_root
        self._max_bytes = max_bytes
        self._max_duration_seconds = max_duration_seconds
        self._converter = converter or _convert_wav_to_ogg

    def render(self, text: str) -> Path:
        """Render ``text`` to an OGG/Opus file. Caller deletes the result."""
        clean = (text or "").strip()
        if not clean:
            raise VoiceReplyError("voice reply text is empty.")
        wav_path = self._synthesize_wav(clean)
        try:
            ogg_path = self._convert(wav_path)
        finally:
            _discard(wav_path)
        size = ogg_path.stat().st_size
        if size > self._max_bytes:
            _discard(ogg_path)
            raise VoiceReplyError("voice reply exceeds the maximum allowed size.")
        duration = _estimate_duration_seconds(ogg_path)
        if duration is not None and duration > self._max_duration_seconds:
            _discard(ogg_path)
            raise VoiceReplyError("voice reply exceeds the maximum allowed duration.")
        return ogg_path

    def _convert(self, wav_path: Path) -> Path:
        import os
        import tempfile

        file_descriptor, ogg_path = tempfile.mkstemp(
            prefix="atlas-voice-", suffix=".ogg", dir=self._temp_root
        )
        # Close and remove the reserved name so PyAV can write to it.
        os.close(file_descriptor)
        Path(ogg_path).unlink(missing_ok=True)
        try:
            self._converter(wav_path, Path(ogg_path))
        except Exception as error:
            _discard(ogg_path)
            raise VoiceReplyError("voice reply conversion failed.") from error
        if not Path(ogg_path).exists():
            raise VoiceReplyError("voice reply conversion produced no file.")
        return Path(ogg_path)

    def _synthesize_wav(self, text: str) -> Path:
        import os
        import tempfile

        engine = None
        file_descriptor, wav_path = tempfile.mkstemp(
            prefix="atlas-voice-", suffix=".wav", dir=self._temp_root
        )
        # Close and remove the reserved name so SAPI can write to it.
        os.close(file_descriptor)
        Path(wav_path).unlink(missing_ok=True)
        try:
            engine = self._engine_factory()
            engine.save_to_file(text, wav_path)
            engine.runAndWait()
        except Exception as error:
            _discard(wav_path)
            raise VoiceReplyError("text to speech synthesis failed.") from error
        rendered = Path(wav_path)
        if not rendered.exists() or rendered.stat().st_size <= 44:
            _discard(wav_path)
            raise VoiceReplyError("text to speech produced no usable audio.")
        return rendered


def _default_engine_factory():
    """Build a fresh pyttsx3 engine per synthesis (SAPI5 requirement)."""
    import pyttsx3

    try:
        engine = pyttsx3.init()
    except Exception as error:
        raise VoiceReplyError("pyttsx3 could not be initialized.") from error
    return engine


def _convert_wav_to_ogg(wav_path: Path, ogg_path: Path) -> None:
    import av

    with av.open(str(wav_path)) as source:
        input_stream = source.streams.audio[0]
        resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=16_000,
        )
        with av.open(str(ogg_path), "w", format="ogg") as target:
            outgoing_stream = target.add_stream("libopus", rate=16_000)
            outgoing_stream.layout = "mono"
            for frame in source.decode(input_stream):
                for resampled in resampler.resample(frame):
                    for packet in outgoing_stream.encode(resampled):
                        target.mux(packet)


def _estimate_duration_seconds(path: Path) -> float | None:
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration is None:
                return None
            return container.duration / 1_000_000
    except Exception:
        return None


def _discard(path: Path | str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("failed to remove whatsapp voice reply temp file")

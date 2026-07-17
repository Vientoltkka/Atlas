from __future__ import annotations

from pathlib import Path

import pytest

import bootstrap.bootstrap as bootstrap
from use_cases.speech_engine import SpeechTranscriptionResult
from use_cases.stt_wake_word_engine import SttWakeWordEngine


def speech_result(
    text: str,
    completed: bool = True,
    cancelled: bool = False,
    no_speech: bool = False,
) -> SpeechTranscriptionResult:
    return SpeechTranscriptionResult(
        text=text,
        language="es" if completed else None,
        audio_duration_seconds=0.5,
        processing_duration_seconds=0.1,
        provider="fake",
        microphone_name="fake mic",
        completed=completed,
        cancelled=cancelled,
        no_speech_detected=no_speech,
    )


class FakeSpeechEngine:
    def __init__(self, results: list[SpeechTranscriptionResult]) -> None:
        self._results = list(results)
        self.capture_settings_seen = []

    def transcribe_once(self, capture_settings=None) -> SpeechTranscriptionResult:
        self.capture_settings_seen.append(capture_settings)

        if not self._results:
            return speech_result("", completed=False, no_speech=True)

        return self._results.pop(0)


def test_normalizes_atlas_variants() -> None:
    assert SttWakeWordEngine.normalize_text("  ÁTLAS,   qué hora es? ") == "atlas, que hora es?"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Atlas", ""),
        ("oye Atlas", ""),
        ("Atlas, que hora es", "que hora es"),
        ("por favor Atlas abre el proyecto", "abre el proyecto"),
    ],
)
def test_extracts_full_word_activation(text: str, expected: str) -> None:
    assert SttWakeWordEngine.extract_activation(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "atlash",
        "atlasico",
        "metatlas",
        "la atlasa nueva",
        "",
    ],
)
def test_rejects_false_positives(text: str) -> None:
    assert SttWakeWordEngine.extract_activation(text) is None


def test_stt_wake_detects_separate_activation() -> None:
    speech = FakeSpeechEngine([speech_result("Atlas")])
    engine = SttWakeWordEngine(speech)

    result = engine.wait_for_wake_word()

    assert result.detected is True
    assert result.phrase is not None
    assert result.phrase.text == ""
    assert speech.capture_settings_seen[0] is not None


def test_stt_wake_detects_activation_and_request_in_one_phrase() -> None:
    speech = FakeSpeechEngine([speech_result("Atlas, que hora es")])
    engine = SttWakeWordEngine(speech)

    result = engine.wait_for_wake_word()

    assert result.detected is True
    assert result.phrase is not None
    assert result.phrase.text == "que hora es"


def test_stt_wake_returns_passive_miss_without_activation() -> None:
    speech = FakeSpeechEngine([speech_result("abre el proyecto")])
    engine = SttWakeWordEngine(speech)

    result = engine.wait_for_wake_word()

    assert result.detected is False
    assert result.warnings == ("sin activacion STT",)


def test_assistant_falls_back_to_stt_without_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLAS_WAKE_WORD_MODEL_PATH", raising=False)

    engine = bootstrap._assistant_wake_word_engine(FakeSpeechEngine([]))

    assert isinstance(engine, SttWakeWordEngine)


def test_assistant_uses_stt_when_onnx_path_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATLAS_WAKE_WORD_MODEL_PATH", str(tmp_path / "Atlas.onnx"))

    engine = bootstrap._assistant_wake_word_engine(FakeSpeechEngine([]))

    assert isinstance(engine, SttWakeWordEngine)

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.orchestrator import AtlasOrchestrator
from use_cases.speech_engine import (
    AudioCaptureResult,
    ProviderTranscriptionResult,
    SpeechEngineUseCase,
    SoundDeviceAudioCapture,
)
from use_cases.wake_word_engine import (
    OpenWakeWordProvider,
    WakeWordDetectionResult,
    WakeWordEngine,
    WakeWordInteractionUseCase,
)


class CloseableFrames:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = iter(frames)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._frames)

    def close(self) -> None:
        self.closed = True


class FakeSpeechEngine:
    def __init__(self, frames: list[np.ndarray] | None = None) -> None:
        self.frames = CloseableFrames(frames or [])
        self.frame_requests: list[tuple[int, int]] = []
        self.transcribe_calls = 0
        self.active_microphone_name = "Selected Mic"
        self.active_microphone_index = 1

    def iter_pcm_frames(self, sample_rate: int, frame_length: int):
        self.frame_requests.append((sample_rate, frame_length))
        return self.frames

    def transcribe_once(self):
        self.transcribe_calls += 1
        return SimpleNamespace(
            text="abre el proyecto",
            language="es",
            audio_duration_seconds=1.0,
            processing_duration_seconds=0.2,
            provider="fake-local",
            microphone_name=self.active_microphone_name,
            completed=True,
            cancelled=False,
            no_speech_detected=False,
            warnings=(),
            summary="fake",
        )


class FakeProvider:
    def __init__(
        self,
        process_results: list[bool],
        sample_rate: int = 16_000,
        frame_length: int = 512,
        fail_initialize: Exception | None = None,
        fail_process: Exception | None = None,
    ) -> None:
        self._process_results = list(process_results)
        self._sample_rate = sample_rate
        self._frame_length = frame_length
        self.fail_initialize = fail_initialize
        self.fail_process = fail_process
        self.initialized = False
        self.closed = False
        self.frames_seen: list[np.ndarray] = []

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_length(self) -> int:
        return self._frame_length

    def initialize(self) -> None:
        if self.fail_initialize is not None:
            raise self.fail_initialize
        self.initialized = True

    def process_frame(self, pcm_frame: np.ndarray) -> bool:
        if self.fail_process is not None:
            raise self.fail_process
        self.frames_seen.append(np.asarray(pcm_frame))
        return self._process_results.pop(0) if self._process_results else False

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._last = values[-1] if values else 0.0

    def __call__(self) -> float:
        if self._values:
            self._last = self._values.pop(0)

        return self._last


def frame(length: int = 512, value: int = 1) -> np.ndarray:
    return np.full(length, value, dtype=np.int16)


def test_openwakeword_configuration_loads_valid_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "Atlas.onnx"
    model.write_bytes(b"model")
    created: dict[str, object] = {}

    class FakeModel:
        def predict(self, _frame):
            return {"Atlas": 0.0}

    def factory(**kwargs):
        created.update(kwargs)
        return FakeModel()

    monkeypatch.setenv("ATLAS_WAKE_WORD_MODEL_PATH", str(model))
    monkeypatch.setenv("ATLAS_WAKE_WORD_SENSITIVITY", "0.7")
    provider = OpenWakeWordProvider.from_environment()
    provider._model_factory = factory

    provider.initialize()

    assert provider.sample_rate == 16_000
    assert provider.frame_length == 1280
    assert created["wakeword_models"] == [str(model.resolve())]
    assert created["inference_framework"] == "onnx"


def test_openwakeword_rejects_missing_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLAS_WAKE_WORD_MODEL_PATH", raising=False)
    provider = OpenWakeWordProvider.from_environment()

    with pytest.raises(RuntimeError) as error:
        provider.initialize()

    assert str(error.value) == (
        "Wake word no configurada. "
        "Define ATLAS_WAKE_WORD_MODEL_PATH con un modelo .onnx local."
    )


def test_openwakeword_rejects_invalid_model_path_and_extension(tmp_path: Path) -> None:
    missing = tmp_path / "Atlas.onnx"
    txt = tmp_path / "Atlas.txt"
    txt.write_text("bad")

    with pytest.raises(RuntimeError, match="no existe"):
        OpenWakeWordProvider(missing).initialize()

    with pytest.raises(RuntimeError, match="archivo .onnx"):
        OpenWakeWordProvider(txt).initialize()


@pytest.mark.parametrize("sensitivity", [-0.1, 1.1, "bad"])
def test_openwakeword_validates_sensitivity(tmp_path: Path, sensitivity) -> None:
    model = tmp_path / "Atlas.onnx"
    model.write_bytes(b"model")

    with pytest.raises(RuntimeError, match="SENSITIVITY"):
        OpenWakeWordProvider(model, sensitivity=sensitivity).initialize()


def test_openwakeword_does_not_expose_unnecessary_paths(tmp_path: Path) -> None:
    model = tmp_path / "Atlas.txt"
    model.write_text("bad")

    with pytest.raises(RuntimeError) as error:
        OpenWakeWordProvider(model).initialize()

    assert str(model) not in str(error.value)


def test_openwakeword_initializes_once_and_detects_above_threshold(tmp_path: Path) -> None:
    model = tmp_path / "Atlas.onnx"
    model.write_bytes(b"model")

    class FakeModel:
        def __init__(self) -> None:
            self.frames: list[np.ndarray] = []
            self.reset_calls = 0

        def predict(self, frame_value):
            self.frames.append(np.asarray(frame_value))
            return {"Atlas": 0.56}

        def reset(self):
            self.reset_calls += 1

    fake = FakeModel()
    created = 0

    def factory(**_kwargs):
        nonlocal created
        created += 1
        return fake

    provider = OpenWakeWordProvider(
        model,
        model_factory=factory,
    )

    assert provider._model is None
    assert provider.process_frame(np.array([[1], [2], [3]], dtype=np.float32)) is True
    assert provider.process_frame(np.array([1, 2, 3], dtype=np.int16)) is True
    assert provider._model is fake
    assert created == 1
    assert fake.frames[0].dtype == np.int16

    provider.close()
    assert fake.reset_calls == 1


def test_openwakeword_rejects_below_threshold(tmp_path: Path) -> None:
    model = tmp_path / "Atlas.onnx"
    model.write_bytes(b"model")

    class FakeModel:
        def predict(self, _frame):
            return {"Atlas": 0.54}

    provider = OpenWakeWordProvider(
        model,
        sensitivity=0.55,
        model_factory=lambda **_kwargs: FakeModel(),
    )

    assert provider.process_frame(np.ones(1280, dtype=np.int16)) is False


def test_openwakeword_rejects_empty_or_invalid_prediction(tmp_path: Path) -> None:
    model = tmp_path / "Atlas.onnx"
    model.write_bytes(b"model")

    class InvalidModel:
        def predict(self, _frame):
            return []

    provider = OpenWakeWordProvider(
        model,
        model_factory=lambda **_kwargs: InvalidModel(),
    )

    with pytest.raises(RuntimeError, match="Frame PCM invalido"):
        provider.process_frame(np.array([], dtype=np.int16))

    with pytest.raises(RuntimeError, match="Prediccion invalida"):
        provider.process_frame(np.ones(1280, dtype=np.int16))


def test_detects_wake_word_from_provider_and_closes_stream_and_provider() -> None:
    speech = FakeSpeechEngine([frame(), frame()])
    provider = FakeProvider([False, True])
    engine = WakeWordEngine(
        speech,
        provider=provider,
        capture_phrase_after_detection=False,
    )

    result = engine.wait_for_wake_word()

    assert result.detected is True
    assert result.attempts == 2
    assert speech.frame_requests == [(16_000, 512)]
    assert speech.frames.closed is True
    assert provider.closed is True
    assert len(provider.frames_seen) == 2
    assert speech.transcribe_calls == 0


def test_continues_when_provider_returns_negative_and_times_out() -> None:
    speech = FakeSpeechEngine([frame(), frame(), frame()])
    provider = FakeProvider([False, False, False])
    clock = FakeClock([0.0, 0.0, 0.5, 2.0])
    engine = WakeWordEngine(
        speech,
        provider=provider,
        timeout_seconds=1.0,
        capture_phrase_after_detection=False,
        clock=clock,
    )

    result = engine.wait_for_wake_word()

    assert result.detected is False
    assert result.warnings == ("timeout de wake word alcanzado",)
    assert speech.frames.closed is True
    assert provider.closed is True


def test_cancels_with_ctrl_c_and_closes_provider() -> None:
    class InterruptingFrames(CloseableFrames):
        def __next__(self):
            raise KeyboardInterrupt

    speech = FakeSpeechEngine([])
    speech.frames = InterruptingFrames([])
    provider = FakeProvider([])
    engine = WakeWordEngine(speech, provider=provider)

    result = engine.wait_for_wake_word()

    assert result.cancelled is True
    assert speech.frames.closed is True
    assert provider.closed is True


def test_closes_provider_after_processing_error() -> None:
    speech = FakeSpeechEngine([frame()])
    provider = FakeProvider([], fail_process=RuntimeError("boom"))
    engine = WakeWordEngine(speech, provider=provider)

    result = engine.wait_for_wake_word()

    assert result.detected is False
    assert result.warnings == ("boom",)
    assert provider.closed is True
    assert speech.frames.closed is True


def test_configuration_error_does_not_open_stream() -> None:
    speech = FakeSpeechEngine([frame()])
    provider = FakeProvider(
        [],
        fail_initialize=RuntimeError(
            "Wake word no configurada. Define ATLAS_WAKE_WORD_MODEL_PATH con un modelo .onnx local."
        ),
    )
    engine = WakeWordEngine(speech, provider=provider)

    result = engine.wait_for_wake_word()

    assert result.configuration_error is True
    assert speech.frame_requests == []
    assert provider.closed is True


def test_standalone_interaction_reports_configuration_error() -> None:
    result = WakeWordDetectionResult(
        wake_word="Atlas",
        detected=False,
        attempts=0,
        elapsed_seconds=0.0,
        configuration_error=True,
        warnings=("Wake word no configurada. Define ATLAS_WAKE_WORD_MODEL_PATH con un modelo .onnx local.",),
    )
    interaction = WakeWordInteractionUseCase(
        SimpleNamespace(wait_for_wake_word=lambda: result)
    )

    assert (
        interaction.execute("atlas")
        == "Wake word no configurada. Define ATLAS_WAKE_WORD_MODEL_PATH con un modelo .onnx local."
    )


def test_capture_can_yield_int16_pcm_from_selected_microphone(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = SoundDeviceAudioCapture()
    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(0, None)),
        query_devices=lambda: [
            {"name": "Default Mic", "max_input_channels": 1},
            {"name": "Selected Mic", "max_input_channels": 1},
        ],
    )
    stream_state: dict[str, object] = {}

    class Stream:
        def __init__(self, **kwargs) -> None:
            stream_state.update(kwargs)
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def read(self, frame_length: int):
            return np.ones((frame_length, 1), dtype=np.int16), False

    fake_sd.InputStream = Stream
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    capture.select_microphone(1)
    iterator = capture.iter_pcm_frames(sample_rate=16_000, frame_length=512)
    pcm = next(iterator)
    iterator.close()

    assert stream_state["device"] == 1
    assert stream_state["samplerate"] == 16_000
    assert stream_state["blocksize"] == 512
    assert stream_state["dtype"] == "int16"
    assert pcm.dtype == np.int16
    assert pcm.shape == (512,)


def test_detection_can_capture_phrase_after_provider_event() -> None:
    speech = FakeSpeechEngine([frame()])
    provider = FakeProvider([True])
    engine = WakeWordEngine(
        speech,
        provider=provider,
        capture_phrase_after_detection=True,
    )

    result = engine.wait_for_wake_word()

    assert result.detected is True
    assert result.phrase is not None
    assert speech.transcribe_calls == 1


def test_orchestrator_wake_word_runs_before_router_and_agent(monkeypatch, capsys) -> None:
    class FailingRouter:
        def route(self, _plan):  # pragma: no cover - must not be called
            raise AssertionError("router should not receive wake word flow")

    response = "Wake word no configurada. Define ATLAS_WAKE_WORD_MODEL_PATH con un modelo .onnx local."
    inputs = iter(["atlas", "salir"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda _prompt: object()),
        router=FailingRouter(),
        model_manager=SimpleNamespace(choose_model=lambda _agent: "unused"),
        memory=SimpleNamespace(add_user=lambda _prompt: None, add_assistant=lambda _response: None, history=list),
        registry=SimpleNamespace(get=lambda _name: None),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        wake_word_interaction=SimpleNamespace(execute=lambda prompt: response if prompt == "atlas" else None),
    )

    orchestrator.start()
    output = capsys.readouterr().out

    assert "Wake word no configurada" in output


def test_bootstrap_injects_wake_word_interaction() -> None:
    from bootstrap.bootstrap import Bootstrap

    orchestrator = Bootstrap.build()

    assert orchestrator._wake_word_interaction is not None


def test_no_active_proprietary_wake_provider_references_remain() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = (
        "Porcu" + "pine",
        "pv" + "porcu" + "pine",
        "ATLAS_" + "PORCUPINE_ACCESS_KEY",
    )
    checked_paths = [
        *root.joinpath("bootstrap").glob("*.py"),
        *root.joinpath("use_cases").glob("*.py"),
        root / "requirements.txt",
    ]

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        assert not any(item in text for item in forbidden), path

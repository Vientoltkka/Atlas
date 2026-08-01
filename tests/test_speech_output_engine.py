from __future__ import annotations

from types import SimpleNamespace

import pytest

from use_cases.speech_output_engine import (
    Pyttsx3SpeechOutputEngine,
    SpeechOutputSettings,
)


class FakePyttsx3Module:
    def __init__(self, engines) -> None:
        self.engines = list(engines)
        self.init_calls = 0

    def init(self):
        self.init_calls += 1
        return self.engines.pop(0)


class FakePyttsx3Engine:
    def __init__(self, fail_run: bool = False, busy: bool = False) -> None:
        self.fail_run = fail_run
        self.busy = busy
        self.say_calls: list[str] = []
        self.run_and_wait_calls = 0
        self.stop_calls = 0
        self.properties: list[tuple[str, object]] = []

    def setProperty(self, name: str, value) -> None:
        self.properties.append((name, value))

    def getProperty(self, name: str):
        if name == "voices":
            return [SimpleNamespace(id="spanish", name="Spanish Voice")]
        return None

    def isBusy(self) -> bool:
        return self.busy

    def stop(self) -> None:
        self.stop_calls += 1
        self.busy = False

    def say(self, text: str) -> None:
        self.say_calls.append(text)

    def runAndWait(self) -> None:
        self.run_and_wait_calls += 1
        if self.fail_run:
            raise RuntimeError("cola rota")


def test_pyttsx3_runs_once_per_response(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = FakePyttsx3Engine()
    fake_module = FakePyttsx3Module([engine])
    monkeypatch.setitem(__import__("sys").modules, "pyttsx3", fake_module)

    output = Pyttsx3SpeechOutputEngine(SpeechOutputSettings())
    output.speak("respuesta")

    assert engine.say_calls == ["respuesta"]
    assert engine.run_and_wait_calls == 1


def test_pyttsx3_rebuilds_engine_for_each_successful_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FakePyttsx3Engine()
    second = FakePyttsx3Engine()
    third = FakePyttsx3Engine()
    fake_module = FakePyttsx3Module([first, second, third])
    monkeypatch.setitem(__import__("sys").modules, "pyttsx3", fake_module)

    output = Pyttsx3SpeechOutputEngine(SpeechOutputSettings())
    output.speak("primera")
    output.speak("segunda")
    output.speak("tercera")

    assert first.say_calls == ["primera"]
    assert second.say_calls == ["segunda"]
    assert third.say_calls == ["tercera"]
    assert fake_module.init_calls == 3


def test_pyttsx3_stops_busy_engine_before_speaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakePyttsx3Engine(busy=True)
    fake_module = FakePyttsx3Module([engine])
    monkeypatch.setitem(__import__("sys").modules, "pyttsx3", fake_module)

    output = Pyttsx3SpeechOutputEngine(SpeechOutputSettings())
    output.speak("respuesta")

    assert engine.stop_calls == 2
    assert engine.run_and_wait_calls == 1


def test_pyttsx3_failure_discards_engine_and_rebuilds_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = FakePyttsx3Engine(fail_run=True)
    healthy = FakePyttsx3Engine()
    fake_module = FakePyttsx3Module([broken, healthy])
    monkeypatch.setitem(__import__("sys").modules, "pyttsx3", fake_module)

    output = Pyttsx3SpeechOutputEngine(SpeechOutputSettings())

    with pytest.raises(RuntimeError):
        output.speak("primera")

    output.speak("segunda")

    assert broken.stop_calls == 1
    assert healthy.say_calls == ["segunda"]
    assert fake_module.init_calls == 2


def test_pyttsx3_reports_separate_monotonic_synthesis_and_playback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakePyttsx3Engine()
    fake_module = FakePyttsx3Module([engine])
    ticks = iter((1.0, 1.1, 1.2, 1.5))
    monkeypatch.setitem(__import__("sys").modules, "pyttsx3", fake_module)
    monkeypatch.setattr(
        "use_cases.speech_output_engine.time.monotonic",
        lambda: next(ticks),
    )

    metrics = Pyttsx3SpeechOutputEngine(SpeechOutputSettings()).speak_with_metrics(
        "respuesta"
    )

    assert metrics.synthesis_seconds == pytest.approx(0.1)
    assert metrics.playback_seconds == pytest.approx(0.3)
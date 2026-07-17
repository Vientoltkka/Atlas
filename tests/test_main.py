from __future__ import annotations

import sys

import main


def test_main_starts_text_mode_by_default(monkeypatch) -> None:
    calls: list[str] = []

    class FakeAtlas:
        def start(self) -> None:
            calls.append("text")

        def start_voice(self) -> None:
            calls.append("voice")

        def start_assistant(self) -> None:
            calls.append("assistant")

    monkeypatch.setattr(main, "Atlas", FakeAtlas)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    main.main()

    assert calls == ["text"]


def test_main_starts_voice_mode_with_flag(monkeypatch) -> None:
    calls: list[str] = []

    class FakeAtlas:
        def start(self) -> None:
            calls.append("text")

        def start_voice(self) -> None:
            calls.append("voice")

        def start_assistant(self) -> None:
            calls.append("assistant")

    monkeypatch.setattr(main, "Atlas", FakeAtlas)
    monkeypatch.setattr(sys, "argv", ["main.py", "--voice"])

    main.main()

    assert calls == ["voice"]


def test_main_starts_assistant_mode_with_flag(monkeypatch) -> None:
    calls: list[str] = []

    class FakeAtlas:
        def start(self) -> None:
            calls.append("text")

        def start_voice(self) -> None:
            calls.append("voice")

        def start_assistant(self) -> None:
            calls.append("assistant")

    monkeypatch.setattr(main, "Atlas", FakeAtlas)
    monkeypatch.setattr(sys, "argv", ["main.py", "--assistant"])

    main.main()

    assert calls == ["assistant"]


def test_main_lists_microphones(monkeypatch, capsys) -> None:
    calls: list[str] = []

    class FakeAtlas:
        def __init__(self) -> None:
            raise AssertionError("Atlas must not be built for microphone listing")

        def start(self) -> None:
            calls.append("text")

        def start_voice(self) -> None:
            calls.append("voice")

        def start_assistant(self) -> None:
            calls.append("assistant")

        def list_microphones(self) -> str:
            calls.append("list")
            return "Microfonos disponibles:\n0. Microfono fisico"

    class FakeCapture:
        def list_microphones(self, include_open_status=False):
            calls.append(f"list:{include_open_status}")
            return [
                type(
                    "Mic",
                    (),
                    {
                        "index": 1,
                        "name": "Microfono fisico",
                        "is_default": False,
                        "host_api": "MME",
                        "channels": 2,
                        "default_samplerate": 44100.0,
                        "can_open": True,
                        "open_error": "",
                    },
                )()
            ]

    monkeypatch.setattr(main, "Atlas", FakeAtlas)
    monkeypatch.setattr(main, "SoundDeviceAudioCapture", FakeCapture)
    monkeypatch.setattr(sys, "argv", ["main.py", "--list-microphones"])

    main.main()

    output = capsys.readouterr().out
    assert calls == ["list:True"]
    assert "Microfonos disponibles:" in output
    assert "Microfono fisico" in output
    assert "host API: MME" in output


def test_main_tests_microphone_without_building_atlas(monkeypatch, capsys) -> None:
    calls: list[str] = []

    class FakeAtlas:
        def __init__(self) -> None:
            raise AssertionError("Atlas must not be built for microphone test")

    class FakeCapture:
        def test_microphone(self, index: int, duration_seconds: float = 3.0):
            calls.append(f"test:{index}:{duration_seconds}")
            microphone = type(
                "Mic",
                (),
                {
                    "index": index,
                    "name": "Microfono fisico",
                    "host_api": "MME",
                    "channels": 2,
                },
            )()
            return type(
                "Result",
                (),
                {
                    "microphone": microphone,
                    "error": "",
                    "duration_seconds": 3.0,
                    "rms": 0.01,
                    "voice_detected": True,
                },
            )()

    monkeypatch.setattr(main, "Atlas", FakeAtlas)
    monkeypatch.setattr(main, "SoundDeviceAudioCapture", FakeCapture)
    monkeypatch.setattr(sys, "argv", ["main.py", "--test-microphone", "1"])

    main.main()

    output = capsys.readouterr().out
    assert calls == ["test:1:3.0"]
    assert "RMS: 0.0100" in output
    assert "Voz detectada: si" in output

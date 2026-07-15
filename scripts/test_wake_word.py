"""Local wake word microphone validation script for Atlas."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from use_cases.speech_engine import (
    ProviderTranscriptionResult,
    SoundDeviceAudioCapture,
    SpeechEngineUseCase,
)
from use_cases.wake_word_engine import OpenWakeWordProvider


class _UnusedSpeechToTextProvider:
    name = "unused"

    def transcribe(self, *_args) -> ProviderTranscriptionResult:
        raise RuntimeError("Este script no ejecuta transcripcion Whisper.")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prueba local de la wake word Atlas con openWakeWord.",
    )
    parser.add_argument(
        "--microphone",
        type=int,
        default=None,
        help="Indice del microfono de entrada. Si se omite, usa el predeterminado.",
    )
    parser.add_argument(
        "--list-microphones",
        action="store_true",
        help="Lista microfonos y termina.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Tiempo maximo de escucha. 0 significa sin limite.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Imprime scores cada N frames.",
    )
    return parser.parse_args()


def _print_microphones(speech_engine: SpeechEngineUseCase) -> None:
    microphones = speech_engine.list_microphones()

    if not microphones:
        print("No hay microfonos de entrada disponibles.")
        return

    print("Microfonos disponibles:")
    for microphone in microphones:
        suffix = " (predeterminado)" if microphone.is_default else ""
        print(f"{microphone.index}. {microphone.name}{suffix}")


def _format_scores(scores: dict[str, float]) -> str:
    if not scores:
        return "sin scores"

    return ", ".join(
        f"{name}={score:.3f}"
        for name, score in sorted(scores.items())
    )


def main() -> int:
    _load_dotenv()
    args = _parse_args()

    capture = SoundDeviceAudioCapture()
    speech_engine = SpeechEngineUseCase(capture, _UnusedSpeechToTextProvider())

    if args.list_microphones:
        _print_microphones(speech_engine)
        return 0

    if args.microphone is not None:
        microphone = speech_engine.select_microphone(args.microphone)
    else:
        microphone = speech_engine.active_microphone()

    provider = OpenWakeWordProvider.from_environment()
    model_path = os.environ.get("ATLAS_WAKE_WORD_MODEL_PATH", "").strip()

    if not model_path:
        print(
            "Error: define ATLAS_WAKE_WORD_MODEL_PATH con la ruta local de Atlas.onnx.",
            file=sys.stderr,
        )
        return 2

    if Path(model_path).suffix.lower() != ".onnx":
        print("Error: ATLAS_WAKE_WORD_MODEL_PATH debe terminar en .onnx.", file=sys.stderr)
        return 2

    print(f"Microfono activo: {microphone.index} - {microphone.name}")
    print(f"Modelo: {Path(model_path).expanduser().resolve()}")
    print("Escuchando wake word Atlas. Pulsa Ctrl+C para salir.")

    started = time.monotonic()
    frames_seen = 0

    try:
        provider.initialize()
        frames = speech_engine.iter_pcm_frames(
            sample_rate=provider.sample_rate,
            frame_length=provider.frame_length,
        )

        try:
            for frame in frames:
                frames_seen += 1
                detected = provider.process_frame(frame)
                score = max(provider.last_scores.values(), default=0.0)

                if frames_seen == 1 or frames_seen % max(1, args.print_every) == 0:
                    print(f"frame={frames_seen} score={score:.3f} {_format_scores(provider.last_scores)}")

                if detected:
                    print(f"Detectado Atlas. score={score:.3f}")

                if args.max_seconds > 0 and time.monotonic() - started >= args.max_seconds:
                    print("Tiempo maximo alcanzado.")
                    return 0
        finally:
            close = getattr(frames, "close", None)

            if callable(close):
                close()
    except KeyboardInterrupt:
        print("\nPrueba cancelada por el usuario.")
        return 130
    except Exception as error:
        print(f"Error accionable: {error}", file=sys.stderr)
        return 1
    finally:
        provider.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

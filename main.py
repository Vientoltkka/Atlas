"""Atlas entry point."""

from __future__ import annotations

import argparse

from core.atlas import Atlas
from use_cases.speech_engine import SoundDeviceAudioCapture


def main() -> None:
    """Start Atlas."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Inicia Atlas en modo conversacion por voz sin wake word.",
    )
    parser.add_argument(
        "--assistant",
        action="store_true",
        help="Inicia Atlas en modo asistente permanente con wake word.",
    )
    parser.add_argument(
        "--list-microphones",
        action="store_true",
        help="Lista los dispositivos de entrada de audio disponibles.",
    )
    parser.add_argument(
        "--test-microphone",
        type=int,
        help="Prueba un microfono por indice sin cargar Whisper, Atlas ni TTS.",
    )
    args = parser.parse_args()

    if args.list_microphones:
        print(_list_microphones())
        return

    if args.test_microphone is not None:
        print(_test_microphone(args.test_microphone))
        return

    atlas = Atlas()

    if args.voice:
        atlas.start_voice()
        return

    if args.assistant:
        atlas.start_assistant()
        return

    atlas.start()


def _list_microphones() -> str:
    capture = SoundDeviceAudioCapture()
    microphones = capture.list_microphones(include_open_status=True)

    if not microphones:
        return "No hay microfonos de entrada disponibles."

    lines = ["Microfonos disponibles:"]

    for microphone in microphones:
        suffix = " (predeterminado)" if microphone.is_default else ""
        samplerate = (
            f"{microphone.default_samplerate:.0f} Hz"
            if microphone.default_samplerate is not None
            else "frecuencia desconocida"
        )
        open_status = ""

        if microphone.can_open is True:
            open_status = " | abre: si"
        elif microphone.can_open is False:
            open_status = f" | abre: no ({microphone.open_error})"

        lines.append(
            f"{microphone.index}. {microphone.name}{suffix} | "
            f"host API: {microphone.host_api or 'desconocida'} | "
            f"canales: {microphone.channels} | {samplerate}{open_status}"
        )

    return "\n".join(lines)


def _test_microphone(index: int) -> str:
    capture = SoundDeviceAudioCapture()
    result = capture.test_microphone(index, duration_seconds=3.0)
    microphone = result.microphone
    lines = [
        f"Microfono probado: {microphone.index}. {microphone.name} | "
        f"host API: {microphone.host_api or 'desconocida'} | "
        f"canales: {microphone.channels}"
    ]

    if result.error:
        lines.append(f"Error: {result.error}")
        return "\n".join(lines)

    lines.append(f"Duracion: {result.duration_seconds:.1f} s")
    lines.append(f"RMS: {result.rms:.4f}")
    lines.append("Voz detectada: si" if result.voice_detected else "Voz detectada: no")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

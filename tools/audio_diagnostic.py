"""Standalone microphone diagnostic script for Atlas development.

This script intentionally avoids Atlas bootstrap, orchestrator, assistant mode,
STT, TTS, and wake-word code paths. It only uses Python audio dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import wave

import numpy as np


DEFAULT_SAMPLE_RATE = 16_000
PROBE_SECONDS = 3.0
RECORD_SECONDS = 5.0
VOICE_RMS_THRESHOLD = 0.004
OUTPUT_PATH = Path("audio_test.wav")
BLOCKING_UNSUPPORTED_HOST_APIS = {"wdm-ks"}


@dataclass(frozen=True)
class InputDevice:
    """Input device visible to sounddevice."""

    index: int
    name: str
    host_api: str
    channels: int
    default_samplerate: float | None
    is_default: bool


@dataclass(frozen=True)
class ProbeResult:
    """Result from a fixed-duration device probe."""

    device: InputDevice
    rms: float
    voice_detected: bool
    compatible: bool
    error: str = ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostico independiente de dispositivos de entrada."
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Indice del dispositivo que se grabara al final.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Frecuencia de captura para pruebas y WAV final. Default: {DEFAULT_SAMPLE_RATE}.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=VOICE_RMS_THRESHOLD,
        help=f"RMS minimo para marcar voz real. Default: {VOICE_RMS_THRESHOLD}.",
    )
    args = parser.parse_args()

    sd = _sounddevice()
    devices = list_input_devices(sd)

    print("Dispositivos de entrada detectados por Python:")
    if not devices:
        print("No se detectaron dispositivos de entrada.")
        return 1

    for device in devices:
        print(format_device(device))

    print()
    print(
        f"Prueba de dispositivos compatibles: {PROBE_SECONDS:.0f} s por dispositivo"
    )
    print(f"Umbral de voz real: RMS >= {args.threshold:.4f}")

    results = [
        probe_device(
            sd=sd,
            device=device,
            duration_seconds=PROBE_SECONDS,
            sample_rate=args.sample_rate,
            threshold=args.threshold,
        )
        for device in devices
    ]

    print()
    print("Resultados:")
    for result in results:
        print(format_probe_result(result))

    compatible_indices = {result.device.index for result in results if result.compatible}
    if not compatible_indices:
        print()
        print("No hay dispositivos compatibles para grabar.")
        return 1

    selected_index = args.index
    if selected_index is None:
        selected_index = ask_for_index(compatible_indices)

    if selected_index not in compatible_indices:
        print(
            f"Indice invalido o incompatible: {selected_index}. "
            f"Indices compatibles: {', '.join(str(item) for item in sorted(compatible_indices))}"
        )
        return 1

    selected = next(device for device in devices if device.index == selected_index)
    print()
    print(
        f"Grabando {RECORD_SECONDS:.0f} s desde {selected.index} - {selected.name}..."
    )
    samples = record_device(
        sd=sd,
        device=selected,
        duration_seconds=RECORD_SECONDS,
        sample_rate=args.sample_rate,
    )
    write_wav(OUTPUT_PATH, samples, args.sample_rate)
    print(f"Archivo guardado: {OUTPUT_PATH.resolve()}")
    print(f"RMS de la grabacion: {rms(samples):.4f}")
    return 0


def list_input_devices(sd) -> list[InputDevice]:
    devices = sd.query_devices()
    host_apis = host_api_names(sd)
    default_input = default_input_index(sd)
    input_devices: list[InputDevice] = []

    for index, device in enumerate(devices):
        channels = int(device.get("max_input_channels", 0))
        if channels <= 0:
            continue

        input_devices.append(
            InputDevice(
                index=index,
                name=" ".join(str(device.get("name", f"Microfono {index}")).split()),
                host_api=host_apis.get(int(device.get("hostapi", -1)), "desconocida"),
                channels=channels,
                default_samplerate=device_samplerate(device),
                is_default=index == default_input,
            )
        )

    return input_devices


def probe_device(
    sd,
    device: InputDevice,
    duration_seconds: float,
    sample_rate: int,
    threshold: float,
) -> ProbeResult:
    if is_blocking_unsupported(device):
        return ProbeResult(
            device=device,
            rms=0.0,
            voice_detected=False,
            compatible=False,
            error="Host API no compatible con captura bloqueante",
        )

    try:
        samples = record_device(
            sd=sd,
            device=device,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
        )
    except Exception as error:
        return ProbeResult(
            device=device,
            rms=0.0,
            voice_detected=False,
            compatible=False,
            error=str(error),
        )

    value = rms(samples)
    return ProbeResult(
        device=device,
        rms=value,
        voice_detected=value >= threshold,
        compatible=True,
    )


def record_device(
    sd,
    device: InputDevice,
    duration_seconds: float,
    sample_rate: int,
) -> np.ndarray:
    frames = int(sample_rate * duration_seconds)
    recording = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device.index,
        blocking=True,
    )
    return mono_float32(recording)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def format_device(device: InputDevice) -> str:
    suffix = " (predeterminado)" if device.is_default else ""
    samplerate = (
        f"{device.default_samplerate:.0f} Hz"
        if device.default_samplerate is not None
        else "frecuencia desconocida"
    )
    return (
        f"{device.index}. {device.name}{suffix} | "
        f"host API: {device.host_api} | "
        f"canales: {device.channels} | "
        f"frecuencia: {samplerate}"
    )


def format_probe_result(result: ProbeResult) -> str:
    if not result.compatible:
        return (
            f"{result.device.index}. {result.device.name} | "
            f"incompatible | RMS: {result.rms:.4f} | Error: {result.error}"
        )

    voice = "si" if result.voice_detected else "no"
    return (
        f"{result.device.index}. {result.device.name} | "
        f"compatible | RMS: {result.rms:.4f} | Voz real detectada: {voice}"
    )


def ask_for_index(compatible_indices: set[int]) -> int:
    prompt = (
        "Elige el indice que quieres grabar "
        f"({', '.join(str(item) for item in sorted(compatible_indices))}): "
    )
    while True:
        raw_value = input(prompt).strip()
        try:
            return int(raw_value)
        except ValueError:
            print("Introduce un indice numerico.")


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0

    return float(np.sqrt(np.mean(np.square(samples))))


def mono_float32(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, 0]
    return array.reshape(-1)


def host_api_names(sd) -> dict[int, str]:
    try:
        host_apis = sd.query_hostapis()
    except Exception:
        return {}

    return {
        index: str(host_api.get("name", ""))
        for index, host_api in enumerate(host_apis)
    }


def default_input_index(sd) -> int | None:
    default = getattr(sd, "default", None)
    device = getattr(default, "device", None)

    if isinstance(device, (list, tuple)) and len(device) >= 1:
        value = device[0]
        return int(value) if value is not None and int(value) >= 0 else None

    return None


def device_samplerate(device) -> float | None:
    value = device.get("default_samplerate")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_blocking_unsupported(device: InputDevice) -> bool:
    normalized = device.host_api.casefold()
    return any(host_api in normalized for host_api in BLOCKING_UNSUPPORTED_HOST_APIS)


def _sounddevice():
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError("Dependencia no disponible: sounddevice.") from error

    return sd


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelado por el usuario.")
        raise SystemExit(130)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

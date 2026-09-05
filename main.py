"""Atlas entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from core.startup import (
    WindowsStartupPreflight,
    close_operational_logging,
    configure_degraded_logging,
    configure_operational_logging,
    install_crash_logging,
    render_startup_banner,
    render_startup_failure,
    render_startup_warnings,
    sanitize_log_message,
)

# Kept as injectable seams for the existing deterministic CLI tests.
Atlas: Any = None
SoundDeviceAudioCapture: Any = None


def main() -> int:
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
        "--whatsapp-webhook",
        action="store_true",
        help="Inicia el webhook de WhatsApp (FastAPI + uvicorn, workers=1).",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Inicia el Orbe de Atlas en modo chat textual sin voz.",
    )
    parser.add_argument(
        "--start-hidden",
        action="store_true",
        help="Inicia el chat oculto; reservado para el autoarranque de Windows.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Inicia el Orbe de Atlas con una sesion de voz real.",
    )
    parser.add_argument(
        "--test-microphone",
        type=int,
        help="Prueba un microfono por indice sin cargar Whisper, Atlas ni TTS.",
    )
    args = parser.parse_args()
    mode = _requested_mode(args)
    if args.start_hidden and not args.chat:
        parser.error("--start-hidden solo se puede usar junto a --chat.")
    project_root = Path(__file__).resolve().parent
    _load_environment_file(project_root)
    preflight_mode = "voice" if mode == "ui" else mode
    report = WindowsStartupPreflight(project_root).run(preflight_mode)

    if not report.ready:
        print(render_startup_failure(report))
        return 1

    warnings = render_startup_warnings(report)
    if warnings:
        print(warnings)
        print()

    try:
        logger = configure_operational_logging(project_root)
    except Exception:
        logger = configure_degraded_logging()
        print(
            "Aviso: no se pudo abrir logs\\atlas.log. "
            "Atlas continuara en modo degradado sin log de archivo."
        )
        print()
    logger.info("Inicio solicitado | modo=%s | start_hidden=%s", mode, args.start_hidden)
    install_crash_logging(logger, project_root)
    atlas = None

    try:
        if args.list_microphones:
            print(_list_microphones())
            return 0

        if args.test_microphone is not None:
            print(_test_microphone(args.test_microphone))
            return 0

        if args.whatsapp_webhook:
            return _run_whatsapp_webhook(logger)

        if getattr(args, "chat", False):
            return _run_desktop_ui(
                logger,
                start_voice=False,
                show_on_start=False,
                start_hidden=args.start_hidden,
            )

        if getattr(args, "ui", False):
            return _run_desktop_ui(logger)

        print(render_startup_banner(report))
        print()
        atlas = _atlas_class()()

        if args.voice:
            atlas.start_voice()
            return 0

        if args.assistant:
            atlas.start_assistant()
            return 0

        atlas.start()
        return 0
    except KeyboardInterrupt:
        logger.info("Cierre voluntario por KeyboardInterrupt")
        print("\nInterrupcion recibida. Atlas se ha cerrado correctamente.")
        return 0
    except Exception as exc:
        logger.error(
            "Fallo interno | tipo=%s | detalle=%s",
            type(exc).__name__,
            sanitize_log_message(exc),
        )
        print(
            "Atlas no pudo continuar por un error interno. "
            "Revisa logs\\atlas.log."
        )
        return 1
    finally:
        _close_atlas(atlas, logger)
        logger.info("Proceso Atlas finalizado")
        close_operational_logging(logger)


def _list_microphones() -> str:
    capture = _audio_capture_class()()
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
    capture = _audio_capture_class()()
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


def _load_environment_file(project_root: Path) -> None:
    """Load .env without overriding variables already present in the
    environment. Values are never logged."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(project_root / ".env")


def _requested_mode(args: argparse.Namespace) -> str:
    if args.list_microphones or args.test_microphone is not None:
        return "microphone"
    if getattr(args, "whatsapp_webhook", False):
        return "whatsapp"
    if getattr(args, "chat", False):
        return "chat"
    if getattr(args, "ui", False):
        return "ui"
    if args.voice:
        return "voice"
    if args.assistant:
        return "assistant"
    return "text"


def _run_desktop_ui(
    logger: logging.Logger,
    *,
    start_voice: bool = True,
    show_on_start: bool = True,
    start_hidden: bool = False,
) -> int:
    """Run at most one desktop chat/UI process on Windows."""
    from core.windows_ui_instance import WindowsUiInstance

    instance = WindowsUiInstance()
    if not instance.acquire():
        logger.info("Se omitio una segunda instancia de la interfaz de Atlas")
        return 0
    logger.info("Instancia unica de interfaz adquirida")
    try:
        return _run_ui_orb(
            logger,
            start_voice=start_voice,
            show_on_start=show_on_start,
            start_hidden=start_hidden,
        )
    finally:
        instance.release()


def _run_ui_orb(
    logger: logging.Logger,
    *,
    start_voice: bool = True,
    show_on_start: bool = True,
    start_hidden: bool = False,
) -> int:
    """Run Orbe through its explicit Qt-safe controller."""
    from ui.orbe_controller import OrbeController
    from ui.orbe_app import create_application, create_orb_window, create_transcript_panel

    controller = OrbeController(
        atlas=_atlas_class()(),
        application=create_application([]),
        orb=create_orb_window(),
        transcript_panel=create_transcript_panel(),
        logger=logger,
    )
    exit_code = controller.run(
        start_voice=start_voice,
        show_on_start=show_on_start,
        start_hidden=start_hidden,
    )
    logger.info("Sesion del Orbe finalizada")
    return exit_code

def _run_whatsapp_webhook(logger: logging.Logger) -> int:
    """Start the WhatsApp webhook server (uvicorn, workers=1 for Phase 2)."""
    import os

    from channels.app import create_webhook_app
    from bootstrap.agent_system import build_core_agent_system

    if not os.environ.get("ATLAS_WHATSAPP_VERIFY_TOKEN"):
        logger.error("WhatsApp webhook requiere ATLAS_WHATSAPP_VERIFY_TOKEN")
        print(
            "Falta ATLAS_WHATSAPP_VERIFY_TOKEN. "
            "Configura las variables en .env (ver .env.example)."
        )
        return 1
    if not os.environ.get("ATLAS_WHATSAPP_ACCESS_TOKEN") or not os.environ.get(
        "ATLAS_WHATSAPP_PHONE_NUMBER_ID"
    ):
        logger.error("WhatsApp webhook requiere access token y phone number id")
        print(
            "Faltan ATLAS_WHATSAPP_ACCESS_TOKEN o ATLAS_WHATSAPP_PHONE_NUMBER_ID."
        )
        return 1

    result = build_core_agent_system()
    if result.system is None:
        logger.error("No se pudo construir el sistema de agentes de Atlas")
        print("Atlas no pudo inicializar el sistema de agentes. Revisa logs\\atlas.log.")
        return 1
    executor = result.system.agent_executor

    app = create_webhook_app(executor_fn=executor.execute)

    import uvicorn

    port = int(os.environ.get("ATLAS_WHATSAPP_WEBHOOK_PORT", "8000"))
    print(f"Atlas WhatsApp webhook escuchando en puerto {port} (workers=1)")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
    return 0


def _atlas_class() -> Any:
    if Atlas is not None:
        return Atlas
    from core.atlas import Atlas as AtlasApplication

    return AtlasApplication


def _audio_capture_class() -> Any:
    if SoundDeviceAudioCapture is not None:
        return SoundDeviceAudioCapture
    from use_cases.speech_engine import SoundDeviceAudioCapture as AudioCapture

    return AudioCapture


def _close_atlas(atlas: Any, logger: logging.Logger) -> None:
    if atlas is None:
        return
    close = getattr(atlas, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        logger.error(
            "Fallo al liberar recursos | tipo=%s | detalle=%s",
            type(exc).__name__,
            sanitize_log_message(exc),
        )


if __name__ == "__main__":
    raise SystemExit(main())

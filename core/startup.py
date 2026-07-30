"""Windows-oriented startup checks and operational logging for Atlas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.util
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
import socket
import sys
import tempfile
from typing import Callable, Iterable


ATLAS_VERSION = "1.0"
SUPPORTED_PYTHON_MIN = (3, 11)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 15)


class CheckStatus(str, Enum):
    """Possible outcomes for one startup check."""

    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class StartupCheck:
    """One deterministic, user-actionable startup result."""

    name: str
    status: CheckStatus
    reason: str
    action: str = ""


@dataclass(frozen=True)
class StartupReport:
    """Complete preflight result for one requested operating mode."""

    project_root: Path
    mode: str
    checks: tuple[StartupCheck, ...]
    capabilities: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not any(check.status is CheckStatus.ERROR for check in self.checks)

    @property
    def warnings(self) -> tuple[StartupCheck, ...]:
        return tuple(
            check for check in self.checks if check.status is CheckStatus.WARNING
        )

    @property
    def errors(self) -> tuple[StartupCheck, ...]:
        return tuple(
            check for check in self.checks if check.status is CheckStatus.ERROR
        )


ModuleFinder = Callable[[str], object | None]
SocketProbe = Callable[[str, int, float], bool]
WriteProbe = Callable[[Path], None]


class WindowsStartupPreflight:
    """Validate only the prerequisites needed by the requested Atlas mode."""

    _ESSENTIAL_FILES = (
        "main.py",
        "core/atlas.py",
        "bootstrap/bootstrap.py",
        "requirements.txt",
    )
    _TEXT_MODULES = ("ollama", "numpy")
    _VOICE_MODULES = ("sounddevice", "faster_whisper", "pyttsx3")
    _ASSISTANT_MODULES = ("openwakeword",)
    _OPERATIONAL_DIRECTORIES = (
        "logs",
        ".atlas",
        ".atlas/execution_sessions",
    )

    def __init__(
        self,
        project_root: Path,
        *,
        python_version: tuple[int, int] | None = None,
        module_finder: ModuleFinder | None = None,
        socket_probe: SocketProbe | None = None,
        write_probe: WriteProbe | None = None,
    ) -> None:
        self._project_root = project_root.resolve()
        self._python_version = python_version or sys.version_info[:2]
        self._module_finder = module_finder or importlib.util.find_spec
        self._socket_probe = socket_probe or _probe_tcp_socket
        self._write_probe = write_probe or _probe_directory_write

    def run(self, mode: str = "text") -> StartupReport:
        checks: list[StartupCheck] = []
        checks.append(self._check_python())
        checks.extend(self._check_project_files())
        checks.extend(self._check_operational_directories())
        checks.append(self._check_environment_file())
        checks.extend(self._check_modules(mode))
        checks.append(self._check_local_model_service())

        capabilities = ["texto"]
        if self._modules_available(self._VOICE_MODULES):
            capabilities.append("voz manual")
        if (
            self._modules_available(self._VOICE_MODULES)
            and self._modules_available(self._ASSISTANT_MODULES)
        ):
            capabilities.append("asistente permanente")
        if self._socket_probe("127.0.0.1", 11434, 0.15):
            capabilities.append("modelo local Ollama")

        return StartupReport(
            project_root=self._project_root,
            mode=mode,
            checks=tuple(checks),
            capabilities=tuple(capabilities),
        )

    def _check_python(self) -> StartupCheck:
        version = self._python_version
        if SUPPORTED_PYTHON_MIN <= version < SUPPORTED_PYTHON_MAX_EXCLUSIVE:
            return StartupCheck(
                "Python",
                CheckStatus.OK,
                f"Python {version[0]}.{version[1]} compatible.",
            )
        return StartupCheck(
            "Python",
            CheckStatus.ERROR,
            f"Python {version[0]}.{version[1]} no esta validado para Atlas.",
            "Instala Python 3.11, 3.12, 3.13 o 3.14 y recrea el entorno.",
        )

    def _check_project_files(self) -> Iterable[StartupCheck]:
        for relative_path in self._ESSENTIAL_FILES:
            path = self._project_root / relative_path
            if path.is_file():
                yield StartupCheck(
                    f"Archivo {relative_path}",
                    CheckStatus.OK,
                    "Disponible.",
                )
            else:
                yield StartupCheck(
                    f"Archivo {relative_path}",
                    CheckStatus.ERROR,
                    "No se encontro un archivo obligatorio.",
                    "Restaura el archivo desde una copia valida del proyecto.",
                )

    def _check_operational_directories(self) -> Iterable[StartupCheck]:
        for relative_path in self._OPERATIONAL_DIRECTORIES:
            path = (self._project_root / relative_path).resolve()
            try:
                path.relative_to(self._project_root)
                path.mkdir(parents=True, exist_ok=True)
                self._write_probe(path)
            except (OSError, ValueError) as exc:
                yield StartupCheck(
                    f"Directorio {relative_path}",
                    CheckStatus.ERROR,
                    f"No esta disponible para escritura ({type(exc).__name__}).",
                    "Comprueba permisos y espacio libre en la carpeta del proyecto.",
                )
            else:
                yield StartupCheck(
                    f"Directorio {relative_path}",
                    CheckStatus.OK,
                    "Disponible y escribible.",
                )

    def _check_environment_file(self) -> StartupCheck:
        env_path = self._project_root / ".env"
        if not env_path.exists():
            return StartupCheck(
                "Configuracion .env",
                CheckStatus.WARNING,
                "No existe; las integraciones que requieren credenciales no estaran disponibles.",
                "Crea .env solo si vas a usar integraciones externas.",
            )
        try:
            lines = env_path.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            return StartupCheck(
                "Configuracion .env",
                CheckStatus.ERROR,
                f"No se pudo leer ({type(exc).__name__}).",
                "Comprueba los permisos del archivo .env.",
            )
        malformed = [
            index
            for index, line in enumerate(lines, start=1)
            if line.strip()
            and not line.lstrip().startswith("#")
            and "=" not in line
        ]
        if malformed:
            numbers = ", ".join(str(number) for number in malformed)
            return StartupCheck(
                "Configuracion .env",
                CheckStatus.ERROR,
                f"Formato invalido en las lineas {numbers}.",
                "Usa una asignacion NOMBRE=valor por linea.",
            )
        return StartupCheck(
            "Configuracion .env",
            CheckStatus.OK,
            "Formato valido.",
        )

    def _check_modules(self, mode: str) -> Iterable[StartupCheck]:
        required = list(self._TEXT_MODULES)
        optional = list(self._VOICE_MODULES + self._ASSISTANT_MODULES)
        if mode == "microphone":
            required = ["numpy", "sounddevice"]
            optional = []
        elif mode == "voice":
            required.extend(self._VOICE_MODULES)
            optional = list(self._ASSISTANT_MODULES)
        elif mode == "assistant":
            required.extend(self._VOICE_MODULES + self._ASSISTANT_MODULES)
            optional = []

        for module_name in dict.fromkeys(required):
            if self._module_available(module_name):
                yield StartupCheck(
                    f"Dependencia {module_name}",
                    CheckStatus.OK,
                    "Disponible.",
                )
            else:
                yield StartupCheck(
                    f"Dependencia {module_name}",
                    CheckStatus.ERROR,
                    "No esta instalada y es obligatoria para el modo solicitado.",
                    "Ejecuta: python -m pip install -r requirements.txt",
                )

        for module_name in dict.fromkeys(optional):
            if not self._module_available(module_name):
                yield StartupCheck(
                    f"Dependencia opcional {module_name}",
                    CheckStatus.WARNING,
                    "No disponible; se desactiva la capacidad asociada.",
                    "Instala requirements.txt si necesitas voz o wake word.",
                )

    def _check_local_model_service(self) -> StartupCheck:
        if self._socket_probe("127.0.0.1", 11434, 0.15):
            return StartupCheck(
                "Servicio Ollama",
                CheckStatus.OK,
                "Disponible en 127.0.0.1:11434.",
            )
        return StartupCheck(
            "Servicio Ollama",
            CheckStatus.WARNING,
            "No responde; las funciones que usan el modelo local no estaran disponibles.",
            "Inicia Ollama antes de usar funciones dependientes del modelo.",
        )

    def _modules_available(self, module_names: Iterable[str]) -> bool:
        return all(self._module_available(name) for name in module_names)

    def _module_available(self, module_name: str) -> bool:
        try:
            return self._module_finder(module_name) is not None
        except (ImportError, ValueError):
            return False


_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret)\b(\s*[:=]\s*)([^\s,;]+)"
)


def sanitize_log_message(message: object) -> str:
    """Redact common credential assignments before writing operational logs."""

    return _SECRET_PATTERN.sub(r"\1\2[REDACTED]", str(message))


def configure_operational_logging(project_root: Path) -> logging.Logger:
    """Create one bounded UTF-8 file logger without duplicate handlers."""

    logger = logging.getLogger("atlas.operational")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_path = (project_root / "logs" / "atlas.log").resolve()

    for handler in logger.handlers:
        if getattr(handler, "baseFilename", None) == str(log_path):
            return logger

    handler = RotatingFileHandler(
        log_path,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
    return logger


def configure_degraded_logging() -> logging.Logger:
    """Return a silent fallback logger when the operational file is unavailable."""

    logger = logging.getLogger("atlas.operational.degraded")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


def close_operational_logging(logger: logging.Logger) -> None:
    """Flush, close, and detach handlers owned by one startup logger."""

    for handler in list(logger.handlers):
        try:
            handler.flush()
        finally:
            handler.close()
            logger.removeHandler(handler)


def render_startup_failure(report: StartupReport) -> str:
    lines = ["Atlas no puede arrancar:", ""]
    for check in report.errors:
        lines.append(f"- {check.name}: {check.reason}")
        if check.action:
            lines.append(f"  Accion recomendada: {check.action}")
    return "\n".join(lines)


def render_startup_warnings(report: StartupReport) -> str:
    lines: list[str] = []
    for check in report.warnings:
        lines.append(f"Aviso: {check.name}: {check.reason}")
        if check.action:
            lines.append(f"       {check.action}")
    return "\n".join(lines)


def render_startup_banner(report: StartupReport) -> str:
    text_status = "disponible" if "texto" in report.capabilities else "no disponible"
    voice_status = (
        "disponible"
        if "voz manual" in report.capabilities
        else "no configurada"
    )
    local_model_status = (
        "disponible"
        if "modelo local Ollama" in report.capabilities
        else "no disponible"
    )
    basic_status = (
        "disponible"
        if report.ready and text_status == "disponible"
        else "bloqueado"
    )
    mode_names = {
        "text": "texto",
        "voice": "voz",
        "assistant": "asistente",
        "microphone": "diagnostico de microfono",
    }
    return "\n".join(
        (
            f"Atlas - Base operativa v{ATLAS_VERSION}",
            "Estado: preparado",
            f"Modo: {mode_names.get(report.mode, report.mode)}",
            f"Texto: {text_status}",
            f"Voz: {voice_status}",
            f"Modelo local: {local_model_status}",
            f"Modo operativo basico: {basic_status}",
            f"Directorio de trabajo: {report.project_root}",
            'Escribe una peticion o usa "salir" para cerrar.',
        )
    )


def _probe_tcp_socket(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_directory_write(directory: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".atlas-check-", delete=True):
        pass

"""Desktop interaction use case."""

from __future__ import annotations

from pathlib import Path
import re

from tools.executor import ToolExecutor
from tools.tool_context import ToolContext


class DesktopInteractionUseCase:
    """Execute simple desktop commands through Atlas tools."""

    _DEFAULT_EDITOR = "Visual Studio Code"
    _DEFAULT_WINDOW_TITLE = "Visual Studio Code"

    def __init__(
        self,
        executor: ToolExecutor,
        project_root: Path | None = None,
    ) -> None:
        self._executor = executor
        self._project_root = project_root or Path(".")

    def execute(
        self,
        prompt: str,
    ) -> str | None:
        """Execute a supported desktop command."""
        text = prompt.strip()
        normalized = self._normalize(text)

        try:
            if normalized.startswith("abre "):
                return self._open(text[5:].strip())

            if normalized.startswith("abrir "):
                return self._open(text[6:].strip())

            if normalized.startswith("activa "):
                title = text[7:].strip()
                return self._run(
                    "desktop.activate_window",
                    {"window_title": title},
                    f"Ventana activada: {title}",
                )

            if normalized.startswith("escribe"):
                content = self._extract_text_to_type(text)
                return self._run(
                    "desktop.type_text",
                    {
                        "window_title": self._DEFAULT_WINDOW_TITLE,
                        "text": content,
                    },
                    "Texto escrito.",
                )

            if normalized.startswith("guarda"):
                return self._run(
                    "desktop.save_file",
                    {"window_title": self._DEFAULT_WINDOW_TITLE},
                    "Archivo guardado.",
                )

            if normalized.startswith("pulsa ") or normalized.startswith(
                "presiona "
            ):
                keys = self._extract_keys(text)
                return self._run(
                    "desktop.press_hotkey",
                    {
                        "window_title": self._DEFAULT_WINDOW_TITLE,
                        "keys": keys,
                    },
                    "Atajo enviado.",
                )
        except Exception as exc:
            return f"Error: {exc}"

        return None

    def _open(
        self,
        target: str,
    ) -> str:
        """Open an application, folder, or file."""
        target = self._clean_open_target(target)

        if not target:
            raise ValueError("Falta el objetivo a abrir.")

        path = self._resolve_path(target)

        if path.exists() and path.is_dir():
            return self._run(
                "desktop.open_folder",
                {"path": str(path)},
                f"Carpeta abierta: {path}",
            )

        if path.exists() and path.is_file():
            return self._run(
                "desktop.open_file",
                {
                    "path": str(path),
                    "application": self._DEFAULT_EDITOR,
                },
                f"Archivo abierto en {self._DEFAULT_EDITOR}.",
            )

        if self._looks_like_path(target):
            raise FileNotFoundError(str(path))

        return self._run(
            "desktop.open_application",
            {"application": target},
            f"{target} abierto.",
        )

    def _clean_open_target(
        self,
        target: str,
    ) -> str:
        """Remove natural-language prefixes from an open command."""
        cleaned = target.strip()
        normalized = self._normalize(cleaned)
        prefixes = (
            "la carpeta ",
            "carpeta ",
            "el archivo ",
            "archivo ",
            "la aplicacion ",
            "la aplicación ",
            "aplicacion ",
            "aplicación ",
        )

        for prefix in prefixes:
            if normalized.startswith(prefix):
                return cleaned[len(prefix) :].strip()

        return cleaned

    def _run(
        self,
        tool_name: str,
        parameters: dict[str, object],
        confirmation: str,
    ) -> str:
        """Run a tool and return a user-facing confirmation."""
        self._executor.execute(
            tool_name,
            ToolContext(parameters=parameters),
        )

        return f"✓ {confirmation}"

    def _resolve_path(
        self,
        value: str,
    ) -> Path:
        """Resolve a project-relative or absolute path."""
        expanded = Path(value.strip().strip('"'))

        if expanded.is_absolute():
            return expanded

        return self._project_root / expanded

    def _looks_like_path(
        self,
        value: str,
    ) -> bool:
        """Return whether a value looks like a filesystem path."""
        return (
            "\\" in value
            or "/" in value
            or bool(re.search(r"\.[A-Za-z0-9]{1,8}$", value.strip()))
        )

    def _extract_text_to_type(
        self,
        text: str,
    ) -> str:
        """Extract text after an Escribe command."""
        _, separator, content = text.partition(":")

        if separator:
            return content.lstrip("\r\n ")

        content = text[len("Escribe") :].strip()

        if not content:
            raise ValueError("Falta el texto a escribir.")

        return content

    def _extract_keys(
        self,
        text: str,
    ) -> list[str]:
        """Extract shortcut keys from a command."""
        lowered = self._normalize(text)
        raw = (
            text[6:]
            if lowered.startswith("pulsa ")
            else text[len("presiona ") :]
        )
        keys = [
            key.strip().lower()
            for key in re.split(r"\+|,", raw)
            if key.strip()
        ]

        if not keys:
            raise ValueError("Faltan teclas para el atajo.")

        return keys

    def _normalize(
        self,
        text: str,
    ) -> str:
        """Normalize command text."""
        return text.strip().lower()

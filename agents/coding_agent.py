"""Coding Agent."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
from pathlib import Path
import secrets
import subprocess
import sys

from agents.base_agent import BaseAgent
from models.prompt_client import PromptClient
from use_cases.read_file import ReadFileUseCase
from use_cases.write_file import WriteFileUseCase


class PendingCodingChangeError(ValueError):
    """Raised when a generated change cannot be safely authorized."""


@dataclass(frozen=True)
class PendingCodingChange:
    token: str
    path: Path
    relative_path: str
    original_content: str
    proposed_content: str
    diff: str


class CodingAgent(BaseAgent):
    """Agent specialized in programming tasks."""

    SYSTEM_PROMPT = """
Eres Atlas Coding Agent.

Eres un ingeniero de software senior.

Tu trabajo es:

- Analizar código.
- Corregir errores.
- Refactorizar.
- Mejorar el código.
- Mantener Clean Architecture.
- Mantener SOLID.
- Nunca inventar APIs.
- Devuelve siempre el archivo completo cuando propongas cambios.
"""

    def __init__(
        self,
        prompt_client: PromptClient,
        read_file: ReadFileUseCase,
        write_file: WriteFileUseCase,
    ) -> None:
        self._client = prompt_client
        self._read_file = read_file
        self._write_file = write_file
        self._project_root = Path(__file__).resolve().parents[1]
        self._pending_change: PendingCodingChange | None = None
        self._pending_capability_plan: dict[str, object] | None = None

    @property
    def name(self) -> str:
        return "coding"

    @property
    def description(self) -> str:
        return "Programming assistant."

    def run(self, model: str, messages: list[dict[str, str]]) -> str:
        """Execute a coding request."""
        if not messages:
            return "No hay mensajes."
        prompt = messages[-1]["content"].strip()
        lower = prompt.lower()
        if lower.startswith("lee "):
            path = prompt[4:].strip()
            try:
                return self._read_file.execute(str(self._resolve_project_path(path)))
            except Exception as exc:
                return f"Error leyendo '{path}': {exc}"
        if lower.startswith("corrige "):
            path = prompt[8:].strip()
            try:
                target_path = self._resolve_project_path(path)
                content = self._read_file.execute(str(target_path))
            except Exception as exc:
                return f"Error leyendo '{path}': {exc}"
            conversation = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"""
Corrige el siguiente archivo Python.

Ruta:
{target_path.relative_to(self._project_root).as_posix()}

Devuelve únicamente el archivo completo corregido.

Código:

{content}
"""},
            ]
            generated = self._client.ask(model=model, messages=conversation)
            relative_path = target_path.relative_to(self._project_root).as_posix()
            pending = PendingCodingChange(
                token=secrets.token_urlsafe(24),
                path=target_path,
                relative_path=relative_path,
                original_content=content,
                proposed_content=generated,
                diff=self._render_diff(relative_path, content, generated),
            )
            self._pending_change = pending
            return self._render_pending_change(pending)
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        conversation.extend(messages)
        return self._client.ask(model=model, messages=conversation)

    @property
    def generated_path(self) -> str | None:
        return str(self._pending_change.path) if self._pending_change else None

    @property
    def generated_content(self) -> str | None:
        return self._pending_change.proposed_content if self._pending_change else None

    def clear_generated(self) -> None:
        self._pending_change = None

    def prepare_capability_plan(self, **details) -> str:
        """Prepare one bounded capability plan without applying it."""
        self._pending_capability_plan = dict(details)
        return "\n".join(("Preparación supervisada de mejora:", f"- Capacidad ausente: {details['capability_id']}.", f"- Implementación mínima propuesta: {details['implementation']}.", "- Archivos previsiblemente afectados:", *(f"  - {path}" for path in details['planned_files']), "- Tests focalizados necesarios:", *(f"  - {test}" for test in details['focused_tests']), f"- Riesgo/impacto: {details['risk']}", "- Estado: todavía NO se han realizado cambios."))

    def apply_prepared_capability_plan(self, capability_id: str) -> str:
        """Apply only the previously prepared, fixed capability scope."""
        plan = self._pending_capability_plan
        if plan is None or plan.get("capability_id") != capability_id:
            return "No hay un plan preparado para aplicar. No se han realizado cambios."
        allowed = ("tools/temperature_conversion.py", "bootstrap/bootstrap.py", "tests/test_temperature_conversion.py")
        if tuple(plan.get("planned_files", ())) != allowed:
            return "El alcance del plan no es válido. No se han realizado cambios."
        try:
            for relative_path in allowed:
                target = self._project_root / relative_path
                self._write_file.execute(str(target), target.read_text(encoding="utf-8"))
            result = subprocess.run((sys.executable, "-B", "-m", "pytest", "-q", "tests/test_temperature_conversion.py"), cwd=self._project_root, capture_output=True, text=True, check=False)
        except Exception as error:
            return f"La aplicación se detuvo: {error}"
        self._pending_capability_plan = None
        if result.returncode != 0:
            return "La mejora se aplicó, pero los tests focalizados fallaron. Se detuvo el flujo.\n" + result.stdout + result.stderr
        return "Mejora aplicada de forma controlada. Tests focalizados correctos."
    def authorize_pending_change(self, token: str) -> PendingCodingChange:
        """Consume a single token bound to the exact pending file change."""
        pending = self._pending_change
        if pending is None or not isinstance(token, str):
            raise PendingCodingChangeError("No hay una propuesta pendiente para aplicar.")
        if not secrets.compare_digest(token.strip(), pending.token):
            raise PendingCodingChangeError("La autorización no corresponde a la propuesta pendiente.")
        try:
            current = self._read_file.execute(str(pending.path))
        except Exception as exc:
            raise PendingCodingChangeError("No se pudo verificar el archivo antes de aplicar la propuesta.") from exc
        if current != pending.original_content:
            self.clear_generated()
            raise PendingCodingChangeError("El archivo cambió desde la propuesta; se descartó la autorización.")
        self.clear_generated()
        return pending

    def _resolve_project_path(self, raw_path: str) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("La ruta del archivo es obligatoria.")
        candidate = Path(raw_path.strip())
        resolved = candidate.resolve() if candidate.is_absolute() else (self._project_root / candidate).resolve()
        try:
            resolved.relative_to(self._project_root)
        except ValueError as exc:
            raise ValueError("La ruta debe estar dentro de C:\\AI\\Atlas.") from exc
        if not resolved.is_file():
            raise ValueError("La ruta debe identificar un archivo existente dentro del proyecto.")
        return resolved

    @staticmethod
    def _render_diff(relative_path: str, original: str, proposed: str) -> str:
        return "\n".join(difflib.unified_diff(
            original.splitlines(), proposed.splitlines(), fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}", lineterm="",
        )) or "(sin cambios de contenido)"

    @staticmethod
    def _render_pending_change(change: PendingCodingChange) -> str:
        return "\n".join([
            f"Propuesta preparada para '{change.relative_path}'.",
            "Resumen: reemplazo completo del archivo.", "Diff:", change.diff, "",
            "Revisa el diff. Para aplicar exactamente esta propuesta, escribe:",
            f"APLICAR {change.token}",
            "El token es de un solo uso y se rechazará si el archivo cambia.",
        ])
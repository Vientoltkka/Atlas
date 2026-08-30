"""Coding Agent."""

from __future__ import annotations

from collections.abc import Mapping
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
@dataclass(frozen=True)
class AppliedCapabilityChange:
    """One bounded applied change that can be validated or restored once."""

    capability_id: str
    original_contents: tuple[tuple[str, str | None], ...]
    applied_contents: tuple[tuple[str, str], ...]


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
        self._applied_capability_change: AppliedCapabilityChange | None = None
        self._validated_capability_change: AppliedCapabilityChange | None = None
        self._capability_validation_status: str | None = None

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

    @property
    def capability_validation_status(self) -> str | None:
        """Return the result of the latest supervised capability validation."""
        return self._capability_validation_status

    def prepare_capability_plan(self, **details) -> str:
        """Prepare one bounded capability plan without applying it."""
        self._pending_capability_plan = dict(details)
        return "\n".join(("Preparación supervisada de mejora:", f"- Capacidad ausente: {details['capability_id']}.", f"- Implementación mínima propuesta: {details['implementation']}.", "- Archivos previsiblemente afectados:", *(f"  - {path}" for path in details['planned_files']), "- Tests focalizados necesarios:", *(f"  - {test}" for test in details['focused_tests']), f"- Riesgo/impacto: {details['risk']}", "- Estado: todavía NO se han realizado cambios."))

    def apply_prepared_capability_plan(self, capability_id: str) -> str:
        """Apply, validate and safely restore only one prepared capability plan."""
        plan = self._pending_capability_plan
        allowed = (
            "tools/temperature_conversion.py",
            "bootstrap/bootstrap.py",
            "tests/test_temperature_conversion.py",
        )
        if plan is None or plan.get("capability_id") != capability_id:
            return "No hay un plan preparado para aplicar. No se han realizado cambios."
        if tuple(plan.get("planned_files", ())) != allowed:
            return "El alcance del plan no es válido. No se han realizado cambios."

        proposed_contents = plan.get("proposed_contents", {})
        if not isinstance(proposed_contents, Mapping):
            return "El contenido del plan no es válido. No se han realizado cambios."

        originals: list[tuple[str, str | None]] = []
        applied: list[tuple[str, str]] = []
        for relative_path in allowed:
            target = self._project_root / relative_path
            original_content = target.read_text(encoding="utf-8") if target.exists() else None
            proposed_content = proposed_contents.get(relative_path, original_content)
            if not isinstance(proposed_content, str):
                return "El contenido del plan no es válido. No se han realizado cambios."
            originals.append((relative_path, original_content))
            applied.append((relative_path, proposed_content))

        change = AppliedCapabilityChange(
            capability_id=capability_id,
            original_contents=tuple(originals),
            applied_contents=tuple(applied),
        )
        written_paths: list[str] = []
        self._capability_validation_status = None
        try:
            for relative_path, proposed_content in change.applied_contents:
                self._write_file.execute(str(self._project_root / relative_path), proposed_content)
                written_paths.append(relative_path)
        except Exception as error:
            return self._rollback_capability_change(change, tuple(written_paths), f"Error de aplicación: {error}")

        self._pending_capability_plan = None
        self._applied_capability_change = change
        return self.validate_applied_capability_plan(capability_id)

    def validate_applied_capability_plan(self, capability_id: str) -> str:
        """Validate one applied capability and rollback it when validation fails."""
        change = self._applied_capability_change
        if change is None or change.capability_id != capability_id:
            return "No hay una mejora aplicada pendiente de validación."

        result = self._run_capability_tests()
        if result.returncode != 0:
            return self._rollback_capability_change(change, tuple(path for path, _ in change.applied_contents), "Los tests focalizados fallaron.")

        self._validated_capability_change = change
        self._applied_capability_change = None
        self._capability_validation_status = "VALIDATED"
        return "\n".join((
            "Validación completada correctamente.",
            "La nueva capacidad funciona y los tests focalizados pasan.",
            "No se ha detectado regresión en el alcance validado.",
            "¿Apruebas cerrar y versionar esta mejora?",
        ))

    def close_validated_capability_plan(self, capability_id: str, *, approved: bool) -> str:
        """Commit exactly one still-intact validated capability after human approval."""
        change = self._validated_capability_change
        if (
            change is None
            or change.capability_id != capability_id
            or self._capability_validation_status != "VALIDATED"
        ):
            return "No hay una mejora VALIDATED pendiente de cierre."
        if not approved:
            self._validated_capability_change = None
            self._capability_validation_status = "CLOSURE_DECLINED"
            return "Cierre/versionado no aprobado. La mejora validada se mantiene sin commit."

        paths = tuple(relative_path for relative_path, _ in change.applied_contents)
        applied = dict(change.applied_contents)
        for relative_path in paths:
            target = self._project_root / relative_path
            current_content = target.read_text(encoding="utf-8") if target.exists() else None
            if current_content != applied[relative_path]:
                return "El cierre se detuvo: el archivo aprobado cambió desde la validación: " + relative_path

        scope_status = self._git_command("status", "--porcelain", "--", *paths)
        names = tuple(line[3:].strip() for line in scope_status.stdout.splitlines() if line.strip())
        if scope_status.returncode != 0 or set(names) != set(paths):
            return "El cierre se detuvo: el preflight no coincide exactamente con el alcance validado."

        committed = self._git_command(
            "commit",
            "--only",
            "-m",
            "feat(supervision): close validated capability " + capability_id,
            "--",
            *paths,
        )
        if committed.returncode != 0:
            return "El commit aprobado falló. La mejora permanece VALIDATED. Motivo: " + (committed.stderr or committed.stdout)
        revision = self._git_command("rev-parse", "HEAD")
        if revision.returncode != 0:
            return "El commit fue creado, pero no se pudo confirmar su hash."

        commit_hash = revision.stdout.strip()
        self._clear_capability_state()
        return "Mejora cerrada y versionada correctamente. Commit: " + commit_hash + ". Alcance limpio: " + ", ".join(paths)

    def _git_command(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run one explicit Git command in the project without push or reset."""
        return subprocess.run(
            ("git", *arguments),
            cwd=self._project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    def _run_capability_tests(self) -> subprocess.CompletedProcess[str]:
        """Run only the focused tests bound to this approved capability."""
        return subprocess.run(
            (sys.executable, "-B", "-m", "pytest", "-q", "tests/test_temperature_conversion.py"),
            cwd=self._project_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def _rollback_capability_change(self, change: AppliedCapabilityChange, paths: tuple[str, ...], reason: str) -> str:
        """Restore only unchanged approved files, never overwriting external edits."""
        originals = dict(change.original_contents)
        applied = dict(change.applied_contents)
        for relative_path in paths:
            target = self._project_root / relative_path
            current_content = target.read_text(encoding="utf-8") if target.exists() else None
            if current_content != applied[relative_path]:
                self._clear_capability_state()
                return "La validación falló y el rollback se detuvo para no sobrescribir cambios ajenos en '" + relative_path + "'. Motivo: " + reason

        try:
            for relative_path in paths:
                target = self._project_root / relative_path
                original_content = originals[relative_path]
                if original_content is None:
                    target.unlink()
                else:
                    self._write_file.execute(str(target), original_content)
        except Exception as error:
            self._clear_capability_state()
            return "ERROR CRÍTICO: falló el rollback controlado. Motivo de validación: " + reason + ". Motivo de rollback: " + str(error)

        self._clear_capability_state()
        return "La validación falló. Se restauraron únicamente los archivos aprobados. Motivo: " + reason

    def _clear_capability_state(self) -> None:
        """Forget transient apply state so restarts cannot repeat an action."""
        self._pending_capability_plan = None
        self._applied_capability_change = None
        self._validated_capability_change = None
        self._capability_validation_status = None
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
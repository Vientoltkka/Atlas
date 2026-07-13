"""Interactive correction proposal workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from models.prompt_client import PromptClient
from use_cases.query_architecture_graph import QueryArchitectureGraphUseCase
from use_cases.read_file import ReadFileUseCase


@dataclass(frozen=True)
class CorrectionCommand:
    """Parsed correction command."""

    path: str


class CorrectionInteractionUseCase:
    """Generate correction proposals for one Python file without writing."""

    _COMMAND_PATTERN = re.compile(
        r"^\s*(corrige|arregla|mejora|fix)\s+(?P<path>\S+)\s*$",
        re.IGNORECASE,
    )
    _COMMAND_PREFIXES = ("corrige", "arregla", "mejora", "fix")

    def __init__(
        self,
        read_file: ReadFileUseCase,
        query_architecture_graph: QueryArchitectureGraphUseCase,
        prompt_client: PromptClient,
    ) -> None:
        self._read_file = read_file
        self._query_architecture_graph = query_architecture_graph
        self._prompt_client = prompt_client

    def execute(
        self,
        prompt: str,
        project_root: Path,
        choose_model: Callable[[str], str],
    ) -> str | None:
        """Handle one correction prompt or return None for unrelated prompts."""
        command, error = self.parse(prompt)

        if error is not None:
            return error

        if command is None:
            return None

        try:
            root = project_root.resolve()
            target_path = self._resolve_target_path(
                project_root=root,
                requested_path=command.path,
            )
            relative_path = target_path.relative_to(root).as_posix()
            content = self._read_file.execute(str(target_path))
            impact = self._query_architecture_graph.impact_of(relative_path)
            dependencies = self._query_architecture_graph.dependencies_of(relative_path)
            affected_files = self._affected_files(impact.affected_files)
            risk = self._risk_level(affected_files)
            model = choose_model("coding")
            proposal = self._prompt_client.ask(
                model=model,
                messages=self._build_messages(
                    user_objective=prompt,
                    path=relative_path,
                    content=content,
                    dependencies=dependencies.dependencies,
                    affected_files=affected_files,
                ),
            )

            return self._format_response(
                path=relative_path,
                affected_files=affected_files,
                risk=risk,
                proposal=proposal,
            )
        except Exception as error:
            return f"No se pudo generar la propuesta de corrección: {error}"

    def parse(
        self,
        prompt: str,
    ) -> tuple[CorrectionCommand | None, str | None]:
        """Parse a supported correction command."""
        stripped = prompt.strip()

        if not stripped:
            return (None, None)

        first_word = stripped.split(maxsplit=1)[0].lower()

        if first_word not in self._COMMAND_PREFIXES:
            return (None, None)

        match = self._COMMAND_PATTERN.match(prompt)

        if match is None:
            return (None, "Orden de corrección incompleta o no válida.")

        requested_path = match.group("path")

        if "." in requested_path and not requested_path.endswith(".py"):
            return (None, "Solo se admiten archivos Python .py.")

        return (CorrectionCommand(path=requested_path), None)

    def _resolve_target_path(
        self,
        project_root: Path,
        requested_path: str,
    ) -> Path:
        """Resolve and validate a correction target path."""
        path = Path(requested_path)
        target_path = path if path.is_absolute() else project_root / path
        resolved = target_path.resolve()

        if not self._is_relative_to(resolved, project_root):
            raise ValueError(f"archivo fuera de project_root: {requested_path}")

        if not resolved.exists():
            raise ValueError(f"archivo inexistente: {requested_path}")

        if resolved.suffix != ".py":
            raise ValueError(f"archivo no Python: {requested_path}")

        return resolved

    def _build_messages(
        self,
        user_objective: str,
        path: str,
        content: str,
        dependencies: list[str],
        affected_files: list[str],
    ) -> list[dict[str, str]]:
        """Build the prompt sent to the configured language model."""
        dependencies_text = self._format_bullets(dependencies)
        affected_text = self._format_bullets(affected_files)

        return [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "Eres Atlas Coding Agent.",
                        "Genera una propuesta concreta de corrección para un único archivo Python.",
                        "No inventes módulos, archivos, clases ni APIs inexistentes.",
                        "No modifiques otros archivos.",
                        "Conserva el comportamiento público salvo que el código demuestre un bug.",
                        "Devuelve código Python completo corregido o un diff estructurado.",
                        "Explica brevemente el problema y la solución.",
                    ]
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "Objetivo del usuario:",
                        user_objective,
                        "",
                        "Ruta del archivo:",
                        path,
                        "",
                        "Contexto arquitectónico relevante:",
                        "Dependencias internas:",
                        dependencies_text,
                        "",
                        "Archivos dependientes o afectados:",
                        affected_text,
                        "",
                        "Restricciones:",
                        "- No inventar módulos inexistentes.",
                        "- No modificar otros archivos.",
                        "- Devolver código Python completo o diff estructurado.",
                        "- Explicar brevemente el problema y la solución.",
                        "",
                        "Contenido completo del archivo:",
                        content,
                    ]
                ),
            },
        ]

    def _format_response(
        self,
        path: str,
        affected_files: list[str],
        risk: str,
        proposal: str,
    ) -> str:
        """Format the generated correction proposal."""
        return "\n".join(
            [
                "Propuesta de corrección",
                "",
                "Archivo:",
                path,
                "",
                "Problema detectado:",
                "Ver propuesta generada por el modelo.",
                "",
                "Impacto:",
                *self._format_bullet_lines(affected_files),
                "",
                "Riesgo:",
                risk,
                "",
                "Cambios propuestos:",
                "Ver vista previa.",
                "",
                "Vista previa:",
                proposal,
                "",
                "Cambios todavía no aplicados.",
            ]
        )

    def _affected_files(
        self,
        values: list[str],
    ) -> list[str]:
        """Return deterministic affected files."""
        return sorted(dict.fromkeys(values))

    def _risk_level(
        self,
        affected_files: list[str],
    ) -> str:
        """Return a simple deterministic initial risk level."""
        count = len(affected_files)

        if count <= 2:
            return "low"

        if count <= 5:
            return "medium"

        return "high"

    def _format_bullets(
        self,
        values: list[str],
    ) -> str:
        """Return bullet text for prompt sections."""
        return "\n".join(self._format_bullet_lines(values))

    def _format_bullet_lines(
        self,
        values: list[str],
    ) -> list[str]:
        """Return bullet lines or an explicit empty value."""
        if not values:
            return ["- ninguno"]

        return [f"- {value}" for value in values]

    def _is_relative_to(
        self,
        path: Path,
        parent: Path,
    ) -> bool:
        """Return whether path is inside parent."""
        try:
            path.relative_to(parent)
        except ValueError:
            return False

        return True

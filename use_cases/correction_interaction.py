"""Interactive correction proposal workflow."""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from models.prompt_client import PromptClient
from use_cases.query_architecture_graph import QueryArchitectureGraphUseCase
from use_cases.read_file import ReadFileUseCase
from use_cases.write_file import WriteFileUseCase


@dataclass(frozen=True)
class CorrectionCommand:
    """Parsed correction command."""

    path: str


@dataclass(frozen=True)
class ParsedCorrectionProposal:
    """Structured correction proposal returned by the model."""

    problem: str
    risk: str
    proposed_file: str


@dataclass(frozen=True)
class CorrectionTestResult:
    """Result of running the project test suite."""

    tests_run: int
    tests_passed: int
    tests_failed: int
    success: bool
    output: str


class CorrectionInteractionUseCase:
    """Generate, confirm, apply, validate and rollback one Python correction."""

    _COMMAND_PATTERN = re.compile(
        r"^\s*(corrige|arregla|mejora|fix)\s+(?P<path>\S+)\s*$",
        re.IGNORECASE,
    )
    _PROPOSAL_PATTERN = re.compile(
        r"PROBLEM:\s*(?P<problem>.*?)\s+"
        r"RISK:\s*(?P<risk>low|medium|high)\s+"
        r"PROPOSED_FILE:\s*```python\s*\n(?P<code>.*?)\n?```",
        re.IGNORECASE | re.DOTALL,
    )
    _COMMAND_PREFIXES = ("corrige", "arregla", "mejora", "fix")

    def __init__(
        self,
        read_file: ReadFileUseCase,
        write_file: WriteFileUseCase,
        query_architecture_graph: QueryArchitectureGraphUseCase,
        prompt_client: PromptClient,
    ) -> None:
        self._read_file = read_file
        self._write_file = write_file
        self._query_architecture_graph = query_architecture_graph
        self._prompt_client = prompt_client

    def execute(
        self,
        prompt: str,
        project_root: Path,
        choose_model: Callable[[str], str],
        confirm: Callable[[str], str],
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
                    risk=risk,
                ),
            )
            parsed_proposal = self._parse_model_proposal(proposal)
            proposal_response = self._format_proposal_response(
                path=relative_path,
                affected_files=affected_files,
                risk=risk,
                proposal=parsed_proposal,
            )

            answer = confirm(
                "\n".join(
                    [
                        proposal_response,
                        "",
                        "¿Deseas aplicar la corrección? [s/N]: ",
                    ]
                )
            )

            if answer.strip().lower() not in ("s", "si", "sí", "y", "yes"):
                return "\n".join(
                    [
                        "Corrección cancelada.",
                        "No se modificó ningún archivo.",
                    ]
                )

            final_report = self._apply_confirmed_correction(
                target_path=target_path,
                original_content=content,
                proposed_content=parsed_proposal.proposed_file,
                project_root=root,
            )

            return final_report
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
        risk: str,
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
                        "Devuelve exclusivamente la estructura delimitada solicitada.",
                        "PROPOSED_FILE debe contener el archivo completo corregido.",
                        "No devuelvas diff ni texto libre fuera de la estructura.",
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
                        "Riesgo inicial calculado:",
                        risk,
                        "",
                        "Restricciones:",
                        "- No inventar módulos inexistentes.",
                        "- No modificar otros archivos.",
                        "- Devolver el contenido completo del archivo corregido.",
                        "- No devolver diff.",
                        "- No devolver texto fuera de la estructura obligatoria.",
                        "",
                        "Estructura obligatoria:",
                        "PROBLEM:",
                        "<explicación breve>",
                        "",
                        "RISK:",
                        "low|medium|high",
                        "",
                        "PROPOSED_FILE:",
                        "```python",
                        "<contenido completo del archivo>",
                        "```",
                        "",
                        "Contenido completo del archivo:",
                        content,
                    ]
                ),
            },
        ]

    def _parse_model_proposal(
        self,
        proposal: str,
    ) -> ParsedCorrectionProposal:
        """Parse a safe full-file correction proposal."""
        match = self._PROPOSAL_PATTERN.search(proposal)

        if match is None:
            raise ValueError(
                "la propuesta no contiene PROBLEM, RISK y PROPOSED_FILE "
                "con un archivo Python completo delimitado"
            )

        return ParsedCorrectionProposal(
            problem=match.group("problem").strip(),
            risk=match.group("risk").strip().lower(),
            proposed_file=match.group("code"),
        )

    def _format_proposal_response(
        self,
        path: str,
        affected_files: list[str],
        risk: str,
        proposal: ParsedCorrectionProposal,
    ) -> str:
        """Format the generated correction proposal."""
        return "\n".join(
            [
                "Propuesta de corrección",
                "",
                f"Archivo: {path}",
                "",
                "Problema detectado:",
                proposal.problem,
                "",
                "Archivos afectados:",
                *self._format_bullet_lines(affected_files),
                "",
                f"Riesgo: {risk}",
                "",
                "Vista previa:",
                "```python",
                proposal.proposed_file,
                "```",
                "",
                "Cambios todavía no aplicados.",
            ]
        )

    def _apply_confirmed_correction(
        self,
        target_path: Path,
        original_content: str,
        proposed_content: str,
        project_root: Path,
    ) -> str:
        """Apply a confirmed correction with syntax validation, tests and rollback."""
        written = False

        try:
            self._validate_python_source(proposed_content, target_path)
            self._write_file.execute(str(target_path), proposed_content)
            written = True
            self._validate_written_file(target_path)
            test_result = self._run_tests(project_root)

            if not test_result.success:
                raise RuntimeError(self._format_test_failure(test_result))
        except Exception as error:
            if written:
                self._write_file.execute(str(target_path), original_content)

            return "\n".join(
                [
                    "Aplicando corrección...",
                    "",
                    "La corrección no superó la validación.",
                    "Se restauró el contenido original.",
                    "El proyecto no quedó parcialmente modificado.",
                    f"Motivo: {error}",
                ]
            )

        return "\n".join(
            [
                "Aplicando corrección...",
                "Validación de sintaxis: correcta",
                f"Tests ejecutados: {test_result.tests_run}",
                f"Tests superados: {test_result.tests_passed}",
                f"Tests fallidos: {test_result.tests_failed}",
                "",
                "Corrección aplicada correctamente.",
                "Cambios aún sin commit.",
            ]
        )

    def _validate_python_source(
        self,
        source: str,
        path: Path,
    ) -> None:
        """Validate tokenization and syntax for one Python source."""
        list(tokenize.generate_tokens(io.StringIO(source).readline))
        ast.parse(source, filename=str(path))

    def _validate_written_file(
        self,
        path: Path,
    ) -> None:
        """Validate the file after writing it."""
        self._validate_python_source(
            source=self._read_file.execute(str(path)),
            path=path,
        )

    def _run_tests(
        self,
        project_root: Path,
    ) -> CorrectionTestResult:
        """Run the full project test suite."""
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        output = "\n".join(
            value
            for value in (completed.stdout, completed.stderr)
            if value
        )
        tests_passed, tests_failed = self._parse_pytest_counts(output)

        return CorrectionTestResult(
            tests_run=tests_passed + tests_failed,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            success=completed.returncode == 0,
            output=output,
        )

    def _parse_pytest_counts(
        self,
        output: str,
    ) -> tuple[int, int]:
        """Extract passed and failed test counts from pytest output."""
        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+failed", output)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0

        return (passed, failed)

    def _format_test_failure(
        self,
        test_result: CorrectionTestResult,
    ) -> str:
        """Format a controlled test failure message."""
        return (
            "tests fallidos: "
            f"{test_result.tests_failed}; "
            f"tests superados: {test_result.tests_passed}"
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

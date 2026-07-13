"""Apply safe Python symbol rename plans."""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

from domain.refactoring.refactoring_plan import RefactoringPlan
from domain.refactoring.refactoring_result import RenameSymbolResult
from use_cases.plan_refactoring import (
    InvalidRefactoringNameError,
    PlanRefactoringUseCase,
)


class RenameSymbolError(Exception):
    """Base exception for controlled rename failures."""


class RenameSymbolUseCase:
    """Safely apply a planned Python symbol rename."""

    def __init__(
        self,
        plan_refactoring: PlanRefactoringUseCase,
    ) -> None:
        self._plan_refactoring = plan_refactoring

    def execute(
        self,
        project_root: Path,
        symbol_name: str,
        new_name: str,
    ) -> RenameSymbolResult:
        """Rename a simple Python symbol using a transactional workflow."""
        try:
            plan = self._plan_refactoring.execute(
                symbol_name=symbol_name,
                new_name=new_name,
            )
        except InvalidRefactoringNameError as error:
            return self._blocked_result(
                symbol_name=symbol_name,
                new_name=new_name,
                warnings=[str(error)],
            )

        plan_warning = self._blocking_plan_warning(plan)

        if plan_warning is not None:
            return self._blocked_result(
                symbol_name=symbol_name,
                new_name=new_name,
                warnings=[plan_warning],
            )

        root = project_root.resolve()
        resolved_files, path_warnings = self._resolve_plan_files(
            project_root=root,
            plan=plan,
        )

        if path_warnings:
            return self._blocked_result(
                symbol_name=symbol_name,
                new_name=new_name,
                warnings=path_warnings,
            )

        original_contents = self._read_all(resolved_files)
        transformed_contents: dict[Path, str] = {}
        changed_files: list[Path] = []
        replacements_count = 0

        for path in resolved_files:
            transformed, replacements = self._rename_tokens(
                source=original_contents[path],
                symbol_name=symbol_name,
                new_name=new_name,
            )
            self._validate_python_source(transformed, path)
            transformed_contents[path] = transformed
            replacements_count += replacements

            if transformed != original_contents[path]:
                changed_files.append(path)

        if replacements_count == 0:
            return self._blocked_result(
                symbol_name=symbol_name,
                new_name=new_name,
                warnings=["no se produjo ningún cambio real"],
            )

        return self._apply_transaction(
            symbol_name=symbol_name,
            new_name=new_name,
            original_contents=original_contents,
            transformed_contents=transformed_contents,
            changed_files=tuple(sorted(changed_files)),
            replacements_count=replacements_count,
        )

    def _blocking_plan_warning(
        self,
        plan: RefactoringPlan,
    ) -> str | None:
        """Return a blocking plan warning when the plan is not safe to apply."""
        if not plan.can_apply:
            return "el plan no es aplicable"

        if plan.definition_file is None:
            return "la definición no está resuelta"

        if plan.warnings:
            return "; ".join(plan.warnings)

        return None

    def _resolve_plan_files(
        self,
        project_root: Path,
        plan: RefactoringPlan,
    ) -> tuple[list[Path], list[str]]:
        """Resolve and validate every file path in the plan."""
        warnings: list[str] = []
        paths = self._unique_plan_paths(plan)
        resolved_files: list[Path] = []

        for path_text in paths:
            path = Path(path_text)
            candidate = path if path.is_absolute() else project_root / path
            resolved = candidate.resolve()

            if not self._is_relative_to(resolved, project_root):
                warnings.append(f"archivo fuera de project_root: {path_text}")
                continue

            if not resolved.exists():
                warnings.append(f"archivo inexistente: {path_text}")
                continue

            if resolved.suffix != ".py":
                warnings.append(f"archivo no Python: {path_text}")
                continue

            resolved_files.append(resolved)

        return sorted(dict.fromkeys(resolved_files)), warnings

    def _unique_plan_paths(
        self,
        plan: RefactoringPlan,
    ) -> list[str]:
        """Return plan paths including the definition file without duplicates."""
        paths: list[str] = []

        if plan.definition_file is not None:
            paths.append(plan.definition_file)

        paths.extend(plan.affected_files)

        return sorted(dict.fromkeys(paths))

    def _read_all(
        self,
        paths: list[Path],
    ) -> dict[Path, str]:
        """Read every file before modifying any file."""
        return {
            path: path.read_text(encoding="utf-8")
            for path in paths
        }

    def _rename_tokens(
        self,
        source: str,
        symbol_name: str,
        new_name: str,
    ) -> tuple[str, int]:
        """Rename exact Python NAME tokens only."""
        replacements = 0
        renamed_tokens: list[tokenize.TokenInfo] = []

        for token_info in tokenize.generate_tokens(io.StringIO(source).readline):
            if token_info.type == tokenize.NAME and token_info.string == symbol_name:
                renamed_tokens.append(token_info._replace(string=new_name))
                replacements += 1
                continue

            renamed_tokens.append(token_info)

        transformed = tokenize.untokenize(renamed_tokens)
        list(tokenize.generate_tokens(io.StringIO(transformed).readline))

        return transformed, replacements

    def _validate_python_source(
        self,
        source: str,
        path: Path,
    ) -> None:
        """Validate tokenization and syntax for one Python source."""
        list(tokenize.generate_tokens(io.StringIO(source).readline))
        ast.parse(source, filename=str(path))

    def _apply_transaction(
        self,
        symbol_name: str,
        new_name: str,
        original_contents: dict[Path, str],
        transformed_contents: dict[Path, str],
        changed_files: tuple[Path, ...],
        replacements_count: int,
    ) -> RenameSymbolResult:
        """Write all changes and rollback on any failure."""
        written_files: list[Path] = []

        try:
            for path in changed_files:
                self._write_file(path, transformed_contents[path])
                written_files.append(path)

            self._validate_written_files(changed_files)
        except Exception as error:
            self._rollback(
                original_contents=original_contents,
                written_files=written_files,
            )
            return RenameSymbolResult(
                symbol_name=symbol_name,
                new_name=new_name,
                changed_files=tuple(),
                replacements_count=0,
                applied=False,
                rolled_back=True,
                warnings=(f"rollback ejecutado: {error}",),
                summary=(
                    f"No se aplicó el renombrado de {symbol_name} a {new_name}. "
                    "Se restauró el estado original."
                ),
            )

        return RenameSymbolResult(
            symbol_name=symbol_name,
            new_name=new_name,
            changed_files=changed_files,
            replacements_count=replacements_count,
            applied=True,
            rolled_back=False,
            warnings=tuple(),
            summary=(
                f"Renombrado {symbol_name} a {new_name} aplicado en "
                f"{len(changed_files)} archivo(s), con "
                f"{replacements_count} reemplazo(s)."
            ),
        )

    def _write_file(
        self,
        path: Path,
        content: str,
    ) -> None:
        """Write a UTF-8 file."""
        path.write_text(content, encoding="utf-8")

    def _validate_written_files(
        self,
        changed_files: tuple[Path, ...],
    ) -> None:
        """Validate modified files after writing."""
        for path in changed_files:
            self._validate_python_source(
                source=path.read_text(encoding="utf-8"),
                path=path,
            )

    def _rollback(
        self,
        original_contents: dict[Path, str],
        written_files: list[Path],
    ) -> None:
        """Restore every file already modified by the transaction."""
        for path in written_files:
            self._write_file(path, original_contents[path])

    def _blocked_result(
        self,
        symbol_name: str,
        new_name: str,
        warnings: list[str],
    ) -> RenameSymbolResult:
        """Return a non-applied result."""
        return RenameSymbolResult(
            symbol_name=symbol_name,
            new_name=new_name,
            changed_files=tuple(),
            replacements_count=0,
            applied=False,
            rolled_back=False,
            warnings=tuple(warnings),
            summary=(
                f"No se aplicó el renombrado de {symbol_name} a {new_name}."
            ),
        )

    def _is_relative_to(
        self,
        path: Path,
        parent: Path,
    ) -> bool:
        """Return whether a path is inside a parent path."""
        try:
            path.relative_to(parent)
        except ValueError:
            return False

        return True

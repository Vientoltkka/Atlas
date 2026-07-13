"""Interactive refactoring command workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from domain.refactoring.refactoring_plan import RefactoringPlan
from domain.refactoring.refactoring_result import RenameSymbolResult
from use_cases.plan_refactoring import (
    InvalidRefactoringNameError,
    PlanRefactoringUseCase,
)
from use_cases.rename_symbol import RenameSymbolUseCase


@dataclass(frozen=True)
class RefactoringCommand:
    """Parsed refactoring command."""

    symbol_name: str
    new_name: str


class RefactoringInteractionUseCase:
    """Coordinate interactive rename planning, confirmation, and execution."""

    _COMMAND_PATTERNS = (
        re.compile(
            r"^\s*renombra\s+(?P<symbol>\S+)\s+a\s+(?P<new>\S+)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*renombrar\s+(?P<symbol>\S+)\s+a\s+(?P<new>\S+)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*cambia\s+(?P<symbol>\S+)\s+por\s+(?P<new>\S+)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*rename\s+(?P<symbol>\S+)\s+to\s+(?P<new>\S+)\s*$",
            re.IGNORECASE,
        ),
    )
    _COMMAND_PREFIXES = ("renombra", "renombrar", "cambia", "rename")
    _YES_VALUES = {"s", "si", "sí", "y", "yes"}

    def __init__(
        self,
        plan_refactoring: PlanRefactoringUseCase,
        rename_symbol: RenameSymbolUseCase,
    ) -> None:
        self._plan_refactoring = plan_refactoring
        self._rename_symbol = rename_symbol

    def execute(
        self,
        prompt: str,
        project_root: Path,
        confirm: Callable[[str], str],
    ) -> str | None:
        """Handle one interactive prompt or return None for unrelated prompts."""
        command, error = self.parse(prompt)

        if error is not None:
            return error

        if command is None:
            return None

        try:
            plan = self._plan_refactoring.execute(
                symbol_name=command.symbol_name,
                new_name=command.new_name,
            )
        except InvalidRefactoringNameError as error:
            return f"No se puede planificar la refactorización: {error}"
        except Exception as error:
            return f"No se puede planificar la refactorización: {error}"

        plan_message = self.format_plan(plan)

        if not plan.can_apply:
            return "\n".join(
                [
                    plan_message,
                    "",
                    "No se puede aplicar la refactorización.",
                ]
            )

        confirmation = confirm(f"{plan_message}\n\n¿Deseas aplicar los cambios? [s/N]: ")

        if not self.is_confirmed(confirmation):
            return "\n".join(
                [
                    plan_message,
                    "",
                    "Refactorización cancelada. No se modificó ningún archivo.",
                ]
            )

        try:
            result = self._rename_symbol.execute(
                project_root=project_root,
                symbol_name=command.symbol_name,
                new_name=command.new_name,
            )
        except Exception as error:
            return "\n".join(
                [
                    plan_message,
                    "",
                    f"No se pudo aplicar la refactorización: {error}",
                ]
            )

        return "\n".join(
            [
                plan_message,
                "",
                self.format_result(result),
            ]
        )

    def parse(
        self,
        prompt: str,
    ) -> tuple[RefactoringCommand | None, str | None]:
        """Parse a supported rename command."""
        stripped = prompt.strip()

        if not stripped:
            return (None, None)

        first_word = stripped.split(maxsplit=1)[0].lower()

        if first_word not in self._COMMAND_PREFIXES:
            return (None, None)

        for pattern in self._COMMAND_PATTERNS:
            match = pattern.match(prompt)

            if match is None:
                continue

            symbol_name = match.group("symbol")
            new_name = match.group("new")

            if "." in symbol_name or "." in new_name:
                return (
                    None,
                    "No se admiten expresiones cualificadas para este hito.",
                )

            if not symbol_name.isidentifier() or not new_name.isidentifier():
                return (
                    None,
                    "Orden de renombrado inválida: nombres no válidos.",
                )

            return (
                RefactoringCommand(
                    symbol_name=symbol_name,
                    new_name=new_name,
                ),
                None,
            )

        return (
            None,
            "Orden de renombrado incompleta o no válida.",
        )

    def format_plan(
        self,
        plan: RefactoringPlan,
    ) -> str:
        """Return a user-facing refactoring plan."""
        definition = plan.definition_file or "no resuelta"
        affected_files = plan.affected_files or []
        file_lines = [f"- {path}" for path in affected_files] or ["- ninguno"]
        lines = [
            "Plan de refactorización",
            "",
            f"Operación: {plan.operation}",
            f"Símbolo actual: {plan.symbol_name}",
            f"Nuevo nombre: {plan.new_name}",
            f"Definición: {definition}",
            "",
            "Archivos afectados:",
            *file_lines,
            "",
            f"Riesgo: {plan.risk_level}",
        ]

        if plan.warnings:
            lines.extend(
                [
                    "",
                    "Advertencias:",
                    *[f"- {warning}" for warning in plan.warnings],
                ]
            )

        return "\n".join(lines)

    def format_result(
        self,
        result: RenameSymbolResult,
    ) -> str:
        """Return a user-facing rename result."""
        if result.applied:
            return "\n".join(
                [
                    "Refactorización aplicada correctamente.",
                    f"Archivos modificados: {len(result.changed_files)}",
                    f"Reemplazos realizados: {result.replacements_count}",
                    "",
                    "Archivos modificados:",
                    *[f"- {path}" for path in result.changed_files],
                ]
            )

        if result.rolled_back:
            warning_text = "; ".join(result.warnings) or "rollback ejecutado"
            return "\n".join(
                [
                    "No se pudo aplicar la refactorización.",
                    "Rollback ejecutado.",
                    f"Motivo: {warning_text}",
                ]
            )

        warning_text = "; ".join(result.warnings) or "operación no aplicada"

        return "\n".join(
            [
                "No se aplicó la refactorización.",
                f"Motivo: {warning_text}",
            ]
        )

    def is_confirmed(
        self,
        value: str,
    ) -> bool:
        """Return whether a confirmation value is affirmative."""
        return value.strip().lower() in self._YES_VALUES

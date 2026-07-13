"""Plan refactoring operations without modifying files."""

from __future__ import annotations

from models.architecture_graph import ArchitectureGraph
from domain.refactoring.refactoring_plan import RefactoringPlan, RefactoringRiskLevel
from use_cases.query_architecture_graph import QueryArchitectureGraphUseCase


class RefactoringPlanningError(Exception):
    """Base exception for refactoring planning failures."""


class InvalidRefactoringNameError(RefactoringPlanningError):
    """Raised when a refactoring symbol name is not valid."""


class PlanRefactoringUseCase:
    """Create deterministic refactoring plans from architecture knowledge."""

    def __init__(
        self,
        graph: ArchitectureGraph,
        query_architecture_graph: QueryArchitectureGraphUseCase | None = None,
    ) -> None:
        self._query_architecture_graph = (
            query_architecture_graph
            or QueryArchitectureGraphUseCase(graph)
        )

    def rename_symbol(
        self,
        symbol_name: str,
        new_name: str,
    ) -> RefactoringPlan:
        """Return a plan for renaming a symbol without changing files."""
        self._validate_names(
            symbol_name=symbol_name,
            new_name=new_name,
        )
        definition = self._query_architecture_graph.dependencies_of(symbol_name)

        if definition.matches:
            warnings = [
                "definición ambigua",
                "no se puede elegir una definición arbitrariamente",
                *[f"coincidencia: {match}" for match in definition.matches],
            ]
            return self._build_plan(
                symbol_name=symbol_name,
                new_name=new_name,
                definition_file=None,
                affected_files=[],
                risk_level="high",
                warnings=warnings,
                can_apply=False,
            )

        if definition.target is None:
            return self._build_plan(
                symbol_name=symbol_name,
                new_name=new_name,
                definition_file=None,
                affected_files=[],
                risk_level="high",
                warnings=[f"símbolo inexistente: {symbol_name}"],
                can_apply=False,
            )

        impact = self._query_architecture_graph.impact_of(symbol_name)
        affected_files = self._unique_sorted(
            [
                definition.target,
                *impact.affected_files,
            ]
        )
        warnings = self._warnings(
            affected_files=affected_files,
        )
        risk_level = self._risk_level(
            affected_files=affected_files,
            warnings=warnings,
        )

        return self._build_plan(
            symbol_name=symbol_name,
            new_name=new_name,
            definition_file=definition.target,
            affected_files=affected_files,
            risk_level=risk_level,
            warnings=warnings,
            can_apply=risk_level != "high",
        )

    def execute(
        self,
        symbol_name: str,
        new_name: str,
    ) -> RefactoringPlan:
        """Alias for the initial supported refactoring operation."""
        return self.rename_symbol(
            symbol_name=symbol_name,
            new_name=new_name,
        )

    def _validate_names(
        self,
        symbol_name: str,
        new_name: str,
    ) -> None:
        """Validate refactoring symbol names."""
        if not symbol_name or not symbol_name.isidentifier():
            raise InvalidRefactoringNameError(
                f"Nombre de símbolo inválido: {symbol_name}"
            )

        if not new_name or not new_name.isidentifier():
            raise InvalidRefactoringNameError(
                f"Nombre nuevo inválido: {new_name}"
            )

        if symbol_name == new_name:
            raise InvalidRefactoringNameError(
                "El nombre actual y el nuevo nombre deben ser diferentes."
            )

    def _warnings(
        self,
        affected_files: list[str],
    ) -> list[str]:
        """Return deterministic warnings for a refactoring plan."""
        if len(affected_files) > 5:
            return ["más de 5 archivos afectados"]

        return []

    def _risk_level(
        self,
        affected_files: list[str],
        warnings: list[str],
    ) -> RefactoringRiskLevel:
        """Calculate the initial risk level."""
        affected_count = len(affected_files)

        if warnings or affected_count > 5:
            return "high"

        if affected_count >= 3:
            return "medium"

        return "low"

    def _build_plan(
        self,
        symbol_name: str,
        new_name: str,
        definition_file: str | None,
        affected_files: list[str],
        risk_level: RefactoringRiskLevel,
        warnings: list[str],
        can_apply: bool,
    ) -> RefactoringPlan:
        """Build an immutable refactoring plan."""
        return RefactoringPlan(
            operation="rename_symbol",
            symbol_name=symbol_name,
            new_name=new_name,
            definition_file=definition_file,
            affected_files=affected_files,
            risk_level=risk_level,
            warnings=warnings,
            can_apply=can_apply,
            summary=self._summary(
                symbol_name=symbol_name,
                new_name=new_name,
                definition_file=definition_file,
                affected_files=affected_files,
                risk_level=risk_level,
                can_apply=can_apply,
            ),
        )

    def _summary(
        self,
        symbol_name: str,
        new_name: str,
        definition_file: str | None,
        affected_files: list[str],
        risk_level: RefactoringRiskLevel,
        can_apply: bool,
    ) -> str:
        """Return a readable plan summary."""
        definition = definition_file or "no resuelta"
        applicability = "aplicable" if can_apply else "no aplicable"

        return (
            f"Renombrar {symbol_name} a {new_name}. "
            f"Definición: {definition}. "
            f"Archivos afectados: {len(affected_files)}. "
            f"Riesgo: {risk_level}. "
            f"Estado: {applicability}."
        )

    def _unique_sorted(
        self,
        values: list[str],
    ) -> list[str]:
        """Return unique values in deterministic order."""
        return sorted(dict.fromkeys(values))

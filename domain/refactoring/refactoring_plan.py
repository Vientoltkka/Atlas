"""Domain model for refactoring plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RefactoringOperation = Literal["rename_symbol"]
RefactoringRiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class RefactoringPlan:
    """Immutable plan for a refactoring operation."""

    operation: RefactoringOperation
    symbol_name: str
    new_name: str
    definition_file: str | None
    affected_files: list[str] = field(default_factory=list)
    risk_level: RefactoringRiskLevel = "low"
    warnings: list[str] = field(default_factory=list)
    can_apply: bool = True
    summary: str = ""

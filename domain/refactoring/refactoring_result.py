"""Domain model for refactoring execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RenameSymbolResult:
    """Immutable result for a symbol rename operation."""

    symbol_name: str
    new_name: str
    changed_files: tuple[Path, ...] = field(default_factory=tuple)
    replacements_count: int = 0
    applied: bool = False
    rolled_back: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""

"""Structured references to variables stored in one execution context."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Iterable

from core.structured_reference_path import normalize_reference_path


VARIABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BLOCKED_VARIABLE_NAMES = frozenset(
    {
        "__class__",
        "__dict__",
        "__globals__",
        "__mro__",
        "__subclasses__",
    }
)


class ExecutionVariableError(ValueError):
    """Base error for execution variable failures."""


class InvalidExecutionVariableNameError(ExecutionVariableError):
    """Raised when an execution variable name is unsafe or invalid."""


class ExecutionVariableReferenceError(ExecutionVariableError):
    """Raised when an execution variable reference is invalid."""


@dataclass(frozen=True, slots=True)
class ExecutionVariableReference:
    """Reference an execution variable, optionally selecting a safe path."""

    name: str
    path: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        validate_execution_variable_name(self.name)
        normalized_path = normalize_reference_path(
            self.path,
            label="ExecutionVariableReference",
        )
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "path", normalized_path)

    @classmethod
    def from_path(
        cls,
        name: str,
        path: Iterable[str | int] = (),
    ) -> "ExecutionVariableReference":
        """Build a variable reference from any iterable path."""
        return cls(name=name, path=tuple(path))


def validate_execution_variable_name(
    name: str,
) -> None:
    """Validate a safe flat variable name."""
    if not isinstance(name, str) or not name.strip():
        raise InvalidExecutionVariableNameError(
            "Execution variable name must be a non-empty string."
        )

    normalized = name.strip()
    if normalized in BLOCKED_VARIABLE_NAMES or normalized.startswith("__"):
        raise InvalidExecutionVariableNameError(
            f"Execution variable name is unsafe: {normalized}."
        )

    if VARIABLE_NAME_PATTERN.fullmatch(normalized) is None:
        raise InvalidExecutionVariableNameError(
            f"Execution variable name is invalid: {normalized}."
        )


def copy_execution_variable_reference(
    reference: ExecutionVariableReference,
) -> ExecutionVariableReference:
    """Return a defensive copy of a variable reference."""
    return ExecutionVariableReference(reference.name, deepcopy(reference.path))

"""Declarative binding from a step output into execution variables."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

from core.execution_variable_reference import validate_execution_variable_name
from core.structured_reference_path import normalize_reference_path


class ExecutionVariableBindingError(ValueError):
    """Base error for execution variable binding failures."""


class ExecutionVariableAlreadyExistsError(ExecutionVariableBindingError):
    """Raised when a binding cannot overwrite an existing variable."""


class ExecutionVariableBindingPathError(ExecutionVariableBindingError):
    """Raised when a binding path cannot select a value."""


@dataclass(frozen=True, slots=True)
class ExecutionVariableBinding:
    """Bind a successful step output, or part of it, to an execution variable."""

    variable_name: str
    path: tuple[str | int, ...] = ()
    overwrite: bool = True

    def __post_init__(self) -> None:
        validate_execution_variable_name(self.variable_name)
        if type(self.overwrite) is not bool:
            raise ExecutionVariableBindingError(
                "ExecutionVariableBinding overwrite must be a boolean."
            )
        normalized_path = normalize_reference_path(
            self.path,
            label="ExecutionVariableBinding",
        )
        object.__setattr__(self, "variable_name", self.variable_name.strip())
        object.__setattr__(self, "path", normalized_path)

    @classmethod
    def from_path(
        cls,
        variable_name: str,
        path: Iterable[str | int] = (),
        *,
        overwrite: bool = True,
    ) -> "ExecutionVariableBinding":
        """Build a binding from any iterable path."""
        return cls(variable_name=variable_name, path=tuple(path), overwrite=overwrite)


def copy_execution_variable_binding(
    binding: ExecutionVariableBinding | None,
) -> ExecutionVariableBinding | None:
    """Return a defensive copy of an optional binding."""
    if binding is None:
        return None
    return ExecutionVariableBinding(
        binding.variable_name,
        deepcopy(binding.path),
        binding.overwrite,
    )

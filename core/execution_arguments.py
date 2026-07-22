"""Validated immutable arguments for execution plan steps."""

from __future__ import annotations

from collections.abc import Iterator, KeysView, Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType, ModuleType
from typing import Any

from core.execution_variable_reference import (
    ExecutionVariableReference,
    copy_execution_variable_reference,
)
from core.step_output_reference import StepOutputReference, copy_step_output_reference


class ExecutionArgumentsError(ValueError):
    """Base error for execution argument failures."""


class InvalidExecutionArgumentError(ExecutionArgumentsError):
    """Raised when an argument contains an unsafe structural value."""


class MissingExecutionArgumentError(ExecutionArgumentsError):
    """Raised when a required execution argument is absent."""


@dataclass(frozen=True, slots=True, init=False)
class ExecutionArguments(Mapping[str, object]):
    """Immutable structural argument mapping for one execution step."""

    _values: Mapping[str, object]

    def __init__(
        self,
        values: Mapping[str, object] | None = None,
    ) -> None:
        frozen = _freeze_mapping(values or {}, "arguments")
        object.__setattr__(self, "_values", MappingProxyType(frozen))

    @classmethod
    def empty(cls) -> "ExecutionArguments":
        """Return an empty argument mapping."""
        return cls({})

    def get(
        self,
        name: str,
        default: object = None,
    ) -> object:
        """Return an argument value or default when absent."""
        if name not in self._values:
            return default
        return _thaw_value(self._values[name])

    def require(
        self,
        name: str,
    ) -> object:
        """Return an argument value or raise a contextual missing-argument error."""
        if name not in self._values:
            raise MissingExecutionArgumentError(f"Missing execution argument: {name}.")
        return _thaw_value(self._values[name])

    def contains(
        self,
        name: str,
    ) -> bool:
        """Return whether an argument name is present."""
        return name in self._values

    def as_dict(self) -> dict[str, object]:
        """Return a new mutable structural copy of the arguments."""
        return {
            key: _thaw_value(value)
            for key, value in self._values.items()
        }

    def is_empty(self) -> bool:
        """Return whether no arguments are present."""
        return not self._values

    def keys(self) -> KeysView[str]:
        return self._values.keys()

    def items(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            (key, _thaw_value(value))
            for key, value in self._values.items()
        )

    def __getitem__(
        self,
        key: str,
    ) -> object:
        return _thaw_value(self._values[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if isinstance(other, Mapping):
            return self.as_dict() == {
                key: _thaw_value(value)
                for key, value in other.items()
            }
        return False


def validate_execution_arguments(
    values: Mapping[str, object],
) -> None:
    """Validate an argument mapping without returning a normalized instance."""
    ExecutionArguments(values)


def _freeze_mapping(
    values: Mapping[str, object],
    path: str,
) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise InvalidExecutionArgumentError(
            f"{path}: expected mapping, got {type(values).__name__}."
        )

    frozen: dict[str, object] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise InvalidExecutionArgumentError(
                f"{path}: argument keys must be strings, got {type(key).__name__}."
            )
        if not key:
            raise InvalidExecutionArgumentError(f"{path}: argument keys cannot be empty.")
        frozen[key] = _freeze_value(value, f"{path}.{key}")
    return frozen


def _freeze_value(
    value: object,
    path: str,
) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidExecutionArgumentError(f"{path}: non-finite float is not supported.")
        return value

    if isinstance(value, StepOutputReference):
        return copy_step_output_reference(value)

    if isinstance(value, ExecutionVariableReference):
        return copy_execution_variable_reference(value)

    if isinstance(value, Mapping):
        return MappingProxyType(_freeze_mapping(value, path))

    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )

    if isinstance(value, ModuleType):
        type_name = "module"
    elif callable(value):
        type_name = "callable"
    elif isinstance(value, type):
        type_name = "class"
    else:
        type_name = type(value).__name__
    raise InvalidExecutionArgumentError(f"{path}: unsupported type {type_name}.")


def _thaw_value(
    value: object,
) -> object:
    if isinstance(value, StepOutputReference):
        return copy_step_output_reference(value)

    if isinstance(value, ExecutionVariableReference):
        return copy_execution_variable_reference(value)

    if isinstance(value, Mapping):
        return {
            key: _thaw_value(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]

    return value


def contains_step_output_reference(
    value: object,
) -> bool:
    """Return whether a structural value contains an unresolved step reference."""
    if isinstance(value, StepOutputReference):
        return True

    if isinstance(value, Mapping):
        return any(contains_step_output_reference(item) for item in value.values())

    if isinstance(value, (list, tuple)):
        return any(contains_step_output_reference(item) for item in value)

    return False


def contains_execution_variable_reference(
    value: object,
) -> bool:
    """Return whether a structural value contains an unresolved variable reference."""
    if isinstance(value, ExecutionVariableReference):
        return True

    if isinstance(value, Mapping):
        return any(contains_execution_variable_reference(item) for item in value.values())

    if isinstance(value, (list, tuple)):
        return any(contains_execution_variable_reference(item) for item in value)

    return False


def contains_unresolved_execution_reference(
    value: object,
) -> bool:
    """Return whether a value contains any unresolved structured execution reference."""
    return contains_step_output_reference(value) or contains_execution_variable_reference(value)

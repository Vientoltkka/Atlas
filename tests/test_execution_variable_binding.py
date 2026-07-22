from __future__ import annotations

import pytest

from core.execution_variable_binding import (
    ExecutionVariableBinding,
    ExecutionVariableBindingError,
)
from core.execution_variable_reference import InvalidExecutionVariableNameError


def test_binding_accepts_empty_and_nested_path() -> None:
    direct = ExecutionVariableBinding("workspace_path")
    nested = ExecutionVariableBinding("first_file", ("items", 0, "path"))

    assert direct.variable_name == "workspace_path"
    assert direct.path == ()
    assert direct.overwrite is True
    assert nested.path == ("items", 0, "path")


@pytest.mark.parametrize("name", ["", "workspace path", "1value", "a.b", "__class__"])
def test_binding_rejects_invalid_variable_names(name: str) -> None:
    with pytest.raises(InvalidExecutionVariableNameError):
        ExecutionVariableBinding(name)


@pytest.mark.parametrize("path", [(True,), (-1,), ("_secret",)])
def test_binding_rejects_unsafe_path(path: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        ExecutionVariableBinding("workspace_path", path)  # type: ignore[arg-type]


def test_binding_rejects_non_boolean_overwrite() -> None:
    with pytest.raises(ExecutionVariableBindingError):
        ExecutionVariableBinding("workspace_path", overwrite=1)  # type: ignore[arg-type]

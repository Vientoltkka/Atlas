from __future__ import annotations

import pytest

from core.execution_variable_reference import (
    ExecutionVariableReference,
    InvalidExecutionVariableNameError,
)


def test_variable_reference_accepts_valid_names_and_path() -> None:
    assert ExecutionVariableReference("workspace_path").name == "workspace_path"
    assert ExecutionVariableReference("_temporary").name == "_temporary"
    assert ExecutionVariableReference("result_1").name == "result_1"
    assert ExecutionVariableReference("search_config", ("filters", "limit")).path == (
        "filters",
        "limit",
    )


@pytest.mark.parametrize(
    "name",
    ["", "workspace path", "a.b", "1value", "$secret", "__class__"],
)
def test_variable_reference_rejects_invalid_names(name: str) -> None:
    with pytest.raises(InvalidExecutionVariableNameError):
        ExecutionVariableReference(name)


@pytest.mark.parametrize("path", [(True,), (-1,), ("_secret",)])
def test_variable_reference_rejects_unsafe_path(path: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        ExecutionVariableReference("workspace_path", path)  # type: ignore[arg-type]

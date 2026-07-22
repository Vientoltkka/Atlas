from __future__ import annotations

import math
from types import ModuleType

import pytest

from core.execution_arguments import (
    ExecutionArguments,
    InvalidExecutionArgumentError,
    MissingExecutionArgumentError,
    contains_execution_variable_reference,
    contains_step_output_reference,
    contains_unresolved_execution_reference,
)
from core.execution_variable_reference import ExecutionVariableReference
from core.step_output_reference import StepOutputReference


def test_empty_arguments_api() -> None:
    arguments = ExecutionArguments.empty()

    assert arguments.is_empty() is True
    assert arguments.as_dict() == {}
    assert arguments.get("missing") is None
    assert arguments.get("missing", "fallback") == "fallback"
    assert arguments.contains("missing") is False
    assert tuple(arguments.keys()) == ()
    assert arguments.items() == ()
    assert dict(arguments) == {}


def test_simple_and_nested_values_are_accessible() -> None:
    arguments = ExecutionArguments(
        {
            "query": "is:unread",
            "max_results": 20,
            "enabled": True,
            "score": 1.5,
            "filters": {"labels": ["inbox", "important"], "archived": None},
        }
    )

    assert arguments.is_empty() is False
    assert arguments.get("query") == "is:unread"
    assert arguments.require("max_results") == 20
    assert arguments.contains("filters") is True
    assert tuple(arguments.keys()) == ("query", "max_results", "enabled", "score", "filters")
    assert dict(arguments)["filters"] == {
        "labels": ["inbox", "important"],
        "archived": None,
    }


def test_require_missing_raises_contextual_error() -> None:
    arguments = ExecutionArguments({"query": "is:unread"})

    with pytest.raises(MissingExecutionArgumentError, match="missing"):
        arguments.require("missing")


def test_arguments_are_protected_from_external_mutation() -> None:
    source = {
        "query": "test",
        "filters": {"labels": ["a"]},
    }
    arguments = ExecutionArguments(source)

    source["query"] = "changed"
    source["filters"]["labels"].append("b")  # type: ignore[index]
    exported = arguments.as_dict()
    exported["query"] = "exported"
    exported["filters"]["labels"].append("c")  # type: ignore[index]
    from_get = arguments.get("filters")
    assert isinstance(from_get, dict)
    from_get["labels"].append("d")  # type: ignore[index]

    assert arguments.as_dict() == {
        "query": "test",
        "filters": {"labels": ["a"]},
    }


def test_tuple_values_are_normalized_to_lists_on_export() -> None:
    arguments = ExecutionArguments({"items": ("a", {"nested": (1, 2)})})

    assert arguments.as_dict() == {"items": ["a", {"nested": [1, 2]}]}


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        ({1: "bad"}, "arguments"),
        ({"filters": {"limit": object()}}, "arguments.filters.limit"),
        ({"items": [object()]}, "arguments.items[0]"),
        ({"value": math.nan}, "arguments.value"),
        ({"value": math.inf}, "arguments.value"),
        ({"callback": lambda: None}, "arguments.callback"),
        ({"klass": ExecutionArguments}, "arguments.klass"),
        ({"module": ModuleType("fake")}, "arguments.module"),
    ],
)
def test_invalid_arguments_are_rejected_with_value_path(
    payload: dict[object, object],
    path: str,
) -> None:
    with pytest.raises(InvalidExecutionArgumentError) as error:
        ExecutionArguments(payload)  # type: ignore[arg-type]

    assert path in str(error.value)


def test_as_dict_returns_a_new_copy_each_time() -> None:
    arguments = ExecutionArguments({"nested": {"items": [1]}})

    first = arguments.as_dict()
    second = arguments.as_dict()

    assert first == second
    assert first is not second
    assert first["nested"] is not second["nested"]


def test_execution_arguments_accept_step_output_reference_before_resolution() -> None:
    reference = StepOutputReference("read", ("items", 0, "id"))
    arguments = ExecutionArguments({"value": reference})

    exported = arguments.as_dict()

    assert exported == {"value": reference}
    assert exported["value"] is not reference
    assert contains_step_output_reference(arguments) is True


def test_execution_arguments_accept_execution_variable_reference_before_resolution() -> None:
    reference = ExecutionVariableReference("workspace_path")
    arguments = ExecutionArguments({"path": reference})

    exported = arguments.as_dict()

    assert exported == {"path": reference}
    assert exported["path"] is not reference
    assert contains_execution_variable_reference(arguments) is True
    assert contains_unresolved_execution_reference(arguments) is True


def test_step_output_reference_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="step_id"):
        StepOutputReference("")
    with pytest.raises(ValueError, match="bool"):
        StepOutputReference("read", (True,))
    with pytest.raises(ValueError, match="negative"):
        StepOutputReference("read", (-1,))
    with pytest.raises(ValueError, match="segments"):
        StepOutputReference("read", (object(),))  # type: ignore[arg-type]

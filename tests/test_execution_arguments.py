from __future__ import annotations

import math
from types import ModuleType

import pytest

from core.execution_arguments import (
    ExecutionArguments,
    InvalidExecutionArgumentError,
    MissingExecutionArgumentError,
)


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

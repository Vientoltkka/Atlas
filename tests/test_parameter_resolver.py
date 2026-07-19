from __future__ import annotations

from dataclasses import dataclass
import inspect

from core.execution_plan_executor import StepExecutionResult
from core.parameter_resolver import (
    ParameterResolutionErrorCode,
    ParameterResolutionResult,
    ParameterResolver,
)


@dataclass
class OutputObject:
    path: str
    items: list[dict[str, object]]


def _resolve(
    arguments: dict[str, object],
    previous_results: dict[str, object],
) -> ParameterResolutionResult:
    return ParameterResolver().resolve(arguments, previous_results)


def test_resolves_reference_to_complete_step_output() -> None:
    result = _resolve(
        {"payload": {"$ref": "steps.step_1.output"}},
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"payload": {"path": "README.md"}}
    assert result.used_step_ids == ["step_1"]


def test_resolves_reference_to_dictionary_key() -> None:
    result = _resolve(
        {"path": {"$ref": "steps.step_1.output.path"}},
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"path": "README.md"}


def test_resolves_nested_reference() -> None:
    result = _resolve(
        {"name": {"$ref": "steps.step_1.output.metadata.name"}},
        {"step_1": {"metadata": {"name": "atlas"}}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"name": "atlas"}


def test_resolves_list_index() -> None:
    result = _resolve(
        {"name": {"$ref": "steps.step_1.output.items.0.name"}},
        {"step_1": {"items": [{"name": "first"}]}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"name": "first"}


def test_resolves_multiple_references_in_same_arguments() -> None:
    result = _resolve(
        {
            "path": {"$ref": "steps.step_1.output.path"},
            "content": {"$ref": "steps.step_2.output.content"},
        },
        {
            "step_1": {"path": "README.md"},
            "step_2": {"content": "hello"},
        },
    )

    assert result.success is True
    assert result.resolved_arguments == {
        "path": "README.md",
        "content": "hello",
    }
    assert result.used_step_ids == ["step_1", "step_2"]


def test_resolves_references_inside_nested_lists_and_dictionaries() -> None:
    result = _resolve(
        {
            "destination": {
                "folder": {"$ref": "steps.step_1.output.folder"},
                "filename": {"$ref": "steps.step_2.output.name"},
            },
            "items": [
                {"$ref": "steps.step_3.output.items.0"},
                "literal",
            ],
        },
        {
            "step_1": {"folder": "out"},
            "step_2": {"name": "result.txt"},
            "step_3": {"items": ["alpha"]},
        },
    )

    assert result.success is True
    assert result.resolved_arguments == {
        "destination": {"folder": "out", "filename": "result.txt"},
        "items": ["alpha", "literal"],
    }


def test_literal_values_remain_intact_and_strings_are_not_interpreted() -> None:
    result = _resolve(
        {
            "path": "steps.step_1.output.path",
            "enabled": True,
            "count": 2,
            "empty": None,
        },
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is True
    assert result.resolved_arguments == {
        "path": "steps.step_1.output.path",
        "enabled": True,
        "count": 2,
        "empty": None,
    }
    assert result.used_step_ids == []


def test_original_arguments_and_previous_results_are_not_mutated() -> None:
    previous = {"step_1": {"items": [{"name": "alpha"}]}}
    arguments = {"item": {"$ref": "steps.step_1.output.items.0"}}

    result = _resolve(arguments, previous)
    result.resolved_arguments["item"]["name"] = "changed"  # type: ignore[index]

    assert arguments == {"item": {"$ref": "steps.step_1.output.items.0"}}
    assert previous == {"step_1": {"items": [{"name": "alpha"}]}}


def test_invalid_reference_syntax_fails() -> None:
    result = _resolve(
        {"path": {"$ref": "step.step_1.output.path"}},
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.INVALID_REFERENCE_SYNTAX.value
    assert result.unresolved_references == ["step.step_1.output.path"]


def test_missing_step_fails() -> None:
    result = _resolve(
        {"path": {"$ref": "steps.missing.output.path"}},
        {},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.REFERENCED_STEP_NOT_FOUND.value
    assert result.unresolved_references == ["steps.missing.output.path"]


def test_missing_output_in_step_result_fails() -> None:
    result = _resolve(
        {"path": {"$ref": "steps.step_1.output"}},
        {
            "step_1": StepExecutionResult(
                step_id="step_1",
                status="completed",
                success=True,
                tool_name="safe_tool",
                output=None,
            )
        },
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.REFERENCED_OUTPUT_MISSING.value


def test_missing_field_fails() -> None:
    result = _resolve(
        {"path": {"$ref": "steps.step_1.output.path"}},
        {"step_1": {"name": "README.md"}},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.REFERENCED_FIELD_NOT_FOUND.value


def test_invalid_list_index_fails() -> None:
    result = _resolve(
        {"name": {"$ref": "steps.step_1.output.items.2.name"}},
        {"step_1": {"items": [{"name": "first"}]}},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.INVALID_LIST_INDEX.value


def test_reference_to_incomplete_step_result_fails() -> None:
    result = _resolve(
        {"path": {"$ref": "steps.step_1.output.path"}},
        {
            "step_1": StepExecutionResult(
                step_id="step_1",
                status="failed",
                success=False,
                tool_name="safe_tool",
                output={"path": "README.md"},
            )
        },
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.REFERENCE_TO_INCOMPLETE_STEP.value


def test_object_attribute_access_is_supported_for_public_attributes() -> None:
    result = _resolve(
        {"name": {"$ref": "steps.step_1.output.items.0.name"}},
        {
            "step_1": OutputObject(
                path="README.md",
                items=[{"name": "first"}],
            )
        },
    )

    assert result.success is True
    assert result.resolved_arguments == {"name": "first"}


def test_private_attributes_are_blocked() -> None:
    result = _resolve(
        {"path": {"$ref": "steps.step_1.output._secret"}},
        {"step_1": OutputObject(path="README.md", items=[])},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.INVALID_REFERENCE_SYNTAX.value


def test_dunder_attributes_are_blocked() -> None:
    for blocked in ("__class__", "__dict__", "__globals__"):
        result = _resolve(
            {"path": {"$ref": f"steps.step_1.output.{blocked}"}},
            {"step_1": {"path": "README.md"}},
        )

        assert result.success is False
        assert result.error_code == ParameterResolutionErrorCode.INVALID_REFERENCE_SYNTAX.value


def test_ref_object_with_extra_keys_fails() -> None:
    result = _resolve(
        {"path": {"$ref": "steps.step_1.output.path", "default": "README.md"}},
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.INVALID_REFERENCE_SYNTAX.value


def test_circular_argument_structure_fails_without_mutating_input() -> None:
    arguments: dict[str, object] = {}
    arguments["self"] = arguments

    result = _resolve(arguments, {})

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.PARAMETER_RESOLUTION_FAILED.value
    assert arguments["self"] is arguments


def test_resolver_source_does_not_use_eval_or_exec() -> None:
    source = inspect.getsource(ParameterResolver)

    assert "eval(" not in source
    assert "exec(" not in source

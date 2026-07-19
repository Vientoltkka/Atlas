from __future__ import annotations

from dataclasses import dataclass
import inspect

from core.execution_plan_executor import StepExecutionResult
from core.parameter_resolver import (
    MAX_RESOLUTION_DEPTH,
    MAX_TEMPLATE_LENGTH,
    MAX_TEMPLATE_REFERENCES,
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


def test_template_with_one_reference_resolves_to_text() -> None:
    result = _resolve(
        {"message": {"$template": "Archivo: {{steps.step_1.output.path}}"}},
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"message": "Archivo: README.md"}
    assert result.used_references == ["steps.step_1.output.path"]
    assert result.templates_resolved == 1


def test_template_with_multiple_references_resolves_in_order() -> None:
    result = _resolve(
        {
            "path": {
                "$template": "{{steps.step_1.output.folder}}/{{steps.step_2.output.filename}}"
            }
        },
        {
            "step_1": {"folder": "out"},
            "step_2": {"filename": "result.txt"},
        },
    )

    assert result.success is True
    assert result.resolved_arguments == {"path": "out/result.txt"}
    assert result.used_step_ids == ["step_1", "step_2"]


def test_template_formed_only_by_reference_still_returns_string() -> None:
    result = _resolve(
        {"count": {"$template": "{{steps.step_1.output.count}}"}},
        {"step_1": {"count": 7}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"count": "7"}


def test_template_combines_literal_text_and_nested_reference() -> None:
    result = _resolve(
        {"message": {"$template": "Nombre: {{steps.step_1.output.meta.name}}"}},
        {"step_1": {"meta": {"name": "Atlas"}}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"message": "Nombre: Atlas"}


def test_template_resolves_list_index() -> None:
    result = _resolve(
        {"message": {"$template": "Primero: {{steps.step_1.output.items.0.name}}"}},
        {"step_1": {"items": [{"name": "alpha"}]}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"message": "Primero: alpha"}


def test_template_converts_scalar_values_to_text() -> None:
    result = _resolve(
        {
            "text": {
                "$template": (
                    "{{steps.step_1.output.count}} "
                    "{{steps.step_1.output.ratio}} "
                    "{{steps.step_1.output.enabled}} "
                    "{{steps.step_1.output.empty}}"
                )
            }
        },
        {
            "step_1": {
                "count": 3,
                "ratio": 1.5,
                "enabled": True,
                "empty": None,
            }
        },
    )

    assert result.success is True
    assert result.resolved_arguments == {"text": "3 1.5 true "}


def test_template_converts_dict_to_deterministic_json() -> None:
    result = _resolve(
        {"text": {"$template": "{{steps.step_1.output.payload}}"}},
        {"step_1": {"payload": {"b": 2, "a": 1}}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"text": '{"a":1,"b":2}'}


def test_template_converts_list_to_json() -> None:
    result = _resolve(
        {"text": {"$template": "{{steps.step_1.output.items}}"}},
        {"step_1": {"items": ["a", 1, True]}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"text": '["a",1,true]'}


def test_template_preserves_unicode() -> None:
    result = _resolve(
        {"text": {"$template": "Hola {{steps.step_1.output.name}}"}},
        {"step_1": {"name": "Víctor"}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"text": "Hola Víctor"}


def test_template_escaped_braces_are_literal() -> None:
    result = _resolve(
        {"text": {"$template": "{{{{steps.step_1.output.path}}}}"}},
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"text": "{{steps.step_1.output.path}}"}
    assert result.used_references == []


def test_ref_and_template_can_coexist_in_different_arguments() -> None:
    result = _resolve(
        {
            "count": {"$ref": "steps.step_1.output.count"},
            "message": {"$template": "Cantidad: {{steps.step_1.output.count}}"},
        },
        {"step_1": {"count": 4}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"count": 4, "message": "Cantidad: 4"}


def test_template_inside_nested_lists_and_dictionaries() -> None:
    result = _resolve(
        {
            "items": [
                {"message": {"$template": "{{steps.step_1.output.name}}.bak"}},
            ]
        },
        {"step_1": {"name": "README.md"}},
    )

    assert result.success is True
    assert result.resolved_arguments == {"items": [{"message": "README.md.bak"}]}


def test_template_missing_reference_fails_without_partial_text() -> None:
    result = _resolve(
        {"message": {"$template": "Archivo: {{steps.missing.output.path}}"}},
        {},
    )

    assert result.success is False
    assert result.resolved_arguments == {}
    assert result.error_code == ParameterResolutionErrorCode.UNRESOLVED_TEMPLATE_REFERENCE.value
    assert result.unresolved_references == ["steps.missing.output.path"]


def test_template_missing_field_fails() -> None:
    result = _resolve(
        {"message": {"$template": "Archivo: {{steps.step_1.output.path}}"}},
        {"step_1": {"name": "README.md"}},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.UNRESOLVED_TEMPLATE_REFERENCE.value


def test_template_step_not_completed_fails() -> None:
    result = _resolve(
        {"message": {"$template": "Archivo: {{steps.step_1.output.path}}"}},
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
    assert result.error_code == ParameterResolutionErrorCode.UNRESOLVED_TEMPLATE_REFERENCE.value


def test_template_incomplete_syntax_fails() -> None:
    result = _resolve(
        {"message": {"$template": "Archivo: {{steps.step_1.output.path"}},
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.INVALID_TEMPLATE_SYNTAX.value


def test_template_unbalanced_closing_braces_fail() -> None:
    result = _resolve(
        {"message": {"$template": "Archivo: steps.step_1.output.path}}"}},
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.INVALID_TEMPLATE_SYNTAX.value


def test_template_object_with_extra_keys_fails() -> None:
    result = _resolve(
        {
            "message": {
                "$template": "{{steps.step_1.output.path}}",
                "default": "",
            }
        },
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.INVALID_TEMPLATE_STRUCTURE.value


def test_template_value_must_be_string() -> None:
    result = _resolve(
        {"message": {"$template": 123}},
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.INVALID_TEMPLATE_TYPE.value


def test_template_rejects_function_calls_and_operators() -> None:
    for expression in (
        "steps.step_1.output.path.upper()",
        "steps.step_1.output.count + 1",
        "steps.step_1.output.items[0]",
        "__import__('os').system('dir')",
    ):
        result = _resolve(
            {"message": {"$template": f"{{{{{expression}}}}}"}},
            {"step_1": {"path": "README.md", "count": 1, "items": ["a"]}},
        )

        assert result.success is False
        assert result.error_code == (
            ParameterResolutionErrorCode.UNSUPPORTED_TEMPLATE_EXPRESSION.value
        )


def test_template_rejects_private_and_sensitive_segments() -> None:
    for blocked in ("_secret", "__class__", "password", "token", "credentials"):
        result = _resolve(
            {"message": {"$template": f"{{{{steps.step_1.output.{blocked}}}}}"}},
            {"step_1": {blocked: "hidden"}},
        )

        assert result.success is False
        assert result.error_code == ParameterResolutionErrorCode.UNRESOLVED_TEMPLATE_REFERENCE.value


def test_template_length_limit_is_enforced() -> None:
    result = _resolve(
        {"message": {"$template": "a" * (MAX_TEMPLATE_LENGTH + 1)}},
        {},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.INVALID_TEMPLATE_SYNTAX.value


def test_template_reference_limit_is_enforced() -> None:
    template = "".join(
        "{{steps.step_1.output.path}}"
        for _ in range(MAX_TEMPLATE_REFERENCES + 1)
    )

    result = _resolve(
        {"message": {"$template": template}},
        {"step_1": {"path": "README.md"}},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.INVALID_TEMPLATE_SYNTAX.value


def test_resolution_depth_limit_is_enforced() -> None:
    value: object = {"leaf": "ok"}
    for _ in range(MAX_RESOLUTION_DEPTH + 1):
        value = {"nested": value}

    result = _resolve({"payload": value}, {})

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.PARAMETER_RESOLUTION_FAILED.value


def test_template_value_must_be_json_serializable_for_complex_values() -> None:
    result = _resolve(
        {"message": {"$template": "{{steps.step_1.output.payload}}"}},
        {"step_1": {"payload": {"value": object()}}},
    )

    assert result.success is False
    assert result.error_code == ParameterResolutionErrorCode.TEMPLATE_VALUE_NOT_SERIALIZABLE.value


def test_template_resolution_does_not_mutate_arguments_or_previous_results() -> None:
    arguments = {
        "message": {
            "$template": "Archivo: {{steps.step_1.output.items.0.name}}"
        }
    }
    previous = {"step_1": {"items": [{"name": "README.md"}]}}

    result = _resolve(arguments, previous)

    assert result.success is True
    assert result.resolved_arguments == {"message": "Archivo: README.md"}
    assert arguments == {
        "message": {
            "$template": "Archivo: {{steps.step_1.output.items.0.name}}"
        }
    }
    assert previous == {"step_1": {"items": [{"name": "README.md"}]}}

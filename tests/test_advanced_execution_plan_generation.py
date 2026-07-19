from __future__ import annotations

import inspect
from typing import Any

from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_plan_validator import ExecutionPlanValidator
from core.planner import (
    ExecutionPlan,
    PlanGenerationResult,
    Planner,
    PlannerErrorCode,
)
from tools.argument_schema import (
    ArgumentField,
    ArgumentSchema,
    ArgumentSchemaRegistry,
    ArgumentValidator,
)
from tools.base_tool import BaseTool
from tools.intent_selector import ToolIntentRegistry, ToolSelector
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(
        self,
        name: str,
        output: Any,
        calls: list[str],
        *,
        requires_confirmation: bool = False,
    ) -> None:
        self._name = name
        self._output = output
        self._calls = calls
        self._requires_confirmation = requires_confirmation
        self.contexts: list[ToolContext] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}."

    @property
    def requires_confirmation(self) -> bool:
        return self._requires_confirmation

    def execute(self, context: ToolContext) -> Any:
        self._calls.append(self._name)
        self.contexts.append(context)
        return self._output


def _planner(calls: list[str] | None = None) -> tuple[Planner, ToolRegistry]:
    active_calls = calls if calls is not None else []
    registry = ToolRegistry()
    registry.register(SpyTool("read_file", "contenido", active_calls))
    registry.register(SpyTool("write_file", "written", active_calls, requires_confirmation=True))
    registry.register(SpyTool("list_directory", ["a.txt"], active_calls))

    intent_registry = ToolIntentRegistry()
    intent_registry.register("file.read", "read_file")
    intent_registry.register("file.write", "write_file")
    intent_registry.register("directory.list", "list_directory")
    selector = ToolSelector(registry, intent_registry)

    schema_registry = ArgumentSchemaRegistry()
    schema_registry.register(
        ArgumentSchema("file.read", (ArgumentField("path", str, required=True),))
    )
    schema_registry.register(
        ArgumentSchema(
            "file.write",
            (
                ArgumentField("path", str, required=True),
                ArgumentField("content", (str, dict), required=True),
            ),
        )
    )
    schema_registry.register(
        ArgumentSchema("directory.list", (ArgumentField("path", str, required=True),))
    )

    planner = Planner(
        tool_registry=registry,
        tool_selector=selector,
        schema_registry=schema_registry,
        argument_validator=ArgumentValidator(schema_registry),
    )
    return planner, registry


def test_simple_objective_generates_one_executable_step_with_arguments() -> None:
    planner, _ = _planner()

    plan = planner.create_execution_plan("Lee el archivo README.md")

    assert plan.ordered_steps[0].id == "step_1"
    assert plan.ordered_steps[0].tool == "read_file"
    assert dict(plan.ordered_steps[0].arguments) == {"path": "README.md"}
    assert plan.required_tools == ("read_file",)
    assert plan.estimated_steps == 1
    assert plan.requires_confirmation is False


def test_multi_step_objective_generates_ordered_dependencies_and_ref() -> None:
    planner, _ = _planner()

    plan = planner.create_execution_plan("Lee README.md y copia su contenido en resumen.txt")

    assert [step.id for step in plan.ordered_steps] == ["step_1", "step_2"]
    assert [step.tool for step in plan.ordered_steps] == ["read_file", "write_file"]
    assert plan.ordered_steps[1].dependencies == ("step_1",)
    assert dict(plan.ordered_steps[1].arguments) == {
        "path": "resumen.txt",
        "content": {"$ref": "steps.step_1.output"},
    }
    assert plan.required_tools == ("read_file", "write_file")
    assert plan.requires_confirmation is True


def test_template_is_generated_when_literal_text_must_be_combined_with_output() -> None:
    planner, _ = _planner()

    plan = planner.create_execution_plan(
        'Lee README.md y guarda mensaje "Archivo: " con su contenido en resumen.txt'
    )

    assert dict(plan.ordered_steps[1].arguments) == {
        "path": "resumen.txt",
        "content": {"$template": "Archivo: {{steps.step_1.output}}"},
    }


def test_ref_is_preferred_when_type_should_be_preserved() -> None:
    planner, _ = _planner()

    plan = planner.create_execution_plan("Lee README.md y copia su contenido en resumen.txt")

    assert dict(plan.ordered_steps[1].arguments)["content"] == {
        "$ref": "steps.step_1.output"
    }


def test_required_tools_are_deduplicated_and_initial_statuses_are_valid() -> None:
    planner, _ = _planner()

    plan = planner.create_execution_plan("Lee README.md y copia su contenido en resumen.txt")

    assert plan.required_tools == ("read_file", "write_file")
    assert plan.status == "planned"
    assert all(step.status == "pending" for step in plan.ordered_steps)


def test_empty_objective_returns_structured_error() -> None:
    planner, _ = _planner()

    result = planner.generate_execution_plan("   ")

    assert result.success is False
    assert result.error_code == PlannerErrorCode.EMPTY_OBJECTIVE.value
    assert result.plan is not None


def test_missing_required_argument_is_reported_without_replacing_validator() -> None:
    planner, _ = _planner()

    result = planner.generate_execution_plan("Lee este archivo")

    assert result.success is False
    assert result.error_code == PlannerErrorCode.INSUFFICIENT_INFORMATION.value
    assert any("MISSING_REQUIRED_ARGUMENT: file.read.path" in error for error in result.errors)


def test_invalid_json_response_is_reported() -> None:
    planner = Planner(plan_response_provider=lambda _goal, _catalog: "{bad json")

    result = planner.generate_execution_plan("Lee README.md")

    assert result.success is False
    assert result.error_code == PlannerErrorCode.PLAN_PARSE_ERROR.value
    assert result.raw_response == "{bad json"


def test_incomplete_structured_response_is_rejected() -> None:
    planner = Planner(plan_response_provider=lambda _goal, _catalog: '{"goal":"x"}')

    result = planner.generate_execution_plan("Lee README.md")

    assert result.success is False
    assert result.error_code == PlannerErrorCode.INVALID_PLAN_RESPONSE.value


def test_wrong_structured_response_types_are_rejected() -> None:
    planner = Planner(
        plan_response_provider=lambda _goal, _catalog: '{"goal":"x","steps":"bad"}'
    )

    result = planner.generate_execution_plan("Lee README.md")

    assert result.success is False
    assert result.error_code == PlannerErrorCode.INVALID_PLAN_RESPONSE.value


def test_unknown_tool_in_structured_response_is_rejected() -> None:
    planner = Planner(
        plan_response_provider=lambda _goal, _catalog: (
            '{"goal":"x","steps":[{"id":1,"description":"Delete","tool":"delete_file",'
            '"arguments":{},"dependencies":[]}],"risks":[],"requires_confirmation":false}'
        )
    )

    result = planner.generate_execution_plan("Borra README.md")

    assert result.success is False
    assert result.error_code == PlannerErrorCode.UNKNOWN_TOOL.value


def test_provider_plan_with_invalid_reference_is_left_for_validator_to_reject() -> None:
    planner, _ = _planner()
    planner = Planner(
        tool_registry=planner._tool_registry,  # type: ignore[attr-defined]
        tool_selector=planner._tool_selector,  # type: ignore[attr-defined]
        schema_registry=planner._schema_registry,  # type: ignore[attr-defined]
        argument_validator=planner._argument_validator,  # type: ignore[attr-defined]
        plan_response_provider=lambda _goal, _catalog: (
            '{"goal":"x","steps":['
            '{"id":"step_1","description":"Read","tool":"read_file","arguments":{"path":"a.md"},"dependencies":[]},'
            '{"id":"step_2","description":"Write","tool":"write_file",'
            '"arguments":{"path":"b.md","content":{"$ref":"steps.missing.output"}},'
            '"dependencies":["step_1"]}],"risks":[],"requires_confirmation":true}'
        ),
    )

    result = planner.generate_execution_plan("x")

    assert result.success is True
    assert result.plan is not None
    validation = ExecutionPlanValidator().validate(result.plan)
    assert validation.is_valid is False
    assert "Step 'step_2' references unknown step 'missing'." in validation.errors


def test_dangerous_step_requires_confirmation() -> None:
    planner, _ = _planner()

    plan = planner.create_execution_plan("Escribe hola en resumen.txt")

    assert plan.requires_confirmation is True
    assert "write_file" in plan.required_tools


def test_safe_step_does_not_require_confirmation() -> None:
    planner, _ = _planner()

    plan = planner.create_execution_plan("Lee README.md")

    assert plan.requires_confirmation is False


def test_planner_does_not_execute_tools_or_executor() -> None:
    calls: list[str] = []
    planner, _ = _planner(calls)

    planner.create_execution_plan("Lee README.md y copia su contenido en resumen.txt")

    assert calls == []


def test_planner_source_does_not_use_eval_or_exec() -> None:
    source = inspect.getsource(Planner)

    assert "eval(" not in source
    assert "exec(" not in source


def test_tool_catalog_is_generated_from_registry_and_schema_without_mutation() -> None:
    planner, registry = _planner()
    before = registry.list()

    catalog = planner.tool_catalog()

    assert registry.list() == before
    read_descriptor = next(item for item in catalog if item.name == "read_file")
    assert read_descriptor.argument_names == ("path",)
    assert read_descriptor.required_arguments == ("path",)
    assert read_descriptor.output_description


def test_valid_generated_plan_passes_validator() -> None:
    planner, _ = _planner()
    plan = planner.create_execution_plan("Lee README.md y copia su contenido en resumen.txt")

    result = ExecutionPlanValidator().validate(plan)

    assert result.is_valid is True


def test_minimal_functional_generation_and_validation_without_execution() -> None:
    planner, _ = _planner()

    plan = planner.create_execution_plan("Lee README.md")
    validation = ExecutionPlanValidator().validate(plan)

    assert isinstance(plan, ExecutionPlan)
    assert validation.is_valid is True


def test_generated_plan_executes_with_fake_tools_after_validation() -> None:
    calls: list[str] = []
    planner, registry = _planner(calls)
    plan = planner.create_execution_plan("Lee README.md y copia su contenido en resumen.txt")
    validation = ExecutionPlanValidator().validate(plan)

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        validation,
        confirmation_granted=True,
    )

    assert result.success is True
    assert calls == ["read_file", "write_file"]
    assert registry.get("write_file").contexts[0].parameters == {  # type: ignore[attr-defined]
        "path": "resumen.txt",
        "content": "contenido",
    }

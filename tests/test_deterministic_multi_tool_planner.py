from __future__ import annotations

from typing import Any

from core.deterministic_multi_tool_planner import (
    DeterministicMultiToolPlanner,
    MultiToolPlanningResult,
)
from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_plan_validator import ExecutionPlanValidator
from core.planner import Planner
from core.step_output_reference import StepOutputReference
from tools.argument_schema import ArgumentField, ArgumentSchema, ArgumentSchemaRegistry
from tools.base_tool import BaseTool
from tools.intent_selector import ToolIntentRegistry, ToolSelector
from tools.registry import ToolRegistry
from tools.semantic_catalog import SemanticToolCatalog, SemanticToolDescriptor
from tools.tool_context import ToolContext


class PlanningFakeTool(BaseTool):
    def __init__(
        self,
        name: str,
        *,
        requires_confirmation: bool = False,
    ) -> None:
        self._name = name
        self._requires_confirmation = requires_confirmation
        self.executed = False

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
        self.executed = True
        raise AssertionError("multi-tool planning must not execute tools")


class EmptySelector(ToolSelector):
    def __init__(self) -> None:
        registry = ToolRegistry()
        intent_registry = ToolIntentRegistry()
        super().__init__(registry, intent_registry)

    def select_from_catalog(self, query: str, catalog: SemanticToolCatalog, *, top_k: int = 5):
        from tools.intent_selector import ToolSelectionResult

        return ToolSelectionResult(
            success=True,
            query=query,
            normalized_query=query.lower(),
            candidates=(),
            selected_tool=None,
            requires_clarification=True,
        )


def _descriptor(
    name: str,
    *,
    capabilities: tuple[str, ...],
    intents: tuple[str, ...],
    examples: tuple[str, ...] = (),
    required_arguments: tuple[str, ...] = (),
    optional_arguments: tuple[str, ...] = (),
    output_fields: tuple[str, ...] = (),
    risk_level: str = "low",
    requires_confirmation: bool = False,
) -> SemanticToolDescriptor:
    return SemanticToolDescriptor(
        name=name,
        description=f"Descriptor for {name}.",
        capabilities=capabilities,
        supported_intents=intents,
        input_description="Structured input.",
        required_arguments=required_arguments,
        optional_arguments=optional_arguments,
        output_description="Known output contract.",
        output_fields=output_fields,
        dangerous=requires_confirmation,
        risk_level=risk_level,
        risk_reasons=("requires confirmation",) if requires_confirmation else (),
        requires_confirmation=requires_confirmation,
        preconditions=tuple(f"{argument} must be provided" for argument in required_arguments),
        limitations=(),
        negative_examples=(),
        compatible_tools=(),
        tags=("filesystem", "process", "archivo"),
        positive_examples=examples,
        category="fake",
        technical_arguments=required_arguments + optional_arguments,
    )


def _catalog(*, include_process: bool = True, include_write: bool = True, find_has_pid: bool = True) -> SemanticToolCatalog:
    descriptors = {
        "read_file": _descriptor(
            "read_file",
            capabilities=("read_file",),
            intents=("lee un archivo local",),
            examples=("lee C:/Temp/origen.txt",),
            required_arguments=("path",),
        ),
        "list_directory": _descriptor(
            "list_directory",
            capabilities=("list_directory",),
            intents=("lista archivos en un directorio",),
            examples=("lista C:/Temp y lee notas.txt",),
            optional_arguments=("path",),
        ),
    }
    if include_write:
        descriptors["write_file"] = _descriptor(
            "write_file",
            capabilities=("write_file",),
            intents=("guarda contenido en archivo", "copia contenido en archivo"),
            examples=("lee C:/Temp/origen.txt y guarda el contenido en C:/Temp/copia.txt",),
            required_arguments=("path", "content"),
            risk_level="medium",
            requires_confirmation=True,
        )
    if include_process:
        descriptors["find_process"] = _descriptor(
            "find_process",
            capabilities=("find_process",),
            intents=("encuentra proceso",),
            examples=("encuentra el proceso notepad y terminalo",),
            required_arguments=("query",),
            output_fields=("pid",) if find_has_pid else (),
        )
        descriptors["terminate_process"] = _descriptor(
            "terminate_process",
            capabilities=("terminate_process",),
            intents=("termina proceso",),
            examples=("encuentra el proceso notepad y terminalo",),
            required_arguments=("pid",),
            risk_level="high",
            requires_confirmation=True,
        )
    return SemanticToolCatalog(descriptors)


def _registry_and_selector(*, include_process: bool = True, include_write: bool = True) -> tuple[ToolRegistry, ToolSelector]:
    registry = ToolRegistry()
    registry.register(PlanningFakeTool("read_file"))
    registry.register(PlanningFakeTool("list_directory"))
    if include_write:
        registry.register(PlanningFakeTool("write_file", requires_confirmation=True))
    if include_process:
        registry.register(PlanningFakeTool("find_process"))
        registry.register(PlanningFakeTool("terminate_process", requires_confirmation=True))
    intent_registry = ToolIntentRegistry()
    for action, tool in (
        ("file.read", "read_file"),
        ("file.write", "write_file"),
        ("directory.list", "list_directory"),
        ("process.find", "find_process"),
        ("process.terminate", "terminate_process"),
    ):
        if registry.exists(tool):
            intent_registry.register(action, tool)
    return registry, ToolSelector(registry, intent_registry)


def _planner() -> DeterministicMultiToolPlanner:
    return DeterministicMultiToolPlanner()


def test_read_then_write_complete_pattern_builds_valid_plan() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan(
        "lee C:/Temp/origen.txt y guarda el contenido en C:/Temp/copia.txt",
        _catalog(),
        selector,
    )

    assert result.success is True
    assert result.handled is True
    assert result.matched_pattern == "read_then_write"
    assert result.selected_tools == ("read_file", "write_file")
    assert result.plan is not None
    assert [step.id for step in result.plan.ordered_steps] == ["step_1", "step_2"]
    assert [step.tool for step in result.plan.ordered_steps] == ["read_file", "write_file"]
    assert result.plan.ordered_steps[1].dependencies == ("step_1",)
    assert dict(result.plan.ordered_steps[1].arguments)["content"] == (
        StepOutputReference("step_1")
    )
    assert result.plan.required_tools == ("read_file", "write_file")
    assert result.plan.estimated_steps == 2
    assert result.plan.requires_confirmation is True
    assert ExecutionPlanValidator().validate(result.plan).is_valid is True


def test_read_write_verify_builds_three_dependent_steps() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan(
        (
            "lee C:/Temp/origen.txt, guarda el contenido en "
            "C:/Temp/copia.txt y comprueba que se escribió"
        ),
        _catalog(),
        selector,
    )

    assert result.success is True
    assert result.plan is not None
    assert [step.tool for step in result.plan.ordered_steps] == [
        "read_file",
        "write_file",
        "read_file",
    ]
    assert result.plan.ordered_steps[1].dependencies == ("step_1",)
    assert result.plan.ordered_steps[2].dependencies == ("step_2",)
    assert result.plan.ordered_steps[1].arguments["content"] == (
        StepOutputReference("step_1")
    )
    assert result.plan.ordered_steps[2].arguments["path"] == (
        "C:/Temp/copia.txt"
    )
    assert result.plan.estimated_steps == 3
    assert [
        criterion.criterion_id
        for criterion in result.plan.acceptance_criteria
    ] == [
        "expected_step_count",
        "source_read_tool_used",
        "write_tool_used",
        "verification_read_tool_used",
        "resource_exists",
        "resource_readable",
        "resource_content_equals_source",
        "verification_output_equals_source",
        "no_pending_confirmations",
        "no_critical_failures",
    ]
    assert ExecutionPlanValidator().validate(result.plan).is_valid is True


def test_list_then_read_complete_pattern_builds_valid_plan() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan("lista C:/Temp y lee notas.txt", _catalog(), selector)

    assert result.success is True
    assert result.matched_pattern == "list_then_read"
    assert result.plan is not None
    assert [step.tool for step in result.plan.ordered_steps] == ["list_directory", "read_file"]
    assert result.plan.ordered_steps[1].dependencies == ("step_1",)
    assert dict(result.plan.ordered_steps[0].arguments) == {"path": "C:/Temp"}
    assert dict(result.plan.ordered_steps[1].arguments) == {"path": "C:/Temp/notas.txt"}
    assert ExecutionPlanValidator().validate(result.plan).is_valid is True


def test_process_find_then_terminate_pattern_when_tools_exist() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan("encuentra el proceso notepad y terminalo", _catalog(), selector)

    assert result.success is True
    assert result.matched_pattern == "process_find_then_terminate"
    assert result.plan is not None
    assert [step.tool for step in result.plan.ordered_steps] == ["find_process", "terminate_process"]
    assert dict(result.plan.ordered_steps[1].arguments) == {"pid": {"$ref": "steps.step_1.output.pid"}}
    assert result.plan.requires_confirmation is True
    assert ExecutionPlanValidator().validate(result.plan).is_valid is True


def test_unsupported_objective_is_not_handled() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan("explicame que es un archivo", _catalog(), selector)

    assert result.handled is False
    assert result.plan is None
    assert result.selected_tools == ()


def test_recognized_pattern_with_incomplete_data_requires_clarification() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan("lee un archivo y guardalo", _catalog(), selector)

    assert result.handled is True
    assert result.success is False
    assert result.requires_clarification is True
    assert "source_path" in result.missing_information
    assert "destination_path" in result.missing_information
    assert result.plan is None


def test_source_missing_is_reported() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan("lee y guarda el contenido en C:/Temp/copia.txt", _catalog(), selector)

    assert result.success is False
    assert "source_path" in result.missing_information


def test_destination_missing_is_reported() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan("lee C:/Temp/origen.txt y guarda el contenido", _catalog(), selector)

    assert result.success is False
    assert "destination_path" in result.missing_information


def test_required_tool_unavailable_blocks_plan() -> None:
    _registry, selector = _registry_and_selector(include_write=False)
    result = _planner().plan(
        "lee C:/Temp/origen.txt y guarda el contenido en C:/Temp/copia.txt",
        _catalog(include_write=False),
        selector,
    )

    assert result.success is False
    assert result.error_code == "REQUIRED_TOOL_UNAVAILABLE"
    assert "write_file" in result.missing_information


def test_unknown_output_field_blocks_process_pattern() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan(
        "encuentra el proceso notepad y terminalo",
        _catalog(find_has_pid=False),
        selector,
    )

    assert result.success is False
    assert result.error_code == "OUTPUT_CONTRACT_UNKNOWN"
    assert result.plan is None


def test_generated_ids_dependencies_ref_required_tools_and_estimated_steps() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan(
        "lee C:/Temp/origen.txt y guarda el contenido en C:/Temp/copia.txt",
        _catalog(),
        selector,
    )

    assert result.plan is not None
    assert [step.id for step in result.plan.ordered_steps] == ["step_1", "step_2"]
    assert result.plan.ordered_steps[1].dependencies == ("step_1",)
    assert dict(result.plan.ordered_steps[1].arguments)["content"] == (
        StepOutputReference("step_1")
    )
    assert result.plan.required_tools == ("read_file", "write_file")
    assert result.plan.estimated_steps == 2


def test_confirmation_is_inherited_from_dangerous_tool() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan(
        "lee C:/Temp/origen.txt y guarda el contenido en C:/Temp/copia.txt",
        _catalog(),
        selector,
    )

    assert result.plan is not None
    assert result.plan.requires_confirmation is True
    assert any("requires confirmation" in risk for risk in result.plan.detected_risks)


def test_invalid_plan_is_rejected_by_validator() -> None:
    _registry, selector = _registry_and_selector()
    broken_catalog = SemanticToolCatalog(
        {
            "read_file": _descriptor(
                "read_file",
                capabilities=("read_file",),
                intents=("lee un archivo local",),
                required_arguments=("path",),
            ),
            "write_file": _descriptor(
                "write_file",
                capabilities=("write_file",),
                intents=("guarda contenido en archivo",),
                required_arguments=("path", "content"),
                risk_level="low",
                requires_confirmation=False,
            ),
        }
    )

    result = _planner().plan(
        "lee C:/Temp/origen.txt y guarda el contenido en C:/Temp/copia.txt",
        broken_catalog,
        selector,
    )

    assert result.success is False
    assert result.error_code == "INVALID_MULTI_TOOL_PLAN"


def test_selector_confirms_candidates_for_supported_pattern() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan(
        "lee C:/Temp/origen.txt y guarda el contenido en C:/Temp/copia.txt",
        _catalog(),
        selector,
    )

    assert result.success is True
    assert result.warnings == ()


def test_selector_discrepancy_blocks_plan() -> None:
    result = _planner().plan(
        "lee C:/Temp/origen.txt y guarda el contenido en C:/Temp/copia.txt",
        _catalog(),
        EmptySelector(),
    )

    assert result.success is False
    assert result.error_code == "MULTI_TOOL_PLANNING_FAILED"
    assert "ToolSelector did not return required pattern tools" in result.errors[0]


def test_planning_does_not_execute_tools_or_executor() -> None:
    registry, selector = _registry_and_selector()

    result = _planner().plan(
        "lee C:/Temp/origen.txt y guarda el contenido en C:/Temp/copia.txt",
        _catalog(),
        selector,
    )

    assert result.success is True
    assert all(tool.executed is False for tool in registry.tools.values())  # type: ignore[attr-defined]


def test_executor_is_not_called_by_planning() -> None:
    registry, selector = _registry_and_selector()
    executor = ExecutionPlanExecutor(registry)

    result = _planner().plan("lista C:/Temp y lee notas.txt", _catalog(), selector)

    assert result.success is True
    assert executor is not None
    assert all(tool.executed is False for tool in registry.tools.values())  # type: ignore[attr-defined]


def test_planning_does_not_call_models_or_network() -> None:
    _registry, selector = _registry_and_selector()
    result = _planner().plan("lista C:/Temp y lee notas.txt", _catalog(), selector)

    assert isinstance(result, MultiToolPlanningResult)
    assert result.success is True


def test_planning_does_not_mutate_catalog() -> None:
    _registry, selector = _registry_and_selector()
    catalog = _catalog()
    before = catalog.to_json()

    _planner().plan("lista C:/Temp y lee notas.txt", catalog, selector)

    assert catalog.to_json() == before


def test_compatibility_with_semantic_selector_phase() -> None:
    _registry, selector = _registry_and_selector()
    result = selector.select_from_catalog("lista C:/Temp y lee notas.txt", _catalog())

    assert result.candidates


def test_planner_previous_behavior_without_multi_tool_planner() -> None:
    plan = Planner().create_execution_plan("Lee README.md")

    assert plan.required_tools == ("read_file",)


def test_planner_uses_optional_multi_tool_planner_for_handled_pattern() -> None:
    registry, selector = _registry_and_selector()
    catalog = _catalog()
    planner = Planner(
        tool_registry=registry,
        tool_selector=selector,
        semantic_tool_catalog=catalog,
        multi_tool_planner=_planner(),
    )

    result = planner.generate_execution_plan(
        "lee C:/Temp/origen.txt y guarda el contenido en C:/Temp/copia.txt"
    )

    assert result.success is True
    assert result.plan is not None
    assert [step.tool for step in result.plan.ordered_steps] == ["read_file", "write_file"]


def test_planner_falls_back_when_multi_tool_pattern_not_handled() -> None:
    registry, selector = _registry_and_selector()
    planner = Planner(
        tool_registry=registry,
        tool_selector=selector,
        semantic_tool_catalog=_catalog(),
        multi_tool_planner=_planner(),
    )

    result = planner.generate_execution_plan("Lee README.md")

    assert result.plan is not None
    assert result.plan.required_tools == ("read_file",)

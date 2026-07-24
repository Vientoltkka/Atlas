from __future__ import annotations

from pathlib import Path

import pytest

from bootstrap.bootstrap import Bootstrap
from bootstrap.execution_plan_library import build_core_execution_plan_library
from core.capability_execution_service import CapabilityExecutionStatus
from core.execution_context import ExecutionContext
from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
from core.execution_plan_output import ExecutionPlanOutput
from core.execution_plan_registry import ExecutionPlanReference, ExecutionPlanRegistry
from core.execution_plan_validator import ExecutionPlanValidator
from core.execution_variable_reference import ExecutionVariableReference
from core.multi_capability_planner import (
    MultiCapabilityPlanner,
    MultiCapabilityPlanningRequest,
    MultiCapabilityPlanningStatus,
)
from core.planner import ExecutionPlan, ExecutionStep
from core.step_output_reference import StepOutputReference


PROJECT_TREE_CAPABILITY_ID = "workflow.atlas.core.project.tree.show.1.0"
DIRECTORY_LIST_CAPABILITY_ID = "workflow.atlas.core.directory.list.1.0"


def _workflow(
    plan_id: str,
    *,
    input_name: str,
    outputs: tuple[str, ...],
    tool: str = "list_directory",
) -> WorkflowDefinition:
    step_id = f"run_{plan_id.replace('.', '_')}"
    output_definition: dict[str, object] = {}
    for output in outputs:
        output_definition[output] = (
            ExecutionVariableReference(input_name)
            if output == input_name
            else StepOutputReference(step_id)
        )
    return WorkflowDefinition(
        reference=ExecutionPlanReference(plan_id, "1.0"),
        plan=ExecutionPlan(
            goal=f"Run {plan_id}.",
            ordered_steps=(
                ExecutionStep(
                    step_id,
                    f"Run {plan_id}.",
                    tool,
                    arguments={"path": ExecutionVariableReference(input_name)},
                ),
            ),
            estimated_steps=1,
            required_tools=(tool,),
            detected_risks=(),
            requires_confirmation=False,
            output=ExecutionPlanOutput(output_definition),
        ),
        title=plan_id.replace(".", " ").title(),
        description="Read-only workflow for multi-capability planning tests.",
        category="project.analysis",
        tags=("filesystem", "read_only"),
    )


def _library(*workflows: WorkflowDefinition) -> ExecutionPlanLibrary:
    return ExecutionPlanLibrary("atlas.core", workflows, version="1.0")


def test_plans_one_step_from_available_input_to_required_output() -> None:
    library = build_core_execution_plan_library()
    assert library is not None

    result = MultiCapabilityPlanner(execution_plan_libraries=(library,)).plan(
        MultiCapabilityPlanningRequest(
            initial_inputs=("directory_path",),
            required_outputs=("entries",),
        )
    )

    assert result.status is MultiCapabilityPlanningStatus.PLANNED
    assert result.selected_capability_ids == (DIRECTORY_LIST_CAPABILITY_ID,)
    assert result.plan is not None
    assert len(result.plan.ordered_steps) == 1


def test_plans_two_steps_when_first_output_satisfies_second_input() -> None:
    library = build_core_execution_plan_library()
    assert library is not None

    result = MultiCapabilityPlanner(execution_plan_libraries=(library,)).plan(
        MultiCapabilityPlanningRequest(
            initial_inputs=("path",),
            required_outputs=("entries",),
        )
    )

    assert result.status is MultiCapabilityPlanningStatus.PLANNED
    assert result.selected_capability_ids == (PROJECT_TREE_CAPABILITY_ID, DIRECTORY_LIST_CAPABILITY_ID)
    assert result.plan is not None
    assert tuple(step.subplan_ref for step in result.plan.ordered_steps) == (
        ExecutionPlanReference("project.tree.show", "1.0"),
        ExecutionPlanReference("directory.list", "1.0"),
    )


def test_reuses_variables_through_execution_variable_references() -> None:
    library = build_core_execution_plan_library()
    assert library is not None

    result = MultiCapabilityPlanner(execution_plan_libraries=(library,)).plan(
        MultiCapabilityPlanningRequest(
            initial_inputs=("path",),
            required_outputs=("entries",),
        )
    )

    assert result.plan is not None
    first, second = result.plan.ordered_steps
    assert first.output_binding is not None
    assert first.output_binding.variable_name == "capability_1_output"
    assert second.arguments["directory_path"] == ExecutionVariableReference(
        "capability_1_output",
        ("directory_path",),
    )


def test_dependency_not_satisfied_is_structured_failure() -> None:
    library = build_core_execution_plan_library()
    assert library is not None

    result = MultiCapabilityPlanner(execution_plan_libraries=(library,)).plan(
        MultiCapabilityPlanningRequest(
            initial_inputs=("path",),
            required_outputs=("missing_output",),
        )
    )

    assert result.status is MultiCapabilityPlanningStatus.IMPOSSIBLE_DEPENDENCY
    assert result.plan is None


def test_detects_capability_dependency_cycle() -> None:
    library = _library(
        _workflow("cycle.a", input_name="b", outputs=("a",)),
        _workflow("cycle.b", input_name="a", outputs=("b",)),
    )

    result = MultiCapabilityPlanner(execution_plan_libraries=(library,)).plan(
        MultiCapabilityPlanningRequest(
            initial_inputs=(),
            required_outputs=("a",),
        )
    )

    assert result.status is MultiCapabilityPlanningStatus.CYCLE_DETECTED


def test_rejects_ambiguous_graph() -> None:
    library = _library(
        _workflow("source.a", input_name="seed", outputs=("middle",)),
        _workflow("source.b", input_name="seed", outputs=("middle",)),
        _workflow("target", input_name="middle", outputs=("done",)),
    )

    result = MultiCapabilityPlanner(execution_plan_libraries=(library,)).plan(
        MultiCapabilityPlanningRequest(
            initial_inputs=("seed",),
            required_outputs=("done",),
        )
    )

    assert result.status is MultiCapabilityPlanningStatus.AMBIGUOUS_GRAPH


def test_order_is_deterministic() -> None:
    library = build_core_execution_plan_library()
    assert library is not None
    planner = MultiCapabilityPlanner(execution_plan_libraries=(library,))

    first = planner.plan(MultiCapabilityPlanningRequest(initial_inputs=("path",), required_outputs=("entries",)))
    second = planner.plan(MultiCapabilityPlanningRequest(initial_inputs=("path",), required_outputs=("entries",)))

    assert first.selected_capability_ids == second.selected_capability_ids
    assert first.plan is not None and second.plan is not None
    assert tuple(step.id for step in first.plan.ordered_steps) == tuple(step.id for step in second.plan.ordered_steps)


def test_composed_plan_executes_e2e_with_real_tools(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    library = build_core_execution_plan_library()
    assert library is not None
    registry = ExecutionPlanRegistry()
    library.install(registry)
    result = MultiCapabilityPlanner(execution_plan_libraries=(library,)).plan(
        MultiCapabilityPlanningRequest(
            initial_inputs=("path",),
            required_outputs=("entries",),
        )
    )
    assert result.plan is not None
    tool_registry = Bootstrap.build_tool_registry()
    validator = ExecutionPlanValidator(tool_registry, plan_registry=registry)
    validation = validator.validate(result.plan)
    execution_context = ExecutionContext(initial_variables={"path": str(tmp_path)})

    execution = ExecutionPlanExecutor(tool_registry, plan_registry=registry).execute(
        result.plan,
        validation,
        execution_context=execution_context,
    )

    assert validation.is_valid
    assert execution.success is True
    assert execution.output == {"entries": ["README.md", "pkg"]}
    assert execution.step_results[0].tool_name is None
    assert execution.step_results[1].tool_name is None


def test_core_capability_execution_status_remains_available() -> None:
    assert CapabilityExecutionStatus.COMPLETED.value == "completed"

from __future__ import annotations

from dataclasses import replace

import pytest

from core.execution_dependency_checker import ExecutionDependencyNotFoundError
from core.execution_plan_topology import (
    ExecutionPlanCycleError,
    ExecutionPlanTopologicalSorter,
    ExecutionPlanTopologyValidationError,
)
from core.execution_plan_validator import ExecutionPlanValidator, plan_signature
from core.planner import ExecutionPlan, ExecutionStep


def _step(
    step_id: str,
    *,
    depends_on: tuple[str, ...] = (),
) -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        description=f"Execute {step_id}.",
        tool="safe_tool",
        depends_on=depends_on,
        arguments={},
    )


def _plan(steps: tuple[ExecutionStep, ...]) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Sort execution graph.",
        ordered_steps=steps,
        estimated_steps=len(steps),
        required_tools=("safe_tool",),
        detected_risks=(),
        requires_confirmation=False,
    )


def _ordered_ids(plan: ExecutionPlan) -> tuple[str, ...]:
    return ExecutionPlanTopologicalSorter().sort(plan).ordered_step_ids


def test_topological_sort_keeps_independent_original_order() -> None:
    plan = _plan((_step("b"), _step("a"), _step("c")))

    topology = ExecutionPlanTopologicalSorter().sort(plan)

    assert topology.ordered_step_ids == ("b", "a", "c")
    assert topology.original_step_ids == ("b", "a", "c")
    assert topology.reordered is False
    assert topology.root_step_ids == ("b", "a", "c")
    assert topology.leaf_step_ids == ("b", "a", "c")


def test_topological_sort_allows_future_physical_dependencies() -> None:
    plan = _plan(
        (
            _step("consume", depends_on=("read",)),
            _step("read"),
        )
    )

    topology = ExecutionPlanTopologicalSorter().sort(plan)

    assert topology.ordered_step_ids == ("read", "consume")
    assert topology.original_step_ids == ("consume", "read")
    assert topology.reordered is True
    assert topology.ordered_steps(plan) == (plan.ordered_steps[1], plan.ordered_steps[0])
    assert topology.comes_before("read", "consume") is True


def test_topological_sort_uses_original_index_as_tie_breaker() -> None:
    plan = _plan(
        (
            _step("write", depends_on=("read_a", "read_b")),
            _step("read_b"),
            _step("read_a"),
            _step("notify", depends_on=("read_a",)),
        )
    )

    assert _ordered_ids(plan) == ("read_b", "read_a", "write", "notify")


def test_topological_sort_handles_disconnected_transitive_graph() -> None:
    plan = _plan(
        (
            _step("publish", depends_on=("package",)),
            _step("lint"),
            _step("package", depends_on=("build",)),
            _step("build"),
            _step("docs"),
        )
    )

    topology = ExecutionPlanTopologicalSorter().sort(plan)

    assert topology.ordered_step_ids == ("lint", "build", "package", "publish", "docs")
    assert topology.dependency_count == 2
    assert topology.root_step_ids == ("lint", "build", "docs")
    assert topology.leaf_step_ids == ("publish", "lint", "docs")


def test_topological_sort_rejects_missing_dependency_duplicate_ids_and_cycles() -> None:
    with pytest.raises(ExecutionDependencyNotFoundError):
        _ordered_ids(_plan((_step("a", depends_on=("missing",)),)))

    with pytest.raises(ExecutionPlanTopologyValidationError):
        _ordered_ids(_plan((_step("a"), _step("a"))))

    with pytest.raises(ExecutionPlanCycleError):
        _ordered_ids(
            _plan(
                (
                    _step("a", depends_on=("c",)),
                    _step("b", depends_on=("a",)),
                    _step("c", depends_on=("b",)),
                )
            )
        )


def test_sorter_does_not_mutate_plan_or_signature() -> None:
    plan = _plan(
        (
            _step("second", depends_on=("first",)),
            _step("first"),
        )
    )
    before_steps = plan.ordered_steps
    before_signature = plan_signature(plan)

    assert _ordered_ids(plan) == ("first", "second")

    assert plan.ordered_steps == before_steps
    assert plan_signature(plan) == before_signature
    physically_reordered = replace(plan, ordered_steps=tuple(reversed(plan.ordered_steps)))
    assert plan_signature(plan) != plan_signature(physically_reordered)


def test_validator_accepts_future_dependency_when_graph_is_acyclic() -> None:
    plan = _plan((_step("consume", depends_on=("read",)), _step("read")))

    result = ExecutionPlanValidator().validate(plan)

    assert result.is_valid is True
    assert result.errors == []

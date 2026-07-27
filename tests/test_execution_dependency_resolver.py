from __future__ import annotations

import pytest

from core.execution_dependency_resolver import ExecutionDependencyResolver
from core.execution_plan_topology import ExecutionPlanCycleError
from core.execution_dependency_checker import ExecutionDependencyNotFoundError
from core.planner import ExecutionPlan, ExecutionStep


def _step(
    step_id: str,
    dependencies: tuple[str, ...] = (),
) -> ExecutionStep:
    return ExecutionStep(
        step_id,
        f"step {step_id}",
        "tool",
        dependencies=dependencies,
    )


def _plan(
    steps: tuple[ExecutionStep, ...],
) -> ExecutionPlan:
    return ExecutionPlan(
        goal="graph plan",
        ordered_steps=steps,
        estimated_steps=len(steps),
        required_tools=("tool",),
        detected_risks=(),
        requires_confirmation=False,
    )


def test_linear_plan_returns_declared_order_when_all_steps_are_roots() -> None:
    plan = _plan((_step("a"), _step("b"), _step("c")))

    ready = ExecutionDependencyResolver().get_ready_steps(
        plan,
        completed_step_ids=(),
    )

    assert tuple(step.id for step in ready) == ("a", "b", "c")


def test_simple_dependency_waits_for_completed_parent() -> None:
    plan = _plan((_step("a"), _step("b", ("a",))))
    resolver = ExecutionDependencyResolver()

    initial = resolver.get_ready_steps(plan, completed_step_ids=())
    after_a = resolver.get_ready_steps(plan, completed_step_ids=("a",))

    assert tuple(step.id for step in initial) == ("a",)
    assert tuple(step.id for step in after_a) == ("b",)


def test_multiple_dependencies_wait_for_all_parents() -> None:
    plan = _plan(
        (
            _step("a"),
            _step("b", ("a",)),
            _step("c", ("a",)),
            _step("d", ("b", "c")),
        )
    )
    resolver = ExecutionDependencyResolver()

    after_a = resolver.get_ready_steps(plan, completed_step_ids=("a",))
    after_b = resolver.get_ready_steps(plan, completed_step_ids=("a", "b"))
    after_c = resolver.get_ready_steps(plan, completed_step_ids=("a", "b", "c"))

    assert tuple(step.id for step in after_a) == ("b", "c")
    assert tuple(step.id for step in after_b) == ("c",)
    assert tuple(step.id for step in after_c) == ("d",)


def test_ready_steps_are_deterministic_in_original_plan_order() -> None:
    plan = _plan((_step("c"), _step("a"), _step("b")))

    ready = ExecutionDependencyResolver().get_ready_steps(
        plan,
        completed_step_ids=(),
    )

    assert tuple(step.id for step in ready) == ("c", "a", "b")


def test_failed_step_blocks_direct_and_indirect_dependents_only() -> None:
    plan = _plan(
        (
            _step("a"),
            _step("b", ("a",)),
            _step("c", ("b",)),
            _step("independent"),
        )
    )

    resolved = ExecutionDependencyResolver().resolve(
        plan,
        completed_step_ids=(),
        failed_step_ids=("a",),
    )

    assert resolved.blocked_step_ids == ("b", "c")
    assert tuple(step.id for step in resolved.ready_steps) == ("independent",)


def test_missing_dependency_is_rejected_before_readiness() -> None:
    plan = _plan((_step("a", ("missing",)),))

    with pytest.raises(ExecutionDependencyNotFoundError):
        ExecutionDependencyResolver().get_ready_steps(plan, completed_step_ids=())


def test_cycle_is_rejected_before_readiness() -> None:
    plan = _plan((_step("a", ("b",)), _step("b", ("a",))))

    with pytest.raises(ExecutionPlanCycleError):
        ExecutionDependencyResolver().get_ready_steps(plan, completed_step_ids=())

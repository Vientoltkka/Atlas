"""Explicit factory for capability-based planning."""

from __future__ import annotations

from collections.abc import Iterable

from core.capability_planner import CapabilityPlanner
from core.capability_resolver import CapabilityResolver
from core.execution_plan_library import ExecutionPlanLibrary
from core.execution_plan_registry import ExecutionPlanRegistry
from core.workflow_selector import WorkflowSelector


def build_core_capability_planner(
    *,
    capability_resolver: CapabilityResolver,
    workflow_selector: WorkflowSelector,
    execution_plan_libraries: Iterable[ExecutionPlanLibrary] = (),
    execution_plan_registry: ExecutionPlanRegistry | None = None,
) -> CapabilityPlanner:
    """Build the capability planner from explicitly injected dependencies."""
    return CapabilityPlanner(
        capability_resolver,
        workflow_selector,
        execution_plan_libraries=tuple(execution_plan_libraries),
        execution_plan_registry=execution_plan_registry,
    )

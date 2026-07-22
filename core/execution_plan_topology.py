"""Deterministic topological ordering for Atlas execution plans."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

from core.execution_dependency_checker import (
    ExecutionDependencyCycleError,
    ExecutionDependencyNotFoundError,
)
from core.planner import ExecutionPlan, ExecutionStep


class ExecutionPlanTopologyError(ValueError):
    """Base error for execution plan topology failures."""


class ExecutionPlanCycleError(ExecutionDependencyCycleError, ExecutionPlanTopologyError):
    """Raised when a plan dependency graph contains a cycle."""


class ExecutionPlanTopologyValidationError(ExecutionPlanTopologyError):
    """Raised when topology cannot be built from invalid plan structure."""


class ExecutionPlanTopologyMismatchError(ExecutionPlanTopologyError):
    """Raised when a persisted topology does not match a recalculated topology."""


class ExecutionDependencyStateInconsistencyError(ExecutionPlanTopologyError):
    """Raised when runtime dependency state is impossible for topological order."""


@dataclass(frozen=True, slots=True)
class TopologicalExecutionOrder:
    """Immutable topological execution order derived from an ExecutionPlan."""

    ordered_step_ids: tuple[str, ...]
    original_step_ids: tuple[str, ...]
    reordered: bool
    dependency_count: int = 0
    root_step_ids: tuple[str, ...] = ()
    leaf_step_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_step_ids", tuple(self.ordered_step_ids))
        object.__setattr__(self, "original_step_ids", tuple(self.original_step_ids))
        object.__setattr__(self, "root_step_ids", tuple(self.root_step_ids))
        object.__setattr__(self, "leaf_step_ids", tuple(self.leaf_step_ids))

    def ordered_steps(
        self,
        plan: ExecutionPlan,
    ) -> tuple[ExecutionStep, ...]:
        """Return plan steps in this topological order."""
        step_by_id = {step.id: step for step in plan.ordered_steps}
        return tuple(step_by_id[step_id] for step_id in self.ordered_step_ids)

    def position_of(
        self,
        step_id: str,
    ) -> int:
        """Return the zero-based topological position of one step."""
        try:
            return self.ordered_step_ids.index(step_id)
        except ValueError as error:
            raise ExecutionPlanTopologyValidationError(
                f"step_id={step_id} operation=position_of reason=step is not in topology"
            ) from error

    def comes_before(
        self,
        step_a: str,
        step_b: str,
    ) -> bool:
        """Return whether step_a is before step_b in the topological order."""
        return self.position_of(step_a) < self.position_of(step_b)


class ExecutionPlanTopologicalSorter:
    """Build a stable Kahn topological order for an execution plan."""

    def sort(
        self,
        plan: ExecutionPlan,
    ) -> TopologicalExecutionOrder:
        """Return a deterministic topological order without mutating ``plan``."""
        original_step_ids = tuple(step.id for step in plan.ordered_steps)
        index_by_id = _index_by_step_id(original_step_ids)
        dependencies_by_step: dict[str, tuple[str, ...]] = {}
        dependents_by_step: dict[str, list[str]] = {
            step_id: []
            for step_id in original_step_ids
        }
        indegree: dict[str, int] = {
            step_id: 0
            for step_id in original_step_ids
        }

        for step in plan.ordered_steps:
            dependencies = tuple(step.depends_on)
            dependencies_by_step[step.id] = dependencies
            for dependency_id in dependencies:
                if dependency_id not in index_by_id:
                    raise ExecutionDependencyNotFoundError(
                        f"step_id={step.id} dependency_id={dependency_id} "
                        "operation=topological_sort reason=dependency not found"
                    )
                dependents_by_step[dependency_id].append(step.id)
                indegree[step.id] += 1

        available: list[tuple[int, str]] = [
            (index_by_id[step_id], step_id)
            for step_id, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(available)
        ordered: list[str] = []

        while available:
            _, step_id = heapq.heappop(available)
            ordered.append(step_id)
            for dependent_id in sorted(
                dependents_by_step[step_id],
                key=index_by_id.__getitem__,
            ):
                indegree[dependent_id] -= 1
                if indegree[dependent_id] == 0:
                    heapq.heappush(
                        available,
                        (index_by_id[dependent_id], dependent_id),
                    )

        if len(ordered) != len(original_step_ids):
            unresolved = tuple(
                step_id
                for step_id in original_step_ids
                if step_id not in ordered
            )
            raise ExecutionPlanCycleError(
                "operation=topological_sort "
                f"reason=cycle detected processed={len(ordered)} "
                f"total={len(original_step_ids)} step_ids={list(unresolved)}"
            )

        ordered_step_ids = tuple(ordered)
        roots = tuple(
            step_id
            for step_id in original_step_ids
            if not dependencies_by_step.get(step_id, ())
        )
        leaves = tuple(
            step_id
            for step_id in original_step_ids
            if not dependents_by_step.get(step_id)
        )
        return TopologicalExecutionOrder(
            ordered_step_ids=ordered_step_ids,
            original_step_ids=original_step_ids,
            reordered=ordered_step_ids != original_step_ids,
            dependency_count=sum(len(items) for items in dependencies_by_step.values()),
            root_step_ids=roots,
            leaf_step_ids=leaves,
        )


def _index_by_step_id(
    step_ids: tuple[str, ...],
) -> dict[str, int]:
    index_by_id: dict[str, int] = {}
    for index, step_id in enumerate(step_ids):
        if step_id in index_by_id:
            raise ExecutionPlanTopologyValidationError(
                f"step_id={step_id} operation=topological_sort reason=duplicate step id"
            )
        index_by_id[step_id] = index
    return index_by_id

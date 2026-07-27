"""Pure dependency readiness resolution for Atlas execution plans."""

from __future__ import annotations

from dataclasses import dataclass

from core.execution_plan_topology import ExecutionPlanTopologicalSorter
from core.planner import ExecutionPlan, ExecutionStep


@dataclass(frozen=True, slots=True)
class ExecutionDependencyResolution:
    """Immutable readiness snapshot for one execution plan."""

    ready_steps: tuple[ExecutionStep, ...]
    pending_step_ids: tuple[str, ...]
    blocked_step_ids: tuple[str, ...]
    completed_step_ids: tuple[str, ...]
    failed_step_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ready_steps", tuple(self.ready_steps))
        object.__setattr__(self, "pending_step_ids", tuple(self.pending_step_ids))
        object.__setattr__(self, "blocked_step_ids", tuple(self.blocked_step_ids))
        object.__setattr__(self, "completed_step_ids", tuple(self.completed_step_ids))
        object.__setattr__(self, "failed_step_ids", tuple(self.failed_step_ids))


class ExecutionDependencyResolver:
    """Resolve executable steps from explicit dependencies without execution."""

    def __init__(
        self,
        topological_sorter: ExecutionPlanTopologicalSorter | None = None,
    ) -> None:
        self._topological_sorter = topological_sorter or ExecutionPlanTopologicalSorter()

    def get_ready_steps(
        self,
        plan: ExecutionPlan,
        completed_step_ids: tuple[str, ...],
        failed_step_ids: tuple[str, ...] = (),
    ) -> tuple[ExecutionStep, ...]:
        """Return not-yet-executed steps whose dependencies are completed."""
        return self.resolve(
            plan,
            completed_step_ids=completed_step_ids,
            failed_step_ids=failed_step_ids,
        ).ready_steps

    def resolve(
        self,
        plan: ExecutionPlan,
        *,
        completed_step_ids: tuple[str, ...],
        failed_step_ids: tuple[str, ...] = (),
    ) -> ExecutionDependencyResolution:
        """Return a deterministic dependency-readiness snapshot."""
        self._topological_sorter.sort(plan)
        completed = tuple(dict.fromkeys(completed_step_ids))
        failed = tuple(dict.fromkeys(failed_step_ids))
        completed_set = set(completed)
        failed_set = set(failed)
        failed_or_blocked = set(failed)

        pending: list[str] = []
        blocked: list[str] = []
        ready: list[ExecutionStep] = []

        for step in plan.ordered_steps:
            if step.id in completed_set or step.id in failed_set:
                continue

            dependencies = tuple(step.depends_on)
            if any(dependency in failed_or_blocked for dependency in dependencies):
                blocked.append(step.id)
                failed_or_blocked.add(step.id)
                continue

            if all(dependency in completed_set for dependency in dependencies):
                ready.append(step)
                continue

            pending.append(step.id)

        return ExecutionDependencyResolution(
            ready_steps=tuple(ready),
            pending_step_ids=tuple(pending),
            blocked_step_ids=tuple(blocked),
            completed_step_ids=completed,
            failed_step_ids=failed,
        )

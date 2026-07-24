"""Internal metrics calculated from execution traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.execution_trace import ExecutionTrace, TraceEvent


@dataclass(frozen=True, slots=True)
class ExecutionMetrics:
    """Immutable metrics summary for one execution trace."""

    execution_id: str
    execution_status: str
    total_duration_ms: int
    total_events: int
    started_steps: int
    successful_steps: int
    failed_steps: int
    skipped_steps: int
    success_rate: float
    total_step_duration_ms: int
    average_step_duration_ms: float
    minimum_step_duration_ms: int | None
    maximum_step_duration_ms: int | None
    components: tuple[str, ...]
    actions: tuple[str, ...]
    events_by_component: tuple[tuple[str, int], ...]
    events_by_action: tuple[tuple[str, int], ...]
    blocked_steps: int = 0
    branches_evaluated: int = 0
    then_branches_selected: int = 0
    else_branches_selected: int = 0
    branches_skipped: int = 0
    branches_failed: int = 0


class ExecutionMetricsCalculator:
    """Calculate internal metrics from an ExecutionTrace without mutating it."""

    def calculate(
        self,
        trace: ExecutionTrace,
    ) -> ExecutionMetrics:
        """Return a metrics snapshot derived only from trace data."""
        events = tuple(trace.events)
        started_steps = _count_action(events, "STEP_STARTED")
        successful_steps = _count_action(events, "STEP_FINISHED")
        failed_steps = _count_action(events, "STEP_FAILED")
        skipped_steps = _count_action(events, "STEP_SKIPPED") + _count_action(
            events,
            "execution_step_skipped",
        )
        blocked_steps = _count_action(events, "execution_step_blocked")
        branches_evaluated = _count_action(events, "execution_branch_evaluation_started")
        then_branches_selected = _count_action(events, "execution_branch_then_selected")
        else_branches_selected = _count_action(events, "execution_branch_else_selected")
        branches_skipped = _count_action(events, "execution_branch_skipped")
        branches_failed = _count_action(events, "execution_branch_failed")
        finished_steps = successful_steps + failed_steps
        step_durations = tuple(
            event.duration_ms
            for event in events
            if event.action in {"STEP_FINISHED", "STEP_FAILED"}
            and event.duration_ms is not None
        )

        return ExecutionMetrics(
            execution_id=trace.execution_id,
            execution_status=trace.status,
            total_duration_ms=trace.duration(),
            total_events=len(events),
            started_steps=started_steps,
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            skipped_steps=skipped_steps,
            blocked_steps=blocked_steps,
            success_rate=(
                successful_steps / finished_steps
                if finished_steps
                else 0.0
            ),
            total_step_duration_ms=sum(step_durations),
            average_step_duration_ms=(
                sum(step_durations) / len(step_durations)
                if step_durations
                else 0.0
            ),
            minimum_step_duration_ms=(
                min(step_durations)
                if step_durations
                else None
            ),
            maximum_step_duration_ms=(
                max(step_durations)
                if step_durations
                else None
            ),
            components=tuple(sorted({event.component for event in events})),
            actions=tuple(sorted({event.action for event in events})),
            events_by_component=_count_by(event.component for event in events),
            events_by_action=_count_by(event.action for event in events),
            branches_evaluated=branches_evaluated,
            then_branches_selected=then_branches_selected,
            else_branches_selected=else_branches_selected,
            branches_skipped=branches_skipped,
            branches_failed=branches_failed,
        )


def _count_action(
    events: tuple[TraceEvent, ...],
    action: str,
) -> int:
    return sum(1 for event in events if event.action == action)


def _count_by(
    values: Iterable[str],
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return tuple(sorted(counts.items()))

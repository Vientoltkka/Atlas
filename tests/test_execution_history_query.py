from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from core.execution_history import ExecutionHistory
from core.execution_history_query import (
    ExecutionHistoryQuery,
    ExecutionHistoryQueryResult,
    ExecutionHistoryQueryService,
    ExecutionHistoryQueryStats,
    ExecutionHistorySortField,
)
from core.execution_metrics import ExecutionMetrics
from core.execution_plan_executor import PlanExecutionResult
from core.execution_trace import ExecutionTrace, TraceEventStatus, TraceStatus


def _dt(day: int, hour: int = 10) -> datetime:
    return datetime(2026, 7, day, hour, 0, 0, tzinfo=timezone.utc)


def _naive_dt(day: int) -> datetime:
    return datetime(2026, 7, day, 10, 0, 0)


def _result(
    execution_id: str,
    *,
    status: str = TraceStatus.SUCCESS.value,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    duration_ms: int = 100,
    components: tuple[str, ...] = ("ExecutionPlanExecutor",),
    actions: tuple[str, ...] = ("STEP_STARTED", "STEP_FINISHED"),
) -> PlanExecutionResult:
    started = started_at or _dt(20)
    trace = ExecutionTrace(
        execution_id=execution_id,
        started_at=started,
        status=status,
    )
    for index, action in enumerate(actions):
        trace.add_event(
            timestamp=started,
            component=components[index % len(components)],
            action=action,
            status=(
                TraceEventStatus.FAILED.value
                if action == "STEP_FAILED"
                else TraceEventStatus.FINISHED.value
                if action == "STEP_FINISHED"
                else TraceEventStatus.STARTED.value
            ),
            duration_ms=duration_ms if action in {"STEP_FINISHED", "STEP_FAILED"} else None,
        )
    trace.finished_at = finished_at
    metrics = ExecutionMetrics(
        execution_id=execution_id,
        execution_status=status,
        total_duration_ms=duration_ms,
        total_events=len(actions),
        started_steps=actions.count("STEP_STARTED"),
        successful_steps=actions.count("STEP_FINISHED"),
        failed_steps=actions.count("STEP_FAILED"),
        success_rate=0.0,
        total_step_duration_ms=duration_ms,
        average_step_duration_ms=float(duration_ms),
        minimum_step_duration_ms=duration_ms,
        maximum_step_duration_ms=duration_ms,
        components=tuple(sorted(set(components))),
        actions=tuple(sorted(set(actions))),
        events_by_component=tuple(
            sorted((component, components.count(component)) for component in set(components))
        ),
        events_by_action=tuple(
            sorted((action, actions.count(action)) for action in set(actions))
        ),
    )
    return PlanExecutionResult(
        plan_status=status.lower(),
        success=status == TraceStatus.SUCCESS.value,
        trace=trace,
        metrics=metrics,
    )


def _history() -> ExecutionHistory:
    history = ExecutionHistory(max_entries=10)
    history.add(
        _result(
            "exec-1",
            status=TraceStatus.SUCCESS.value,
            started_at=_dt(20),
            finished_at=_dt(20, 11),
            duration_ms=100,
            components=("ExecutionPlanExecutor",),
            actions=("STEP_STARTED", "STEP_FINISHED"),
        )
    )
    history.add(
        _result(
            "exec-2",
            status=TraceStatus.FAILED.value,
            started_at=_dt(21),
            finished_at=_dt(21, 11),
            duration_ms=300,
            components=("ToolExecutor",),
            actions=("STEP_STARTED", "STEP_FAILED"),
        )
    )
    history.add(
        _result(
            "exec-3",
            status=TraceStatus.CANCELLED.value,
            started_at=_dt(22),
            finished_at=None,
            duration_ms=200,
            components=("ExecutionPlanExecutor", "ToolExecutor"),
            actions=("STEP_STARTED", "STEP_FAILED"),
        )
    )
    history.add(
        _result(
            "exec-4",
            status=TraceStatus.RUNNING.value,
            started_at=_dt(23),
            finished_at=None,
            duration_ms=400,
            components=("Scheduler",),
            actions=("STEP_STARTED",),
        )
    )
    return history


def _query(
    history: ExecutionHistory,
    criteria: ExecutionHistoryQuery,
) -> ExecutionHistoryQueryResult:
    return ExecutionHistoryQueryService().query(history, criteria)


def test_query_empty_history_returns_empty_stats() -> None:
    result = _query(ExecutionHistory(), ExecutionHistoryQuery())

    assert result.entries == ()
    assert result.stats == ExecutionHistoryQueryStats(
        total_matches=0,
        successful_count=0,
        failed_count=0,
        cancelled_count=0,
        average_duration_ms=0.0,
        minimum_duration_ms=None,
        maximum_duration_ms=None,
        total_duration_ms=0,
    )


def test_query_without_filters_returns_all_entries_sorted_by_started_at() -> None:
    result = _query(_history(), ExecutionHistoryQuery())

    assert tuple(entry.execution_id for entry in result.entries) == (
        "exec-1",
        "exec-2",
        "exec-3",
        "exec-4",
    )
    assert isinstance(result.entries, tuple)


def test_filters_by_statuses_and_empty_statuses() -> None:
    history = _history()

    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(statuses=(TraceStatus.SUCCESS,))).entries
    ) == ("exec-1",)
    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(statuses=(TraceStatus.FAILED.value,))).entries
    ) == ("exec-2",)
    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(statuses=(TraceStatus.CANCELLED,))).entries
    ) == ("exec-3",)
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(
                statuses=(TraceStatus.SUCCESS, TraceStatus.FAILED),
            ),
        ).entries
    ) == ("exec-1", "exec-2")
    assert _query(history, ExecutionHistoryQuery(statuses=())).entries == ()


def test_filters_by_components_with_exact_or_semantics() -> None:
    history = _history()

    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(components=("ExecutionPlanExecutor",)),
        ).entries
    ) == ("exec-1", "exec-3")
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(components=("Missing",)),
        ).entries
    ) == ()
    assert _query(history, ExecutionHistoryQuery(components=())).entries == ()
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(components=("ExecutionPlanExecutor", "Scheduler")),
        ).entries
    ) == ("exec-1", "exec-3", "exec-4")


def test_filters_by_actions_with_exact_or_semantics() -> None:
    history = _history()

    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(actions=("STEP_FINISHED",))).entries
    ) == ("exec-1",)
    assert _query(history, ExecutionHistoryQuery(actions=("UNKNOWN",))).entries == ()
    assert _query(history, ExecutionHistoryQuery(actions=())).entries == ()
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(actions=("STEP_FINISHED", "STEP_FAILED")),
        ).entries
    ) == ("exec-1", "exec-2", "exec-3")


def test_filters_by_duration_inclusive_and_rejects_invalid_ranges() -> None:
    history = _history()

    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(min_duration_ms=200)).entries
    ) == ("exec-2", "exec-3", "exec-4")
    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(max_duration_ms=200)).entries
    ) == ("exec-1", "exec-3")
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(min_duration_ms=200, max_duration_ms=300),
        ).entries
    ) == ("exec-2", "exec-3")
    with pytest.raises(ValueError):
        _query(history, ExecutionHistoryQuery(min_duration_ms=-1))
    with pytest.raises(ValueError):
        _query(history, ExecutionHistoryQuery(max_duration_ms=-1))
    with pytest.raises(ValueError):
        _query(history, ExecutionHistoryQuery(min_duration_ms=3, max_duration_ms=2))


def test_filters_by_started_and_finished_ranges() -> None:
    history = _history()

    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(started_from=_dt(21))).entries
    ) == ("exec-2", "exec-3", "exec-4")
    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(started_until=_dt(21))).entries
    ) == ("exec-1", "exec-2")
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(started_from=_dt(21), started_until=_dt(22)),
        ).entries
    ) == ("exec-2", "exec-3")
    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(finished_from=_dt(21))).entries
    ) == ("exec-2",)
    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(finished_until=_dt(20, 11))).entries
    ) == ("exec-1",)
    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(finished_from=_dt(20))).entries
    ) == ("exec-1", "exec-2")


def test_rejects_invalid_temporal_ranges_and_incompatible_datetimes() -> None:
    history = _history()

    with pytest.raises(ValueError):
        _query(history, ExecutionHistoryQuery(started_from=_dt(22), started_until=_dt(21)))
    with pytest.raises(ValueError):
        _query(history, ExecutionHistoryQuery(finished_from=_dt(22), finished_until=_dt(21)))
    with pytest.raises(ValueError):
        _query(history, ExecutionHistoryQuery(started_from=_naive_dt(20)))
    with pytest.raises(ValueError):
        _query(history, ExecutionHistoryQuery(finished_until=_naive_dt(22)))


def test_combines_filters_with_and_logic() -> None:
    history = _history()

    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(
                statuses=(TraceStatus.FAILED,),
                components=("ToolExecutor",),
            ),
        ).entries
    ) == ("exec-2",)
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(
                components=("ToolExecutor",),
                actions=("STEP_FAILED",),
            ),
        ).entries
    ) == ("exec-2", "exec-3")
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(
                statuses=(TraceStatus.CANCELLED,),
                components=("ToolExecutor",),
                actions=("STEP_FAILED",),
                min_duration_ms=200,
                max_duration_ms=200,
                started_from=_dt(22),
                started_until=_dt(22),
            ),
        ).entries
    ) == ("exec-3",)


def test_sorting_fields_and_finished_at_none_policy() -> None:
    history = _history()

    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(sort_by=ExecutionHistorySortField.STARTED_AT),
        ).entries
    ) == ("exec-1", "exec-2", "exec-3", "exec-4")
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(
                sort_by=ExecutionHistorySortField.STARTED_AT,
                descending=True,
            ),
        ).entries
    ) == ("exec-4", "exec-3", "exec-2", "exec-1")
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(
                sort_by=ExecutionHistorySortField.FINISHED_AT,
                descending=True,
            ),
        ).entries
    ) == ("exec-2", "exec-1", "exec-3", "exec-4")
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(sort_by=ExecutionHistorySortField.DURATION_MS),
        ).entries
    ) == ("exec-1", "exec-3", "exec-2", "exec-4")
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(sort_by=ExecutionHistorySortField.EXECUTION_ID),
        ).entries
    ) == ("exec-1", "exec-2", "exec-3", "exec-4")
    assert tuple(
        entry.execution_id
        for entry in _query(history, ExecutionHistoryQuery(sort_by="status")).entries
    ) == ("exec-3", "exec-2", "exec-4", "exec-1")
    with pytest.raises(ValueError):
        _query(history, ExecutionHistoryQuery(sort_by="unknown"))


def test_sorting_is_stable_and_limit_is_applied_after_sorting() -> None:
    history = ExecutionHistory()
    history.add(_result("exec-a", started_at=_dt(20), duration_ms=100))
    history.add(_result("exec-b", started_at=_dt(20), duration_ms=100))
    history.add(_result("exec-c", started_at=_dt(20), duration_ms=100))

    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(sort_by=ExecutionHistorySortField.DURATION_MS),
        ).entries
    ) == ("exec-a", "exec-b", "exec-c")
    assert tuple(
        entry.execution_id
        for entry in _query(
            history,
            ExecutionHistoryQuery(
                sort_by=ExecutionHistorySortField.EXECUTION_ID,
                descending=True,
                limit=2,
            ),
        ).entries
    ) == ("exec-c", "exec-b")
    assert _query(history, ExecutionHistoryQuery(limit=0)).entries == ()
    assert len(_query(history, ExecutionHistoryQuery(limit=10)).entries) == 3
    with pytest.raises(ValueError):
        _query(history, ExecutionHistoryQuery(limit=-1))


def test_stats_for_single_and_mixed_results() -> None:
    service = ExecutionHistoryQueryService()
    history = _history()

    single = _query(history, ExecutionHistoryQuery(statuses=(TraceStatus.SUCCESS,))).stats
    mixed = service.analyze(tuple(history))

    assert single.total_matches == 1
    assert single.successful_count == 1
    assert single.failed_count == 0
    assert single.cancelled_count == 0
    assert single.total_duration_ms == 100
    assert single.average_duration_ms == 100.0
    assert single.minimum_duration_ms == 100
    assert single.maximum_duration_ms == 100
    assert mixed.total_matches == 4
    assert mixed.successful_count == 1
    assert mixed.failed_count == 1
    assert mixed.cancelled_count == 1
    assert mixed.total_duration_ms == 1000
    assert mixed.average_duration_ms == 250.0
    assert mixed.minimum_duration_ms == 100
    assert mixed.maximum_duration_ms == 400


def test_query_objects_are_immutable_and_query_does_not_modify_history_or_entries() -> None:
    history = _history()
    criteria = ExecutionHistoryQuery(statuses=(TraceStatus.SUCCESS,))
    before = tuple(history)
    first = _query(history, criteria)
    second = _query(history, criteria)

    with pytest.raises(FrozenInstanceError):
        criteria.limit = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.stats.total_matches = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.entries[0].status = TraceStatus.FAILED.value  # type: ignore[misc]
    assert tuple(history) == before
    assert first == second


def test_compatible_with_history_duplicate_replacement_and_capacity_limit() -> None:
    history = ExecutionHistory(max_entries=2)
    history.add(_result("exec-1", duration_ms=100))
    history.add(_result("exec-2", duration_ms=200))
    history.add(_result("exec-1", duration_ms=300))
    history.add(_result("exec-3", duration_ms=400))

    result = _query(
        history,
        ExecutionHistoryQuery(sort_by=ExecutionHistorySortField.EXECUTION_ID),
    )

    assert tuple(entry.execution_id for entry in result.entries) == ("exec-1", "exec-3")
    assert result.stats.total_duration_ms == 700

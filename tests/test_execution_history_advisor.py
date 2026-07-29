from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.execution_history import ExecutionHistoryRecord, ExecutionSessionHistory
from core.execution_history_advisor import (
    ExecutionHistoryAdvisor,
    HistoricalAdvisorPolicy,
    HistoricalAnalysisRequest,
    HistoricalRecommendationType,
    HistoricalTimeWindow,
)
from core.execution_report import (
    OperationalExecutionReport,
    OperationalExecutionStatus,
)
from core.execution_supervisor import ExecutionState
from core.planner import Planner


_START = datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc)


class _StaticHistory(ExecutionSessionHistory):
    def __init__(self, records: tuple[ExecutionHistoryRecord, ...]) -> None:
        self.records = records
        self.requested_limits: list[int] = []

    def latest_executions(self, limit: int) -> tuple[ExecutionHistoryRecord, ...]:
        self.requested_limits.append(limit)
        return self.records[:limit]


def _report(
    execution_id: str,
    objective: str,
    status: OperationalExecutionStatus,
    *,
    duration: float,
    retries: int,
    actions: tuple[str, ...],
    replan_status: str,
) -> OperationalExecutionReport:
    return OperationalExecutionReport(
        session_id=execution_id,
        objective=objective,
        status=status,
        title="Resultado histórico",
        total_steps=2,
        completed_steps=2 if status in {
            OperationalExecutionStatus.COMPLETED,
            OperationalExecutionStatus.COMPLETED_WITH_RECOVERY,
        } else 1,
        failed_steps=1 if status is OperationalExecutionStatus.FAILED else 0,
        skipped_steps=0,
        cancelled_steps=0,
        progress_percent=100.0,
        duration_seconds=duration,
        retry_count=retries,
        retried_step_ids=("read",) if retries else (),
        replan_status=replan_status,
        replan_count=1 if replan_status != "not_needed" else 0,
        warnings=(),
        pending_user_actions=actions,
        steps=(),
        final_message="Final",
    )


def _record(
    execution_id: str,
    *,
    objective: str = "Preparar informe mensual",
    minutes_ago: int = 0,
    result: OperationalExecutionStatus = OperationalExecutionStatus.COMPLETED,
    duration: float = 10.0,
    retries: int = 0,
    failed_steps: tuple[str, ...] = (),
    omitted_steps: tuple[str, ...] = (),
    recovery_types: tuple[str, ...] = (),
    replan_status: str = "not_needed",
    actions: tuple[str, ...] = (),
    tool: str = "read_file",
) -> ExecutionHistoryRecord:
    state = (
        ExecutionState.FAILED
        if result is OperationalExecutionStatus.FAILED
        else ExecutionState.COMPLETED
    )
    report = _report(
        execution_id,
        objective,
        result,
        duration=duration,
        retries=retries,
        actions=actions,
        replan_status=replan_status,
    )
    return ExecutionHistoryRecord(
        id=execution_id,
        date=_START - timedelta(minutes=minutes_ago),
        objective=objective,
        duration_seconds=duration,
        final_result=result,
        progress_percent=100.0,
        executed_step_ids=("read", "write"),
        failed_step_ids=failed_steps,
        omitted_step_ids=omitted_steps,
        replanned_step_ids=(),
        retry_count=retries,
        failure_reason="failed" if failed_steps else None,
        required_actions=actions,
        operational_report=report,
        recovery_types=recovery_types,
        state=state,
        tool_names=(tool,),
        tools_by_step={"read": tool, "write": "write_file"},
    )


def _advisor(
    *records: ExecutionHistoryRecord,
    policy: HistoricalAdvisorPolicy | None = None,
) -> tuple[ExecutionHistoryAdvisor, _StaticHistory]:
    history = _StaticHistory(tuple(records))
    return ExecutionHistoryAdvisor(history, policy=policy), history


def _types(result) -> tuple[HistoricalRecommendationType, ...]:
    return tuple(item.type for item in result.recommendations)


def test_empty_and_single_execution_are_insufficient() -> None:
    empty, _ = _advisor()
    single, _ = _advisor(_record("one"))

    assert _types(empty.analyze(HistoricalAnalysisRequest("Preparar informe"))) == (
        HistoricalRecommendationType.INSUFFICIENT_HISTORY,
    )
    assert _types(single.analyze(HistoricalAnalysisRequest("Preparar informe"))) == (
        HistoricalRecommendationType.INSUFFICIENT_HISTORY,
    )


def test_similar_objectives_are_selected_and_exact_match_scores_higher() -> None:
    advisor, _ = _advisor(
        _record("shared", objective="Preparar informe semanal", minutes_ago=0),
        _record("exact", objective="PREPARAR   INFORME mensual", minutes_ago=5),
        _record("other", objective="Abrir calculadora", minutes_ago=1),
    )

    result = advisor.analyze(
        HistoricalAnalysisRequest("preparar informe mensual")
    )

    assert tuple(record.id for record in result.relevant_records) == (
        "exact",
        "shared",
    )
    assert result.relevance_scores["exact"] > result.relevance_scores["shared"]


def test_tool_capability_and_operation_type_contribute_only_when_observable() -> None:
    advisor, _ = _advisor(
        _record("tool", objective="Procesar archivo", tool="read_file"),
        _record(
            "capability",
            objective="Generar reporte financiero",
            tool="write_file",
            minutes_ago=1,
        ),
        _record(
            "unrelated",
            objective="Abrir calculadora",
            minutes_ago=2,
            tool="desktop.open_application",
        ),
    )
    request = HistoricalAnalysisRequest(
        objective="Procesar datos",
        operation_type="reporte",
        candidate_tools=("read_file",),
        candidate_capabilities=("financiero",),
    )

    result = advisor.analyze(request)

    assert tuple(record.id for record in result.relevant_records) == (
        "tool",
        "capability",
    )


def test_relevance_tie_prefers_recent_and_duplicate_ids_are_removed() -> None:
    recent = _record("duplicate", objective="Preparar informe", minutes_ago=0)
    older_duplicate = _record(
        "duplicate",
        objective="Preparar informe",
        minutes_ago=10,
    )
    older = _record("older", objective="Preparar informe", minutes_ago=5)
    advisor, _ = _advisor(recent, older_duplicate, older)

    result = advisor.analyze(HistoricalAnalysisRequest("Preparar informe"))

    assert tuple(record.id for record in result.relevant_records) == (
        "duplicate",
        "older",
    )
    assert result.relevant_records[0].date == recent.date


def test_query_limit_and_time_and_failure_recovery_filters_are_enforced() -> None:
    advisor, history = _advisor(
        _record(
            "failed",
            result=OperationalExecutionStatus.FAILED,
            failed_steps=("read",),
        ),
        _record("recovered", recovery_types=("retry",), retries=1),
        _record("old", minutes_ago=60),
    )
    request = HistoricalAnalysisRequest(
        objective="Preparar informe",
        max_history_records=3,
        time_window=HistoricalTimeWindow(
            started_from=_START - timedelta(minutes=5),
        ),
        include_failed=False,
        include_recovered=False,
    )

    result = advisor.analyze(request)

    assert history.requested_limits == [3]
    assert result.relevant_records == ()


def test_repeated_success_generates_previous_success() -> None:
    advisor, _ = _advisor(_record("one"), _record("two", minutes_ago=1))

    result = advisor.analyze(HistoricalAnalysisRequest("Preparar informe mensual"))

    assert HistoricalRecommendationType.PREVIOUS_SUCCESS in _types(result)


def test_frequent_failure_requires_two_distinct_executions() -> None:
    repeated, _ = _advisor(
        _record(
            "one",
            result=OperationalExecutionStatus.FAILED,
            failed_steps=("read",),
        ),
        _record(
            "two",
            result=OperationalExecutionStatus.FAILED,
            failed_steps=("read",),
            minutes_ago=1,
        ),
    )
    isolated, _ = _advisor(
        _record(
            "one",
            result=OperationalExecutionStatus.FAILED,
            failed_steps=("read",),
        ),
        _record("two", minutes_ago=1),
    )

    repeated_result = repeated.analyze(
        HistoricalAnalysisRequest("Preparar informe mensual")
    )
    isolated_result = isolated.analyze(
        HistoricalAnalysisRequest("Preparar informe mensual")
    )

    recommendation = next(
        item
        for item in repeated_result.recommendations
        if item.type is HistoricalRecommendationType.FREQUENT_FAILURE
    )
    assert recommendation.related_tool == "read_file"
    assert HistoricalRecommendationType.FREQUENT_FAILURE not in _types(
        isolated_result
    )


def test_retry_risk_requires_repeated_retries() -> None:
    advisor, _ = _advisor(
        _record("one", retries=1, recovery_types=("retry",)),
        _record("two", retries=2, recovery_types=("retry",), minutes_ago=1),
    )

    result = advisor.analyze(HistoricalAnalysisRequest("Preparar informe mensual"))

    assert HistoricalRecommendationType.RETRY_RISK in _types(result)


def test_only_successful_recovery_is_recommended() -> None:
    advisor, _ = _advisor(
        _record(
            "success",
            result=OperationalExecutionStatus.COMPLETED_WITH_RECOVERY,
            recovery_types=("replan:recoverable_failure",),
            replan_status="succeeded",
        ),
        _record(
            "failed",
            result=OperationalExecutionStatus.FAILED,
            failed_steps=("write",),
            recovery_types=("replan:recoverable_failure",),
            replan_status="failed",
            minutes_ago=1,
        ),
    )

    result = advisor.analyze(HistoricalAnalysisRequest("Preparar informe mensual"))

    recovery = next(
        item
        for item in result.recommendations
        if item.type is HistoricalRecommendationType.RECOVERY_AVAILABLE
    )
    assert recovery.session_ids == ("success",)


def test_high_duration_requires_comparable_history() -> None:
    advisor, _ = _advisor(
        _record("slow", duration=40),
        _record("normal-1", duration=10, minutes_ago=1),
        _record("normal-2", duration=10, minutes_ago=2),
    )

    result = advisor.analyze(HistoricalAnalysisRequest("Preparar informe mensual"))

    assert HistoricalRecommendationType.HIGH_DURATION in _types(result)


def test_optional_and_user_action_patterns_require_repetition() -> None:
    advisor, _ = _advisor(
        _record(
            "one",
            omitted_steps=("write",),
            actions=("Confirma la publicación.",),
        ),
        _record(
            "two",
            omitted_steps=("write",),
            actions=("Confirma la publicación.",),
            minutes_ago=1,
        ),
    )

    result = advisor.analyze(HistoricalAnalysisRequest("Preparar informe mensual"))

    assert HistoricalRecommendationType.OPTIONAL_STEP_PATTERN in _types(result)
    assert HistoricalRecommendationType.USER_ACTION_PATTERN in _types(result)


def test_summary_is_deterministic_bounded_and_sanitizes_secrets() -> None:
    advisor, _ = _advisor(
        _record(
            "one",
            actions=("token sk-secretvalue123456",),
        ),
        _record(
            "two",
            actions=("token sk-secretvalue123456",),
            minutes_ago=1,
        ),
    )
    request = HistoricalAnalysisRequest("Preparar informe mensual")

    first = advisor.analyze(request).planning_context
    second = advisor.analyze(request).planning_context

    assert first.summary == second.summary
    assert len(first.summary) <= advisor.policy.max_summary_chars
    assert "No se aplican cambios automáticos al plan." in first.summary
    assert "secretvalue" not in str(first.to_planner_context())
    assert "[redacted]" in str(first.to_planner_context())


def test_recommendation_limit_is_hard_and_models_are_serializable() -> None:
    policy = HistoricalAdvisorPolicy(max_recommendations=2)
    advisor, _ = _advisor(
        _record(
            "one",
            retries=1,
            failed_steps=("read",),
            omitted_steps=("write",),
            actions=("Confirma.",),
            recovery_types=("retry",),
        ),
        _record(
            "two",
            retries=1,
            failed_steps=("read",),
            omitted_steps=("write",),
            actions=("Confirma.",),
            recovery_types=("retry",),
            minutes_ago=1,
        ),
        policy=policy,
    )

    result = advisor.analyze(HistoricalAnalysisRequest("Preparar informe mensual"))

    assert len(result.recommendations) == 2
    assert all(
        recommendation.to_dict()["informational"] is True
        for recommendation in result.recommendations
    )


def test_planner_optional_context_does_not_modify_generated_plan() -> None:
    advisor, _ = _advisor(_record("one"), _record("two", minutes_ago=1))
    context = advisor.build_planning_context("Conversar sobre el informe")
    planner = Planner()

    without_history = planner.generate_execution_plan(
        "Conversar sobre el informe"
    )
    with_history = planner.generate_execution_plan(
        "Conversar sobre el informe",
        planning_context=context.to_planner_context(),
    )

    assert with_history.plan == without_history.plan
    assert with_history.success == without_history.success
    assert context.to_planner_context()["informational_only"] is True


@pytest.mark.parametrize("limit", [0, 51, True])
def test_request_rejects_unbounded_or_invalid_history_limits(limit) -> None:
    expected = TypeError if limit is True else ValueError
    with pytest.raises(expected):
        HistoricalAnalysisRequest("Objetivo", max_history_records=limit)

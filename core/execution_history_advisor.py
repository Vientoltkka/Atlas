"""Deterministic, consultation-only recommendations from execution history."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from statistics import median
from types import MappingProxyType
import unicodedata

from core.execution_history import ExecutionHistoryRecord, ExecutionSessionHistory
from core.execution_report import OperationalExecutionStatus, _safe_text


DEFAULT_MAX_HISTORY_RECORDS = 20
MAX_HISTORY_RECORDS = 50
DEFAULT_MAX_RECOMMENDATIONS = 8
MAX_RECOMMENDATIONS = 20
DEFAULT_MAX_SUMMARY_CHARS = 800
_SUCCESS_RESULTS = frozenset(
    {
        OperationalExecutionStatus.COMPLETED,
        OperationalExecutionStatus.COMPLETED_WITH_RECOVERY,
    }
)
_STOP_WORDS = frozenset(
    {
        "con",
        "del",
        "desde",
        "el",
        "en",
        "la",
        "las",
        "los",
        "para",
        "por",
        "que",
        "the",
        "una",
        "uno",
        "unos",
        "unas",
        "and",
        "for",
        "from",
        "with",
    }
)


class HistoricalRecommendationType(str, Enum):
    PREVIOUS_SUCCESS = "PREVIOUS_SUCCESS"
    FREQUENT_FAILURE = "FREQUENT_FAILURE"
    RETRY_RISK = "RETRY_RISK"
    RECOVERY_AVAILABLE = "RECOVERY_AVAILABLE"
    HIGH_DURATION = "HIGH_DURATION"
    OPTIONAL_STEP_PATTERN = "OPTIONAL_STEP_PATTERN"
    USER_ACTION_PATTERN = "USER_ACTION_PATTERN"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


class HistoricalRecommendationSeverity(str, Enum):
    INFORMATION = "INFORMATION"
    CAUTION = "CAUTION"
    WARNING = "WARNING"


@dataclass(frozen=True, slots=True)
class HistoricalTimeWindow:
    """Optional inclusive execution-start interval."""

    started_from: datetime | None = None
    started_until: datetime | None = None

    def __post_init__(self) -> None:
        if self.started_from is None or self.started_until is None:
            return
        try:
            invalid = self.started_from > self.started_until
        except TypeError as error:
            raise ValueError("Historical time-window datetimes are incompatible.") from error
        if invalid:
            raise ValueError("started_from cannot be later than started_until.")

    def contains(self, value: datetime) -> bool:
        try:
            if self.started_from is not None and value < self.started_from:
                return False
            if self.started_until is not None and value > self.started_until:
                return False
        except TypeError as error:
            raise ValueError("Historical datetime is incompatible with the time window.") from error
        return True


@dataclass(frozen=True, slots=True)
class HistoricalAnalysisRequest:
    """Bounded structured request for historical planning context."""

    objective: str
    operation_type: str | None = None
    candidate_tools: tuple[str, ...] = ()
    candidate_capabilities: tuple[str, ...] = ()
    max_history_records: int = DEFAULT_MAX_HISTORY_RECORDS
    time_window: HistoricalTimeWindow | None = None
    include_failed: bool = True
    include_recovered: bool = True

    def __post_init__(self) -> None:
        objective = _required_text(self.objective, "objective")
        operation_type = (
            None
            if self.operation_type is None
            else _required_text(self.operation_type, "operation_type")
        )
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "operation_type", operation_type)
        object.__setattr__(
            self,
            "candidate_tools",
            _normalized_names(self.candidate_tools, "candidate_tools"),
        )
        object.__setattr__(
            self,
            "candidate_capabilities",
            _normalized_names(
                self.candidate_capabilities,
                "candidate_capabilities",
            ),
        )
        _bounded_int(
            self.max_history_records,
            "max_history_records",
            minimum=1,
            maximum=MAX_HISTORY_RECORDS,
        )
        if self.time_window is not None and not isinstance(
            self.time_window,
            HistoricalTimeWindow,
        ):
            raise TypeError("time_window must be HistoricalTimeWindow or None.")
        if type(self.include_failed) is not bool:
            raise TypeError("include_failed must be a bool.")
        if type(self.include_recovered) is not bool:
            raise TypeError("include_recovered must be a bool.")


@dataclass(frozen=True, slots=True)
class HistoricalAdvisorPolicy:
    """Conservative configurable evidence and output limits."""

    minimum_relevant_executions: int = 2
    minimum_repeated_evidence: int = 2
    minimum_successful_recoveries: int = 1
    high_duration_minimum_records: int = 3
    high_duration_ratio: float = 1.5
    high_duration_minimum_seconds: float = 1.0
    max_recommendations: int = DEFAULT_MAX_RECOMMENDATIONS
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS

    def __post_init__(self) -> None:
        for name in (
            "minimum_relevant_executions",
            "minimum_repeated_evidence",
            "minimum_successful_recoveries",
            "high_duration_minimum_records",
        ):
            _bounded_int(getattr(self, name), name, minimum=1, maximum=50)
        if self.high_duration_ratio <= 1.0:
            raise ValueError("high_duration_ratio must be greater than one.")
        if self.high_duration_minimum_seconds < 0:
            raise ValueError("high_duration_minimum_seconds cannot be negative.")
        _bounded_int(
            self.max_recommendations,
            "max_recommendations",
            minimum=1,
            maximum=MAX_RECOMMENDATIONS,
        )
        _bounded_int(
            self.max_summary_chars,
            "max_summary_chars",
            minimum=120,
            maximum=2_000,
        )


@dataclass(frozen=True, slots=True)
class HistoricalEvidence:
    """Sanitized factual evidence for one recommendation."""

    fact: str
    occurrence_count: int
    session_ids: tuple[str, ...]
    observed_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact", _sanitized_text(self.fact))
        _bounded_int(
            self.occurrence_count,
            "occurrence_count",
            minimum=1,
            maximum=MAX_HISTORY_RECORDS,
        )
        object.__setattr__(self, "session_ids", _unique(self.session_ids))
        object.__setattr__(
            self,
            "observed_values",
            tuple(_sanitized_text(value) for value in self.observed_values[:10]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "fact": self.fact,
            "occurrence_count": self.occurrence_count,
            "session_ids": list(self.session_ids),
            "observed_values": list(self.observed_values),
        }


@dataclass(frozen=True, slots=True)
class HistoricalRecommendation:
    """Closed, serializable and strictly informative recommendation."""

    type: HistoricalRecommendationType
    severity: HistoricalRecommendationSeverity
    message: str
    evidence: tuple[HistoricalEvidence, ...]
    supporting_execution_count: int
    session_ids: tuple[str, ...]
    related_tool: str | None = None
    related_capability: str | None = None
    related_step: str | None = None
    informational: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.type, HistoricalRecommendationType):
            raise TypeError("type must be HistoricalRecommendationType.")
        if not isinstance(self.severity, HistoricalRecommendationSeverity):
            raise TypeError("severity must be HistoricalRecommendationSeverity.")
        object.__setattr__(self, "message", _sanitized_text(self.message))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        _bounded_int(
            self.supporting_execution_count,
            "supporting_execution_count",
            minimum=0 if self.type is HistoricalRecommendationType.INSUFFICIENT_HISTORY else 1,
            maximum=MAX_HISTORY_RECORDS,
        )
        object.__setattr__(self, "session_ids", _unique(self.session_ids))
        for name in ("related_tool", "related_capability", "related_step"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sanitized_text(value))
        if self.informational is not True:
            raise ValueError("Historical recommendations must remain informational.")

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": [item.to_dict() for item in self.evidence],
            "supporting_execution_count": self.supporting_execution_count,
            "session_ids": list(self.session_ids),
            "related_tool": self.related_tool,
            "related_capability": self.related_capability,
            "related_step": self.related_step,
            "informational": self.informational,
        }


@dataclass(frozen=True, slots=True)
class HistoricalPlanningContext:
    """Bounded historical facts that a planner may inspect but need not apply."""

    objective: str
    reviewed_execution_count: int
    relevant_execution_ids: tuple[str, ...]
    recommendations: tuple[HistoricalRecommendation, ...]
    historical_risks: tuple[str, ...]
    known_recoveries: tuple[str, ...]
    incident_tools_or_steps: tuple[str, ...]
    summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective", _sanitized_text(self.objective))
        object.__setattr__(
            self,
            "relevant_execution_ids",
            _unique(self.relevant_execution_ids),
        )
        object.__setattr__(self, "recommendations", tuple(self.recommendations))
        object.__setattr__(
            self,
            "historical_risks",
            tuple(_sanitized_text(value) for value in self.historical_risks),
        )
        object.__setattr__(
            self,
            "known_recoveries",
            tuple(_sanitized_text(value) for value in self.known_recoveries),
        )
        object.__setattr__(
            self,
            "incident_tools_or_steps",
            tuple(_sanitized_text(value) for value in self.incident_tools_or_steps),
        )
        object.__setattr__(self, "summary", _sanitized_text(self.summary, limit=2_000))

    def to_planner_context(self) -> Mapping[str, object]:
        """Return a bounded immutable mapping for the existing optional contract."""
        return MappingProxyType(
            {
                "objective": self.objective,
                "reviewed_execution_count": self.reviewed_execution_count,
                "relevant_execution_ids": self.relevant_execution_ids,
                "recommendations": tuple(
                    recommendation.to_dict()
                    for recommendation in self.recommendations
                ),
                "historical_risks": self.historical_risks,
                "known_recoveries": self.known_recoveries,
                "incident_tools_or_steps": self.incident_tools_or_steps,
                "summary": self.summary,
                "informational_only": True,
            }
        )


@dataclass(frozen=True, slots=True)
class HistoricalAnalysisResult:
    """Structured result of one bounded historical analysis."""

    request: HistoricalAnalysisRequest
    reviewed_execution_count: int
    relevant_records: tuple[ExecutionHistoryRecord, ...]
    recommendations: tuple[HistoricalRecommendation, ...]
    planning_context: HistoricalPlanningContext
    relevance_scores: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "relevant_records", tuple(self.relevant_records))
        object.__setattr__(self, "recommendations", tuple(self.recommendations))
        object.__setattr__(
            self,
            "relevance_scores",
            MappingProxyType(dict(self.relevance_scores)),
        )


class ExecutionHistoryAdvisor:
    """Derive bounded historical recommendations without changing any plan."""

    def __init__(
        self,
        history: ExecutionSessionHistory,
        *,
        policy: HistoricalAdvisorPolicy | None = None,
    ) -> None:
        if not isinstance(history, ExecutionSessionHistory):
            raise TypeError("history must be an ExecutionSessionHistory.")
        self._history = history
        self._policy = policy or HistoricalAdvisorPolicy()

    @property
    def policy(self) -> HistoricalAdvisorPolicy:
        return self._policy

    def analyze(self, request: HistoricalAnalysisRequest) -> HistoricalAnalysisResult:
        if not isinstance(request, HistoricalAnalysisRequest):
            raise TypeError("request must be a HistoricalAnalysisRequest.")
        reviewed = self._history.latest_executions(request.max_history_records)
        relevant_with_scores = self._select_relevant(reviewed, request)
        relevant = tuple(record for _, record in relevant_with_scores)
        recommendations = self._recommend(relevant)
        context = self._context(
            request.objective,
            len(reviewed),
            relevant,
            recommendations,
        )
        return HistoricalAnalysisResult(
            request=request,
            reviewed_execution_count=len(reviewed),
            relevant_records=relevant,
            recommendations=recommendations,
            planning_context=context,
            relevance_scores={
                record.id: score for score, record in relevant_with_scores
            },
        )

    def recommend_for_objective(
        self,
        objective: str,
        *,
        max_history_records: int = DEFAULT_MAX_HISTORY_RECORDS,
    ) -> tuple[HistoricalRecommendation, ...]:
        return self.analyze(
            HistoricalAnalysisRequest(
                objective=objective,
                max_history_records=max_history_records,
            )
        ).recommendations

    def recommend_for_tool(
        self,
        tool_name: str,
        *,
        objective: str | None = None,
        max_history_records: int = DEFAULT_MAX_HISTORY_RECORDS,
    ) -> tuple[HistoricalRecommendation, ...]:
        tool = _required_text(tool_name, "tool_name")
        return self.analyze(
            HistoricalAnalysisRequest(
                objective=objective or tool,
                candidate_tools=(tool,),
                max_history_records=max_history_records,
            )
        ).recommendations

    def build_planning_context(
        self,
        objective: str,
        *,
        max_history_records: int = DEFAULT_MAX_HISTORY_RECORDS,
    ) -> HistoricalPlanningContext:
        return self.analyze(
            HistoricalAnalysisRequest(
                objective=objective,
                max_history_records=max_history_records,
            )
        ).planning_context

    def _select_relevant(
        self,
        records: tuple[ExecutionHistoryRecord, ...],
        request: HistoricalAnalysisRequest,
    ) -> tuple[tuple[int, ExecutionHistoryRecord], ...]:
        candidates: list[tuple[int, ExecutionHistoryRecord]] = []
        seen_ids: set[str] = set()
        for record in records:
            if record.id in seen_ids:
                continue
            seen_ids.add(record.id)
            if request.time_window is not None and not request.time_window.contains(
                record.date
            ):
                continue
            if not request.include_failed and record.state.value == "failed":
                continue
            if not request.include_recovered and record.recovery_types:
                continue
            score = _relevance_score(record, request)
            if score > 0:
                candidates.append((score, record))
        candidates.sort(key=lambda item: item[1].id)
        candidates.sort(key=lambda item: item[1].date, reverse=True)
        candidates.sort(key=lambda item: item[0], reverse=True)
        return tuple(candidates[: request.max_history_records])

    def _recommend(
        self,
        records: tuple[ExecutionHistoryRecord, ...],
    ) -> tuple[HistoricalRecommendation, ...]:
        if len(records) < self._policy.minimum_relevant_executions:
            return (
                _insufficient_history(records),
            )

        recommendations: list[HistoricalRecommendation] = []
        successful = tuple(
            record for record in records if record.final_result in _SUCCESS_RESULTS
        )
        if len(successful) >= self._policy.minimum_repeated_evidence:
            recommendations.append(_previous_success(successful))

        failed_items = _item_occurrences(records, "failed")
        recommendations.extend(
            _frequent_failure(item, supporting)
            for item, supporting in failed_items.items()
            if len(supporting) >= self._policy.minimum_repeated_evidence
        )

        retried = tuple(record for record in records if record.retry_count > 0)
        if len(retried) >= self._policy.minimum_repeated_evidence:
            recommendations.append(_retry_risk(retried))

        recoveries = _successful_recoveries(records)
        recommendations.extend(
            _recovery_available(recovery, supporting)
            for recovery, supporting in recoveries.items()
            if len(supporting) >= self._policy.minimum_successful_recoveries
        )

        high_duration = _high_duration_records(records, self._policy)
        if high_duration:
            recommendations.append(_high_duration(high_duration, records))

        omitted_items = _item_occurrences(records, "omitted")
        recommendations.extend(
            _optional_pattern(item, supporting)
            for item, supporting in omitted_items.items()
            if len(supporting) >= self._policy.minimum_repeated_evidence
        )

        action_records = tuple(record for record in records if record.required_actions)
        if len(action_records) >= self._policy.minimum_repeated_evidence:
            recommendations.append(_user_action_pattern(action_records))

        if not recommendations:
            return (_insufficient_history(records),)
        ordered = sorted(
            recommendations,
            key=lambda item: (
                tuple(HistoricalRecommendationType).index(item.type),
                item.related_tool or "",
                item.related_step or "",
                item.message,
            ),
        )
        return tuple(ordered[: self._policy.max_recommendations])

    def _context(
        self,
        objective: str,
        reviewed_count: int,
        records: tuple[ExecutionHistoryRecord, ...],
        recommendations: tuple[HistoricalRecommendation, ...],
    ) -> HistoricalPlanningContext:
        risks = tuple(
            recommendation.message
            for recommendation in recommendations
            if recommendation.severity
            in {
                HistoricalRecommendationSeverity.CAUTION,
                HistoricalRecommendationSeverity.WARNING,
            }
        )
        recoveries = _unique(
            recommendation.related_step or recommendation.message
            for recommendation in recommendations
            if recommendation.type is HistoricalRecommendationType.RECOVERY_AVAILABLE
        )
        incidents = _unique(
            value
            for recommendation in recommendations
            for value in (
                recommendation.related_tool,
                recommendation.related_step,
                recommendation.related_capability,
            )
            if value is not None
        )
        summary = _summary_text(records, recommendations)
        return HistoricalPlanningContext(
            objective=objective,
            reviewed_execution_count=reviewed_count,
            relevant_execution_ids=tuple(record.id for record in records),
            recommendations=recommendations,
            historical_risks=risks,
            known_recoveries=recoveries,
            incident_tools_or_steps=incidents,
            summary=summary[: self._policy.max_summary_chars],
        )


def _relevance_score(
    record: ExecutionHistoryRecord,
    request: HistoricalAnalysisRequest,
) -> int:
    objective = _normalize_text(request.objective)
    historical = _normalize_text(record.objective)
    score = 0
    if objective == historical:
        score += 100
    shared = _significant_tokens(objective) & _significant_tokens(historical)
    score += len(shared) * 10
    record_tools = {_normalize_text(tool) for tool in record.tool_names}
    requested_tools = {_normalize_text(tool) for tool in request.candidate_tools}
    score += len(record_tools & requested_tools) * 30
    capability_tokens = {
        token
        for capability in request.candidate_capabilities
        for token in _significant_tokens(_normalize_text(capability))
    }
    score += len(capability_tokens & _significant_tokens(historical)) * 15
    if request.operation_type is not None:
        operation = _normalize_text(request.operation_type)
        if operation in historical:
            score += 20
    return score


def _item_occurrences(
    records: tuple[ExecutionHistoryRecord, ...],
    kind: str,
) -> dict[str, tuple[ExecutionHistoryRecord, ...]]:
    grouped: dict[str, list[ExecutionHistoryRecord]] = {}
    for record in records:
        step_ids = (
            record.failed_step_ids if kind == "failed" else record.omitted_step_ids
        )
        items = {
            record.tools_by_step.get(step_id, step_id)
            for step_id in step_ids
        }
        for item in sorted(items):
            grouped.setdefault(item, []).append(record)
    return {item: tuple(values) for item, values in sorted(grouped.items())}


def _successful_recoveries(
    records: tuple[ExecutionHistoryRecord, ...],
) -> dict[str, tuple[ExecutionHistoryRecord, ...]]:
    grouped: dict[str, list[ExecutionHistoryRecord]] = {}
    for record in records:
        if record.final_result not in _SUCCESS_RESULTS:
            continue
        for recovery in record.recovery_types:
            if recovery.startswith("replan:") and (
                record.operational_report.replan_status != "succeeded"
            ):
                continue
            grouped.setdefault(recovery, []).append(record)
    return {item: tuple(values) for item, values in sorted(grouped.items())}


def _high_duration_records(
    records: tuple[ExecutionHistoryRecord, ...],
    policy: HistoricalAdvisorPolicy,
) -> tuple[ExecutionHistoryRecord, ...]:
    if len(records) < policy.high_duration_minimum_records:
        return ()
    durations = tuple(record.duration_seconds for record in records)
    baseline = median(durations)
    if baseline <= 0:
        return ()
    return tuple(
        record
        for record in records
        if record.duration_seconds >= policy.high_duration_minimum_seconds
        and record.duration_seconds >= baseline * policy.high_duration_ratio
    )


def _previous_success(
    records: tuple[ExecutionHistoryRecord, ...],
) -> HistoricalRecommendation:
    return _recommendation(
        HistoricalRecommendationType.PREVIOUS_SUCCESS,
        HistoricalRecommendationSeverity.INFORMATION,
        f"Se observaron {len(records)} ejecuciones similares completadas correctamente.",
        records,
        fact="Ejecuciones similares completadas correctamente.",
    )


def _frequent_failure(
    item: str,
    records: tuple[ExecutionHistoryRecord, ...],
) -> HistoricalRecommendation:
    tools = {tool for record in records for tool in record.tool_names}
    related_tool = item if item in tools else None
    return _recommendation(
        HistoricalRecommendationType.FREQUENT_FAILURE,
        HistoricalRecommendationSeverity.WARNING,
        f"{item} presentó fallos en {len(records)} ejecuciones similares.",
        records,
        fact=f"Fallo repetido asociado a {item}.",
        related_tool=related_tool,
        related_step=None if related_tool else item,
    )


def _retry_risk(
    records: tuple[ExecutionHistoryRecord, ...],
) -> HistoricalRecommendation:
    retried_steps = Counter(
        step_id
        for record in records
        for step_id in set(record.operational_report.retried_step_ids)
    )
    related_step = next(
        (
            step_id
            for step_id, count in sorted(
                retried_steps.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count >= 2
        ),
        None,
    )
    return _recommendation(
        HistoricalRecommendationType.RETRY_RISK,
        HistoricalRecommendationSeverity.CAUTION,
        f"{len(records)} ejecuciones similares necesitaron reintentos.",
        records,
        fact="Uso repetido de reintentos.",
        observed_values=tuple(str(record.retry_count) for record in records),
        related_step=related_step,
    )


def _recovery_available(
    recovery: str,
    records: tuple[ExecutionHistoryRecord, ...],
) -> HistoricalRecommendation:
    return _recommendation(
        HistoricalRecommendationType.RECOVERY_AVAILABLE,
        HistoricalRecommendationSeverity.INFORMATION,
        f"La recuperación {recovery} terminó correctamente en {len(records)} ejecuciones.",
        records,
        fact=f"Recuperación histórica correcta: {recovery}.",
        related_step=recovery,
    )


def _high_duration(
    records: tuple[ExecutionHistoryRecord, ...],
    compared: tuple[ExecutionHistoryRecord, ...],
) -> HistoricalRecommendation:
    return _recommendation(
        HistoricalRecommendationType.HIGH_DURATION,
        HistoricalRecommendationSeverity.CAUTION,
        f"{len(records)} ejecuciones similares tuvieron una duración elevada frente a la mediana comparable.",
        records,
        fact=f"Duración comparada con {len(compared)} ejecuciones similares.",
        observed_values=tuple(f"{record.duration_seconds:.3f}s" for record in records),
    )


def _optional_pattern(
    item: str,
    records: tuple[ExecutionHistoryRecord, ...],
) -> HistoricalRecommendation:
    return _recommendation(
        HistoricalRecommendationType.OPTIONAL_STEP_PATTERN,
        HistoricalRecommendationSeverity.CAUTION,
        f"{item} fue omitido en {len(records)} ejecuciones similares.",
        records,
        fact=f"Omisión repetida de {item}.",
        related_step=item,
    )


def _user_action_pattern(
    records: tuple[ExecutionHistoryRecord, ...],
) -> HistoricalRecommendation:
    actions = _unique(
        action for record in records for action in record.required_actions
    )
    return _recommendation(
        HistoricalRecommendationType.USER_ACTION_PATTERN,
        HistoricalRecommendationSeverity.CAUTION,
        f"{len(records)} ejecuciones similares requirieron intervención del usuario.",
        records,
        fact="Intervención del usuario requerida repetidamente.",
        observed_values=actions,
    )


def _insufficient_history(
    records: tuple[ExecutionHistoryRecord, ...],
) -> HistoricalRecommendation:
    return HistoricalRecommendation(
        type=HistoricalRecommendationType.INSUFFICIENT_HISTORY,
        severity=HistoricalRecommendationSeverity.INFORMATION,
        message="No existe historial relevante suficiente para formular recomendaciones.",
        evidence=(),
        supporting_execution_count=len(records),
        session_ids=tuple(record.id for record in records),
    )


def _recommendation(
    recommendation_type: HistoricalRecommendationType,
    severity: HistoricalRecommendationSeverity,
    message: str,
    records: tuple[ExecutionHistoryRecord, ...],
    *,
    fact: str,
    observed_values: tuple[str, ...] = (),
    related_tool: str | None = None,
    related_step: str | None = None,
) -> HistoricalRecommendation:
    session_ids = tuple(record.id for record in records)
    evidence = HistoricalEvidence(
        fact=fact,
        occurrence_count=len(records),
        session_ids=session_ids,
        observed_values=observed_values,
    )
    return HistoricalRecommendation(
        type=recommendation_type,
        severity=severity,
        message=message,
        evidence=(evidence,),
        supporting_execution_count=len(records),
        session_ids=session_ids,
        related_tool=related_tool,
        related_step=related_step,
    )


def _summary_text(
    records: tuple[ExecutionHistoryRecord, ...],
    recommendations: tuple[HistoricalRecommendation, ...],
) -> str:
    successful = sum(record.final_result in _SUCCESS_RESULTS for record in records)
    lines = [
        "Contexto histórico:",
        f"- Se encontraron {len(records)} ejecuciones similares.",
        f"- {successful} se completaron correctamente.",
    ]
    lines.extend(
        f"- {recommendation.message}"
        for recommendation in recommendations
        if recommendation.type
        is not HistoricalRecommendationType.INSUFFICIENT_HISTORY
    )
    if recommendations and recommendations[0].type is HistoricalRecommendationType.INSUFFICIENT_HISTORY:
        lines.append(f"- {recommendations[0].message}")
    lines.append("- No se aplican cambios automáticos al plan.")
    return "\n".join(lines)


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return _sanitized_text(value)


def _sanitized_text(value: object, *, limit: int = 240) -> str:
    lines = str(value).splitlines() or [str(value)]
    return "\n".join(_safe_text(line) for line in lines)[:limit]


def _normalized_names(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a tuple of strings.")
    return _unique(_required_text(value, name) for value in values)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(
        "".join(
            character if character.isalnum() or character in "._-" else " "
            for character in without_accents
        ).split()
    )


def _significant_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in value.split()
        if len(token) >= 3 and token not in _STOP_WORDS
    )


def _bounded_int(
    value: int,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))

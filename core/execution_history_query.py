"""Structured query service for in-memory execution history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable

from core.execution_history import ExecutionHistory, ExecutionHistoryEntry
from core.execution_trace import TraceStatus


class ExecutionHistorySortField(str, Enum):
    """Supported fields for execution history sorting."""

    STARTED_AT = "started_at"
    FINISHED_AT = "finished_at"
    DURATION_MS = "duration_ms"
    EXECUTION_ID = "execution_id"
    STATUS = "status"


@dataclass(frozen=True, slots=True)
class ExecutionHistoryQuery:
    """Immutable structured criteria for querying execution history.

    Datetime filters are compared exactly as provided. Mixing naive and
    timezone-aware datetimes is rejected with ValueError.
    """

    statuses: tuple[str | TraceStatus, ...] | None = None
    components: tuple[str, ...] | None = None
    actions: tuple[str, ...] | None = None
    min_duration_ms: int | None = None
    max_duration_ms: int | None = None
    started_from: datetime | None = None
    started_until: datetime | None = None
    finished_from: datetime | None = None
    finished_until: datetime | None = None
    limit: int | None = None
    sort_by: str | ExecutionHistorySortField = ExecutionHistorySortField.STARTED_AT
    descending: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionHistoryQueryStats:
    """Immutable statistics for a filtered execution-history result."""

    total_matches: int
    successful_count: int
    failed_count: int
    cancelled_count: int
    average_duration_ms: float
    minimum_duration_ms: int | None
    maximum_duration_ms: int | None
    total_duration_ms: int


@dataclass(frozen=True, slots=True)
class ExecutionHistoryQueryResult:
    """Entries and statistics returned by a history query."""

    entries: tuple[ExecutionHistoryEntry, ...]
    stats: ExecutionHistoryQueryStats


class ExecutionHistoryQueryService:
    """Validate, filter, sort and analyze execution-history entries."""

    def query(
        self,
        history: ExecutionHistory,
        criteria: ExecutionHistoryQuery,
    ) -> ExecutionHistoryQueryResult:
        """Return entries matching criteria and stats for those entries."""
        self._validate(criteria)
        entries = tuple(history)
        filtered = tuple(entry for entry in entries if self._matches(entry, criteria))
        ordered = self._sort(filtered, criteria)
        limited = self._limit(ordered, criteria.limit)
        return ExecutionHistoryQueryResult(
            entries=limited,
            stats=self.analyze(limited),
        )

    def analyze(
        self,
        entries: tuple[ExecutionHistoryEntry, ...],
    ) -> ExecutionHistoryQueryStats:
        """Return simple statistics for the supplied entries only."""
        durations = tuple(entry.metrics.total_duration_ms for entry in entries)
        total_duration = sum(durations)
        return ExecutionHistoryQueryStats(
            total_matches=len(entries),
            successful_count=sum(
                1 for entry in entries if entry.status == TraceStatus.SUCCESS.value
            ),
            failed_count=sum(
                1 for entry in entries if entry.status == TraceStatus.FAILED.value
            ),
            cancelled_count=sum(
                1 for entry in entries if entry.status == TraceStatus.CANCELLED.value
            ),
            average_duration_ms=(
                total_duration / len(durations)
                if durations
                else 0.0
            ),
            minimum_duration_ms=min(durations) if durations else None,
            maximum_duration_ms=max(durations) if durations else None,
            total_duration_ms=total_duration,
        )

    def _validate(
        self,
        criteria: ExecutionHistoryQuery,
    ) -> None:
        if criteria.min_duration_ms is not None and criteria.min_duration_ms < 0:
            raise ValueError("min_duration_ms cannot be negative.")
        if criteria.max_duration_ms is not None and criteria.max_duration_ms < 0:
            raise ValueError("max_duration_ms cannot be negative.")
        if (
            criteria.min_duration_ms is not None
            and criteria.max_duration_ms is not None
            and criteria.min_duration_ms > criteria.max_duration_ms
        ):
            raise ValueError("min_duration_ms cannot be greater than max_duration_ms.")
        if criteria.limit is not None and criteria.limit < 0:
            raise ValueError("History query limit cannot be negative.")
        if criteria.started_from is not None and criteria.started_until is not None:
            _ensure_datetime_order(
                criteria.started_from,
                criteria.started_until,
                "started_at",
            )
        if criteria.finished_from is not None and criteria.finished_until is not None:
            _ensure_datetime_order(
                criteria.finished_from,
                criteria.finished_until,
                "finished_at",
            )
        _sort_field(criteria.sort_by)

    def _matches(
        self,
        entry: ExecutionHistoryEntry,
        criteria: ExecutionHistoryQuery,
    ) -> bool:
        if not _matches_optional_values(entry.status, _status_values(criteria.statuses)):
            return False
        if not _matches_event_field(entry, "component", criteria.components):
            return False
        if not _matches_event_field(entry, "action", criteria.actions):
            return False
        duration = entry.metrics.total_duration_ms
        if criteria.min_duration_ms is not None and duration < criteria.min_duration_ms:
            return False
        if criteria.max_duration_ms is not None and duration > criteria.max_duration_ms:
            return False
        if criteria.started_from is not None and not _datetime_gte(
            entry.started_at,
            criteria.started_from,
            "started_at",
        ):
            return False
        if criteria.started_until is not None and not _datetime_lte(
            entry.started_at,
            criteria.started_until,
            "started_at",
        ):
            return False
        if criteria.finished_from is not None:
            if entry.finished_at is None or not _datetime_gte(
                entry.finished_at,
                criteria.finished_from,
                "finished_at",
            ):
                return False
        if criteria.finished_until is not None:
            if entry.finished_at is None or not _datetime_lte(
                entry.finished_at,
                criteria.finished_until,
                "finished_at",
            ):
                return False
        return True

    def _sort(
        self,
        entries: tuple[ExecutionHistoryEntry, ...],
        criteria: ExecutionHistoryQuery,
    ) -> tuple[ExecutionHistoryEntry, ...]:
        sort_by = _sort_field(criteria.sort_by)
        if sort_by == ExecutionHistorySortField.FINISHED_AT:
            present = tuple(entry for entry in entries if entry.finished_at is not None)
            missing = tuple(entry for entry in entries if entry.finished_at is None)
            return tuple(
                sorted(
                    present,
                    key=lambda entry: entry.finished_at,
                    reverse=criteria.descending,
                )
            ) + missing

        return tuple(
            sorted(
                entries,
                key=lambda entry: _sort_value(entry, sort_by),
                reverse=criteria.descending,
            )
        )

    def _limit(
        self,
        entries: tuple[ExecutionHistoryEntry, ...],
        limit: int | None,
    ) -> tuple[ExecutionHistoryEntry, ...]:
        if limit is None:
            return entries
        if limit == 0:
            return ()
        return entries[:limit]


def _status_values(
    statuses: tuple[str | TraceStatus, ...] | None,
) -> tuple[str, ...] | None:
    if statuses is None:
        return None
    return tuple(status.value if isinstance(status, TraceStatus) else status for status in statuses)


def _matches_optional_values(
    value: str,
    allowed: tuple[str, ...] | None,
) -> bool:
    if allowed is None:
        return True
    if not allowed:
        return False
    return value in allowed


def _matches_event_field(
    entry: ExecutionHistoryEntry,
    field_name: str,
    allowed: tuple[str, ...] | None,
) -> bool:
    if allowed is None:
        return True
    if not allowed:
        return False
    allowed_set = set(allowed)
    return any(getattr(event, field_name) in allowed_set for event in entry.trace.events)


def _sort_field(
    value: str | ExecutionHistorySortField,
) -> ExecutionHistorySortField:
    if isinstance(value, ExecutionHistorySortField):
        return value
    try:
        return ExecutionHistorySortField(value)
    except ValueError as error:
        raise ValueError(f"Unsupported execution history sort field: {value}") from error


def _sort_value(
    entry: ExecutionHistoryEntry,
    sort_by: ExecutionHistorySortField,
) -> object:
    if sort_by == ExecutionHistorySortField.STARTED_AT:
        return entry.started_at
    if sort_by == ExecutionHistorySortField.DURATION_MS:
        return entry.metrics.total_duration_ms
    if sort_by == ExecutionHistorySortField.EXECUTION_ID:
        return entry.execution_id
    if sort_by == ExecutionHistorySortField.STATUS:
        return entry.status
    raise ValueError(f"Unsupported execution history sort field: {sort_by}")


def _ensure_datetime_order(
    start: datetime,
    end: datetime,
    field_name: str,
) -> None:
    try:
        invalid = start > end
    except TypeError as error:
        raise ValueError(
            f"Incompatible datetime values for {field_name} range."
        ) from error
    if invalid:
        raise ValueError(f"Invalid {field_name} range.")


def _datetime_gte(
    value: datetime,
    boundary: datetime,
    field_name: str,
) -> bool:
    try:
        return value >= boundary
    except TypeError as error:
        raise ValueError(
            f"Incompatible datetime values for {field_name} filter."
        ) from error


def _datetime_lte(
    value: datetime,
    boundary: datetime,
    field_name: str,
) -> bool:
    try:
        return value <= boundary
    except TypeError as error:
        raise ValueError(
            f"Incompatible datetime values for {field_name} filter."
        ) from error

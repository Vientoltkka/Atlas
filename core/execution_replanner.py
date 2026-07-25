"""Deterministic replanning decisions for failed execution goals."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from core.execution_plan_registry import ExecutionPlanReference

if TYPE_CHECKING:
    from core.planner import ExecutionPlan


MAX_REPLANS = 3


class ReplanningError(ValueError):
    """Raised when a replanning contract is malformed."""


class ReplanningStrategy(str, Enum):
    """Supported deterministic replanning strategies."""

    NO_REPLAN = "NO_REPLAN"
    ALTERNATIVE_WORKFLOW = "ALTERNATIVE_WORKFLOW"
    ALTERNATIVE_PLAN = "ALTERNATIVE_PLAN"


class ReplanningStatus(str, Enum):
    """Stable statuses for replanning decisions."""

    NOT_REQUESTED = "NOT_REQUESTED"
    DISABLED = "DISABLED"
    LIMIT_REACHED = "LIMIT_REACHED"
    GOAL_ALREADY_SATISFIED = "GOAL_ALREADY_SATISFIED"
    FAILURE_NOT_REPLANNABLE = "FAILURE_NOT_REPLANNABLE"
    NO_ALTERNATIVE_PLAN = "NO_ALTERNATIVE_PLAN"
    REPLANNED = "REPLANNED"
    INVALID_REQUEST = "INVALID_REQUEST"
    REPLANNING_FAILED = "REPLANNING_FAILED"


@dataclass(frozen=True, slots=True)
class ReplanningPolicy:
    """Immutable policy controlling whether failed goals can be replanned."""

    enabled: bool = False
    max_replans: int = 0
    strategy: ReplanningStrategy = ReplanningStrategy.NO_REPLAN
    retryable_goal_reasons: tuple[str, ...] = ()
    retryable_execution_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ReplanningError("enabled must be a bool.")
        if isinstance(self.strategy, str):
            object.__setattr__(self, "strategy", ReplanningStrategy(self.strategy))
        if isinstance(self.max_replans, bool) or not isinstance(self.max_replans, int):
            raise ReplanningError("max_replans must be an integer.")
        if self.max_replans < 0 or self.max_replans > MAX_REPLANS:
            raise ReplanningError(f"max_replans must be between 0 and {MAX_REPLANS}.")
        object.__setattr__(
            self,
            "retryable_goal_reasons",
            _normalize_reason_codes(self.retryable_goal_reasons, "retryable_goal_reasons"),
        )
        object.__setattr__(
            self,
            "retryable_execution_errors",
            _normalize_reason_codes(self.retryable_execution_errors, "retryable_execution_errors"),
        )
        if not self.enabled and self.max_replans != 0:
            raise ReplanningError("disabled replanning requires max_replans=0.")
        if self.enabled and self.strategy is ReplanningStrategy.NO_REPLAN and self.max_replans != 0:
            raise ReplanningError("NO_REPLAN strategy requires max_replans=0.")


@dataclass(frozen=True, slots=True)
class ReplanningCandidate:
    """Existing plan candidate that may replace a failed plan."""

    plan: "ExecutionPlan"
    plan_signature: str | None = None
    workflow_reference: ExecutionPlanReference | None = None
    library_id: str | None = None
    score: int = 0

    def __post_init__(self) -> None:
        if not _is_execution_plan(self.plan):
            raise ReplanningError("candidate plan must be ExecutionPlan.")
        signature = self.plan_signature or _plan_signature(self.plan)
        if not isinstance(signature, str) or not signature:
            raise ReplanningError("candidate plan_signature must be a non-empty string.")
        object.__setattr__(self, "plan_signature", signature)
        if self.workflow_reference is not None and not isinstance(self.workflow_reference, ExecutionPlanReference):
            raise ReplanningError("workflow_reference must be ExecutionPlanReference or None.")
        if self.library_id is not None and (not isinstance(self.library_id, str) or not self.library_id.strip()):
            raise ReplanningError("library_id must be a non-empty string or None.")
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise ReplanningError("score must be an integer.")


@dataclass(frozen=True, slots=True)
class ReplanningHistoryEntry:
    """Safe history entry for one deterministic replanning decision."""

    attempt: int
    status: ReplanningStatus
    reason: str
    previous_plan_signature: str | None
    replacement_plan_signature: str | None
    workflow_plan_id: str | None = None
    workflow_version: str | None = None
    library_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 0:
            raise ReplanningError("history attempt must be a non-negative int.")
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(self, "reason", _safe_text(self.reason, "reason"))
        for field_name in (
            "previous_plan_signature",
            "replacement_plan_signature",
            "workflow_plan_id",
            "workflow_version",
            "library_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _safe_text(value, field_name))


@dataclass(frozen=True, slots=True)
class ReplanningRequest:
    """Inputs for a pure replanning decision."""

    original_plan: "ExecutionPlan"
    failed_plan: "ExecutionPlan"
    execution_result: object | None = None
    goal_verification_result: object | None = None
    candidates: tuple[ReplanningCandidate, ...] = ()
    replan_attempts: int = 0
    excluded_plan_signatures: tuple[str, ...] = ()
    history: tuple[ReplanningHistoryEntry, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_execution_plan(self.original_plan):
            raise ReplanningError("original_plan must be ExecutionPlan.")
        if not _is_execution_plan(self.failed_plan):
            raise ReplanningError("failed_plan must be ExecutionPlan.")
        if isinstance(self.replan_attempts, bool) or not isinstance(self.replan_attempts, int):
            raise ReplanningError("replan_attempts must be an integer.")
        if self.replan_attempts < 0:
            raise ReplanningError("replan_attempts cannot be negative.")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not all(isinstance(candidate, ReplanningCandidate) for candidate in self.candidates):
            raise ReplanningError("candidates must contain ReplanningCandidate values.")
        object.__setattr__(
            self,
            "excluded_plan_signatures",
            _normalize_text_tuple(self.excluded_plan_signatures, "excluded_plan_signatures"),
        )
        object.__setattr__(self, "history", tuple(self.history))
        if not all(isinstance(entry, ReplanningHistoryEntry) for entry in self.history):
            raise ReplanningError("history must contain ReplanningHistoryEntry values.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class ReplanningDecision:
    """Decision produced by the deterministic replanner."""

    should_replan: bool
    status: ReplanningStatus
    reason: str
    replacement_plan: "ExecutionPlan | None" = None
    replan_attempts: int = 0
    previous_plan_signature: str | None = None
    replacement_plan_signature: str | None = None
    history_entry: ReplanningHistoryEntry | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.should_replan, bool):
            raise ReplanningError("should_replan must be a bool.")
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(self, "reason", _safe_text(self.reason, "reason"))
        if self.replacement_plan is not None and not _is_execution_plan(self.replacement_plan):
            raise ReplanningError("replacement_plan must be ExecutionPlan or None.")


class ExecutionReplanner:
    """Choose a replacement from existing candidates without executing anything."""

    def decide(
        self,
        policy: ReplanningPolicy | None,
        request: ReplanningRequest,
    ) -> ReplanningDecision:
        """Return a deterministic replanning decision."""
        try:
            if not isinstance(request, ReplanningRequest):
                raise ReplanningError("request must be ReplanningRequest.")
            active_policy = ReplanningPolicy() if policy is None else policy
            if not isinstance(active_policy, ReplanningPolicy):
                raise ReplanningError("policy must be ReplanningPolicy or None.")
            if not active_policy.enabled:
                return _decision(ReplanningStatus.DISABLED, "replanning policy is disabled", request)
            if active_policy.strategy is ReplanningStrategy.NO_REPLAN:
                return _decision(ReplanningStatus.DISABLED, "replanning strategy is NO_REPLAN", request)
            if _goal_is_satisfied(request):
                return _decision(ReplanningStatus.GOAL_ALREADY_SATISFIED, "goal already satisfied", request)
            if request.replan_attempts >= active_policy.max_replans:
                return _decision(ReplanningStatus.LIMIT_REACHED, "replanning limit reached", request)
            if not _failure_is_replannable(active_policy, request):
                return _decision(ReplanningStatus.FAILURE_NOT_REPLANNABLE, "failure is not replannable", request)

            failed_signature = _plan_signature(request.failed_plan)
            excluded = set(request.excluded_plan_signatures)
            excluded.add(failed_signature)
            excluded.add(_plan_signature(request.original_plan))
            for entry in request.history:
                if entry.replacement_plan_signature:
                    excluded.add(entry.replacement_plan_signature)
                if entry.previous_plan_signature:
                    excluded.add(entry.previous_plan_signature)

            for candidate in request.candidates:
                candidate_signature = str(candidate.plan_signature)
                if candidate_signature in excluded:
                    continue
                if candidate_signature == failed_signature:
                    continue
                attempt = request.replan_attempts + 1
                history_entry = ReplanningHistoryEntry(
                    attempt=attempt,
                    status=ReplanningStatus.REPLANNED,
                    reason="alternative plan selected",
                    previous_plan_signature=failed_signature,
                    replacement_plan_signature=candidate_signature,
                    workflow_plan_id=(
                        candidate.workflow_reference.plan_id
                        if candidate.workflow_reference is not None
                        else None
                    ),
                    workflow_version=(
                        candidate.workflow_reference.version
                        if candidate.workflow_reference is not None
                        else None
                    ),
                    library_id=candidate.library_id,
                )
                return ReplanningDecision(
                    should_replan=True,
                    status=ReplanningStatus.REPLANNED,
                    reason="alternative plan selected",
                    replacement_plan=candidate.plan,
                    replan_attempts=attempt,
                    previous_plan_signature=failed_signature,
                    replacement_plan_signature=candidate_signature,
                    history_entry=history_entry,
                )

            return _decision(ReplanningStatus.NO_ALTERNATIVE_PLAN, "no alternative plan available", request)
        except ReplanningError as error:
            return ReplanningDecision(
                should_replan=False,
                status=ReplanningStatus.INVALID_REQUEST,
                reason=str(error),
            )
        except (TypeError, ValueError) as error:
            return ReplanningDecision(
                should_replan=False,
                status=ReplanningStatus.REPLANNING_FAILED,
                reason=type(error).__name__,
            )


def copy_replanning_policy(
    policy: ReplanningPolicy | None,
) -> ReplanningPolicy | None:
    """Return an immutable replanning policy copy."""
    if policy is None:
        return None
    if not isinstance(policy, ReplanningPolicy):
        raise TypeError("replanning_policy must be ReplanningPolicy or None.")
    return ReplanningPolicy(
        enabled=policy.enabled,
        max_replans=policy.max_replans,
        strategy=policy.strategy,
        retryable_goal_reasons=policy.retryable_goal_reasons,
        retryable_execution_errors=policy.retryable_execution_errors,
    )


def replanning_policy_to_dict(
    policy: ReplanningPolicy | None,
) -> dict[str, Any] | None:
    """Serialize a replanning policy to safe JSON-compatible data."""
    if policy is None:
        return None
    return {
        "$type": "replanning_policy",
        "enabled": policy.enabled,
        "max_replans": policy.max_replans,
        "strategy": policy.strategy.value,
        "retryable_goal_reasons": list(policy.retryable_goal_reasons),
        "retryable_execution_errors": list(policy.retryable_execution_errors),
    }


def replanning_policy_from_dict(
    payload: Any,
) -> ReplanningPolicy | None:
    """Load a replanning policy from checkpoint data."""
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "$type",
        "enabled",
        "max_replans",
        "strategy",
        "retryable_goal_reasons",
        "retryable_execution_errors",
    }:
        raise ValueError("replanning policy must be an explicit object or null.")
    if payload.get("$type") != "replanning_policy":
        raise ValueError("replanning policy type is invalid.")
    return ReplanningPolicy(
        enabled=_required_bool(payload, "enabled"),
        max_replans=_required_int(payload, "max_replans"),
        strategy=ReplanningStrategy(_required_str(payload, "strategy")),
        retryable_goal_reasons=_str_tuple(payload, "retryable_goal_reasons"),
        retryable_execution_errors=_str_tuple(payload, "retryable_execution_errors"),
    )


def replanning_history_to_dict(
    history: tuple[ReplanningHistoryEntry, ...],
) -> list[dict[str, Any]]:
    """Serialize replanning history."""
    return [
        {
            "attempt": entry.attempt,
            "status": entry.status.value,
            "reason": entry.reason,
            "previous_plan_signature": entry.previous_plan_signature,
            "replacement_plan_signature": entry.replacement_plan_signature,
            "workflow_plan_id": entry.workflow_plan_id,
            "workflow_version": entry.workflow_version,
            "library_id": entry.library_id,
        }
        for entry in history
    ]


def replanning_history_from_dict(
    payload: Any,
) -> tuple[ReplanningHistoryEntry, ...]:
    """Load safe replanning history from checkpoint data."""
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError("replanning history must be a list.")
    entries: list[ReplanningHistoryEntry] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("replanning history entries must be objects.")
        entries.append(
            ReplanningHistoryEntry(
                attempt=_required_int(item, "attempt"),
                status=ReplanningStatus(_required_str(item, "status")),
                reason=_required_str(item, "reason"),
                previous_plan_signature=_optional_str(item, "previous_plan_signature"),
                replacement_plan_signature=_optional_str(item, "replacement_plan_signature"),
                workflow_plan_id=_optional_str(item, "workflow_plan_id"),
                workflow_version=_optional_str(item, "workflow_version"),
                library_id=_optional_str(item, "library_id"),
            )
        )
    return tuple(entries)


def _decision(
    status: ReplanningStatus,
    reason: str,
    request: ReplanningRequest,
) -> ReplanningDecision:
    failed_signature = _plan_signature(request.failed_plan)
    return ReplanningDecision(
        should_replan=False,
        status=status,
        reason=reason,
        replan_attempts=request.replan_attempts,
        previous_plan_signature=failed_signature,
    )


def _failure_is_replannable(
    policy: ReplanningPolicy,
    request: ReplanningRequest,
) -> bool:
    if request.goal_verification_result is not None:
        reason = request.goal_verification_result.reason.value
        if reason in policy.retryable_goal_reasons:
            return True
    if request.execution_result is not None:
        error_code = str(request.execution_result.error_code or "").upper()
        if error_code in policy.retryable_execution_errors:
            return True
        for result in request.execution_result.step_results:
            if str(result.error_code or "").upper() in policy.retryable_execution_errors:
                return True
    return False


def _goal_is_satisfied(
    request: ReplanningRequest,
) -> bool:
    return request.goal_verification_result is not None and request.goal_verification_result.satisfied


def _plan_signature(
    plan: "ExecutionPlan",
) -> str:
    from core.execution_plan_validator import plan_signature

    return plan_signature(plan)


def _is_execution_plan(
    value: object,
) -> bool:
    from core.planner import ExecutionPlan

    return isinstance(value, ExecutionPlan)


def _normalize_reason_codes(
    values: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    normalized = _normalize_text_tuple(values, field_name)
    return tuple(dict.fromkeys(item.upper() for item in normalized))


def _normalize_text_tuple(
    values: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, tuple):
        raise ReplanningError(f"{field_name} must be a tuple of strings.")
    normalized: list[str] = []
    for value in values:
        normalized.append(_safe_text(value, field_name))
    return tuple(normalized)


def _safe_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise ReplanningError("metadata must be a mapping.")
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        safe[_safe_text(key, "metadata key")] = _safe_value(value)
    return safe


def _safe_value(
    value: object,
) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ReplanningError("metadata values must be primitive safe values.")


def _safe_text(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise ReplanningError(f"{field_name} must be a string.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ReplanningError(f"{field_name} cannot be empty.")
    return normalized


def _status(
    value: ReplanningStatus | str,
) -> ReplanningStatus:
    if isinstance(value, ReplanningStatus):
        return value
    if isinstance(value, str):
        return ReplanningStatus(value)
    raise ReplanningError("status must be ReplanningStatus.")


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null.")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return value


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a bool.")
    return value


def _str_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings.")
    return tuple(value)

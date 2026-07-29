"""Deterministic authorization and one-shot dispatch for validated executions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
from threading import RLock
from types import MappingProxyType
from typing import Callable, Mapping

from core.execution_plan_validator import PlanValidationResult, plan_signature
from core.execution_report import _safe_text
from core.execution_retry import RetryStrategy
from core.execution_strategy import (
    ExecutionStrategySelectionResult,
    ExecutionStrategyType,
    GlobalExecutionSafetyPolicy,
)
from core.planner import ExecutionPlan


_TERMINAL_OR_ACTIVE_PLAN_STATES = frozenset(
    {"executing", "running", "completed", "cancelled", "failed", "terminal"}
)
_SUPERVISED_STRATEGIES = frozenset(
    {
        ExecutionStrategyType.SUPERVISED,
        ExecutionStrategyType.CONFIRMATION_REQUIRED,
        ExecutionStrategyType.RECOVERY_PREPARED,
    }
)
_MAX_TRACE_ITEMS = 30


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionAuthorizationDecision(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    CONFIRMATION_PENDING = "CONFIRMATION_PENDING"
    MANUAL_REVIEW_PENDING = "MANUAL_REVIEW_PENDING"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"
    ALREADY_DISPATCHED = "ALREADY_DISPATCHED"


class AuthorizationValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    BLOCKED = "BLOCKED"
    PENDING = "PENDING"


class ConfirmationScope(str, Enum):
    PLAN = "PLAN"
    STEP = "STEP"


class DispatchStatus(str, Enum):
    DISPATCHED = "DISPATCHED"
    ALREADY_DISPATCHED = "ALREADY_DISPATCHED"
    BLOCKED = "BLOCKED"
    FAILED_AFTER_DISPATCH = "FAILED_AFTER_DISPATCH"


@dataclass(frozen=True, slots=True)
class ExecutionAuthorizationPolicy:
    """Conservative local controls for authorization and confirmation validity."""

    confirmation_ttl_seconds: int = 900
    require_single_use_confirmations: bool = True
    require_plan_signature: bool = True
    require_executor: bool = True
    require_supervisor_for_reinforced_strategy: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.confirmation_ttl_seconds, bool)
            or not isinstance(self.confirmation_ttl_seconds, int)
        ):
            raise TypeError("confirmation_ttl_seconds must be an int.")
        if not 1 <= self.confirmation_ttl_seconds <= 86_400:
            raise ValueError(
                "confirmation_ttl_seconds must be between 1 and 86400."
            )
        for name in (
            "require_single_use_confirmations",
            "require_plan_signature",
            "require_executor",
            "require_supervisor_for_reinforced_strategy",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")


@dataclass(frozen=True, slots=True)
class ExecutionConfirmationReference:
    """Non-sensitive reference to one confirmation in the existing flow."""

    confirmation_id: str
    plan_id: str
    plan_signature: str
    step_id: str
    tool: str | None
    scope: ConfirmationScope
    issued_at: datetime
    expires_at: datetime
    granted: bool = False
    revoked: bool = False
    consumed: bool = False

    def __post_init__(self) -> None:
        for name in ("confirmation_id", "plan_id", "plan_signature", "step_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if self.tool is not None:
            object.__setattr__(self, "tool", _safe_text(self.tool))
        if not isinstance(self.scope, ConfirmationScope):
            raise TypeError("scope must be ConfirmationScope.")
        if not isinstance(self.issued_at, datetime) or not isinstance(
            self.expires_at,
            datetime,
        ):
            raise TypeError("confirmation timestamps must be datetime values.")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at.")
        for name in ("granted", "revoked", "consumed"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")

    def granted_copy(self) -> "ExecutionConfirmationReference":
        return replace(self, granted=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "confirmation_id": self.confirmation_id,
            "plan_id": self.plan_id,
            "plan_signature": self.plan_signature,
            "step_id": self.step_id,
            "tool": self.tool,
            "scope": self.scope.value,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "granted": self.granted,
            "revoked": self.revoked,
            "consumed": self.consumed,
        }


@dataclass(frozen=True, slots=True)
class ExecutionAuthorizationRequest:
    """Closed request at the boundary immediately before dispatch."""

    plan: ExecutionPlan
    plan_validation: PlanValidationResult
    strategy_selection: ExecutionStrategySelectionResult
    plan_state: str = "planned"
    session_state: str | None = None
    required_confirmations: tuple[ExecutionConfirmationReference, ...] = ()
    granted_confirmations: tuple[ExecutionConfirmationReference, ...] = ()
    protected_step_ids: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    executor_available: bool = True
    supervisor_available: bool = True
    replanner_available: bool = False
    source: str | None = None
    safety_policy: GlobalExecutionSafetyPolicy = field(
        default_factory=GlobalExecutionSafetyPolicy
    )
    authorization_policy: ExecutionAuthorizationPolicy = field(
        default_factory=ExecutionAuthorizationPolicy
    )

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExecutionPlan):
            raise TypeError("plan must be an ExecutionPlan.")
        if not isinstance(self.plan_validation, PlanValidationResult):
            raise TypeError("plan_validation must be PlanValidationResult.")
        if not isinstance(
            self.strategy_selection,
            ExecutionStrategySelectionResult,
        ):
            raise TypeError(
                "strategy_selection must be ExecutionStrategySelectionResult."
            )
        if not isinstance(self.safety_policy, GlobalExecutionSafetyPolicy):
            raise TypeError("safety_policy must be GlobalExecutionSafetyPolicy.")
        if not isinstance(
            self.authorization_policy,
            ExecutionAuthorizationPolicy,
        ):
            raise TypeError(
                "authorization_policy must be ExecutionAuthorizationPolicy."
            )
        if not isinstance(self.plan_state, str) or not self.plan_state.strip():
            raise ValueError("plan_state must be a non-empty string.")
        if self.session_state is not None and not self.session_state.strip():
            raise ValueError("session_state must be non-empty or None.")
        for name in (
            "executor_available",
            "supervisor_available",
            "replanner_available",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        object.__setattr__(
            self,
            "required_confirmations",
            tuple(self.required_confirmations),
        )
        object.__setattr__(
            self,
            "granted_confirmations",
            tuple(self.granted_confirmations),
        )
        if any(
            not isinstance(item, ExecutionConfirmationReference)
            for item in (*self.required_confirmations, *self.granted_confirmations)
        ):
            raise TypeError(
                "confirmation collections must contain "
                "ExecutionConfirmationReference values."
            )
        known_steps = {step.id for step in self.plan.ordered_steps}
        protected = _unique_safe(self.protected_step_ids)
        if any(step_id not in known_steps for step_id in protected):
            raise ValueError("protected_step_ids contains an unknown step.")
        object.__setattr__(self, "protected_step_ids", protected)
        object.__setattr__(
            self,
            "required_capabilities",
            _unique_safe(self.required_capabilities),
        )
        if self.source is not None:
            object.__setattr__(self, "source", _safe_text(self.source))


@dataclass(frozen=True, slots=True)
class ExecutionAuthorizationResult:
    """Traceable authorization decision bound to one exact plan signature."""

    authorization_id: str
    plan_id: str
    plan_signature: str
    strategy: ExecutionStrategyType
    decision: ExecutionAuthorizationDecision
    reason: str
    activated_rules: tuple[str, ...]
    pending_confirmation_ids: tuple[str, ...]
    affected_step_ids: tuple[str, ...]
    missing_components: tuple[str, ...]
    validation_status: AuthorizationValidationStatus
    dispatch_allowed: bool
    decided_at: datetime
    trace: Mapping[str, object]
    plan: ExecutionPlan
    plan_validation: PlanValidationResult
    strategy_selection: ExecutionStrategySelectionResult

    def __post_init__(self) -> None:
        for name in ("authorization_id", "plan_id", "plan_signature"):
            if not isinstance(getattr(self, name), str) or not getattr(
                self,
                name,
            ).strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.strategy, ExecutionStrategyType):
            raise TypeError("strategy must be ExecutionStrategyType.")
        if not isinstance(self.decision, ExecutionAuthorizationDecision):
            raise TypeError("decision must be ExecutionAuthorizationDecision.")
        if not isinstance(
            self.validation_status,
            AuthorizationValidationStatus,
        ):
            raise TypeError(
                "validation_status must be AuthorizationValidationStatus."
            )
        if type(self.dispatch_allowed) is not bool:
            raise TypeError("dispatch_allowed must be a bool.")
        object.__setattr__(self, "reason", _safe_text(self.reason))
        object.__setattr__(
            self,
            "activated_rules",
            _unique_safe(self.activated_rules)[:_MAX_TRACE_ITEMS],
        )
        object.__setattr__(
            self,
            "pending_confirmation_ids",
            _unique_safe(self.pending_confirmation_ids),
        )
        object.__setattr__(
            self,
            "affected_step_ids",
            _unique_safe(self.affected_step_ids),
        )
        object.__setattr__(
            self,
            "missing_components",
            _unique_safe(self.missing_components),
        )
        object.__setattr__(
            self,
            "trace",
            MappingProxyType(_safe_mapping(self.trace)),
        )

    def persisted_snapshot(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "authorization_id": self.authorization_id,
                "plan_signature": self.plan_signature,
                "strategy": self.strategy.value,
                "decision": self.decision.value,
                "reason": self.reason,
                "pending_confirmation_ids": list(
                    self.pending_confirmation_ids
                ),
                "satisfied_confirmation_ids": list(
                    self.trace.get("satisfied_confirmation_ids", ())
                ),
                "dispatch_allowed": self.dispatch_allowed,
                "consumed": False,
                "decided_at": self.decided_at.isoformat(),
                "dispatch": None,
                "session_id": None,
                "legacy_default": False,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "authorization_id": self.authorization_id,
            "plan_id": self.plan_id,
            "plan_signature": self.plan_signature,
            "strategy": self.strategy.value,
            "decision": self.decision.value,
            "reason": self.reason,
            "activated_rules": list(self.activated_rules),
            "pending_confirmation_ids": list(self.pending_confirmation_ids),
            "affected_step_ids": list(self.affected_step_ids),
            "missing_components": list(self.missing_components),
            "validation_status": self.validation_status.value,
            "dispatch_allowed": self.dispatch_allowed,
            "decided_at": self.decided_at.isoformat(),
            "trace": dict(self.trace),
        }


@dataclass(frozen=True, slots=True)
class ExecutionDispatchResult:
    authorization_id: str
    status: DispatchStatus
    dispatched: bool
    consumed: bool
    session_id: str | None
    reason: str
    dispatched_at: datetime
    error: str | None = None
    execution_result: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _safe_text(self.reason))
        if self.error is not None:
            object.__setattr__(self, "error", _safe_text(self.error))

    def persisted_snapshot(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "status": self.status.value,
                "dispatched": self.dispatched,
                "consumed": self.consumed,
                "session_id": self.session_id,
                "reason": self.reason,
                "dispatched_at": self.dispatched_at.isoformat(),
                "error": self.error,
            }
        )


class ExecutionAuthorizationGate:
    """Authorize an exact validated plan and strategy without executing it."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        already_dispatched: Callable[[str], bool] | None = None,
        confirmation_consumed: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._clock = clock or _utc_now
        self._already_dispatched = already_dispatched or (lambda _: False)
        self._confirmation_consumed = (
            confirmation_consumed or (lambda _confirmation_id, _step_id: False)
        )

    def authorize(
        self,
        request: ExecutionAuthorizationRequest,
    ) -> ExecutionAuthorizationResult:
        if not isinstance(request, ExecutionAuthorizationRequest):
            raise TypeError("request must be ExecutionAuthorizationRequest.")
        now = self._clock()
        signature = plan_signature(request.plan)
        strategy = request.strategy_selection.strategy
        authorization_id = _authorization_id(request, signature)
        rules: list[str] = ["plan_signature_bound"]
        missing: list[str] = []
        affected = request.protected_step_ids

        if self._already_dispatched(authorization_id):
            return self._result(
                request,
                authorization_id,
                signature,
                ExecutionAuthorizationDecision.ALREADY_DISPATCHED,
                "This dispatch permit was already used.",
                (*rules, "authorization_already_dispatched"),
                AuthorizationValidationStatus.BLOCKED,
                False,
                now,
                affected=affected,
            )
        if not request.plan_validation.is_valid:
            return self._rejected(
                request,
                authorization_id,
                signature,
                "The execution plan is invalid.",
                (*rules, "invalid_plan_rejected"),
                now,
            )
        if (
            request.plan_validation.plan_signature != signature
            or strategy.plan_id != signature
            or request.strategy_selection.original_plan is not request.plan
        ):
            return self._rejected(
                request,
                authorization_id,
                signature,
                "Plan or strategy signature no longer matches the approved request.",
                (*rules, "plan_or_strategy_signature_mismatch"),
                now,
            )
        if (
            strategy.type is ExecutionStrategyType.MANUAL_REVIEW_REQUIRED
            and not strategy.configuration.execution_allowed
        ):
            return self._result(
                request,
                authorization_id,
                signature,
                ExecutionAuthorizationDecision.MANUAL_REVIEW_PENDING,
                "Manual review is required before execution.",
                (*rules, "manual_review_blocks_automatic_dispatch"),
                AuthorizationValidationStatus.PENDING,
                False,
                now,
                affected=affected,
            )
        if (
            not request.strategy_selection.validation.is_valid
            or strategy.validation_status.value == "INVALID"
        ):
            return self._rejected(
                request,
                authorization_id,
                signature,
                "The execution strategy is invalid.",
                (*rules, "invalid_strategy_rejected"),
                now,
            )
        normalized_state = request.plan_state.strip().lower()
        normalized_session = (
            ""
            if request.session_state is None
            else request.session_state.strip().lower()
        )
        if (
            normalized_state in _TERMINAL_OR_ACTIVE_PLAN_STATES
            or normalized_session in _TERMINAL_OR_ACTIVE_PLAN_STATES
        ):
            return self._rejected(
                request,
                authorization_id,
                signature,
                "The plan or session is already active or terminal.",
                (*rules, "active_or_terminal_state_rejected"),
                now,
            )
        if request.authorization_policy.require_executor and not request.executor_available:
            missing.append("Executor")
        if (
            request.authorization_policy.require_supervisor_for_reinforced_strategy
            and strategy.type in _SUPERVISED_STRATEGIES
            and not request.supervisor_available
        ):
            missing.append("ExecutionSupervisor")
        if (
            strategy.type is ExecutionStrategyType.RECOVERY_PREPARED
            and strategy.configuration.allow_replanning
            and not request.replanner_available
        ):
            missing.append("ExecutionReplanner")
        if missing:
            return self._result(
                request,
                authorization_id,
                signature,
                ExecutionAuthorizationDecision.BLOCKED,
                "A required execution component is unavailable.",
                (*rules, "required_component_missing"),
                AuthorizationValidationStatus.BLOCKED,
                False,
                now,
                missing=tuple(missing),
                affected=affected,
            )
        retry_error = _retry_invariant_error(request)
        if retry_error is not None:
            return self._rejected(
                request,
                authorization_id,
                signature,
                retry_error,
                (*rules, "retry_policy_invariant_failed"),
                now,
            )
        if any(
            self._confirmation_consumed(item.confirmation_id, item.step_id)
            for item in request.granted_confirmations
        ):
            return self._rejected(
                request,
                authorization_id,
                signature,
                "Confirmation was already consumed.",
                (*rules, "confirmation_reference_already_consumed"),
                now,
                affected=affected,
            )
        confirmation_error, pending, satisfied = _evaluate_confirmations(
            request,
            signature,
            now,
        )
        if confirmation_error is not None:
            return self._rejected(
                request,
                authorization_id,
                signature,
                confirmation_error,
                (*rules, "confirmation_reference_rejected"),
                now,
                affected=affected,
            )
        if pending:
            return self._result(
                request,
                authorization_id,
                signature,
                ExecutionAuthorizationDecision.CONFIRMATION_PENDING,
                "Required confirmations are still pending.",
                (*rules, "confirmation_required_before_dispatch"),
                AuthorizationValidationStatus.PENDING,
                False,
                now,
                pending=pending,
                affected=affected,
                satisfied=satisfied,
            )
        return self._result(
            request,
            authorization_id,
            signature,
            ExecutionAuthorizationDecision.AUTHORIZED,
            "Validated plan and strategy are authorized for one dispatch.",
            (*rules, "all_authorization_checks_passed"),
            AuthorizationValidationStatus.VALID,
            True,
            now,
            affected=affected,
            satisfied=satisfied,
        )

    def _rejected(
        self,
        request: ExecutionAuthorizationRequest,
        authorization_id: str,
        signature: str,
        reason: str,
        rules: tuple[str, ...],
        now: datetime,
        *,
        affected: tuple[str, ...] = (),
    ) -> ExecutionAuthorizationResult:
        return self._result(
            request,
            authorization_id,
            signature,
            ExecutionAuthorizationDecision.REJECTED,
            reason,
            rules,
            AuthorizationValidationStatus.INVALID,
            False,
            now,
            affected=affected,
        )

    @staticmethod
    def _result(
        request: ExecutionAuthorizationRequest,
        authorization_id: str,
        signature: str,
        decision: ExecutionAuthorizationDecision,
        reason: str,
        rules: tuple[str, ...],
        validation_status: AuthorizationValidationStatus,
        dispatch_allowed: bool,
        now: datetime,
        *,
        pending: tuple[str, ...] = (),
        missing: tuple[str, ...] = (),
        affected: tuple[str, ...] = (),
        satisfied: tuple[str, ...] = (),
    ) -> ExecutionAuthorizationResult:
        return ExecutionAuthorizationResult(
            authorization_id=authorization_id,
            plan_id=request.strategy_selection.strategy.plan_id,
            plan_signature=signature,
            strategy=request.strategy_selection.strategy.type,
            decision=decision,
            reason=reason,
            activated_rules=tuple(sorted(set(rules))),
            pending_confirmation_ids=pending,
            affected_step_ids=affected,
            missing_components=missing,
            validation_status=validation_status,
            dispatch_allowed=dispatch_allowed,
            decided_at=now,
            trace={
                "source": request.source,
                "required_capabilities": request.required_capabilities,
                "required_confirmation_count": len(
                    request.required_confirmations
                ),
                "satisfied_confirmation_ids": satisfied,
                "satisfied_confirmation_references": tuple(
                    f"{item.confirmation_id}:{item.step_id}"
                    for item in request.granted_confirmations
                    if item.confirmation_id in satisfied
                ),
                "plan_unchanged": True,
            },
            plan=request.plan,
            plan_validation=request.plan_validation,
            strategy_selection=request.strategy_selection,
        )


class ExecutionDispatcher:
    """Atomically consume one authorization and deliver it to existing execution."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or _utc_now
        self._lock = RLock()
        self._records: dict[str, ExecutionDispatchResult] = {}
        self._consumed_confirmations: set[tuple[str, str]] = set()

    def has_dispatched(self, authorization_id: str) -> bool:
        with self._lock:
            return authorization_id in self._records

    def get(self, authorization_id: str) -> ExecutionDispatchResult | None:
        with self._lock:
            return self._records.get(authorization_id)

    def confirmation_consumed(
        self,
        confirmation_id: str,
        step_id: str,
    ) -> bool:
        with self._lock:
            return (confirmation_id, step_id) in self._consumed_confirmations

    def dispatch(
        self,
        authorization: ExecutionAuthorizationResult,
        deliver: Callable[
            [
                ExecutionPlan,
                PlanValidationResult,
                ExecutionStrategySelectionResult,
            ],
            object,
        ],
    ) -> ExecutionDispatchResult:
        if not isinstance(authorization, ExecutionAuthorizationResult):
            raise TypeError("authorization must be ExecutionAuthorizationResult.")
        if not callable(deliver):
            raise TypeError("deliver must be callable.")
        now = self._clock()
        with self._lock:
            existing = self._records.get(authorization.authorization_id)
            if existing is not None:
                return replace(
                    existing,
                    status=DispatchStatus.ALREADY_DISPATCHED,
                    reason="Dispatch permit was already consumed.",
                )
            precondition = _dispatch_precondition_error(authorization)
            if precondition is not None:
                return ExecutionDispatchResult(
                    authorization_id=authorization.authorization_id,
                    status=DispatchStatus.BLOCKED,
                    dispatched=False,
                    consumed=False,
                    session_id=None,
                    reason=precondition,
                    dispatched_at=now,
                )
            in_progress = ExecutionDispatchResult(
                authorization_id=authorization.authorization_id,
                status=DispatchStatus.DISPATCHED,
                dispatched=True,
                consumed=True,
                session_id=None,
                reason="Dispatch permit consumed and delivered once.",
                dispatched_at=now,
            )
            self._records[authorization.authorization_id] = in_progress
            for reference in authorization.trace.get(
                "satisfied_confirmation_references",
                (),
            ):
                if not isinstance(reference, str) or ":" not in reference:
                    continue
                confirmation_id, step_id = reference.rsplit(":", 1)
                self._consumed_confirmations.add((confirmation_id, step_id))
        try:
            execution_result = deliver(
                authorization.plan,
                authorization.plan_validation,
                authorization.strategy_selection,
            )
        except Exception as error:
            failed = replace(
                in_progress,
                status=DispatchStatus.FAILED_AFTER_DISPATCH,
                reason="Dispatch was delivered but execution raised an error.",
                error=f"{type(error).__name__}: {error}",
            )
            with self._lock:
                self._records[authorization.authorization_id] = failed
            return failed
        completed = replace(
            in_progress,
            session_id=_execution_session_id(execution_result),
            execution_result=execution_result,
        )
        with self._lock:
            self._records[authorization.authorization_id] = completed
        return completed


def build_confirmation_references(
    plan: ExecutionPlan,
    strategy_selection: ExecutionStrategySelectionResult,
    *,
    confirmation_id: str,
    issued_at: datetime | None = None,
    ttl_seconds: int = 900,
) -> tuple[ExecutionConfirmationReference, ...]:
    """Bind the coordinator's existing confirmation token to protected steps."""
    now = issued_at or _utc_now()
    expires_at = now + timedelta(seconds=ttl_seconds)
    signature = plan_signature(plan)
    steps = {step.id: step for step in plan.ordered_steps}
    protected = strategy_selection.strategy.configuration.confirmation_step_ids
    return tuple(
        ExecutionConfirmationReference(
            confirmation_id=confirmation_id,
            plan_id=strategy_selection.strategy.plan_id,
            plan_signature=signature,
            step_id=step_id,
            tool=steps[step_id].tool,
            scope=ConfirmationScope.STEP,
            issued_at=now,
            expires_at=expires_at,
        )
        for step_id in protected
    )


def legacy_authorization_snapshot() -> Mapping[str, object]:
    """Represent old sessions without inventing an authorization decision."""
    return MappingProxyType(
        {
            "decision": "LEGACY_NOT_RECORDED",
            "legacy_default": True,
            "reason": (
                "An explicit dispatch permit was not recorded for this session."
            ),
            "consumed": None,
            "dispatch": None,
            "session_id": None,
        }
    )


def authorization_with_dispatch(
    authorization: ExecutionAuthorizationResult,
    dispatch: ExecutionDispatchResult,
) -> Mapping[str, object]:
    snapshot = dict(authorization.persisted_snapshot())
    snapshot["consumed"] = dispatch.consumed
    snapshot["dispatch"] = dict(dispatch.persisted_snapshot())
    snapshot["session_id"] = dispatch.session_id
    return MappingProxyType(snapshot)


def _evaluate_confirmations(
    request: ExecutionAuthorizationRequest,
    signature: str,
    now: datetime,
) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    required = request.required_confirmations
    granted = request.granted_confirmations
    requires_confirmation = (
        request.plan.requires_confirmation
        or request.strategy_selection.strategy.configuration.requires_confirmation
        or bool(required)
    )
    if not requires_confirmation:
        return None, (), ()
    if not required:
        return "Confirmation is required but no scoped reference exists.", (), ()
    granted_by_key = {
        (item.confirmation_id, item.step_id): item for item in granted
    }
    required_keys = {
        (item.confirmation_id, item.step_id) for item in required
    }
    for item in granted:
        key = (item.confirmation_id, item.step_id)
        if key not in required_keys:
            return "Confirmation belongs to another plan or step.", (), ()
        if item.plan_id != request.strategy_selection.strategy.plan_id:
            return "Confirmation belongs to another plan.", (), ()
        if item.plan_signature != signature:
            return "Confirmation targets an obsolete plan version.", (), ()
        expected = next(
            value
            for value in required
            if (value.confirmation_id, value.step_id) == key
        )
        if item.tool != expected.tool or item.scope is not expected.scope:
            return "Confirmation scope does not match the protected action.", (), ()
        if not item.granted:
            return "Confirmation was not granted.", (), ()
        if item.revoked:
            return "Confirmation was revoked.", (), ()
        if item.expires_at <= now:
            return "Confirmation expired.", (), ()
        if (
            request.authorization_policy.require_single_use_confirmations
            and item.consumed
        ):
            return "Confirmation was already consumed.", (), ()
    pending = tuple(
        item.confirmation_id
        for item in required
        if (item.confirmation_id, item.step_id) not in granted_by_key
    )
    satisfied = tuple(
        item.confirmation_id
        for item in required
        if (item.confirmation_id, item.step_id) in granted_by_key
    )
    return None, _unique_safe(pending), _unique_safe(satisfied)


def _retry_invariant_error(
    request: ExecutionAuthorizationRequest,
) -> str | None:
    config = request.strategy_selection.strategy.configuration
    if not config.preserve_plan_retry_policies:
        return "Strategy does not preserve plan retry policies."
    plan_limit = max(
        (
            step.retry_policy.max_attempts
            for step in request.plan.ordered_steps
            if step.retry_policy is not None
        ),
        default=1,
    )
    if config.effective_max_retry_attempts > plan_limit:
        return "Strategy cannot elevate retries above the validated plan."
    for step in request.plan.ordered_steps:
        policy = step.retry_policy
        if (
            policy is not None
            and policy.strategy is RetryStrategy.NO_RETRY
            and policy.max_attempts != 1
        ):
            return "NO_RETRY must remain limited to one attempt."
    return None


def _dispatch_precondition_error(
    authorization: ExecutionAuthorizationResult,
) -> str | None:
    if (
        authorization.decision is not ExecutionAuthorizationDecision.AUTHORIZED
        or not authorization.dispatch_allowed
        or authorization.validation_status is not AuthorizationValidationStatus.VALID
    ):
        return "Only a valid AUTHORIZED result can be dispatched."
    if plan_signature(authorization.plan) != authorization.plan_signature:
        return "Plan changed after approval."
    if authorization.strategy_selection.strategy.plan_id != authorization.plan_id:
        return "Strategy changed after approval."
    if not authorization.strategy_selection.executable:
        return "Blocking strategy cannot be dispatched."
    return None


def _authorization_id(
    request: ExecutionAuthorizationRequest,
    signature: str,
) -> str:
    payload = {
        "plan_signature": signature,
        "strategy": request.strategy_selection.strategy.type.value,
        "required_confirmations": sorted(
            (item.confirmation_id, item.step_id)
            for item in request.required_confirmations
        ),
        "source": request.source,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "authorization." + hashlib.sha256(encoded).hexdigest()[:24]


def _execution_session_id(result: object) -> str | None:
    if isinstance(result, str) and result.strip():
        return _safe_text(result)
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("execution_session_id")
        if isinstance(value, str) and value.strip():
            return _safe_text(value)
    value = getattr(result, "session_id", None)
    return _safe_text(value) if isinstance(value, str) and value.strip() else None


def _safe_mapping(value: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        safe_key = _safe_text(key)
        if safe_key == "[redacted]":
            continue
        if item is None or isinstance(item, (str, int, float, bool)):
            result[safe_key] = _safe_text(item) if isinstance(item, str) else item
        elif isinstance(item, (tuple, list)):
            result[safe_key] = tuple(_safe_text(entry) for entry in item[:20])
    return result


def _unique_safe(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = _safe_text(value)
        if text and text not in result:
            result.append(text)
    return tuple(result)

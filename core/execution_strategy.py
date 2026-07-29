"""Deterministic selection of operational strategies for validated plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping

from core.execution_history_advisor import (
    HistoricalPlanningContext,
    HistoricalRecommendationType,
)
from core.execution_plan_validator import (
    ExecutionPlanValidator,
    PlanValidationResult,
    plan_signature,
)
from core.execution_report import _safe_text
from core.historical_plan_adjustment import HistoricalPlanAdjustmentResult
from core.planner import ExecutionPlan


MAX_STRATEGY_FACTORS = 20
MAX_STRATEGY_TRACE_ENTRIES = 30


class ExecutionStrategyType(str, Enum):
    STANDARD = "STANDARD"
    CONSERVATIVE = "CONSERVATIVE"
    SUPERVISED = "SUPERVISED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    RECOVERY_PREPARED = "RECOVERY_PREPARED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


class ExecutionStrategyRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SupervisionMode(str, Enum):
    NORMAL = "NORMAL"
    DETAILED = "DETAILED"
    REINFORCED = "REINFORCED"


class FailureBehavior(str, Enum):
    CURRENT_POLICY = "CURRENT_POLICY"
    STOP_ON_AMBIGUOUS_ERROR = "STOP_ON_AMBIGUOUS_ERROR"
    STOP_ON_CRITICAL_FAILURE = "STOP_ON_CRITICAL_FAILURE"
    CONTROLLED_RECOVERY = "CONTROLLED_RECOVERY"
    BLOCK_EXECUTION = "BLOCK_EXECUTION"


class StrategyValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class GlobalExecutionSafetyPolicy:
    """Closed limits that a strategy may tighten but never relax."""

    max_retry_attempts: int = 10
    max_replans: int = 1
    max_steps: int = 100
    require_supervision_for_criticality: int = 1
    repeated_history_threshold: int = 2
    allow_replanning: bool = True

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("max_retry_attempts", 1, 10),
            ("max_replans", 0, 10),
            ("max_steps", 1, 10_000),
            ("require_supervision_for_criticality", 1, 100),
            ("repeated_history_threshold", 1, 50),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int.")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        if type(self.allow_replanning) is not bool:
            raise TypeError("allow_replanning must be a bool.")


@dataclass(frozen=True, slots=True)
class ExecutionStrategyConfiguration:
    """Resolved operational controls; it contains no functional plan data."""

    supervision_mode: SupervisionMode
    progress_required: bool
    record_all_transitions: bool
    cancel_on_critical_failure: bool
    effective_max_retry_attempts: int
    preserve_plan_retry_policies: bool
    failure_behavior: FailureBehavior
    requires_confirmation: bool
    confirmation_step_ids: tuple[str, ...]
    allow_replanning: bool
    max_replans: int
    max_steps: int
    execution_allowed: bool
    recovery_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.supervision_mode, SupervisionMode):
            raise TypeError("supervision_mode must be SupervisionMode.")
        if not isinstance(self.failure_behavior, FailureBehavior):
            raise TypeError("failure_behavior must be FailureBehavior.")
        for name in (
            "progress_required",
            "record_all_transitions",
            "cancel_on_critical_failure",
            "preserve_plan_retry_policies",
            "requires_confirmation",
            "allow_replanning",
            "execution_allowed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        if not 1 <= self.effective_max_retry_attempts <= 10:
            raise ValueError("effective_max_retry_attempts must be between 1 and 10.")
        if not 0 <= self.max_replans <= 10:
            raise ValueError("max_replans must be between 0 and 10.")
        if not 1 <= self.max_steps <= 10_000:
            raise ValueError("max_steps must be between 1 and 10000.")
        object.__setattr__(
            self,
            "confirmation_step_ids",
            _unique_safe(self.confirmation_step_ids),
        )
        object.__setattr__(
            self,
            "recovery_hints",
            _unique_safe(self.recovery_hints),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "supervision_mode": self.supervision_mode.value,
            "progress_required": self.progress_required,
            "record_all_transitions": self.record_all_transitions,
            "cancel_on_critical_failure": self.cancel_on_critical_failure,
            "effective_max_retry_attempts": self.effective_max_retry_attempts,
            "preserve_plan_retry_policies": self.preserve_plan_retry_policies,
            "failure_behavior": self.failure_behavior.value,
            "requires_confirmation": self.requires_confirmation,
            "confirmation_step_ids": list(self.confirmation_step_ids),
            "allow_replanning": self.allow_replanning,
            "max_replans": self.max_replans,
            "max_steps": self.max_steps,
            "execution_allowed": self.execution_allowed,
            "recovery_hints": list(self.recovery_hints),
        }


@dataclass(frozen=True, slots=True)
class ExecutionStrategy:
    """Closed, serializable strategy associated with one immutable plan."""

    type: ExecutionStrategyType
    plan_id: str
    reason: str
    factors: tuple[str, ...]
    risk_level: ExecutionStrategyRisk
    configuration: ExecutionStrategyConfiguration
    validation_status: StrategyValidationStatus

    def __post_init__(self) -> None:
        if not isinstance(self.type, ExecutionStrategyType):
            raise TypeError("type must be ExecutionStrategyType.")
        if not isinstance(self.risk_level, ExecutionStrategyRisk):
            raise TypeError("risk_level must be ExecutionStrategyRisk.")
        if not isinstance(self.validation_status, StrategyValidationStatus):
            raise TypeError("validation_status must be StrategyValidationStatus.")
        if not isinstance(self.configuration, ExecutionStrategyConfiguration):
            raise TypeError("configuration must be ExecutionStrategyConfiguration.")
        if not self.plan_id.strip():
            raise ValueError("plan_id must be non-empty.")
        object.__setattr__(self, "reason", _safe_text(self.reason))
        object.__setattr__(
            self,
            "factors",
            _unique_safe(self.factors[:MAX_STRATEGY_FACTORS]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type.value,
            "plan_id": self.plan_id,
            "reason": self.reason,
            "factors": list(self.factors),
            "risk_level": self.risk_level.value,
            "configuration": self.configuration.to_dict(),
            "validation_status": self.validation_status.value,
        }


@dataclass(frozen=True, slots=True)
class ExecutionStrategySelectionRequest:
    """Immutable input assembled after plan validation."""

    plan: ExecutionPlan
    plan_validation: PlanValidationResult
    historical_adjustment: HistoricalPlanAdjustmentResult | None = None
    historical_context: HistoricalPlanningContext | None = None
    detected_risks: tuple[str, ...] = ()
    maximum_criticality: int | None = None
    has_side_effects: bool | None = None
    confirmation_step_ids: tuple[str, ...] = ()
    supervisor_available: bool = True
    replanner_available: bool = False
    confirmation_available: bool = True
    safety_policy: GlobalExecutionSafetyPolicy = field(
        default_factory=GlobalExecutionSafetyPolicy
    )

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExecutionPlan):
            raise TypeError("plan must be an ExecutionPlan.")
        if not isinstance(self.plan_validation, PlanValidationResult):
            raise TypeError("plan_validation must be PlanValidationResult.")
        if self.historical_adjustment is not None and not isinstance(
            self.historical_adjustment,
            HistoricalPlanAdjustmentResult,
        ):
            raise TypeError(
                "historical_adjustment must be HistoricalPlanAdjustmentResult or None."
            )
        if self.historical_context is not None and not isinstance(
            self.historical_context,
            HistoricalPlanningContext,
        ):
            raise TypeError(
                "historical_context must be HistoricalPlanningContext or None."
            )
        if not isinstance(self.safety_policy, GlobalExecutionSafetyPolicy):
            raise TypeError("safety_policy must be GlobalExecutionSafetyPolicy.")
        for name in (
            "supervisor_available",
            "replanner_available",
            "confirmation_available",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        if self.has_side_effects is not None and type(self.has_side_effects) is not bool:
            raise TypeError("has_side_effects must be a bool or None.")
        criticality = (
            max((step.criticality for step in self.plan.ordered_steps), default=0)
            if self.maximum_criticality is None
            else self.maximum_criticality
        )
        if isinstance(criticality, bool) or not isinstance(criticality, int):
            raise TypeError("maximum_criticality must be an int or None.")
        if criticality < 0:
            raise ValueError("maximum_criticality cannot be negative.")
        object.__setattr__(self, "maximum_criticality", criticality)
        object.__setattr__(
            self,
            "has_side_effects",
            (
                any(not step.side_effect_free for step in self.plan.ordered_steps)
                if self.has_side_effects is None
                else self.has_side_effects
            ),
        )
        object.__setattr__(
            self,
            "detected_risks",
            _unique_safe((*self.plan.detected_risks, *self.detected_risks)),
        )
        confirmation_ids = self.confirmation_step_ids
        if self.plan.requires_confirmation and not confirmation_ids:
            confirmation_ids = tuple(
                step.id
                for step in self.plan.ordered_steps
                if not step.side_effect_free
            ) or tuple(step.id for step in self.plan.ordered_steps)
        known_ids = {step.id for step in self.plan.ordered_steps}
        if any(step_id not in known_ids for step_id in confirmation_ids):
            raise ValueError("confirmation_step_ids contains an unknown step.")
        object.__setattr__(
            self,
            "confirmation_step_ids",
            _unique_safe(confirmation_ids),
        )


@dataclass(frozen=True, slots=True)
class StrategySelectionTrace:
    considered: tuple[ExecutionStrategyType, ...]
    activated_rules: tuple[str, ...]
    discarded: tuple[str, ...]
    historical_context_used: bool
    safe_fallback: ExecutionStrategyType | None
    final_decision: str

    def to_dict(self) -> dict[str, object]:
        return {
            "considered": [item.value for item in self.considered],
            "activated_rules": list(self.activated_rules),
            "discarded": list(self.discarded),
            "historical_context_used": self.historical_context_used,
            "safe_fallback": (
                None if self.safe_fallback is None else self.safe_fallback.value
            ),
            "final_decision": self.final_decision,
        }


@dataclass(frozen=True, slots=True)
class ExecutionStrategyValidationResult:
    status: StrategyValidationStatus
    is_valid: bool
    executable: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", _unique_safe(self.errors))
        object.__setattr__(self, "warnings", _unique_safe(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "executable": self.executable,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ExecutionStrategySelectionResult:
    original_plan: ExecutionPlan
    strategy: ExecutionStrategy
    validation: ExecutionStrategyValidationResult
    trace: StrategySelectionTrace
    summary: str

    @property
    def executable(self) -> bool:
        return self.validation.executable

    def persisted_snapshot(self) -> Mapping[str, object]:
        """Return only the bounded operational subset stored with a session."""
        return MappingProxyType(
            {
                "strategy": self.strategy.type.value,
                "plan_id": self.strategy.plan_id,
                "reason": self.strategy.reason,
                "risk_level": self.strategy.risk_level.value,
                "configuration": self.strategy.configuration.to_dict(),
                "trace": {
                    "activated_rules": list(self.trace.activated_rules),
                    "safe_fallback": (
                        None
                        if self.trace.safe_fallback is None
                        else self.trace.safe_fallback.value
                    ),
                    "final_decision": self.trace.final_decision,
                },
                "validation": self.validation.to_dict(),
            }
        )


class ExecutionStrategyValidator:
    """Validate operational configuration without executing or changing a plan."""

    def validate(
        self,
        request: ExecutionStrategySelectionRequest,
        strategy: ExecutionStrategy,
    ) -> ExecutionStrategyValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        config = strategy.configuration
        expected_plan_id = (
            request.plan_validation.plan_signature
            if request.plan_validation.is_valid
            else None
        )
        if not request.plan_validation.is_valid:
            errors.append("The execution plan is invalid.")
        if request.plan.status != "planned" or any(
            step.status != "pending" for step in request.plan.ordered_steps
        ):
            errors.append("The plan is already executing or terminal.")
        if expected_plan_id is not None and strategy.plan_id != expected_plan_id:
            errors.append("The strategy does not target the validated plan.")
        if len(request.plan.ordered_steps) > config.max_steps:
            errors.append("The plan exceeds the strategy execution limit.")
        if (
            strategy.type
            in {
                ExecutionStrategyType.SUPERVISED,
                ExecutionStrategyType.CONFIRMATION_REQUIRED,
                ExecutionStrategyType.RECOVERY_PREPARED,
            }
            and not request.supervisor_available
        ):
            errors.append("The selected strategy requires an available supervisor.")
        if config.requires_confirmation and not request.confirmation_available:
            errors.append("The confirmation mechanism is unavailable.")
        if config.allow_replanning and not request.replanner_available:
            errors.append("The selected strategy requires an available replanner.")
        if request.plan.requires_confirmation and not config.requires_confirmation:
            errors.append("The strategy reduces an existing confirmation requirement.")
        if config.effective_max_retry_attempts > request.safety_policy.max_retry_attempts:
            errors.append("The strategy exceeds the global retry limit.")
        plan_retry_limit = max(
            (
                step.retry_policy.max_attempts
                for step in request.plan.ordered_steps
                if step.retry_policy is not None
            ),
            default=1,
        )
        if plan_retry_limit > config.effective_max_retry_attempts:
            errors.append(
                "The plan retry policy exceeds the resolved operational limit."
            )
        if config.max_replans > request.safety_policy.max_replans:
            errors.append("The strategy exceeds the global replanning limit.")
        if config.allow_replanning and not request.safety_policy.allow_replanning:
            errors.append("The strategy weakens the global replanning policy.")
        if not config.preserve_plan_retry_policies:
            errors.append("The strategy must preserve plan retry policies.")
        if strategy.type is ExecutionStrategyType.MANUAL_REVIEW_REQUIRED:
            if config.execution_allowed:
                errors.append("Manual review strategy cannot allow execution.")
            return ExecutionStrategyValidationResult(
                status=StrategyValidationStatus.BLOCKED,
                is_valid=not errors,
                executable=False,
                errors=tuple(errors),
                warnings=tuple(warnings),
            )
        if not config.execution_allowed:
            errors.append("An executable strategy cannot block execution.")
        return ExecutionStrategyValidationResult(
            status=(
                StrategyValidationStatus.VALID
                if not errors
                else StrategyValidationStatus.INVALID
            ),
            is_valid=not errors,
            executable=not errors and config.execution_allowed,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )


class ExecutionStrategySelector:
    """Select the safest applicable strategy using explicit deterministic rules."""

    _PRECEDENCE = (
        ExecutionStrategyType.MANUAL_REVIEW_REQUIRED,
        ExecutionStrategyType.CONFIRMATION_REQUIRED,
        ExecutionStrategyType.SUPERVISED,
        ExecutionStrategyType.RECOVERY_PREPARED,
        ExecutionStrategyType.CONSERVATIVE,
        ExecutionStrategyType.STANDARD,
    )

    def __init__(
        self,
        validator: ExecutionStrategyValidator | None = None,
    ) -> None:
        self._validator = validator or ExecutionStrategyValidator()

    @property
    def precedence(self) -> tuple[ExecutionStrategyType, ...]:
        return self._PRECEDENCE

    def select(
        self,
        request: ExecutionStrategySelectionRequest,
    ) -> ExecutionStrategySelectionResult:
        if not isinstance(request, ExecutionStrategySelectionRequest):
            raise TypeError("request must be ExecutionStrategySelectionRequest.")
        original_signature = plan_signature(request.plan)
        candidates, rules, factors = self._candidates(request)
        selected = next(item for item in self._PRECEDENCE if item in candidates)
        fallback: ExecutionStrategyType | None = None
        discarded: list[str] = []

        strategy = self._build_strategy(selected, request, rules, factors)
        validation = self._validator.validate(request, strategy)
        if not validation.is_valid and selected is not ExecutionStrategyType.MANUAL_REVIEW_REQUIRED:
            discarded.append(
                f"{selected.value}: " + "; ".join(validation.errors)
            )
            fallback = ExecutionStrategyType.MANUAL_REVIEW_REQUIRED
            strategy = self._build_strategy(
                fallback,
                request,
                (*rules, "invalid_strategy_safe_block"),
                (*factors, *validation.errors),
            )
            validation = self._validator.validate(request, strategy)

        if plan_signature(request.plan) != original_signature:
            raise RuntimeError("Execution strategy selection mutated the plan.")
        for candidate in self._PRECEDENCE:
            if candidate is strategy.type:
                continue
            if candidate in candidates:
                discarded.append(
                    f"{candidate.value}: lower precedence than {strategy.type.value}."
                )
            else:
                discarded.append(
                    f"{candidate.value}: activation rule not met."
                )
        final_decision = "EXECUTE" if validation.executable else "BLOCK"
        trace = StrategySelectionTrace(
            considered=self._PRECEDENCE,
            activated_rules=tuple(sorted(set(rules))),
            discarded=tuple(sorted(discarded))[:MAX_STRATEGY_TRACE_ENTRIES],
            historical_context_used=request.historical_context is not None,
            safe_fallback=fallback,
            final_decision=final_decision,
        )
        strategy = ExecutionStrategy(
            type=strategy.type,
            plan_id=strategy.plan_id,
            reason=strategy.reason,
            factors=strategy.factors,
            risk_level=strategy.risk_level,
            configuration=strategy.configuration,
            validation_status=validation.status,
        )
        return ExecutionStrategySelectionResult(
            original_plan=request.plan,
            strategy=strategy,
            validation=validation,
            trace=trace,
            summary=_selection_summary(strategy, final_decision),
        )

    def _candidates(
        self,
        request: ExecutionStrategySelectionRequest,
    ) -> tuple[
        set[ExecutionStrategyType],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        candidates = {ExecutionStrategyType.STANDARD}
        rules: list[str] = ["valid_low_risk_standard"]
        factors: list[str] = []
        if not request.plan_validation.is_valid:
            candidates.add(ExecutionStrategyType.MANUAL_REVIEW_REQUIRED)
            rules.append("invalid_plan_blocks_execution")
            factors.extend(request.plan_validation.errors)
        if (
            request.historical_adjustment is not None
            and request.historical_adjustment.requires_manual_review
        ):
            candidates.add(ExecutionStrategyType.MANUAL_REVIEW_REQUIRED)
            rules.append("historical_manual_review_pending")
            factors.append("Historical adjustment requires manual review.")
        if request.plan.requires_confirmation or request.confirmation_step_ids:
            candidates.add(ExecutionStrategyType.CONFIRMATION_REQUIRED)
            rules.append("existing_confirmation_preserved")
            factors.append("The validated plan requires confirmation.")
        if (
            request.maximum_criticality
            >= request.safety_policy.require_supervision_for_criticality
        ):
            candidates.add(ExecutionStrategyType.SUPERVISED)
            rules.append("critical_step_requires_supervision")
            factors.append(
                f"Maximum plan criticality is {request.maximum_criticality}."
            )
        repeated_history = self._has_repeated_historical_risk(request)
        if repeated_history:
            candidates.add(ExecutionStrategyType.CONSERVATIVE)
            rules.append("repeated_historical_risk")
            factors.append("Historical risk is supported by repeated executions.")
        if self._has_recovery(request):
            if request.replanner_available and request.safety_policy.allow_replanning:
                candidates.add(ExecutionStrategyType.RECOVERY_PREPARED)
                rules.append("validated_recovery_available")
                factors.append("Historical recovery guidance is available.")
            else:
                candidates.add(ExecutionStrategyType.SUPERVISED)
                rules.append("recovery_unavailable_safe_supervision")
                factors.append(
                    "Recovery guidance exists but the replanner is unavailable."
                )
        if request.detected_risks:
            candidates.add(ExecutionStrategyType.CONSERVATIVE)
            rules.append("detected_risk_conservative")
            factors.extend(f"Detected risk: {risk}" for risk in request.detected_risks)
        if request.has_side_effects:
            candidates.add(ExecutionStrategyType.CONSERVATIVE)
            rules.append("side_effects_conservative")
            factors.append("The plan contains steps with declared side effects.")
        return candidates, tuple(rules), tuple(factors)

    @staticmethod
    def _has_repeated_historical_risk(
        request: ExecutionStrategySelectionRequest,
    ) -> bool:
        context = request.historical_context
        if context is None:
            return False
        risk_types = {
            HistoricalRecommendationType.FREQUENT_FAILURE,
            HistoricalRecommendationType.RETRY_RISK,
        }
        return any(
            recommendation.type in risk_types
            and recommendation.supporting_execution_count
            >= request.safety_policy.repeated_history_threshold
            for recommendation in context.recommendations
        )

    @staticmethod
    def _has_recovery(request: ExecutionStrategySelectionRequest) -> bool:
        context = request.historical_context
        if context is None:
            return False
        return bool(context.known_recoveries) or any(
            recommendation.type is HistoricalRecommendationType.RECOVERY_AVAILABLE
            for recommendation in context.recommendations
        )

    def _build_strategy(
        self,
        strategy_type: ExecutionStrategyType,
        request: ExecutionStrategySelectionRequest,
        rules: tuple[str, ...],
        factors: tuple[str, ...],
    ) -> ExecutionStrategy:
        policy = request.safety_policy
        plan_retry_limit = max(
            (
                step.retry_policy.max_attempts
                for step in request.plan.ordered_steps
                if step.retry_policy is not None
            ),
            default=1,
        )
        retry_limit = min(plan_retry_limit, policy.max_retry_attempts)
        confirmation = request.plan.requires_confirmation or bool(
            request.confirmation_step_ids
        )
        recovery_hints = (
            ()
            if request.historical_context is None
            else request.historical_context.known_recoveries
        )
        config = ExecutionStrategyConfiguration(
            supervision_mode=SupervisionMode.NORMAL,
            progress_required=False,
            record_all_transitions=False,
            cancel_on_critical_failure=False,
            effective_max_retry_attempts=retry_limit,
            preserve_plan_retry_policies=True,
            failure_behavior=FailureBehavior.CURRENT_POLICY,
            requires_confirmation=confirmation,
            confirmation_step_ids=request.confirmation_step_ids,
            allow_replanning=(
                policy.allow_replanning
                and request.replanner_available
                and policy.max_replans > 0
            ),
            max_replans=(
                policy.max_replans
                if (
                    policy.allow_replanning
                    and request.replanner_available
                    and policy.max_replans > 0
                )
                else 0
            ),
            max_steps=min(policy.max_steps, max(1, len(request.plan.ordered_steps))),
            execution_allowed=True,
        )
        reason = "The validated plan is simple and has no restrictive signal."
        risk = ExecutionStrategyRisk.LOW
        if strategy_type is ExecutionStrategyType.CONSERVATIVE:
            config = ExecutionStrategyConfiguration(
                supervision_mode=SupervisionMode.DETAILED,
                progress_required=True,
                record_all_transitions=True,
                cancel_on_critical_failure=True,
                effective_max_retry_attempts=retry_limit,
                preserve_plan_retry_policies=True,
                failure_behavior=FailureBehavior.STOP_ON_AMBIGUOUS_ERROR,
                requires_confirmation=confirmation,
                confirmation_step_ids=request.confirmation_step_ids,
                allow_replanning=False,
                max_replans=0,
                max_steps=config.max_steps,
                execution_allowed=True,
            )
            reason = "Repeated historical risk requires conservative limits."
            risk = ExecutionStrategyRisk.MEDIUM
        elif strategy_type is ExecutionStrategyType.RECOVERY_PREPARED:
            allow_replanning = (
                policy.allow_replanning
                and request.replanner_available
                and policy.max_replans > 0
            )
            config = ExecutionStrategyConfiguration(
                supervision_mode=SupervisionMode.DETAILED,
                progress_required=True,
                record_all_transitions=True,
                cancel_on_critical_failure=True,
                effective_max_retry_attempts=retry_limit,
                preserve_plan_retry_policies=True,
                failure_behavior=FailureBehavior.CONTROLLED_RECOVERY,
                requires_confirmation=confirmation,
                confirmation_step_ids=request.confirmation_step_ids,
                allow_replanning=allow_replanning,
                max_replans=policy.max_replans if allow_replanning else 0,
                max_steps=config.max_steps,
                execution_allowed=True,
                recovery_hints=recovery_hints,
            )
            reason = "Validated recovery guidance is prepared but not executed."
            risk = ExecutionStrategyRisk.MEDIUM
        elif strategy_type is ExecutionStrategyType.SUPERVISED:
            config = ExecutionStrategyConfiguration(
                supervision_mode=SupervisionMode.REINFORCED,
                progress_required=True,
                record_all_transitions=True,
                cancel_on_critical_failure=True,
                effective_max_retry_attempts=retry_limit,
                preserve_plan_retry_policies=True,
                failure_behavior=FailureBehavior.STOP_ON_CRITICAL_FAILURE,
                requires_confirmation=confirmation,
                confirmation_step_ids=request.confirmation_step_ids,
                allow_replanning=False,
                max_replans=0,
                max_steps=config.max_steps,
                execution_allowed=True,
            )
            reason = "A critical plan step requires reinforced supervision."
            risk = ExecutionStrategyRisk.HIGH
        elif strategy_type is ExecutionStrategyType.CONFIRMATION_REQUIRED:
            config = ExecutionStrategyConfiguration(
                supervision_mode=SupervisionMode.REINFORCED,
                progress_required=True,
                record_all_transitions=True,
                cancel_on_critical_failure=True,
                effective_max_retry_attempts=retry_limit,
                preserve_plan_retry_policies=True,
                failure_behavior=FailureBehavior.STOP_ON_CRITICAL_FAILURE,
                requires_confirmation=True,
                confirmation_step_ids=request.confirmation_step_ids,
                allow_replanning=False,
                max_replans=0,
                max_steps=config.max_steps,
                execution_allowed=True,
            )
            reason = "Protected plan steps require explicit confirmation."
            risk = ExecutionStrategyRisk.HIGH
        elif strategy_type is ExecutionStrategyType.MANUAL_REVIEW_REQUIRED:
            config = ExecutionStrategyConfiguration(
                supervision_mode=SupervisionMode.REINFORCED,
                progress_required=True,
                record_all_transitions=True,
                cancel_on_critical_failure=True,
                effective_max_retry_attempts=1,
                preserve_plan_retry_policies=True,
                failure_behavior=FailureBehavior.BLOCK_EXECUTION,
                requires_confirmation=confirmation,
                confirmation_step_ids=request.confirmation_step_ids,
                allow_replanning=False,
                max_replans=0,
                max_steps=config.max_steps,
                execution_allowed=False,
            )
            reason = (
                "Execution is blocked until the existing authorization mechanism "
                "completes manual review."
            )
            risk = ExecutionStrategyRisk.CRITICAL
        return ExecutionStrategy(
            type=strategy_type,
            plan_id=(
                request.plan_validation.plan_signature
                or plan_signature(request.plan)
            ),
            reason=reason,
            factors=tuple(sorted(set(factors))),
            risk_level=risk,
            configuration=config,
            validation_status=StrategyValidationStatus.VALID,
        )


class ExecutionStrategyGate:
    """Prevent a blocking or invalid strategy from reaching an executor callback."""

    def execute(
        self,
        selection: ExecutionStrategySelectionResult,
        executor: Callable[[ExecutionStrategyConfiguration], object],
    ) -> object | None:
        if not isinstance(selection, ExecutionStrategySelectionResult):
            raise TypeError("selection must be ExecutionStrategySelectionResult.")
        if not selection.executable:
            return None
        return executor(selection.strategy.configuration)


def build_strategy_request(
    plan: ExecutionPlan,
    validation: PlanValidationResult,
    *,
    historical_adjustment: HistoricalPlanAdjustmentResult | None = None,
    historical_context: HistoricalPlanningContext | None = None,
    supervisor_available: bool = True,
    replanner_available: bool = False,
    confirmation_available: bool = True,
    safety_policy: GlobalExecutionSafetyPolicy | None = None,
) -> ExecutionStrategySelectionRequest:
    """Build a request from existing plan metadata without duplicating it."""
    return ExecutionStrategySelectionRequest(
        plan=plan,
        plan_validation=validation,
        historical_adjustment=historical_adjustment,
        historical_context=historical_context,
        supervisor_available=supervisor_available,
        replanner_available=replanner_available,
        confirmation_available=confirmation_available,
        safety_policy=safety_policy or GlobalExecutionSafetyPolicy(),
    )


def legacy_strategy_snapshot() -> Mapping[str, object]:
    """Represent absent strategy data without inventing historical facts."""
    return MappingProxyType(
        {
            "strategy": ExecutionStrategyType.STANDARD.value,
            "legacy_default": True,
            "reason": "Strategy was not recorded; previous execution behavior applies.",
        }
    )


def _selection_summary(strategy: ExecutionStrategy, decision: str) -> str:
    config = strategy.configuration
    return "\n".join(
        (
            "Execution strategy:",
            f"- Selected: {strategy.type.value}.",
            f"- Reason: {strategy.reason}",
            f"- Confirmations required: {len(config.confirmation_step_ids)}.",
            (
                "- Replanning allowed: "
                + (
                    f"yes, maximum {config.max_replans}."
                    if config.allow_replanning
                    else "no."
                )
            ),
            f"- Maximum retries: {config.effective_max_retry_attempts}.",
            f"- Final decision: {decision}.",
            "- Plan content was not modified.",
        )
    )


def _unique_safe(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = _safe_text(str(value))
        if text and text not in result:
            result.append(text)
    return tuple(result)

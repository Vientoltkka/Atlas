"""Controlled post-planning adjustments backed by historical evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib

from core.execution_history_advisor import (
    HistoricalEvidence,
    HistoricalPlanningContext,
    HistoricalRecommendation,
    HistoricalRecommendationSeverity,
    HistoricalRecommendationType,
)
from core.execution_plan_validator import (
    ExecutionPlanValidator,
    PlanValidationResult,
    plan_signature,
)
from core.execution_report import _safe_text
from core.execution_retry import RetryPolicy, RetryStrategy
from core.planner import ExecutionPlan, ExecutionStep


PLAN_SCOPE_STEP_ID = "__plan__"


class HistoricalAdjustmentType(str, Enum):
    ADD_HISTORICAL_WARNING = "ADD_HISTORICAL_WARNING"
    MARK_RETRY_RISK = "MARK_RETRY_RISK"
    INCREASE_RETRY_LIMIT = "INCREASE_RETRY_LIMIT"
    REQUIRE_CONFIRMATION = "REQUIRE_CONFIRMATION"
    ATTACH_RECOVERY_HINT = "ATTACH_RECOVERY_HINT"
    REQUEST_MANUAL_REVIEW = "REQUEST_MANUAL_REVIEW"


class HistoricalAdjustmentTarget(str, Enum):
    DETECTED_RISKS = "DETECTED_RISKS"
    RETRY_LIMIT = "RETRY_LIMIT"
    CONFIRMATION = "CONFIRMATION"
    RECOVERY_HINT = "RECOVERY_HINT"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    TOOL = "TOOL"
    STEP_SET = "STEP_SET"
    OBJECTIVE = "OBJECTIVE"
    CRITICALITY = "CRITICALITY"
    OPTIONAL = "OPTIONAL"


class HistoricalAdjustmentRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class HistoricalAdjustmentStatus(str, Enum):
    PROPOSED = "PROPOSED"
    INFORMATIONAL = "INFORMATIONAL"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class HistoricalAdjustmentValidation(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    VALID = "VALID"
    INVALID = "INVALID"


class HistoricalPolicyDecision(str, Enum):
    NO_ADJUSTMENT = "NO_ADJUSTMENT"
    INFORMATIONAL = "INFORMATIONAL"
    APPLICABLE = "APPLICABLE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    REJECTED = "REJECTED"


class PlanAdjustmentLifecycle(str, Enum):
    GENERATED = "GENERATED"
    EXECUTING = "EXECUTING"
    TERMINAL = "TERMINAL"


class HistoricalAdjustmentValueKind(str, Enum):
    NONE = "NONE"
    TEXT = "TEXT"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"


@dataclass(frozen=True, slots=True)
class HistoricalAdjustmentValue:
    """Closed serializable value used before and after one proposal."""

    kind: HistoricalAdjustmentValueKind
    text: str | None = None
    integer: int | None = None
    boolean: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, HistoricalAdjustmentValueKind):
            raise TypeError("kind must be HistoricalAdjustmentValueKind.")
        populated = sum(
            value is not None for value in (self.text, self.integer, self.boolean)
        )
        expected = 0 if self.kind is HistoricalAdjustmentValueKind.NONE else 1
        if populated != expected:
            raise ValueError("Adjustment value does not match its declared kind.")
        if self.kind is HistoricalAdjustmentValueKind.TEXT:
            if not isinstance(self.text, str) or not self.text.strip():
                raise ValueError("TEXT adjustment values require non-empty text.")
            object.__setattr__(self, "text", _safe_text(self.text))
        elif self.kind is HistoricalAdjustmentValueKind.INTEGER:
            if isinstance(self.integer, bool) or not isinstance(self.integer, int):
                raise ValueError("INTEGER adjustment values require an integer.")
        elif self.kind is HistoricalAdjustmentValueKind.BOOLEAN:
            if type(self.boolean) is not bool:
                raise ValueError("BOOLEAN adjustment values require a bool.")

    @classmethod
    def none(cls) -> "HistoricalAdjustmentValue":
        return cls(HistoricalAdjustmentValueKind.NONE)

    @classmethod
    def text_value(cls, value: str) -> "HistoricalAdjustmentValue":
        return cls(HistoricalAdjustmentValueKind.TEXT, text=value)

    @classmethod
    def integer_value(cls, value: int) -> "HistoricalAdjustmentValue":
        return cls(HistoricalAdjustmentValueKind.INTEGER, integer=value)

    @classmethod
    def boolean_value(cls, value: bool) -> "HistoricalAdjustmentValue":
        return cls(HistoricalAdjustmentValueKind.BOOLEAN, boolean=value)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "text": self.text,
            "integer": self.integer,
            "boolean": self.boolean,
        }


@dataclass(frozen=True, slots=True)
class HistoricalAdjustmentPolicyLimits:
    """Conservative local limits for historical plan adjustments."""

    max_proposals_per_plan: int = 5
    max_proposals_per_step: int = 2
    max_retry_increment: int = 1
    absolute_retry_limit: int = 3
    minimum_severity: HistoricalRecommendationSeverity = (
        HistoricalRecommendationSeverity.CAUTION
    )
    minimum_evidence_count: int = 2

    def __post_init__(self) -> None:
        _bounded_int(self.max_proposals_per_plan, "max_proposals_per_plan", 1, 20)
        _bounded_int(self.max_proposals_per_step, "max_proposals_per_step", 1, 5)
        _bounded_int(self.max_retry_increment, "max_retry_increment", 1, 3)
        _bounded_int(self.absolute_retry_limit, "absolute_retry_limit", 1, 10)
        _bounded_int(self.minimum_evidence_count, "minimum_evidence_count", 1, 50)
        if self.max_retry_increment >= self.absolute_retry_limit:
            raise ValueError(
                "max_retry_increment must be lower than absolute_retry_limit."
            )
        if not isinstance(
            self.minimum_severity,
            HistoricalRecommendationSeverity,
        ):
            raise TypeError(
                "minimum_severity must be HistoricalRecommendationSeverity."
            )


@dataclass(frozen=True, slots=True)
class HistoricalAdjustmentRequest:
    """One immutable post-planning adjustment request."""

    plan: ExecutionPlan
    historical_context: HistoricalPlanningContext
    lifecycle: PlanAdjustmentLifecycle = PlanAdjustmentLifecycle.GENERATED
    completed_step_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExecutionPlan):
            raise TypeError("plan must be an ExecutionPlan.")
        if not isinstance(self.historical_context, HistoricalPlanningContext):
            raise TypeError(
                "historical_context must be a HistoricalPlanningContext."
            )
        if not isinstance(self.lifecycle, PlanAdjustmentLifecycle):
            raise TypeError("lifecycle must be PlanAdjustmentLifecycle.")
        completed = tuple(dict.fromkeys(self.completed_step_ids))
        known = {step.id for step in self.plan.ordered_steps}
        if any(step_id not in known for step_id in completed):
            raise ValueError("completed_step_ids contains an unknown step.")
        object.__setattr__(self, "completed_step_ids", completed)


@dataclass(frozen=True, slots=True)
class HistoricalAdjustmentProposal:
    """Closed, traceable proposal generated from one recommendation."""

    proposal_id: str
    plan_id: str
    step_id: str
    adjustment_type: HistoricalAdjustmentType
    target: HistoricalAdjustmentTarget
    previous_value: HistoricalAdjustmentValue
    proposed_value: HistoricalAdjustmentValue
    source_recommendation: HistoricalRecommendation
    evidence: tuple[HistoricalEvidence, ...]
    reason: str
    risk: HistoricalAdjustmentRisk
    policy_rule: str
    status: HistoricalAdjustmentStatus = HistoricalAdjustmentStatus.PROPOSED
    validation: HistoricalAdjustmentValidation = (
        HistoricalAdjustmentValidation.PENDING
    )
    validation_messages: tuple[str, ...] = ()
    applied: bool = False
    rejected: bool = False

    def __post_init__(self) -> None:
        for name in ("proposal_id", "plan_id", "step_id", "policy_rule"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.adjustment_type, HistoricalAdjustmentType):
            raise TypeError("adjustment_type must be HistoricalAdjustmentType.")
        if not isinstance(self.target, HistoricalAdjustmentTarget):
            raise TypeError("target must be HistoricalAdjustmentTarget.")
        if not isinstance(self.previous_value, HistoricalAdjustmentValue):
            raise TypeError("previous_value must be HistoricalAdjustmentValue.")
        if not isinstance(self.proposed_value, HistoricalAdjustmentValue):
            raise TypeError("proposed_value must be HistoricalAdjustmentValue.")
        if not isinstance(self.source_recommendation, HistoricalRecommendation):
            raise TypeError(
                "source_recommendation must be HistoricalRecommendation."
            )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "reason", _safe_text(self.reason))
        object.__setattr__(self, "policy_rule", _safe_text(self.policy_rule))
        object.__setattr__(
            self,
            "validation_messages",
            tuple(_safe_text(value) for value in self.validation_messages[:10]),
        )
        if self.applied and self.rejected:
            raise ValueError("A proposal cannot be both applied and rejected.")

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "adjustment_type": self.adjustment_type.value,
            "target": self.target.value,
            "previous_value": self.previous_value.to_dict(),
            "proposed_value": self.proposed_value.to_dict(),
            "source_recommendation": self.source_recommendation.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "reason": self.reason,
            "risk": self.risk.value,
            "policy_rule": self.policy_rule,
            "status": self.status.value,
            "validation": self.validation.value,
            "validation_messages": list(self.validation_messages),
            "applied": self.applied,
            "rejected": self.rejected,
        }


@dataclass(frozen=True, slots=True)
class HistoricalAdjustmentTrace:
    """Sanitized final decision trace for one proposal."""

    proposal_id: str
    recommendation_type: HistoricalRecommendationType
    evidence_session_ids: tuple[str, ...]
    policy_rule: str
    requested_change: str
    decision: HistoricalAdjustmentStatus
    validation: HistoricalAdjustmentValidation
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_session_ids",
            tuple(dict.fromkeys(self.evidence_session_ids)),
        )
        object.__setattr__(self, "requested_change", _safe_text(self.requested_change))
        object.__setattr__(self, "reason", _safe_text(self.reason))

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "recommendation_type": self.recommendation_type.value,
            "evidence_session_ids": list(self.evidence_session_ids),
            "policy_rule": self.policy_rule,
            "requested_change": self.requested_change,
            "decision": self.decision.value,
            "validation": self.validation.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class HistoricalPlanAdjustmentResult:
    """Immutable result retaining both the original and selected plan."""

    original_plan: ExecutionPlan
    selected_plan: ExecutionPlan
    original_plan_id: str
    selected_plan_id: str
    recommendations_reviewed: int
    proposals: tuple[HistoricalAdjustmentProposal, ...]
    traces: tuple[HistoricalAdjustmentTrace, ...]
    base_validation: PlanValidationResult
    final_validation: PlanValidationResult
    summary: str

    @property
    def generated_count(self) -> int:
        return len(self.proposals)

    @property
    def applied_count(self) -> int:
        return sum(proposal.applied for proposal in self.proposals)

    @property
    def rejected_count(self) -> int:
        return sum(proposal.rejected for proposal in self.proposals)

    @property
    def requires_manual_review(self) -> bool:
        return any(
            proposal.status is HistoricalAdjustmentStatus.MANUAL_REVIEW
            for proposal in self.proposals
        )


@dataclass(frozen=True, slots=True)
class _PolicyEvaluation:
    decision: HistoricalPolicyDecision
    rule: str
    reason: str


class HistoricalPlanAdjustmentPolicy:
    """Map historical recommendations to a small safe proposal vocabulary."""

    def __init__(
        self,
        limits: HistoricalAdjustmentPolicyLimits | None = None,
    ) -> None:
        self._limits = limits or HistoricalAdjustmentPolicyLimits()

    @property
    def limits(self) -> HistoricalAdjustmentPolicyLimits:
        return self._limits

    def evaluate_recommendation(
        self,
        recommendation: HistoricalRecommendation,
    ) -> _PolicyEvaluation:
        if recommendation.type is HistoricalRecommendationType.INSUFFICIENT_HISTORY:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.NO_ADJUSTMENT,
                "insufficient_history_no_change",
                "Insufficient history cannot produce an adjustment.",
            )
        if recommendation.type is HistoricalRecommendationType.PREVIOUS_SUCCESS:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.INFORMATIONAL,
                "previous_success_informational_only",
                "Previous success is informative and does not change a plan.",
            )
        if (
            recommendation.type
            is not HistoricalRecommendationType.RECOVERY_AVAILABLE
            and _severity_rank(recommendation.severity)
            < _severity_rank(self._limits.minimum_severity)
        ):
            return _PolicyEvaluation(
                HistoricalPolicyDecision.REJECTED,
                "minimum_severity_not_met",
                "Recommendation severity does not meet the configured minimum.",
            )
        if (
            recommendation.supporting_execution_count
            < self._limits.minimum_evidence_count
        ):
            return _PolicyEvaluation(
                HistoricalPolicyDecision.REJECTED,
                "minimum_evidence_not_met",
                "Historical evidence does not meet the configured minimum.",
            )
        if recommendation.type is HistoricalRecommendationType.FREQUENT_FAILURE:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.APPLICABLE,
                "frequent_failure_warning",
                "Repeated failures may add a bounded historical warning.",
            )
        if recommendation.type is HistoricalRecommendationType.RETRY_RISK:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.APPLICABLE,
                "retry_risk_bounded_adjustment",
                "Repeated retries may mark risk and increase attempts within limits.",
            )
        if recommendation.type is HistoricalRecommendationType.RECOVERY_AVAILABLE:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.APPLICABLE,
                "successful_recovery_hint_only",
                "Successful recovery may be attached only as an informational hint.",
            )
        if recommendation.type is HistoricalRecommendationType.USER_ACTION_PATTERN:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.APPLICABLE,
                "user_action_requires_confirmation",
                "Repeated user intervention may elevate plan confirmation.",
            )
        return _PolicyEvaluation(
            HistoricalPolicyDecision.NO_ADJUSTMENT,
            "recommendation_not_actionable",
            "This recommendation type is consultative in the current phase.",
        )

    def evaluate_proposal(
        self,
        proposal: HistoricalAdjustmentProposal,
        request: HistoricalAdjustmentRequest,
    ) -> _PolicyEvaluation:
        if request.lifecycle is not PlanAdjustmentLifecycle.GENERATED:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.REJECTED,
                "plan_not_generated",
                "Executing or terminal plans cannot be adjusted.",
            )
        if proposal.rejected:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.REJECTED,
                "proposal_previously_rejected",
                proposal.validation_messages[0]
                if proposal.validation_messages
                else "Proposal was already rejected.",
            )
        if proposal.status is HistoricalAdjustmentStatus.INFORMATIONAL:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.INFORMATIONAL,
                proposal.policy_rule,
                proposal.reason,
            )
        if request.plan.status != "planned":
            return _PolicyEvaluation(
                HistoricalPolicyDecision.REJECTED,
                "plan_status_not_planned",
                "Only planned execution plans can be adjusted.",
            )
        if proposal.step_id in request.completed_step_ids:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.REJECTED,
                "completed_step_immutable",
                "Completed steps cannot be adjusted.",
            )
        if (
            proposal.step_id != PLAN_SCOPE_STEP_ID
            and proposal.step_id
            not in {step.id for step in request.plan.ordered_steps}
        ):
            return _PolicyEvaluation(
                HistoricalPolicyDecision.REJECTED,
                "unknown_target_step",
                "Proposal identifies an unknown execution step.",
            )
        if proposal.target in {
            HistoricalAdjustmentTarget.TOOL,
            HistoricalAdjustmentTarget.STEP_SET,
            HistoricalAdjustmentTarget.OBJECTIVE,
            HistoricalAdjustmentTarget.CRITICALITY,
            HistoricalAdjustmentTarget.OPTIONAL,
        }:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.REJECTED,
                "forbidden_plan_mutation",
                "The proposal targets an immutable plan property.",
            )
        expected_target = {
            HistoricalAdjustmentType.ADD_HISTORICAL_WARNING: (
                HistoricalAdjustmentTarget.DETECTED_RISKS
            ),
            HistoricalAdjustmentType.MARK_RETRY_RISK: (
                HistoricalAdjustmentTarget.DETECTED_RISKS
            ),
            HistoricalAdjustmentType.INCREASE_RETRY_LIMIT: (
                HistoricalAdjustmentTarget.RETRY_LIMIT
            ),
            HistoricalAdjustmentType.REQUIRE_CONFIRMATION: (
                HistoricalAdjustmentTarget.CONFIRMATION
            ),
            HistoricalAdjustmentType.ATTACH_RECOVERY_HINT: (
                HistoricalAdjustmentTarget.RECOVERY_HINT
            ),
            HistoricalAdjustmentType.REQUEST_MANUAL_REVIEW: (
                HistoricalAdjustmentTarget.MANUAL_REVIEW
            ),
        }[proposal.adjustment_type]
        if proposal.target is not expected_target:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.REJECTED,
                "adjustment_target_mismatch",
                "Adjustment type and target are incompatible.",
            )
        if (
            proposal.adjustment_type
            is HistoricalAdjustmentType.REQUIRE_CONFIRMATION
            and proposal.proposed_value.boolean is not True
        ):
            return _PolicyEvaluation(
                HistoricalPolicyDecision.REJECTED,
                "confirmation_cannot_be_reduced",
                "Historical adjustment cannot reduce confirmation.",
            )
        if proposal.adjustment_type is HistoricalAdjustmentType.REQUEST_MANUAL_REVIEW:
            return _PolicyEvaluation(
                HistoricalPolicyDecision.HUMAN_REVIEW,
                "manual_review_required",
                "The proposal requires explicit human review.",
            )
        return _PolicyEvaluation(
            HistoricalPolicyDecision.APPLICABLE,
            proposal.policy_rule,
            proposal.reason,
        )


class HistoricalPlanAdjuster:
    """Build and validate safe plan candidates without mutating the base plan."""

    def __init__(
        self,
        validator: ExecutionPlanValidator,
        *,
        policy: HistoricalPlanAdjustmentPolicy | None = None,
    ) -> None:
        if not isinstance(validator, ExecutionPlanValidator):
            raise TypeError("validator must be an ExecutionPlanValidator.")
        self._validator = validator
        self._policy = policy or HistoricalPlanAdjustmentPolicy()

    @property
    def policy(self) -> HistoricalPlanAdjustmentPolicy:
        return self._policy

    def propose(
        self,
        request: HistoricalAdjustmentRequest,
    ) -> tuple[HistoricalAdjustmentProposal, ...]:
        if not isinstance(request, HistoricalAdjustmentRequest):
            raise TypeError("request must be HistoricalAdjustmentRequest.")
        proposals: list[HistoricalAdjustmentProposal] = []
        counts_by_step: CounterLike = {}
        for recommendation in request.historical_context.recommendations:
            evaluation = self._policy.evaluate_recommendation(recommendation)
            if evaluation.decision not in {
                HistoricalPolicyDecision.APPLICABLE,
                HistoricalPolicyDecision.INFORMATIONAL,
            }:
                continue
            for proposal in self._proposals_for_recommendation(
                request.plan,
                recommendation,
                evaluation,
            ):
                if len(proposals) >= self._policy.limits.max_proposals_per_plan:
                    return tuple(proposals)
                count = counts_by_step.get(proposal.step_id, 0)
                if count >= self._policy.limits.max_proposals_per_step:
                    continue
                counts_by_step[proposal.step_id] = count + 1
                proposals.append(proposal)
        return tuple(proposals)

    def adjust(
        self,
        request: HistoricalAdjustmentRequest,
        *,
        proposals: tuple[HistoricalAdjustmentProposal, ...] | None = None,
    ) -> HistoricalPlanAdjustmentResult:
        if not isinstance(request, HistoricalAdjustmentRequest):
            raise TypeError("request must be HistoricalAdjustmentRequest.")
        base = request.plan
        base_id = plan_signature(base)
        base_validation = self._validator.validate(base)
        supplied = self.propose(request) if proposals is None else tuple(proposals)
        limited = supplied[: self._policy.limits.max_proposals_per_plan]
        current = base
        final_proposals: list[HistoricalAdjustmentProposal] = []
        traces: list[HistoricalAdjustmentTrace] = []
        per_step: CounterLike = {}

        for proposal in limited:
            proposal = self._normalize_proposal_plan_id(proposal, base_id)
            if proposal.rejected:
                final_proposals.append(proposal)
                traces.append(_trace(proposal))
                continue
            count = per_step.get(proposal.step_id, 0)
            if count >= self._policy.limits.max_proposals_per_step:
                rejected = self._rejected(
                    proposal,
                    "proposal_limit_per_step",
                    "Maximum proposals per step was reached.",
                )
                final_proposals.append(rejected)
                traces.append(_trace(rejected))
                continue
            per_step[proposal.step_id] = count + 1
            evaluation = self._policy.evaluate_proposal(proposal, request)
            if evaluation.decision is HistoricalPolicyDecision.REJECTED:
                rejected = self._rejected(
                    proposal,
                    evaluation.rule,
                    evaluation.reason,
                )
                final_proposals.append(rejected)
                traces.append(_trace(rejected))
                continue
            if evaluation.decision is HistoricalPolicyDecision.INFORMATIONAL:
                informational = replace(
                    proposal,
                    status=HistoricalAdjustmentStatus.INFORMATIONAL,
                    validation=HistoricalAdjustmentValidation.NOT_REQUIRED,
                    validation_messages=(evaluation.reason,),
                    applied=False,
                    rejected=False,
                )
                final_proposals.append(informational)
                traces.append(_trace(informational))
                continue
            if evaluation.decision is HistoricalPolicyDecision.HUMAN_REVIEW:
                manual = replace(
                    proposal,
                    status=HistoricalAdjustmentStatus.MANUAL_REVIEW,
                    validation=HistoricalAdjustmentValidation.NOT_REQUIRED,
                    validation_messages=(evaluation.reason,),
                )
                final_proposals.append(manual)
                traces.append(_trace(manual))
                continue

            candidate, application_error = self._candidate(current, proposal)
            if application_error is not None:
                rejected = self._rejected(
                    proposal,
                    "proposal_application_rejected",
                    application_error,
                )
                final_proposals.append(rejected)
                traces.append(_trace(rejected))
                continue
            invariant_error = _invariant_error(base, candidate)
            if invariant_error is not None:
                rejected = self._rejected(
                    proposal,
                    "plan_invariant_rejected",
                    invariant_error,
                )
                final_proposals.append(rejected)
                traces.append(_trace(rejected))
                continue
            validation = self._validator.validate(candidate)
            if not validation.is_valid:
                rejected = replace(
                    proposal,
                    status=HistoricalAdjustmentStatus.REJECTED,
                    validation=HistoricalAdjustmentValidation.INVALID,
                    validation_messages=tuple(validation.errors[:10]),
                    applied=False,
                    rejected=True,
                )
                final_proposals.append(rejected)
                traces.append(_trace(rejected))
                continue

            current = candidate
            applied = replace(
                proposal,
                status=HistoricalAdjustmentStatus.APPLIED,
                validation=HistoricalAdjustmentValidation.VALID,
                validation_messages=tuple(validation.warnings[:10]),
                applied=True,
                rejected=False,
            )
            final_proposals.append(applied)
            traces.append(_trace(applied))

        final_validation = self._validator.validate(current)
        if not final_validation.is_valid:
            current = base
            final_validation = base_validation
        summary = _summary(
            request,
            tuple(final_proposals),
            base,
            current,
        )
        return HistoricalPlanAdjustmentResult(
            original_plan=base,
            selected_plan=current,
            original_plan_id=base_id,
            selected_plan_id=plan_signature(current),
            recommendations_reviewed=len(
                request.historical_context.recommendations
            ),
            proposals=tuple(final_proposals),
            traces=tuple(traces),
            base_validation=base_validation,
            final_validation=final_validation,
            summary=summary,
        )

    def _proposals_for_recommendation(
        self,
        plan: ExecutionPlan,
        recommendation: HistoricalRecommendation,
        evaluation: _PolicyEvaluation,
    ) -> tuple[HistoricalAdjustmentProposal, ...]:
        if recommendation.type is HistoricalRecommendationType.PREVIOUS_SUCCESS:
            proposal = self._proposal(
                plan,
                recommendation,
                PLAN_SCOPE_STEP_ID,
                HistoricalAdjustmentType.ADD_HISTORICAL_WARNING,
                HistoricalAdjustmentTarget.DETECTED_RISKS,
                HistoricalAdjustmentValue.none(),
                HistoricalAdjustmentValue.text_value(
                    recommendation.message
                ),
                HistoricalAdjustmentRisk.LOW,
                evaluation,
            )
            return (
                replace(
                    proposal,
                    status=HistoricalAdjustmentStatus.INFORMATIONAL,
                    validation=HistoricalAdjustmentValidation.NOT_REQUIRED,
                ),
            )
        if recommendation.type is HistoricalRecommendationType.FREQUENT_FAILURE:
            step = _related_step(plan, recommendation)
            step_id = step.id if step is not None else PLAN_SCOPE_STEP_ID
            proposals = [
                self._proposal(
                    plan,
                    recommendation,
                    step_id,
                    HistoricalAdjustmentType.ADD_HISTORICAL_WARNING,
                    HistoricalAdjustmentTarget.DETECTED_RISKS,
                    HistoricalAdjustmentValue.none(),
                    HistoricalAdjustmentValue.text_value(
                        recommendation.message
                    ),
                    HistoricalAdjustmentRisk.LOW,
                    evaluation,
                )
            ]
            if step is not None and step.criticality > 0:
                proposals.append(
                    self._proposal(
                        plan,
                        recommendation,
                        step.id,
                        HistoricalAdjustmentType.REQUEST_MANUAL_REVIEW,
                        HistoricalAdjustmentTarget.MANUAL_REVIEW,
                        HistoricalAdjustmentValue.none(),
                        HistoricalAdjustmentValue.text_value(
                            "Review repeated failure on a critical step."
                        ),
                        HistoricalAdjustmentRisk.HIGH,
                        evaluation,
                    )
                )
            return tuple(proposals)
        if recommendation.type is HistoricalRecommendationType.RETRY_RISK:
            step = _related_step(plan, recommendation)
            if step is None:
                return ()
            current_attempts = (
                step.retry_policy.max_attempts
                if step.retry_policy is not None
                else 1
            )
            proposed_attempts = min(
                current_attempts + self._policy.limits.max_retry_increment,
                self._policy.limits.absolute_retry_limit,
            )
            proposals = [
                self._proposal(
                    plan,
                    recommendation,
                    step.id,
                    HistoricalAdjustmentType.MARK_RETRY_RISK,
                    HistoricalAdjustmentTarget.DETECTED_RISKS,
                    HistoricalAdjustmentValue.none(),
                    HistoricalAdjustmentValue.text_value(
                        recommendation.message
                    ),
                    HistoricalAdjustmentRisk.LOW,
                    evaluation,
                )
            ]
            safe_for_retry_increase = (
                step.idempotent
                and step.recovery_safe
                and step.side_effect_free
                and step.criticality == 0
            )
            if (
                proposed_attempts > current_attempts
                and safe_for_retry_increase
                and not (
                    step.retry_policy is not None
                    and step.retry_policy.strategy is RetryStrategy.NO_RETRY
                )
            ):
                proposals.append(
                    self._proposal(
                        plan,
                        recommendation,
                        step.id,
                        HistoricalAdjustmentType.INCREASE_RETRY_LIMIT,
                        HistoricalAdjustmentTarget.RETRY_LIMIT,
                        HistoricalAdjustmentValue.integer_value(current_attempts),
                        HistoricalAdjustmentValue.integer_value(proposed_attempts),
                        HistoricalAdjustmentRisk.MEDIUM,
                        evaluation,
                    )
                )
            elif not safe_for_retry_increase:
                proposals.append(
                    self._proposal(
                        plan,
                        recommendation,
                        step.id,
                        HistoricalAdjustmentType.REQUEST_MANUAL_REVIEW,
                        HistoricalAdjustmentTarget.MANUAL_REVIEW,
                        HistoricalAdjustmentValue.none(),
                        HistoricalAdjustmentValue.text_value(
                            "Retry increase requires an idempotent, recovery-safe, side-effect-free step."
                        ),
                        HistoricalAdjustmentRisk.HIGH,
                        evaluation,
                    )
                )
            return tuple(proposals)
        if recommendation.type is HistoricalRecommendationType.RECOVERY_AVAILABLE:
            return (
                self._proposal(
                    plan,
                    recommendation,
                    PLAN_SCOPE_STEP_ID,
                    HistoricalAdjustmentType.ATTACH_RECOVERY_HINT,
                    HistoricalAdjustmentTarget.RECOVERY_HINT,
                    HistoricalAdjustmentValue.none(),
                    HistoricalAdjustmentValue.text_value(
                        recommendation.message
                    ),
                    HistoricalAdjustmentRisk.LOW,
                    evaluation,
                ),
            )
        if recommendation.type is HistoricalRecommendationType.USER_ACTION_PATTERN:
            if plan.requires_confirmation:
                return ()
            return (
                self._proposal(
                    plan,
                    recommendation,
                    PLAN_SCOPE_STEP_ID,
                    HistoricalAdjustmentType.REQUIRE_CONFIRMATION,
                    HistoricalAdjustmentTarget.CONFIRMATION,
                    HistoricalAdjustmentValue.boolean_value(
                        plan.requires_confirmation
                    ),
                    HistoricalAdjustmentValue.boolean_value(True),
                    HistoricalAdjustmentRisk.MEDIUM,
                    evaluation,
                ),
            )
        return ()

    def _proposal(
        self,
        plan: ExecutionPlan,
        recommendation: HistoricalRecommendation,
        step_id: str,
        adjustment_type: HistoricalAdjustmentType,
        target: HistoricalAdjustmentTarget,
        previous: HistoricalAdjustmentValue,
        proposed: HistoricalAdjustmentValue,
        risk: HistoricalAdjustmentRisk,
        evaluation: _PolicyEvaluation,
    ) -> HistoricalAdjustmentProposal:
        plan_id = plan_signature(plan)
        proposal_id = _proposal_id(
            plan_id,
            recommendation.type,
            step_id,
            adjustment_type,
        )
        return HistoricalAdjustmentProposal(
            proposal_id=proposal_id,
            plan_id=plan_id,
            step_id=step_id,
            adjustment_type=adjustment_type,
            target=target,
            previous_value=previous,
            proposed_value=proposed,
            source_recommendation=recommendation,
            evidence=recommendation.evidence,
            reason=evaluation.reason,
            risk=risk,
            policy_rule=evaluation.rule,
        )

    def _candidate(
        self,
        plan: ExecutionPlan,
        proposal: HistoricalAdjustmentProposal,
    ) -> tuple[ExecutionPlan, str | None]:
        if proposal.adjustment_type in {
            HistoricalAdjustmentType.ADD_HISTORICAL_WARNING,
            HistoricalAdjustmentType.MARK_RETRY_RISK,
            HistoricalAdjustmentType.ATTACH_RECOVERY_HINT,
        }:
            warning = proposal.proposed_value.text
            if warning is None:
                return plan, "Historical warning proposal requires text."
            risks = tuple(dict.fromkeys(plan.detected_risks + (_safe_text(warning),)))
            return replace(plan, detected_risks=risks), None
        if proposal.adjustment_type is HistoricalAdjustmentType.REQUIRE_CONFIRMATION:
            if proposal.proposed_value.boolean is not True:
                return plan, "Confirmation proposals can only set confirmation to true."
            return replace(plan, requires_confirmation=True), None
        if proposal.adjustment_type is HistoricalAdjustmentType.INCREASE_RETRY_LIMIT:
            attempts = proposal.proposed_value.integer
            step = next(
                (
                    item
                    for item in plan.ordered_steps
                    if item.id == proposal.step_id
                ),
                None,
            )
            if step is None or attempts is None:
                return plan, "Retry proposal does not identify a valid step or limit."
            current_attempts = (
                step.retry_policy.max_attempts
                if step.retry_policy is not None
                else 1
            )
            if attempts <= current_attempts:
                return plan, "Retry limit cannot be reduced or left unchanged."
            if (
                attempts - current_attempts
                > self._policy.limits.max_retry_increment
                or attempts > self._policy.limits.absolute_retry_limit
            ):
                return plan, "Retry proposal exceeds configured safety limits."
            if (
                step.retry_policy is not None
                and step.retry_policy.strategy is RetryStrategy.NO_RETRY
            ):
                return plan, "Explicit NO_RETRY policy cannot be weakened."
            retry_policy = RetryPolicy(
                max_attempts=attempts,
                strategy=(
                    step.retry_policy.strategy
                    if step.retry_policy is not None
                    else RetryStrategy.IMMEDIATE
                ),
                classifier=(
                    step.retry_policy.classifier
                    if step.retry_policy is not None
                    else RetryPolicy().classifier
                ),
                delay_ms=(
                    step.retry_policy.delay_ms
                    if step.retry_policy is not None
                    else 0
                ),
            )
            replacement_step = _copy_step(step, retry_policy=retry_policy)
            steps = tuple(
                replacement_step if item.id == step.id else item
                for item in plan.ordered_steps
            )
            return replace(plan, ordered_steps=steps), None
        return plan, "Adjustment type is not automatically applicable."

    @staticmethod
    def _normalize_proposal_plan_id(
        proposal: HistoricalAdjustmentProposal,
        plan_id: str,
    ) -> HistoricalAdjustmentProposal:
        if proposal.plan_id == plan_id:
            return proposal
        return replace(
            proposal,
            status=HistoricalAdjustmentStatus.REJECTED,
            validation=HistoricalAdjustmentValidation.INVALID,
            validation_messages=("Proposal belongs to a different plan.",),
            applied=False,
            rejected=True,
        )

    @staticmethod
    def _rejected(
        proposal: HistoricalAdjustmentProposal,
        rule: str,
        reason: str,
    ) -> HistoricalAdjustmentProposal:
        return replace(
            proposal,
            policy_rule=rule,
            status=HistoricalAdjustmentStatus.REJECTED,
            validation=HistoricalAdjustmentValidation.INVALID,
            validation_messages=(_safe_text(reason),),
            applied=False,
            rejected=True,
        )


CounterLike = dict[str, int]


def _copy_step(
    step: ExecutionStep,
    *,
    retry_policy: RetryPolicy | None = None,
) -> ExecutionStep:
    return ExecutionStep(
        step.id,
        step.description,
        step.tool,
        step.depends_on,
        subplan=step.subplan,
        subplan_ref=step.subplan_ref,
        branch=step.branch,
        loop=step.loop,
        status=step.status,
        arguments=step.arguments,
        output_binding=step.output_binding,
        condition=step.condition,
        retry_policy=retry_policy
        if retry_policy is not None
        else step.retry_policy,
        parallel_safe=step.parallel_safe,
        resource_keys=step.resource_keys,
        idempotent=step.idempotent,
        recovery_safe=step.recovery_safe,
        side_effect_free=step.side_effect_free,
        optional=step.optional,
        priority=step.priority,
        urgency=step.urgency,
        estimated_cost=step.estimated_cost,
        estimated_duration_seconds=step.estimated_duration_seconds,
        criticality=step.criticality,
        deadline=step.deadline,
        resource_requirements=step.resource_requirements,
    )


def _related_step(
    plan: ExecutionPlan,
    recommendation: HistoricalRecommendation,
) -> ExecutionStep | None:
    if recommendation.related_step is not None:
        exact = next(
            (
                step
                for step in plan.ordered_steps
                if step.id == recommendation.related_step
            ),
            None,
        )
        if exact is not None:
            return exact
    if recommendation.related_tool is not None:
        return next(
            (
                step
                for step in plan.ordered_steps
                if step.tool == recommendation.related_tool
            ),
            None,
        )
    return None


def _invariant_error(
    original: ExecutionPlan,
    candidate: ExecutionPlan,
) -> str | None:
    if candidate.goal != original.goal:
        return "Historical adjustment cannot change the plan objective."
    if len(candidate.ordered_steps) != len(original.ordered_steps):
        return "Historical adjustment cannot insert or remove steps."
    if tuple(step.id for step in candidate.ordered_steps) != tuple(
        step.id for step in original.ordered_steps
    ):
        return "Historical adjustment cannot reorder or rename steps."
    for before, after in zip(original.ordered_steps, candidate.ordered_steps):
        if before.tool != after.tool:
            return "Historical adjustment cannot change tools."
        if before.depends_on != after.depends_on:
            return "Historical adjustment cannot change dependencies."
        if after.criticality < before.criticality:
            return "Historical adjustment cannot reduce criticality."
        if before.optional != after.optional:
            return "Historical adjustment cannot change step optionality."
    if original.requires_confirmation and not candidate.requires_confirmation:
        return "Historical adjustment cannot reduce confirmation."
    return None


def _proposal_id(
    plan_id: str,
    recommendation_type: HistoricalRecommendationType,
    step_id: str,
    adjustment_type: HistoricalAdjustmentType,
) -> str:
    payload = "|".join(
        (plan_id, recommendation_type.value, step_id, adjustment_type.value)
    )
    return "historical.adjustment." + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]


def _trace(proposal: HistoricalAdjustmentProposal) -> HistoricalAdjustmentTrace:
    return HistoricalAdjustmentTrace(
        proposal_id=proposal.proposal_id,
        recommendation_type=proposal.source_recommendation.type,
        evidence_session_ids=proposal.source_recommendation.session_ids,
        policy_rule=proposal.policy_rule,
        requested_change=(
            f"{proposal.adjustment_type.value} on {proposal.step_id}"
        ),
        decision=proposal.status,
        validation=proposal.validation,
        reason=(
            proposal.validation_messages[0]
            if proposal.validation_messages
            else proposal.reason
        ),
    )


def _summary(
    request: HistoricalAdjustmentRequest,
    proposals: tuple[HistoricalAdjustmentProposal, ...],
    original: ExecutionPlan,
    selected: ExecutionPlan,
) -> str:
    applied = tuple(item for item in proposals if item.applied)
    rejected = tuple(item for item in proposals if item.rejected)
    steps = tuple(
        dict.fromkeys(
            item.step_id
            for item in proposals
            if item.step_id != PLAN_SCOPE_STEP_ID
        )
    )
    confirmations = sum(
        item.applied
        and item.adjustment_type is HistoricalAdjustmentType.REQUIRE_CONFIRMATION
        for item in proposals
    )
    retries = sum(
        item.applied
        and item.adjustment_type is HistoricalAdjustmentType.INCREASE_RETRY_LIMIT
        for item in proposals
    )
    warnings = sum(
        item.applied
        and item.adjustment_type
        in {
            HistoricalAdjustmentType.ADD_HISTORICAL_WARNING,
            HistoricalAdjustmentType.MARK_RETRY_RISK,
            HistoricalAdjustmentType.ATTACH_RECOVERY_HINT,
        }
        for item in proposals
    )
    lines = [
        "Ajustes históricos:",
        (
            f"- Se revisaron {len(request.historical_context.relevant_execution_ids)} "
            "ejecuciones similares."
        ),
        f"- Se revisaron {len(request.historical_context.recommendations)} recomendaciones.",
        f"- Se generaron {len(proposals)} propuestas.",
        f"- Se aplicaron {len(applied)} propuestas.",
        f"- Se rechazaron {len(rejected)} propuestas.",
        f"- Pasos afectados: {', '.join(steps) if steps else 'ninguno'}.",
        f"- Confirmaciones añadidas: {confirmations}.",
        f"- Cambios de reintentos: {retries}.",
        f"- Advertencias incorporadas: {warnings}.",
        (
            "- El objetivo, las herramientas y el orden del plan no fueron modificados."
            if _invariant_error(original, selected) is None
            else "- El plan original fue conservado."
        ),
    ]
    return "\n".join(_safe_text(line) for line in lines)[:1_200]


def _bounded_int(value: int, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")


def _severity_rank(value: HistoricalRecommendationSeverity) -> int:
    return {
        HistoricalRecommendationSeverity.INFORMATION: 0,
        HistoricalRecommendationSeverity.CAUTION: 1,
        HistoricalRecommendationSeverity.WARNING: 2,
    }[value]

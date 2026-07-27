"""Deterministic ready-step prioritization for Atlas executions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from types import MappingProxyType

from core.planner import ExecutionPlan, ExecutionStep


MAX_PRIORITY_HISTORY_ENTRIES = 100


@dataclass(frozen=True, slots=True)
class ExecutionPriorityPolicy:
    """Immutable scoring policy for already-ready steps.

    Formula:
    final = priority*w + urgency*w + criticality*w + deadline*w
    + dependency_impact*w + age*w - cost*w - duration*w - risk*w.
    Disabled policy preserves the incoming ready-step order exactly.
    """

    enabled: bool = False
    priority_weight: float = 1.0
    urgency_weight: float = 1.0
    criticality_weight: float = 1.0
    deadline_weight: float = 1.0
    dependency_impact_weight: float = 1.0
    age_weight: float = 1.0
    cost_weight: float = 1.0
    duration_weight: float = 1.0
    risk_weight: float = 1.0
    prefer_short_tasks: bool = False
    preserve_plan_order_on_tie: bool = True
    max_age_score: float = 10.0
    policy_name: str = "default_execution_priority"

    def __post_init__(self) -> None:
        for name in ("enabled", "prefer_short_tasks", "preserve_plan_order_on_tie"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        for name in (
            "priority_weight",
            "urgency_weight",
            "criticality_weight",
            "deadline_weight",
            "dependency_impact_weight",
            "age_weight",
            "cost_weight",
            "duration_weight",
            "risk_weight",
            "max_age_score",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite non-negative number.")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, float(value))
        if not self.policy_name.strip():
            raise ValueError("policy_name must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class PriorityScore:
    """Auditable score for one ready step."""

    step_id: str
    declared_priority: int
    urgency_score: float
    criticality_score: float
    deadline_score: float
    dependency_impact_score: float
    age_score: float
    cost_penalty: float
    duration_penalty: float
    risk_penalty: float
    final_score: float


@dataclass(frozen=True, slots=True)
class PriorityDecision:
    """Deterministic prioritization decision for one ready-step set."""

    ordered_step_ids: tuple[str, ...]
    scores: tuple[PriorityScore, ...]
    selected_step_ids: tuple[str, ...]
    policy_name: str
    generated_at: datetime
    tie_breaker_used: str | None
    rationale_summary: str


class DependencyImpactAnalyzer:
    """Pure dependency-impact calculator for execution plans."""

    def impact_scores(
        self,
        plan: ExecutionPlan,
        ready_step_ids: tuple[str, ...],
        completed_step_ids: tuple[str, ...],
    ) -> MappingProxyType:
        completed = set(completed_step_ids)
        ready = set(ready_step_ids)
        steps = {step.id: step for step in plan.ordered_steps}
        children: dict[str, list[str]] = {step.id: [] for step in plan.ordered_steps}
        for step in plan.ordered_steps:
            for dependency in step.depends_on:
                if dependency in children:
                    children[dependency].append(step.id)

        scores: dict[str, float] = {}
        for step_id in ready_step_ids:
            seen: set[str] = set()
            stack = list(children.get(step_id, ()))
            score = 0.0
            while stack:
                child_id = stack.pop(0)
                if child_id in seen:
                    continue
                seen.add(child_id)
                child = steps.get(child_id)
                if child is None or child_id in completed:
                    continue
                remaining = [
                    dependency
                    for dependency in child.depends_on
                    if dependency not in completed and dependency != step_id
                ]
                if not remaining or all(dependency in ready for dependency in remaining):
                    score += 1.0
                    stack.extend(children.get(child_id, ()))
            scores[step_id] = score
        return MappingProxyType(scores)


class ReadyStepPrioritizer:
    """Score and order executable steps without resolving dependencies."""

    def __init__(
        self,
        policy: ExecutionPriorityPolicy | None = None,
        impact_analyzer: DependencyImpactAnalyzer | None = None,
    ) -> None:
        self._policy = policy or ExecutionPriorityPolicy()
        self._impact_analyzer = impact_analyzer or DependencyImpactAnalyzer()

    @property
    def policy(self) -> ExecutionPriorityPolicy:
        return self._policy

    def prioritize(
        self,
        ready_steps: tuple[ExecutionStep, ...],
        *,
        plan: ExecutionPlan,
        completed_step_ids: tuple[str, ...] = (),
        ready_since_by_step_id: MappingProxyType | dict[str, datetime] | None = None,
        now: datetime,
        selected_step_count: int | None = None,
    ) -> PriorityDecision:
        if not ready_steps:
            return PriorityDecision(
                ordered_step_ids=(),
                scores=(),
                selected_step_ids=(),
                policy_name=self._policy.policy_name,
                generated_at=now,
                tie_breaker_used=None,
                rationale_summary="no ready steps",
            )

        if not self._policy.enabled:
            ordered = tuple(step.id for step in ready_steps)
            selected = ordered if selected_step_count is None else ordered[:selected_step_count]
            scores = tuple(
                PriorityScore(
                    step_id=step.id,
                    declared_priority=getattr(step, "priority", 0),
                    urgency_score=0.0,
                    criticality_score=0.0,
                    deadline_score=0.0,
                    dependency_impact_score=0.0,
                    age_score=0.0,
                    cost_penalty=0.0,
                    duration_penalty=0.0,
                    risk_penalty=0.0,
                    final_score=0.0,
                )
                for step in ready_steps
            )
            return PriorityDecision(
                ordered_step_ids=ordered,
                scores=scores,
                selected_step_ids=selected,
                policy_name=self._policy.policy_name,
                generated_at=now,
                tie_breaker_used="plan_order",
                rationale_summary="priority policy disabled",
            )

        ready_since = ready_since_by_step_id or {}
        plan_order = {step.id: index for index, step in enumerate(plan.ordered_steps)}
        impact = self._impact_analyzer.impact_scores(
            plan,
            tuple(step.id for step in ready_steps),
            completed_step_ids,
        )
        scores = tuple(
            self._score_step(
                step,
                now=now,
                ready_since=ready_since.get(step.id),
                dependency_impact=float(impact.get(step.id, 0.0)),
            )
            for step in ready_steps
        )
        score_by_step_id = {score.step_id: score for score in scores}
        ordered_steps = tuple(
            sorted(
                ready_steps,
                key=lambda step: self._sort_key(
                    step,
                    score_by_step_id[step.id],
                    now,
                    ready_since.get(step.id),
                    plan_order.get(step.id, len(plan_order)),
                ),
            )
        )
        ordered = tuple(step.id for step in ordered_steps)
        selected = ordered if selected_step_count is None else ordered[:selected_step_count]
        tie_breaker = _tie_breaker(scores, ordered, tuple(step.id for step in ready_steps))
        return PriorityDecision(
            ordered_step_ids=ordered,
            scores=tuple(score_by_step_id[step_id] for step_id in ordered),
            selected_step_ids=selected,
            policy_name=self._policy.policy_name,
            generated_at=now,
            tie_breaker_used=tie_breaker,
            rationale_summary="ready steps prioritized by explicit deterministic policy",
        )

    def _score_step(
        self,
        step: ExecutionStep,
        *,
        now: datetime,
        ready_since: datetime | None,
        dependency_impact: float,
    ) -> PriorityScore:
        deadline_score = _deadline_score(step.deadline, now)
        age_score = _age_score(ready_since, now, self._policy.max_age_score)
        cost_penalty = 0.0 if step.estimated_cost is None else step.estimated_cost
        duration_penalty = (
            0.0
            if step.estimated_duration_seconds is None or not self._policy.prefer_short_tasks
            else step.estimated_duration_seconds
        )
        risk_penalty = _risk_penalty(step)
        final = (
            step.priority * self._policy.priority_weight
            + step.urgency * self._policy.urgency_weight
            + step.criticality * self._policy.criticality_weight
            + deadline_score * self._policy.deadline_weight
            + dependency_impact * self._policy.dependency_impact_weight
            + age_score * self._policy.age_weight
            - cost_penalty * self._policy.cost_weight
            - duration_penalty * self._policy.duration_weight
            - risk_penalty * self._policy.risk_weight
        )
        return PriorityScore(
            step_id=step.id,
            declared_priority=step.priority,
            urgency_score=float(step.urgency),
            criticality_score=float(step.criticality),
            deadline_score=deadline_score,
            dependency_impact_score=dependency_impact,
            age_score=age_score,
            cost_penalty=cost_penalty,
            duration_penalty=duration_penalty,
            risk_penalty=risk_penalty,
            final_score=final,
        )

    def _sort_key(
        self,
        step: ExecutionStep,
        score: PriorityScore,
        now: datetime,
        ready_since: datetime | None,
        plan_index: int,
    ) -> tuple[float, int, int, int, float, float, int, str]:
        deadline = step.deadline
        deadline_seconds = (
            float("inf")
            if deadline is None
            else (deadline.astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds()
        )
        age = _age_score(ready_since, now, self._policy.max_age_score)
        order = plan_index if self._policy.preserve_plan_order_on_tie else 0
        return (
            -score.final_score,
            -step.priority,
            -step.urgency,
            -step.criticality,
            deadline_seconds,
            -age,
            order,
            step.id,
        )


def _deadline_score(deadline: datetime | None, now: datetime) -> float:
    if deadline is None:
        return 0.0
    remaining = (deadline.astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds()
    if remaining <= 0:
        return 10.0
    if remaining <= 3600:
        return 8.0
    if remaining <= 86400:
        return 4.0
    return 1.0


def _age_score(ready_since: datetime | None, now: datetime, max_age_score: float) -> float:
    if ready_since is None:
        return 0.0
    seconds = max(
        0.0,
        (now.astimezone(timezone.utc) - ready_since.astimezone(timezone.utc)).total_seconds(),
    )
    return min(max_age_score, seconds / 3600.0)


def _risk_penalty(step: ExecutionStep) -> float:
    penalty = 0.0
    if getattr(step, "requires_confirmation", False):
        penalty += 5.0
    if not getattr(step, "parallel_safe", False):
        penalty += 1.0
    if not getattr(step, "recovery_safe", False):
        penalty += 1.0
    if not getattr(step, "idempotent", False):
        penalty += 1.0
    return penalty


def _tie_breaker(
    scores: tuple[PriorityScore, ...],
    ordered: tuple[str, ...],
    original: tuple[str, ...],
) -> str | None:
    final_scores = [score.final_score for score in scores]
    if len(set(final_scores)) != len(final_scores):
        return "deterministic_tie_break"
    if ordered == original:
        return None
    return "score"

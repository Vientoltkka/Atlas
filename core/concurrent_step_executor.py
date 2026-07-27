"""Bounded opt-in concurrent execution for independent Atlas plan steps."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any

from core.planner import ExecutionStep


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class ExecutionConcurrencyPolicy:
    """Explicit policy required before any independent steps run concurrently."""

    enabled: bool = False
    max_concurrency: int = 1
    fail_fast: bool = True
    allow_parallel_confirmations: bool = False
    safe_steps_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "fail_fast",
            "allow_parallel_confirmations",
            "safe_steps_only",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        if isinstance(self.max_concurrency, bool) or not isinstance(
            self.max_concurrency,
            int,
        ):
            raise TypeError("max_concurrency must be an integer.")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be greater than zero.")


@dataclass(frozen=True, slots=True)
class ExecutionBatch:
    """A deterministic batch of currently ready steps."""

    batch_id: str
    step_ids: tuple[str, ...]
    created_at: str
    concurrency_limit: int
    execution_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.batch_id.strip():
            raise ValueError("batch_id must be a non-empty string.")
        object.__setattr__(self, "step_ids", tuple(self.step_ids))
        object.__setattr__(self, "execution_order", tuple(self.execution_order))
        if self.concurrency_limit < 1:
            raise ValueError("concurrency_limit must be greater than zero.")


@dataclass(frozen=True, slots=True)
class ConcurrentStepResult:
    """Result of one step executed inside a bounded concurrent batch."""

    step_id: str
    status: str
    result: Any | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionBatchResult:
    """Aggregate result of one bounded concurrent batch."""

    batch_id: str
    step_results: tuple[ConcurrentStepResult, ...]
    completed_step_ids: tuple[str, ...] = ()
    failed_step_ids: tuple[str, ...] = ()
    cancelled_step_ids: tuple[str, ...] = ()
    duration_ms: int = 0
    fail_fast_triggered: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_results", tuple(self.step_results))
        if not self.completed_step_ids:
            object.__setattr__(
                self,
                "completed_step_ids",
                tuple(
                    result.step_id
                    for result in self.step_results
                    if result.status == "completed"
                ),
            )
        if not self.failed_step_ids:
            object.__setattr__(
                self,
                "failed_step_ids",
                tuple(
                    result.step_id
                    for result in self.step_results
                    if result.status == "failed"
                ),
            )
        if not self.cancelled_step_ids:
            object.__setattr__(
                self,
                "cancelled_step_ids",
                tuple(
                    result.step_id
                    for result in self.step_results
                    if result.status == "cancelled"
                ),
            )


class ExecutionResourceConflictDetector:
    """Detect simple declared resource conflicts between ready steps."""

    def conflicts(
        self,
        step: ExecutionStep,
        selected_steps: Sequence[ExecutionStep],
    ) -> bool:
        resource_keys = set(getattr(step, "resource_keys", ()))
        if not resource_keys:
            return False
        for selected in selected_steps:
            if resource_keys.intersection(getattr(selected, "resource_keys", ())):
                return True
        return False


def build_execution_batch(
    ready_steps: Sequence[ExecutionStep],
    policy: ExecutionConcurrencyPolicy,
    *,
    batch_id: str,
    created_at: str | None = None,
    conflict_detector: ExecutionResourceConflictDetector | None = None,
) -> ExecutionBatch:
    """Select a deterministic, bounded subset of ready steps for one batch."""

    if not ready_steps:
        return ExecutionBatch(
            batch_id=batch_id,
            step_ids=(),
            created_at=created_at or _utc_iso(),
            concurrency_limit=policy.max_concurrency,
            execution_order=(),
        )

    if not policy.enabled or policy.max_concurrency <= 1:
        first = ready_steps[0]
        return ExecutionBatch(
            batch_id=batch_id,
            step_ids=(first.id,),
            created_at=created_at or _utc_iso(),
            concurrency_limit=1,
            execution_order=(first.id,),
        )

    detector = conflict_detector or ExecutionResourceConflictDetector()
    selected: list[ExecutionStep] = []
    for step in ready_steps:
        if policy.safe_steps_only and not getattr(step, "parallel_safe", False):
            if not selected:
                selected.append(step)
            break
        if (
            not policy.allow_parallel_confirmations
            and getattr(step, "requires_confirmation", False)
        ):
            if not selected:
                selected.append(step)
            break
        if detector.conflicts(step, selected):
            continue
        selected.append(step)
        if len(selected) >= policy.max_concurrency:
            break

    if not selected:
        selected.append(ready_steps[0])
    step_ids = tuple(step.id for step in selected)
    return ExecutionBatch(
        batch_id=batch_id,
        step_ids=step_ids,
        created_at=created_at or _utc_iso(),
        concurrency_limit=min(policy.max_concurrency, max(1, len(step_ids))),
        execution_order=step_ids,
    )


class ConcurrentStepExecutor:
    """Run already-selected independent steps using a bounded thread pool."""

    def __init__(
        self,
        step_runner: Callable[[ExecutionStep], Any],
    ) -> None:
        self._step_runner = step_runner

    def run_batch(
        self,
        batch: ExecutionBatch,
        steps: Sequence[ExecutionStep],
        policy: ExecutionConcurrencyPolicy,
    ) -> ExecutionBatchResult:
        """Execute one batch and return deterministic step results."""

        ordered_steps = [step for step in steps if step.id in set(batch.step_ids)]
        if not ordered_steps:
            return ExecutionBatchResult(batch.batch_id, ())

        started = time.monotonic()
        results_by_step: dict[str, ConcurrentStepResult] = {}
        pending_steps = list(ordered_steps)
        in_flight: dict[Future[ConcurrentStepResult], ExecutionStep] = {}
        fail_fast_triggered = False
        max_workers = min(policy.max_concurrency, len(ordered_steps))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while pending_steps and len(in_flight) < max_workers:
                step = pending_steps.pop(0)
                in_flight[pool.submit(self._run_step, step)] = step

            while in_flight:
                done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    step = in_flight.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = ConcurrentStepResult(
                            step_id=step.id,
                            status="failed",
                            error=str(error) or type(error).__name__,
                        )
                    results_by_step[step.id] = result
                    if result.status == "failed" and policy.fail_fast:
                        fail_fast_triggered = True

                if fail_fast_triggered:
                    for future, step in tuple(in_flight.items()):
                        if future.cancel():
                            in_flight.pop(future)
                            results_by_step[step.id] = ConcurrentStepResult(
                                step_id=step.id,
                                status="cancelled",
                                error="cancelled by fail-fast policy",
                            )
                    pending_steps.clear()
                    continue

                while pending_steps and len(in_flight) < max_workers:
                    step = pending_steps.pop(0)
                    in_flight[pool.submit(self._run_step, step)] = step

        ordered_results = tuple(
            results_by_step.get(
                step.id,
                ConcurrentStepResult(
                    step_id=step.id,
                    status="cancelled",
                    error="step was not started",
                ),
            )
            for step in ordered_steps
        )
        return ExecutionBatchResult(
            batch_id=batch.batch_id,
            step_results=ordered_results,
            duration_ms=int((time.monotonic() - started) * 1000),
            fail_fast_triggered=fail_fast_triggered,
        )

    def _run_step(self, step: ExecutionStep) -> ConcurrentStepResult:
        started_at = _utc_iso()
        try:
            result = self._step_runner(step)
        except Exception as error:
            return ConcurrentStepResult(
                step_id=step.id,
                status="failed",
                error=str(error) or type(error).__name__,
                started_at=started_at,
                finished_at=_utc_iso(),
            )
        return ConcurrentStepResult(
            step_id=step.id,
            status="completed",
            result=result,
            started_at=started_at,
            finished_at=_utc_iso(),
        )

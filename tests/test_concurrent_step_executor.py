from __future__ import annotations

from threading import Barrier, Lock
import time

import pytest

from core.concurrent_step_executor import (
    ConcurrentStepExecutor,
    ExecutionConcurrencyPolicy,
    ExecutionResourceConflictDetector,
    build_execution_batch,
)
from core.planner import ExecutionStep


def _step(
    step_id: str,
    *,
    parallel_safe: bool = True,
    resource_keys: tuple[str, ...] = (),
) -> ExecutionStep:
    return ExecutionStep(
        step_id,
        f"run {step_id}",
        "tool",
        parallel_safe=parallel_safe,
        resource_keys=resource_keys,
    )


def test_policy_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValueError):
        ExecutionConcurrencyPolicy(enabled=True, max_concurrency=0)


def test_disabled_policy_builds_single_step_batch() -> None:
    batch = build_execution_batch(
        (_step("a"), _step("b")),
        ExecutionConcurrencyPolicy(enabled=False, max_concurrency=4),
        batch_id="batch.1",
    )

    assert batch.step_ids == ("a",)
    assert batch.concurrency_limit == 1


def test_safe_independent_steps_run_concurrently() -> None:
    barrier = Barrier(2)
    seen: list[str] = []
    lock = Lock()

    def runner(step: ExecutionStep) -> str:
        barrier.wait(timeout=2)
        with lock:
            seen.append(step.id)
        return step.id

    policy = ExecutionConcurrencyPolicy(enabled=True, max_concurrency=2)
    steps = (_step("a"), _step("b"))
    batch = build_execution_batch(steps, policy, batch_id="batch.1")

    result = ConcurrentStepExecutor(runner).run_batch(batch, steps, policy)

    assert tuple(sorted(seen)) == ("a", "b")
    assert result.completed_step_ids == ("a", "b")
    assert result.failed_step_ids == ()


def test_max_concurrency_one_runs_sequentially() -> None:
    active = 0
    max_seen = 0
    lock = Lock()

    def runner(step: ExecutionStep) -> str:
        nonlocal active, max_seen
        with lock:
            active += 1
            max_seen = max(max_seen, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return step.id

    policy = ExecutionConcurrencyPolicy(enabled=True, max_concurrency=1)
    steps = (_step("a"), _step("b"))
    batch = build_execution_batch(steps, policy, batch_id="batch.1")

    result = ConcurrentStepExecutor(runner).run_batch(batch, steps, policy)

    assert batch.step_ids == ("a",)
    assert result.completed_step_ids == ("a",)
    assert max_seen == 1


def test_resource_conflict_keeps_step_out_of_same_batch() -> None:
    policy = ExecutionConcurrencyPolicy(enabled=True, max_concurrency=3)
    steps = (
        _step("a", resource_keys=("file:README.md",)),
        _step("b", resource_keys=("file:README.md",)),
        _step("c", resource_keys=("file:CHANGELOG.md",)),
    )

    batch = build_execution_batch(steps, policy, batch_id="batch.1")

    assert batch.step_ids == ("a", "c")
    assert ExecutionResourceConflictDetector().conflicts(steps[1], (steps[0],))


def test_unsafe_step_runs_alone_when_safe_steps_only() -> None:
    policy = ExecutionConcurrencyPolicy(enabled=True, max_concurrency=3)
    steps = (_step("a", parallel_safe=False), _step("b"))

    batch = build_execution_batch(steps, policy, batch_id="batch.1")

    assert batch.step_ids == ("a",)


def test_fail_fast_false_collects_success_and_failure() -> None:
    def runner(step: ExecutionStep) -> str:
        if step.id == "a":
            raise RuntimeError("boom")
        return step.id

    policy = ExecutionConcurrencyPolicy(
        enabled=True,
        max_concurrency=2,
        fail_fast=False,
    )
    steps = (_step("a"), _step("b"))
    batch = build_execution_batch(steps, policy, batch_id="batch.1")

    result = ConcurrentStepExecutor(runner).run_batch(batch, steps, policy)

    assert result.failed_step_ids == ("a",)
    assert result.completed_step_ids == ("b",)
    assert result.fail_fast_triggered is False


def test_fail_fast_true_marks_policy_triggered() -> None:
    def runner(step: ExecutionStep) -> str:
        if step.id == "a":
            raise RuntimeError("boom")
        time.sleep(0.01)
        return step.id

    policy = ExecutionConcurrencyPolicy(enabled=True, max_concurrency=2)
    steps = (_step("a"), _step("b"))
    batch = build_execution_batch(steps, policy, batch_id="batch.1")

    result = ConcurrentStepExecutor(runner).run_batch(batch, steps, policy)

    assert result.failed_step_ids == ("a",)
    assert result.fail_fast_triggered is True

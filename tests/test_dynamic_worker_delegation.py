"""Deterministic dynamic worker delegation: selection, fallback, cap, metadata."""

from __future__ import annotations

import pytest

from core.async_task_scheduler import (
    AsyncTaskScheduler,
    JsonGoalTaskStore,
    Task,
    TaskStatus,
    ToolTaskExecutor,
)
from core.worker_delegation import (
    DynamicWorkerDelegator,
    ModelWorker,
    is_synthesis_transform,
    sanitize_worker_error,
)


class FakeWorker:
    """Deterministic worker double with an explicit call log."""

    def __init__(
        self,
        worker_id: str,
        *,
        tier: int = 0,
        available: bool = True,
        error: str | None = None,
        output: str | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.tier = tier
        self._available = available
        self._error = error
        self._output = output
        self.calls: list[str] = []

    def available(self) -> bool:
        return self._available

    def supports(self, task_kind: str) -> bool:
        return task_kind == "transform"

    def execute(self, instruction: str) -> str:
        self.calls.append(instruction)
        if self._error is not None:
            raise RuntimeError(self._error)
        return self._output or f"{self.worker_id} resultado"


def _delegator(local: FakeWorker, gemini: FakeWorker) -> DynamicWorkerDelegator:
    return DynamicWorkerDelegator((local, gemini))


def test_light_transform_selects_local_and_never_calls_gemini() -> None:
    local = FakeWorker("LOCAL", tier=0, output="resumen local")
    gemini = FakeWorker("GEMINI", tier=1, output="resumen gemini")
    delegator = _delegator(local, gemini)

    result = delegator.delegate("resume el texto")

    assert result.success
    assert result.output == "resumen local"
    assert result.final_worker == "LOCAL"
    assert result.preferred_worker == "LOCAL"
    assert [record.worker_id for record in result.attempted] == ["LOCAL"]
    assert gemini.calls == []


def test_synthesis_transform_prefers_the_capable_worker() -> None:
    local = FakeWorker("LOCAL", tier=0)
    gemini = FakeWorker("GEMINI", tier=1, output="síntesis gemini")
    delegator = _delegator(local, gemini)

    result = delegator.delegate("compara las fuentes", synthesis=True)

    assert result.success
    assert result.final_worker == "GEMINI"
    assert local.calls == []
    assert is_synthesis_transform({"input_tasks": ["a", "b"]}, "resume")
    assert not is_synthesis_transform({"input_task": "a"}, "resume el texto")


def test_failed_primary_falls_back_once_to_gemini() -> None:
    local = FakeWorker("LOCAL", tier=0, error="ollama no responde")
    gemini = FakeWorker("GEMINI", tier=1, output="resumen gemini")
    delegator = _delegator(local, gemini)

    result = delegator.delegate("texto")

    assert result.success
    assert result.output == "resumen gemini"
    assert result.final_worker == "GEMINI"
    assert result.preferred_worker == "LOCAL"
    assert [record.worker_id for record in result.attempted] == ["LOCAL", "GEMINI"]
    assert result.attempted[0].error == "ollama no responde"
    assert result.attempted[1].error is None
    assert len(local.calls) == 1
    assert len(gemini.calls) == 1


def test_unavailable_primary_is_skipped_without_an_attempt() -> None:
    local = FakeWorker("LOCAL", tier=0, available=False)
    gemini = FakeWorker("GEMINI", tier=1, output="resumen gemini")
    delegator = _delegator(local, gemini)

    result = delegator.delegate("texto")

    assert result.success
    assert result.final_worker == "GEMINI"
    assert local.calls == []
    assert [record.worker_id for record in result.attempted] == ["GEMINI"]


def test_attempt_cap_is_two_and_all_workers_failing_fails_the_task() -> None:
    local = FakeWorker("LOCAL", tier=0, error="fallo local")
    gemini = FakeWorker("GEMINI", tier=1, error="fallo gemini")
    delegator = _delegator(local, gemini)

    result = delegator.delegate("texto")

    assert not result.success
    assert result.final_worker is None
    assert len(result.attempted) == 2
    assert result.attempted[0].error == "fallo local"
    assert result.attempted[1].error == "fallo gemini"
    assert result.error is not None
    assert len(local.calls) == 1
    assert len(gemini.calls) == 1


def test_sanitized_errors_stay_bounded_and_single_line() -> None:
    noisy = RuntimeError("secreto-valor\n" + "x" * 500)
    sanitized = sanitize_worker_error(noisy)

    assert "\n" not in sanitized
    assert len(sanitized) <= 200
    assert sanitized.endswith("...")


def test_empty_output_is_a_failed_attempt_and_falls_back() -> None:
    local = FakeWorker("LOCAL", tier=0, output="   ")
    gemini = FakeWorker("GEMINI", tier=1, output="resumen gemini")
    delegator = _delegator(local, gemini)

    result = delegator.delegate("texto")

    assert result.success
    assert result.final_worker == "GEMINI"
    assert result.attempted[0].error == "worker output failed verification"


def _scheduler_with_transform(tmp_path, delegator):
    executor = ToolTaskExecutor(object(), worker_delegator=delegator)
    scheduler = AsyncTaskScheduler(
        executor,
        store=JsonGoalTaskStore(tmp_path / "goals"),
    )
    executor.bind_result_lookup(scheduler.task)
    return scheduler


def test_transform_task_records_worker_metadata_and_persists_it(tmp_path) -> None:
    local = FakeWorker("LOCAL", tier=0, output="resumen del archivo")
    gemini = FakeWorker("GEMINI", tier=1)
    scheduler = _scheduler_with_transform(tmp_path, _delegator(local, gemini))
    goal_id = scheduler.submit_goal(
        "resumir",
        [
            {
                "task_id": "read_source",
                "description": "leer",
                "payload": {"tool": "read_file", "arguments": {"path": "a.txt"}},
            },
            {
                "task_id": "summarize_source",
                "description": "resumir",
                "dependencies": ["read_source"],
                "payload": {
                    "kind": "transform",
                    "instruction": "resume: {input}",
                    "input_task": "read_source",
                },
            },
        ],
    )
    state = scheduler.goal(goal_id)
    state.tasks["read_source"].status = TaskStatus.DONE
    state.tasks["read_source"].result = "contenido"

    scheduler.run_ready()

    task = state.tasks["summarize_source"]
    assert task.status is TaskStatus.DONE
    assert task.result == "resumen del archivo"
    assert task.metadata == {
        "preferred_worker": "LOCAL",
        "attempted_workers": [{"worker_id": "LOCAL", "error": None}],
        "final_worker": "LOCAL",
    }
    assert gemini.calls == []

    reloaded = scheduler.load_goal(goal_id)
    assert reloaded.tasks["summarize_source"].metadata == task.metadata


def test_task_metadata_round_trip_stays_backward_compatible() -> None:
    legacy_payload = {
        "task_id": "t",
        "goal_id": "g",
        "description": "d",
        "status": "done",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    assert Task.from_dict(legacy_payload).metadata is None
    assert Task.from_dict(Task(task_id="t2", goal_id="g", description="d").to_dict()).metadata is None
    with_metadata = Task(
        task_id="t3",
        goal_id="g",
        description="d",
        metadata={"final_worker": "LOCAL"},
    )
    assert Task.from_dict(with_metadata.to_dict()).metadata == {"final_worker": "LOCAL"}


def test_model_worker_availability_failure_is_swallowed() -> None:
    def broken() -> bool:
        raise RuntimeError("down")

    worker = ModelWorker(
        worker_id="local",
        invoke=lambda instruction: "ok",
        tier=0,
        availability=broken,
    )
    assert worker.available() is False
    healthy = ModelWorker(worker_id="gemini", invoke=lambda instruction: "ok", tier=1)
    assert healthy.available() is True


def test_delegator_rejects_non_positive_attempt_cap() -> None:
    with pytest.raises(ValueError):
        DynamicWorkerDelegator((), max_attempts=0)

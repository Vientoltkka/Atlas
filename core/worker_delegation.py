"""Bounded dynamic worker delegation for transform tasks (V1).

The AsyncTaskScheduler stays the coordinator: delegation happens inside the
execution of a single transform task. Workers are thin adapters over the
existing model abstractions; selection is a deterministic declarative policy
(cheap/local first for light transforms, most capable first for synthesis),
with at most two workers attempted per task before the task fails.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

MAX_DELEGATION_ATTEMPTS = 2

_ERROR_DETAIL_LIMIT = 200

_SYNTHESIS_HINTS: tuple[str, ...] = (
    "compara",
    "sintetiza",
    "síntesis",
    "sintesis",
    "extrae los puntos",
    "puntos clave",
    "concluye",
    "conclusiones",
)


class WorkerCapability(Protocol):
    """Minimal contract every delegation worker satisfies."""

    @property
    def worker_id(self) -> str:
        """Stable identifier used in metadata and policies."""

    def available(self) -> bool:
        """Whether this worker can currently accept work."""

    def supports(self, task_kind: str) -> bool:
        """Whether this worker accepts tasks of the given kind."""

    def execute(self, instruction: str) -> str:
        """Run the instruction and return the produced text."""


@dataclass(frozen=True)
class AttemptRecord:
    """One bounded attempt against a worker."""

    worker_id: str
    error: str | None = None


@dataclass(frozen=True)
class DelegationResult:
    """Outcome of one delegated execution attempt chain."""

    output: str | None
    final_worker: str | None
    preferred_worker: str | None
    attempted: tuple[AttemptRecord, ...]
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.output is not None and self.final_worker is not None

    def metadata(self) -> dict[str, Any]:
        """Minimal worker metadata safe to persist with the task."""
        payload: dict[str, Any] = {
            "attempted_workers": [
                {"worker_id": record.worker_id, "error": record.error}
                for record in self.attempted
            ]
        }
        if self.preferred_worker is not None:
            payload["preferred_worker"] = self.preferred_worker
        if self.final_worker is not None:
            payload["final_worker"] = self.final_worker
        return payload


class DelegationPolicy:
    """Deterministic ordering policy: cheap first for light work."""

    def select(
        self,
        workers: Sequence[WorkerCapability],
        *,
        task_kind: str,
        synthesis: bool,
    ) -> tuple[WorkerCapability, ...]:
        candidates = [
            worker
            for worker in workers
            if worker.supports(task_kind) and worker.available()
        ]
        tiers = {worker.worker_id: getattr(worker, "tier", 0) for worker in candidates}
        if synthesis:
            candidates.sort(key=lambda worker: (-tiers[worker.worker_id],))
        else:
            candidates.sort(key=lambda worker: (tiers[worker.worker_id],))
        return tuple(candidates)


def sanitize_worker_error(error: BaseException | str) -> str:
    """Return a bounded, single-line, secret-free failure detail."""
    detail = str(error).replace("\r", " ").replace("\n", " ").strip()
    if len(detail) > _ERROR_DETAIL_LIMIT:
        detail = detail[: _ERROR_DETAIL_LIMIT - 3] + "..."
    return detail or type(error).__name__


def default_output_verifier(output: Any, task_kind: str) -> bool:
    """Minimal deterministic check: the worker must return non-empty text."""
    return isinstance(output, str) and bool(output.strip())


def is_synthesis_transform(payload: dict[str, Any] | None, instruction: str) -> bool:
    """Classify one transform payload: multi-source or synthesis wording."""
    input_tasks = (payload or {}).get("input_tasks")
    if isinstance(input_tasks, (list, tuple)) and len(input_tasks) > 1:
        return True
    normalized = instruction.lower()
    return any(hint in normalized for hint in _SYNTHESIS_HINTS)


class DynamicWorkerDelegator:
    """Delegate one transform instruction to at most two declared workers."""

    def __init__(
        self,
        workers: Iterable[WorkerCapability] = (),
        *,
        policy: DelegationPolicy | None = None,
        max_attempts: int = MAX_DELEGATION_ATTEMPTS,
        output_verifier: Callable[[Any, str], bool] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")
        self._workers: tuple[WorkerCapability, ...] = tuple(workers)
        self._policy = policy or DelegationPolicy()
        self._max_attempts = max_attempts
        self._output_verifier = output_verifier or default_output_verifier

    @property
    def workers(self) -> tuple[WorkerCapability, ...]:
        return self._workers

    def delegate(
        self,
        instruction: str,
        *,
        task_kind: str = "transform",
        synthesis: bool = False,
    ) -> DelegationResult:
        candidates = self._policy.select(
            self._workers,
            task_kind=task_kind,
            synthesis=synthesis,
        )
        preferred = candidates[0].worker_id if candidates else None
        attempted: list[AttemptRecord] = []
        for worker in candidates[: self._max_attempts]:
            try:
                output = worker.execute(instruction)
            except Exception as error:  # noqa: BLE001 - fallback must absorb failures
                attempted.append(
                    AttemptRecord(worker.worker_id, sanitize_worker_error(error))
                )
                continue
            if not self._output_verifier(output, task_kind):
                attempted.append(
                    AttemptRecord(worker.worker_id, "worker output failed verification")
                )
                continue
            attempted.append(AttemptRecord(worker.worker_id))
            return DelegationResult(
                output=output,
                final_worker=worker.worker_id,
                preferred_worker=preferred,
                attempted=tuple(attempted),
            )
        return DelegationResult(
            output=None,
            final_worker=None,
            preferred_worker=preferred,
            attempted=tuple(attempted),
            error="no candidate worker completed the task.",
        )


@dataclass(frozen=True)
class ModelWorker:
    """Adapter binding an existing agent/model invocation into a worker."""

    worker_id: str
    invoke: Callable[[str], str]
    tier: int = 0
    availability: Callable[[], bool] | None = None
    task_kinds: frozenset[str] = field(default=frozenset({"transform"}))

    @property
    def tier_key(self) -> int:
        return self.tier

    def available(self) -> bool:
        if self.availability is None:
            return True
        try:
            return bool(self.availability())
        except Exception:  # noqa: BLE001 - an unavailable worker must not crash
            return False

    def supports(self, task_kind: str) -> bool:
        return task_kind in self.task_kinds

    def execute(self, instruction: str) -> str:
        return self.invoke(instruction)

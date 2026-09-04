"""Minimal non-blocking task scheduler with per-task approval pauses."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import uuid
from enum import Enum
from typing import Any, Callable, Protocol

_SCHEMA_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    """Per-task lifecycle states; WAITING_APPROVAL blocks only that task."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    RESUMABLE = "resumable"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


_TERMINAL_TASK_STATES = frozenset(
    {
        TaskStatus.DONE,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }
)


class AsyncTaskSchedulerError(RuntimeError):
    """Base error for the async task scheduler."""


class UnknownTaskError(AsyncTaskSchedulerError):
    """Raised when a task id does not exist."""


class InvalidApprovalError(AsyncTaskSchedulerError):
    """Raised when an approval token is unknown, stale or already consumed."""


class UnknownGoalError(AsyncTaskSchedulerError):
    """Raised when a goal id does not exist."""


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """Result returned by a task executor for one invocation."""

    completed: bool
    result: Any = None
    error: str | None = None
    needs_approval: bool = False
    approval_prompt: str | None = None
    resumable_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @staticmethod
    def succeed(result: Any = None, metadata: dict[str, Any] | None = None) -> "TaskOutcome":
        return TaskOutcome(completed=True, result=result, metadata=metadata)

    @staticmethod
    def pause_for_approval(
        prompt: str,
        resumable_payload: dict[str, Any] | None = None,
    ) -> "TaskOutcome":
        return TaskOutcome(
            completed=False,
            needs_approval=True,
            approval_prompt=prompt,
            resumable_payload=resumable_payload,
        )

    @staticmethod
    def fail(error: str) -> "TaskOutcome":
        return TaskOutcome(completed=False, error=error)


TaskExecutor = Callable[[Any, dict[str, Any] | None], TaskOutcome]
TaskVerifier = Callable[[Any, Any], bool]


@dataclass(slots=True)
class Task:
    """Independent unit of work with its own lifecycle state."""

    task_id: str
    goal_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: tuple[str, ...] = ()
    requires_approval: bool = False
    initial_payload: dict[str, Any] | None = None
    resumable_payload: dict[str, Any] | None = None
    result: Any = None
    error: str | None = None
    pending_confirmation_id: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string.")
        if not self.goal_id.strip():
            raise ValueError("goal_id must be a non-empty string.")
        for dependency in self.dependencies:
            if dependency == self.task_id:
                raise ValueError("a task cannot depend on itself.")
        if self.status is TaskStatus.PENDING and not self.dependencies:
            self.status = TaskStatus.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "description": self.description,
            "status": self.status.value,
            "dependencies": list(self.dependencies),
            "requires_approval": self.requires_approval,
            "initial_payload": self.initial_payload,
            "resumable_payload": self.resumable_payload,
            "result": self.result,
            "error": self.error,
            "pending_confirmation_id": self.pending_confirmation_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "Task":
        return Task(
            task_id=payload["task_id"],
            goal_id=payload["goal_id"],
            description=payload["description"],
            status=TaskStatus(payload["status"]),
            dependencies=tuple(payload.get("dependencies", ())),
            requires_approval=bool(payload.get("requires_approval", False)),
            initial_payload=payload.get("initial_payload"),
            resumable_payload=payload.get("resumable_payload"),
            result=payload.get("result"),
            error=payload.get("error"),
            pending_confirmation_id=payload.get("pending_confirmation_id"),
            created_at=datetime.fromisoformat(payload["created_at"]),
            started_at=(
                datetime.fromisoformat(payload["started_at"])
                if payload.get("started_at")
                else None
            ),
            finished_at=(
                datetime.fromisoformat(payload["finished_at"])
                if payload.get("finished_at")
                else None
            ),
            metadata=payload.get("metadata"),
        )


@dataclass(slots=True)
class PendingApproval:
    """Single-use approval token bound to exactly one task."""

    confirmation_id: str
    task_id: str
    goal_id: str
    prompt: str | None = None
    issued_at: datetime = field(default_factory=_utc_now)
    consumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmation_id": self.confirmation_id,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "prompt": self.prompt,
            "issued_at": self.issued_at.isoformat(),
            "consumed": self.consumed,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "PendingApproval":
        return PendingApproval(
            confirmation_id=payload["confirmation_id"],
            task_id=payload["task_id"],
            goal_id=payload["goal_id"],
            prompt=payload.get("prompt"),
            issued_at=datetime.fromisoformat(payload["issued_at"]),
            consumed=bool(payload.get("consumed", False)),
        )


@dataclass(slots=True)
class GoalState:
    """Set of independent tasks sharing one objective."""

    goal_id: str
    description: str
    tasks: dict[str, Task] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "tasks": [task.to_dict() for task in self.tasks.values()],
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "GoalState":
        return GoalState(
            goal_id=payload["goal_id"],
            description=payload["description"],
            tasks={
                task["task_id"]: Task.from_dict(task)
                for task in payload.get("tasks", [])
            },
            created_at=datetime.fromisoformat(payload["created_at"]),
        )


class GoalTaskStore(Protocol):
    """Persistence contract for one goal's pending task state."""

    def save(self, goal_id: str, payload: dict[str, Any]) -> None:
        """Persist the serialized goal state."""

    def load(self, goal_id: str) -> dict[str, Any] | None:
        """Load a serialized goal state if present."""

    def delete(self, goal_id: str) -> None:
        """Delete a persisted goal state."""


class JsonGoalTaskStore:
    """JSON-file store following the local execution-state persistence pattern."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path(self, goal_id: str) -> Path:
        return self._directory / f"{goal_id}.json"

    def save(self, goal_id: str, payload: dict[str, Any]) -> None:
        path = self._path(goal_id)
        temporary_path = path.with_suffix(".tmp")
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with open(temporary_path, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        os.replace(temporary_path, path)

    def load(self, goal_id: str) -> dict[str, Any] | None:
        path = self._path(goal_id)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def delete(self, goal_id: str) -> None:
        self._path(goal_id).unlink(missing_ok=True)


class AsyncTaskScheduler:
    """Cooperative scheduler where one WAITING_APPROVAL never blocks READY tasks."""

    def __init__(
        self,
        task_executor: TaskExecutor,
        *,
        store: GoalTaskStore | None = None,
        verifier: TaskVerifier | None = None,
    ) -> None:
        self._task_executor = task_executor
        self._store = store
        self._verifier = verifier
        self._goals: dict[str, GoalState] = {}
        self._approvals: dict[str, PendingApproval] = {}
        self._lock = threading.RLock()

    # -- goal / task management -------------------------------------------------

    def submit_goal(
        self,
        description: str,
        tasks: list[dict[str, Any]],
        *,
        goal_id: str | None = None,
    ) -> str:
        with self._lock:
            resolved_goal_id = goal_id or uuid.uuid4().hex
            if resolved_goal_id in self._goals:
                raise AsyncTaskSchedulerError(
                    f"goal id already exists: {resolved_goal_id}"
                )
            state = GoalState(
                goal_id=resolved_goal_id,
                description=description,
            )
            for spec in tasks:
                task = Task(
                    task_id=spec["task_id"],
                    goal_id=resolved_goal_id,
                    description=spec["description"],
                    dependencies=tuple(spec.get("dependencies", ())),
                    requires_approval=bool(spec.get("requires_approval", False)),
                    initial_payload=spec.get("payload"),
                )
                state.tasks[task.task_id] = task
            for task in state.tasks.values():
                for dependency in task.dependencies:
                    if dependency not in state.tasks:
                        raise AsyncTaskSchedulerError(
                            f"task {task.task_id} depends on unknown "
                            f"task {dependency}"
                        )
            self._goals[resolved_goal_id] = state
            self._persist(resolved_goal_id)
            return resolved_goal_id

    def goal(self, goal_id: str) -> GoalState:
        with self._lock:
            return self._require_goal(goal_id)

    def task(self, task_id: str) -> Task:
        with self._lock:
            return self._require_task(task_id)

    # -- scheduling -------------------------------------------------------------

    def run_ready(self) -> list[str]:
        """Run every READY task; a pause only stops the paused task itself."""
        completed_now: list[str] = []
        with self._lock:
            for goal_id in list(self._goals):
                state = self._goals[goal_id]
                for task in state.tasks.values():
                    task.status = self._initial_status(task, state)
            progressed = True
            while progressed:
                progressed = False
                for goal_id in list(self._goals):
                    state = self._goals[goal_id]
                    for task_id in list(state.tasks):
                        task = state.tasks[task_id]
                        if task.status is TaskStatus.RESUMABLE:
                            progressed = self._run_task(task) or progressed
                            if task.status is TaskStatus.DONE:
                                completed_now.append(task_id)
                    for task_id in list(state.tasks):
                        task = state.tasks[task_id]
                        if task.status is TaskStatus.READY:
                            progressed = self._run_task(task) or progressed
                            if task.status is TaskStatus.DONE:
                                completed_now.append(task_id)
                    for task in state.tasks.values():
                        if task.status is TaskStatus.PENDING:
                            promoted = self._initial_status(task, state)
                            if promoted is not task.status:
                                task.status = promoted
                                progressed = True
            self._persist_all()
        return completed_now

    def _initial_status(self, task: Task, state: GoalState) -> TaskStatus:
        if task.status is not TaskStatus.PENDING:
            return task.status
        dependency_states = [
            state.tasks[dependency].status for dependency in task.dependencies
        ]
        if any(
            status in _TERMINAL_TASK_STATES and status is not TaskStatus.DONE
            for status in dependency_states
        ):
            return TaskStatus.BLOCKED
        if all(status is TaskStatus.DONE for status in dependency_states):
            return TaskStatus.READY
        return TaskStatus.PENDING

    def _run_task(self, task: Task) -> bool:
        """Execute one task; returns True when it reached DONE during this call."""
        task.status = TaskStatus.RUNNING
        task.started_at = task.started_at or _utc_now()
        outcome = self._safe_execute(task)
        if outcome.needs_approval:
            approval = PendingApproval(
                confirmation_id=uuid.uuid4().hex,
                task_id=task.task_id,
                goal_id=task.goal_id,
                prompt=outcome.approval_prompt,
            )
            self._approvals[approval.confirmation_id] = approval
            task.status = TaskStatus.WAITING_APPROVAL
            task.pending_confirmation_id = approval.confirmation_id
            task.resumable_payload = outcome.resumable_payload
            task.error = None
            return False
        return self._finalize(task, outcome)

    def _safe_execute(self, task: Task) -> TaskOutcome:
        try:
            return self._task_executor(task, task.resumable_payload)
        except Exception as error:  # noqa: BLE001 - scheduler must never crash
            return TaskOutcome.fail(f"executor error: {error}")

    def _finalize(self, task: Task, outcome: TaskOutcome) -> bool:
        if not outcome.completed:
            task.status = TaskStatus.FAILED
            task.error = outcome.error
            task.finished_at = _utc_now()
            return False
        if self._verifier is not None and not self._verifier(task, outcome.result):
            task.status = TaskStatus.FAILED
            task.error = "verifier rejected the task result"
            task.finished_at = _utc_now()
            return False
        task.status = TaskStatus.DONE
        task.result = outcome.result
        task.error = None
        task.metadata = outcome.metadata
        task.pending_confirmation_id = None
        task.finished_at = _utc_now()
        return True

    # -- approval bridge --------------------------------------------------------

    def approve(self, confirmation_id: str) -> str | None:
        """Consume a single-use approval token; resumes only the bound task."""
        with self._lock:
            approval = self._approvals.get(confirmation_id)
            if approval is None or approval.consumed:
                raise InvalidApprovalError(
                    f"unknown or already consumed confirmation: {confirmation_id}"
                )
            task = self._require_task(approval.task_id)
            if task.task_id != approval.task_id or task.goal_id != approval.goal_id:
                raise InvalidApprovalError("approval does not match its task.")
            if task.status is not TaskStatus.WAITING_APPROVAL:
                raise InvalidApprovalError(
                    f"task {task.task_id} is not waiting for approval."
                )
            approval.consumed = True
            task.pending_confirmation_id = None
            task.status = TaskStatus.RESUMABLE
            goal_id = task.goal_id
        resumed = self.run_ready()
        with self._lock:
            self._persist(goal_id)
        return approval.task_id if approval.task_id in resumed else None

    def deny(self, confirmation_id: str) -> str:
        """Deny one pending approval; only the bound task is affected."""
        with self._lock:
            approval = self._approvals.get(confirmation_id)
            if approval is None or approval.consumed:
                raise InvalidApprovalError(
                    f"unknown or already consumed confirmation: {confirmation_id}"
                )
            task = self._require_task(approval.task_id)
            if task.status is not TaskStatus.WAITING_APPROVAL:
                raise InvalidApprovalError(
                    f"task {task.task_id} is not waiting for approval."
                )
            approval.consumed = True
            task.pending_confirmation_id = None
            task.status = TaskStatus.BLOCKED
            task.error = "approval denied by user"
            task.finished_at = _utc_now()
            self._persist(task.goal_id)
            return task.task_id

    def pending_approvals(self, goal_id: str | None = None) -> list[PendingApproval]:
        with self._lock:
            return [
                approval
                for approval in self._approvals.values()
                if not approval.consumed
                and (goal_id is None or approval.goal_id == goal_id)
            ]

    # -- objective status -------------------------------------------------------

    def goal_status(self, goal_id: str) -> TaskStatus:
        with self._lock:
            state = self._require_goal(goal_id)
            statuses = [task.status for task in state.tasks.values()]
            if all(status in _TERMINAL_TASK_STATES for status in statuses):
                return TaskStatus.DONE
            if TaskStatus.WAITING_APPROVAL in statuses:
                return TaskStatus.WAITING_APPROVAL
            if any(status is TaskStatus.RESUMABLE for status in statuses):
                return TaskStatus.RESUMABLE
            if any(
                status
                in {TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.PENDING}
                for status in statuses
            ):
                return TaskStatus.RUNNING
            return TaskStatus.BLOCKED

    def goal_finished(self, goal_id: str) -> bool:
        return self.goal_status(goal_id) is TaskStatus.DONE

    # -- persistence ------------------------------------------------------------

    def persist_goal(self, goal_id: str) -> None:
        with self._lock:
            self._persist(goal_id)

    def load_goal(self, goal_id: str) -> GoalState:
        """Rebuild a pending goal from the store after a restart."""
        with self._lock:
            if self._store is None:
                raise AsyncTaskSchedulerError("no persistence store configured.")
            payload = self._store.load(goal_id)
            if payload is None:
                raise UnknownGoalError(f"no persisted state for goal: {goal_id}")
            state = GoalState.from_dict(payload)
            self._goals[goal_id] = state
            for task in state.tasks.values():
                confirmation_id = task.pending_confirmation_id
                if task.status is TaskStatus.WAITING_APPROVAL and confirmation_id:
                    self._approvals[confirmation_id] = PendingApproval(
                        confirmation_id=confirmation_id,
                        task_id=task.task_id,
                        goal_id=task.goal_id,
                    )
                elif task.status is TaskStatus.WAITING_APPROVAL:
                    task.status = TaskStatus.BLOCKED
                    task.error = "restored without a pending approval token"
            return state

    def _require_goal(self, goal_id: str) -> GoalState:
        state = self._goals.get(goal_id)
        if state is None:
            raise UnknownGoalError(f"unknown goal: {goal_id}")
        return state

    def _require_task(self, task_id: str) -> Task:
        for state in self._goals.values():
            if task_id in state.tasks:
                return state.tasks[task_id]
        raise UnknownTaskError(f"unknown task: {task_id}")

    def _persist(self, goal_id: str) -> None:
        if self._store is None:
            return
        state = self._goals.get(goal_id)
        if state is None:
            return
        self._store.save(goal_id, state.to_dict())

    def _persist_all(self) -> None:
        for goal_id in self._goals:
            self._persist(goal_id)


class ToolTaskExecutor:
    """Adapter that runs scheduler tasks through the existing SingleToolRunner.

    Tool confirmation policies are not modified: the runner keeps deciding
    which tools require confirmation, and its own pending confirmation is
    consumed with an explicit ``yes`` only after the scheduler approval for
    this exact task has been granted.
    """

    def __init__(
        self,
        runner,
        *,
        model_transformer: "Callable[[str], str] | None" = None,
        worker_delegator: "Any | None" = None,
    ) -> None:
        self._runner = runner
        self._model_transformer = model_transformer
        self._worker_delegator = worker_delegator
        self._result_lookup: "Callable[[str], Any] | None" = None

    def bind_result_lookup(self, result_lookup: "Callable[[str], Any]") -> None:
        """Attach the scheduler task lookup used to chain dependency results."""
        self._result_lookup = result_lookup

    def __call__(self, task: Task, resumable_payload: dict[str, Any] | None) -> TaskOutcome:
        from tools.intent_selector import ToolIntent  # local import: adapter hook

        payload = resumable_payload or task.initial_payload or {}
        runner_confirmation_id = payload.get("runner_confirmation_id")
        if runner_confirmation_id:
            return self._resume(runner_confirmation_id)
        if payload.get("kind") == "transform":
            return self._run_transform(task, payload)
        tool_name = payload.get("tool")
        if not tool_name:
            return TaskOutcome.fail("task payload is missing the tool name.")
        arguments = dict(payload.get("arguments", {}))
        content_task_id = payload.get("content_task")
        if content_task_id:
            arguments["content"] = self._dependency_result(content_task_id)
        result = self._runner.run_registered_tool(tool_name, arguments)
        if result is None:
            return TaskOutcome.fail(f"no intent mapping for tool: {tool_name}")
        if result.status == "confirmation_required":
            stored = self._runner.pending_confirmation(result.confirmation_id)
            return TaskOutcome.pause_for_approval(
                stored.prompt if stored else f"Confirma la tarea {task.task_id}.",
                resumable_payload={
                    "tool": tool_name,
                    "arguments": arguments,
                    "runner_confirmation_id": result.confirmation_id,
                },
            )
        return self._from_result(result)

    def _resume(self, runner_confirmation_id: str) -> TaskOutcome:
        result = self._runner.confirm(runner_confirmation_id, "yes")
        if result is None:
            return TaskOutcome.fail("runner confirmation was not available.")
        return self._from_result(result)

    def _run_transform(self, task: Task, payload: dict[str, Any]) -> TaskOutcome:
        """Run a model-only text transformation over a dependency result."""
        instruction = str(payload.get("instruction", ""))
        if self._worker_delegator is not None:
            return self._delegate_transform(instruction, payload)
        if self._model_transformer is None:
            return TaskOutcome.fail(
                "no model transformer is available for transform tasks."
            )
        try:
            source_text = self._composed_transform_input(payload)
            return TaskOutcome.succeed(
                self._model_transformer(instruction.replace("{input}", source_text))
            )
        except Exception as error:  # noqa: BLE001 - transform must not crash goal
            return TaskOutcome.fail(f"transform task failed: {error}")

    def _delegate_transform(self, instruction: str, payload: dict[str, Any]) -> TaskOutcome:
        """Delegate the transform to the bounded dynamic worker delegator."""
        from core.worker_delegation import is_synthesis_transform

        try:
            source_text = self._composed_transform_input(payload)
            delegation = self._worker_delegator.delegate(
                instruction.replace("{input}", source_text),
                task_kind="transform",
                synthesis=is_synthesis_transform(payload, instruction),
            )
        except Exception as error:  # noqa: BLE001 - transform must not crash goal
            return TaskOutcome.fail(f"transform task failed: {error}")
        if not delegation.success:
            failure = delegation.error or "no candidate worker completed the task."
            return TaskOutcome.fail(f"transform task failed: {failure}")
        return TaskOutcome.succeed(delegation.output, metadata=delegation.metadata())

    def _composed_transform_input(self, payload: dict[str, Any]) -> str:
        """Compose transform input from one or several labeled dependencies.

        Multiple sources keep their order and task ids visible so the model
        receives clearly identified sections, never a mixed blob.
        """
        input_tasks = payload.get("input_tasks")
        if input_tasks:
            sections = [
                f"[{index}] {task_id}:\n{self._dependency_result(task_id)}"
                for index, task_id in enumerate(input_tasks, start=1)
            ]
            return "\n\n".join(sections)
        input_task_id = payload.get("input_task")
        return str(self._dependency_result(input_task_id)) if input_task_id else ""

    def _dependency_result(self, dependency_task_id: str) -> Any:
        """Return the stored result of an already completed dependency."""
        if self._result_lookup is None:
            raise RuntimeError("no result lookup is bound for dependency chaining.")
        dependency = self._result_lookup(dependency_task_id)
        if dependency is None or dependency.result is None:
            raise RuntimeError(
                f"dependency task {dependency_task_id} has no result yet."
            )
        return dependency.result

    def _from_result(self, result) -> TaskOutcome:
        if result.status == "success":
            return TaskOutcome.succeed(result.result)
        return TaskOutcome.fail(
            f"{result.error_code or result.status}: {result.error_message or ''}".strip()
        )

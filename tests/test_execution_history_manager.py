from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from core.execution_history import ExecutionHistory
from core.execution_history_manager import (
    ExecutionHistoryAutosaveBlockedError,
    ExecutionHistoryClosedError,
    ExecutionHistoryInitializationError,
    ExecutionHistoryManager,
    ExecutionHistoryManagerConfig,
    ExecutionHistorySaveError,
)
from core.execution_history_query import (
    ExecutionHistoryQuery,
    ExecutionHistoryQueryService,
)
from core.execution_history_repository import (
    ExecutionHistoryLoadError,
    ExecutionHistoryPersistenceError,
    ExecutionHistoryRepository,
)
from core.execution_metrics import ExecutionMetricsCalculator
from core.execution_plan_executor import (
    ExecutionControl,
    ExecutionPlanExecutor,
    PlanExecutionResult,
    ResumableExecutionState,
)
from core.execution_plan_validator import ExecutionPlanValidator, PlanValidationResult
from core.execution_trace import ExecutionTrace, TraceEventStatus, TraceStatus
from core.planner import ExecutionPlan, ExecutionStep
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class FakeHistoryRepository(ExecutionHistoryRepository):
    def __init__(
        self,
        loaded_history: ExecutionHistory | None = None,
        *,
        load_error: ExecutionHistoryLoadError | None = None,
        save_error: ExecutionHistoryPersistenceError | None = None,
    ) -> None:
        self.loaded_history = loaded_history
        self.load_error = load_error
        self.save_error = save_error
        self.load_calls = 0
        self.save_calls = 0
        self.clear_calls = 0
        self.saved_entry_ids: list[tuple[str, ...]] = []
        self.exists_result = False

    def save(self, history: ExecutionHistory) -> None:
        self.save_calls += 1
        self.saved_entry_ids.append(tuple(entry.execution_id for entry in history))
        if self.save_error is not None:
            raise self.save_error

    def load(self, max_entries: int | None = None) -> ExecutionHistory:
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        if self.loaded_history is None:
            return ExecutionHistory(max_entries=max_entries or 100)
        if max_entries is None:
            return self.loaded_history
        restored = ExecutionHistory(max_entries=max_entries)
        for entry in self.loaded_history:
            restored.add_entry(entry)
        return restored

    def exists(self) -> bool:
        return self.exists_result

    def clear(self) -> None:
        self.clear_calls += 1


class SpyTool(BaseTool):
    def __init__(
        self,
        name: str,
        calls: list[str],
        output: object = "ok",
        *,
        fail: bool = False,
    ) -> None:
        self._name = name
        self._calls = calls
        self._output = output
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake tool {self._name}."

    def execute(self, context: ToolContext) -> object:
        self._calls.append(context.step_id or self._name)
        if self._fail:
            raise RuntimeError("tool failed")
        return self._output


def _timestamp(seconds: int = 0) -> datetime:
    return datetime(2026, 7, 20, 10, 0, seconds, tzinfo=timezone.utc)


def _result(
    execution_id: str,
    *,
    status: str = TraceStatus.SUCCESS.value,
) -> PlanExecutionResult:
    trace = ExecutionTrace(execution_id=execution_id, started_at=_timestamp())
    trace.add_event(
        timestamp=_timestamp(),
        component="ExecutionPlanExecutor",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED.value,
    )
    trace.add_event(
        timestamp=_timestamp(1),
        component="ExecutionPlanExecutor",
        action=(
            "STEP_FINISHED"
            if status == TraceStatus.SUCCESS.value
            else "STEP_FAILED"
        ),
        status=(
            TraceEventStatus.FINISHED.value
            if status == TraceStatus.SUCCESS.value
            else TraceEventStatus.FAILED.value
        ),
        duration_ms=100,
    )
    trace.finish(status, finished_at=_timestamp() + timedelta(milliseconds=100))
    return PlanExecutionResult(
        plan_status="completed" if status == TraceStatus.SUCCESS.value else "failed",
        success=status == TraceStatus.SUCCESS.value,
        trace=trace,
        metrics=ExecutionMetricsCalculator().calculate(trace),
    )


def _history(*results: PlanExecutionResult, max_entries: int = 100) -> ExecutionHistory:
    history = ExecutionHistory(max_entries=max_entries)
    for result in results:
        history.add(result)
    return history


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Execute.",
        ordered_steps=(ExecutionStep(id="step_1", description="Run.", tool="safe_tool"),),
        estimated_steps=1,
        required_tools=("safe_tool",),
        detected_risks=(),
        requires_confirmation=False,
    )


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _validation(plan: ExecutionPlan) -> PlanValidationResult:
    return ExecutionPlanValidator().validate(plan)


def _state_from_result(
    plan: ExecutionPlan,
    validation: PlanValidationResult,
    result: PlanExecutionResult,
) -> ResumableExecutionState:
    return ResumableExecutionState(
        objective="resume",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=tuple(result.completed_steps),
        pending_step_ids=tuple(result.pending_steps),
        failed_step_ids=tuple(result.failed_steps),
        interrupted_step_id=result.current_step,
        previous_results={
            step.step_id: step.output
            for step in result.step_results
            if step.success
        },
        resumable=result.resumable,
    )


def test_config_defaults_are_immutable_and_validated() -> None:
    config = ExecutionHistoryManagerConfig()

    assert config.max_entries == 100
    assert config.autoload is True
    assert config.autosave is True
    assert config.fail_on_load_error is True
    assert config.fail_on_save_error is False
    assert config.save_only_when_dirty is True
    with pytest.raises(FrozenInstanceError):
        config.autosave = False  # type: ignore[misc]
    with pytest.raises(ValueError):
        ExecutionHistoryManagerConfig(max_entries=0)
    with pytest.raises(ValueError):
        ExecutionHistoryManagerConfig(max_entries=-1)
    with pytest.raises(ValueError):
        ExecutionHistoryManagerConfig(autoload="yes")  # type: ignore[arg-type]


def test_creation_without_repository_uses_memory_only_and_save_is_explicit_skip() -> None:
    manager = ExecutionHistoryManager(None)

    history = manager.initialize()
    result = manager.save()
    entry = manager.add(_result("exec-1"))

    assert history.max_entries == 100
    assert manager.persistence_available is False
    assert result.skipped is True
    assert result.reason == "no_repository"
    assert entry.execution_id == "exec-1"
    assert manager.history.count() == 1


def test_initialize_autoloads_once_and_uses_configured_capacity() -> None:
    repository = FakeHistoryRepository(
        _history(_result("exec-1"), _result("exec-2"), max_entries=10)
    )
    manager = ExecutionHistoryManager(
        repository,
        ExecutionHistoryManagerConfig(max_entries=1),
    )

    first = manager.initialize()
    second = manager.initialize()

    assert first is second
    assert repository.load_calls == 1
    assert first.max_entries == 1
    assert tuple(entry.execution_id for entry in first) == ("exec-2",)
    assert manager.load_succeeded is True
    assert manager.dirty is False


def test_initialize_with_autoload_false_or_provided_history_does_not_touch_repository() -> None:
    repository = FakeHistoryRepository(_history(_result("persisted")))
    provided = _history(_result("provided"))

    no_autoload = ExecutionHistoryManager(
        repository,
        ExecutionHistoryManagerConfig(autoload=False),
    )
    provided_manager = ExecutionHistoryManager(repository, history=provided)

    assert no_autoload.initialize().count() == 0
    assert provided_manager.initialize() is provided
    assert tuple(entry.execution_id for entry in provided_manager.history) == ("provided",)
    assert repository.load_calls == 0


def test_load_error_can_fail_or_continue_with_autosave_blocked() -> None:
    load_error = ExecutionHistoryLoadError("corrupt")
    strict = ExecutionHistoryManager(FakeHistoryRepository(load_error=load_error))
    lenient = ExecutionHistoryManager(
        FakeHistoryRepository(load_error=load_error),
        ExecutionHistoryManagerConfig(fail_on_load_error=False),
    )

    with pytest.raises(ExecutionHistoryInitializationError) as error:
        strict.initialize()

    history = lenient.initialize()
    lenient.add(_result("exec-1"))
    save_result = lenient.save()

    assert error.value.__cause__ is load_error
    assert history.count() == 1
    assert lenient.last_error is load_error
    assert lenient.load_succeeded is False
    assert lenient.autosave_blocked is True
    assert save_result.skipped is True
    assert save_result.reason == "autosave_blocked"


def test_explicit_acknowledge_unblocks_persistence_after_load_failure() -> None:
    repository = FakeHistoryRepository(load_error=ExecutionHistoryLoadError("bad"))
    manager = ExecutionHistoryManager(
        repository,
        ExecutionHistoryManagerConfig(fail_on_load_error=False),
    )

    manager.initialize()
    manager.add(_result("exec-1"))
    manager.acknowledge_load_error()
    save_result = manager.save()

    assert manager.autosave_blocked is False
    assert save_result.saved is True
    assert repository.save_calls == 1
    assert manager.dirty is False


def test_save_raises_autosave_blocked_error_when_policy_requires_failure() -> None:
    manager = ExecutionHistoryManager(
        FakeHistoryRepository(load_error=ExecutionHistoryLoadError("bad")),
        ExecutionHistoryManagerConfig(
            fail_on_load_error=False,
            fail_on_save_error=True,
        ),
    )

    manager.initialize()

    with pytest.raises(ExecutionHistoryAutosaveBlockedError):
        manager.save()


def test_add_marks_dirty_and_autosaves_once_when_enabled() -> None:
    repository = FakeHistoryRepository()
    manager = ExecutionHistoryManager(repository)

    entry = manager.add(_result("exec-1"))

    assert entry.execution_id == "exec-1"
    assert manager.dirty is False
    assert manager.successful_save_count == 1
    assert repository.save_calls == 1
    assert repository.saved_entry_ids == [("exec-1",)]


def test_add_with_autosave_disabled_does_not_touch_repository_until_explicit_save() -> None:
    repository = FakeHistoryRepository()
    manager = ExecutionHistoryManager(
        repository,
        ExecutionHistoryManagerConfig(autosave=False),
    )

    manager.add(_result("exec-1"))
    save_result = manager.save()

    assert repository.save_calls == 1
    assert save_result.saved is True
    assert manager.dirty is False


def test_save_without_changes_is_skipped_and_dirty_save_clears_dirty() -> None:
    repository = FakeHistoryRepository()
    manager = ExecutionHistoryManager(
        repository,
        ExecutionHistoryManagerConfig(autoload=False, autosave=False),
    )

    skipped = manager.save()
    manager.add(_result("exec-1"))
    saved = manager.save()

    assert skipped.skipped is True
    assert skipped.reason == "no_changes"
    assert saved.saved is True
    assert manager.dirty is False
    assert manager.successful_save_count == 1


def test_save_failure_keeps_entry_dirty_and_does_not_change_trace_status() -> None:
    save_error = ExecutionHistoryPersistenceError("disk full")
    repository = FakeHistoryRepository(save_error=save_error)
    manager = ExecutionHistoryManager(repository)
    result = _result("exec-1", status=TraceStatus.SUCCESS.value)

    entry = manager.add(result)

    assert entry.trace.status == TraceStatus.SUCCESS.value
    assert result.trace.status == TraceStatus.SUCCESS.value  # type: ignore[union-attr]
    assert manager.history.count() == 1
    assert manager.dirty is True
    assert manager.failed_save_count == 1
    assert manager.last_error is save_error
    assert manager.last_save_result is not None
    assert manager.last_save_result.error is save_error


def test_save_failure_can_be_propagated_after_entry_is_kept() -> None:
    save_error = ExecutionHistoryPersistenceError("denied")
    manager = ExecutionHistoryManager(
        FakeHistoryRepository(save_error=save_error),
        ExecutionHistoryManagerConfig(fail_on_save_error=True),
    )

    with pytest.raises(ExecutionHistorySaveError) as error:
        manager.add(_result("exec-1"))

    assert error.value.__cause__ is save_error
    assert manager.history.count() == 1
    assert manager.dirty is True
    assert manager.failed_save_count == 1


def test_clear_history_and_clear_persistence_are_separate_operations() -> None:
    repository = FakeHistoryRepository()
    manager = ExecutionHistoryManager(repository, ExecutionHistoryManagerConfig(autosave=False))
    manager.add(_result("exec-1"))

    result = manager.clear_history()
    manager.clear_persistence()

    assert result is None
    assert manager.history.count() == 0
    assert manager.dirty is True
    assert repository.clear_calls == 1


def test_clear_history_autosaves_when_enabled_but_clear_persistence_keeps_memory() -> None:
    repository = FakeHistoryRepository()
    manager = ExecutionHistoryManager(repository)
    manager.add(_result("exec-1"))

    manager.clear_history()
    manager.add(_result("exec-2"))
    manager.clear_persistence()

    assert repository.save_calls == 3
    assert repository.saved_entry_ids[-2] == ()
    assert manager.history.count() == 1
    assert repository.clear_calls == 1


def test_close_saves_dirty_once_and_is_idempotent() -> None:
    repository = FakeHistoryRepository()
    manager = ExecutionHistoryManager(
        repository,
        ExecutionHistoryManagerConfig(autosave=False),
    )
    manager.add(_result("exec-1"))

    first = manager.close()
    second = manager.close()

    assert first.saved is True
    assert second.skipped is True
    assert second.reason == "closed"
    assert manager.closed is True
    assert repository.save_calls == 1


def test_autosave_followed_by_close_does_not_save_twice() -> None:
    repository = FakeHistoryRepository()
    manager = ExecutionHistoryManager(repository)

    manager.add(_result("exec-1"))
    close_result = manager.close()

    assert close_result.skipped is True
    assert close_result.reason == "no_changes"
    assert repository.save_calls == 1


def test_closed_manager_rejects_add_and_clear_history_and_save_is_skipped() -> None:
    manager = ExecutionHistoryManager(None)
    manager.close()

    with pytest.raises(ExecutionHistoryClosedError):
        manager.add(_result("exec-1"))
    with pytest.raises(ExecutionHistoryClosedError):
        manager.clear_history()
    assert manager.save().reason == "closed"


def test_context_manager_initializes_closes_and_does_not_hide_external_exception() -> None:
    repository = FakeHistoryRepository()
    manager = ExecutionHistoryManager(repository)

    with pytest.raises(RuntimeError):
        with manager as active:
            assert active.initialized is True
            active.add(_result("exec-1"))
            raise RuntimeError("external")

    assert manager.closed is True


def test_history_manager_is_compatible_with_query_service_and_new_entries() -> None:
    manager = ExecutionHistoryManager(None)
    manager.add(_result("exec-1", status=TraceStatus.FAILED.value))
    manager.add(_result("exec-2"))

    result = ExecutionHistoryQueryService().query(
        manager.history,
        ExecutionHistoryQuery(statuses=(TraceStatus.SUCCESS,)),
    )

    assert tuple(entry.execution_id for entry in result.entries) == ("exec-2",)


def test_execution_plan_executor_can_record_to_manager_without_double_registration() -> None:
    calls: list[str] = []
    repository = FakeHistoryRepository()
    manager = ExecutionHistoryManager(repository)
    plan = _plan()

    result = ExecutionPlanExecutor(
        _registry(SpyTool("safe_tool", calls)),
        execution_history=manager,
    ).execute(plan, _validation(plan))

    assert result.success is True
    assert calls == ["step_1"]
    assert manager.history.count() == 1
    assert manager.history.latest().result is result  # type: ignore[union-attr]
    assert repository.save_calls == 1


def test_executor_without_manager_keeps_existing_behavior() -> None:
    calls: list[str] = []
    plan = _plan()

    result = ExecutionPlanExecutor(_registry(SpyTool("safe_tool", calls))).execute(
        plan,
        _validation(plan),
    )

    assert result.success is True
    assert calls == ["step_1"]


def test_success_failed_and_cancelled_executions_are_kept_in_memory_and_saved() -> None:
    repository = FakeHistoryRepository()
    manager = ExecutionHistoryManager(repository)
    plan = _plan()

    success = ExecutionPlanExecutor(
        _registry(SpyTool("safe_tool", [])),
        execution_history=manager,
    ).execute(plan, _validation(plan))
    failed = ExecutionPlanExecutor(
        _registry(SpyTool("safe_tool", [], fail=True)),
        execution_history=manager,
    ).execute(plan, _validation(plan))
    cancelled = ExecutionPlanExecutor(
        _registry(SpyTool("safe_tool", [])),
        execution_history=manager,
    ).execute(plan, _validation(plan), control=ExecutionControl(should_cancel=lambda: True))

    assert success.trace.status == TraceStatus.SUCCESS.value  # type: ignore[union-attr]
    assert failed.trace.status == TraceStatus.FAILED.value  # type: ignore[union-attr]
    assert cancelled.trace.status == TraceStatus.CANCELLED.value  # type: ignore[union-attr]
    assert tuple(entry.status for entry in manager.history) == (
        TraceStatus.SUCCESS.value,
        TraceStatus.FAILED.value,
        TraceStatus.CANCELLED.value,
    )
    assert repository.save_calls == 3


def test_resumed_execution_is_registered_once() -> None:
    plan = ExecutionPlan(
        goal="Resume.",
        ordered_steps=(
            ExecutionStep(id="step_1", description="First.", tool="first_tool"),
            ExecutionStep(
                id="step_2",
                description="Second.",
                tool="second_tool",
                dependencies=("step_1",),
            ),
        ),
        estimated_steps=2,
        required_tools=("first_tool", "second_tool"),
        detected_risks=(),
        requires_confirmation=False,
    )
    validation = _validation(plan)
    interrupted_calls: list[str] = []
    interrupted = ExecutionPlanExecutor(
        _registry(
            SpyTool("first_tool", interrupted_calls, {"value": "ok"}),
            SpyTool("second_tool", interrupted_calls),
        )
    ).execute(
        plan,
        validation,
        control=ExecutionControl(should_stop=lambda: len(interrupted_calls) >= 1),
    )
    manager = ExecutionHistoryManager(FakeHistoryRepository())

    resumed = ExecutionPlanExecutor(
        _registry(
            SpyTool("first_tool", []),
            SpyTool("second_tool", []),
        ),
        execution_history=manager,
    ).resume(_state_from_result(plan, validation, interrupted))

    assert resumed.success is True
    assert manager.history.count() == 1
    assert manager.history.latest().result is resumed  # type: ignore[union-attr]


def test_loaded_history_can_receive_new_entry_and_later_save_preserves_both() -> None:
    repository = FakeHistoryRepository(_history(_result("loaded")))
    manager = ExecutionHistoryManager(
        repository,
        ExecutionHistoryManagerConfig(autosave=False),
    )

    manager.initialize()
    manager.add(_result("new"))
    manager.save()

    assert repository.saved_entry_ids == [("loaded", "new")]
    assert manager.dirty is False


def test_manager_source_has_no_background_or_logging_side_effects() -> None:
    source = Path("core/execution_history_manager.py").read_text(encoding="utf-8")

    assert "threading" not in source
    assert "asyncio" not in source
    assert "atexit" not in source
    assert "__del__" not in source
    assert "logging" not in source
    assert "print(" not in source

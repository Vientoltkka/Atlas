"""Lifecycle coordination for execution history persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.execution_history import ExecutionHistory, ExecutionHistoryEntry
from core.execution_history_repository import (
    DEFAULT_EXECUTION_HISTORY_MAX_ENTRIES,
    ExecutionHistoryLoadError,
    ExecutionHistoryPersistenceError,
    ExecutionHistoryRepository,
)

if TYPE_CHECKING:
    from core.execution_plan_executor import PlanExecutionResult


@dataclass(frozen=True, slots=True)
class ExecutionHistoryManagerConfig:
    """Immutable lifecycle policy for an execution history manager."""

    max_entries: int = DEFAULT_EXECUTION_HISTORY_MAX_ENTRIES
    autoload: bool = True
    autosave: bool = True
    fail_on_load_error: bool = True
    fail_on_save_error: bool = False
    save_only_when_dirty: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_entries, bool) or not isinstance(self.max_entries, int):
            raise ValueError("max_entries must be an integer.")
        if self.max_entries <= 0:
            raise ValueError("max_entries must be greater than zero.")
        for field_name in (
            "autoload",
            "autosave",
            "fail_on_load_error",
            "fail_on_save_error",
            "save_only_when_dirty",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean.")


@dataclass(frozen=True, slots=True)
class ExecutionHistorySaveResult:
    """Structured result for explicit or automatic save attempts."""

    saved: bool
    skipped: bool
    reason: str | None
    entry_count: int
    error: Exception | None = None


class ExecutionHistoryManagerError(Exception):
    """Base error for execution history manager failures."""


class ExecutionHistoryInitializationError(ExecutionHistoryManagerError):
    """Raised when initial history loading fails and policy requires failure."""


class ExecutionHistorySaveError(ExecutionHistoryManagerError):
    """Raised when saving fails and policy requires failure."""


class ExecutionHistoryClosedError(ExecutionHistoryManagerError):
    """Raised when a mutating operation is attempted after close."""


class ExecutionHistoryAutosaveBlockedError(ExecutionHistoryManagerError):
    """Raised when autosave is blocked after an unresolved load failure."""


class ExecutionHistoryManager:
    """Coordinate active history, persistence lifecycle and autosave policy."""

    def __init__(
        self,
        repository: ExecutionHistoryRepository | None,
        config: ExecutionHistoryManagerConfig | None = None,
        history: ExecutionHistory | None = None,
    ) -> None:
        self._repository = repository
        self._config = config or ExecutionHistoryManagerConfig()
        self._history = history
        self._history_was_provided = history is not None
        self._initialized = False
        self._closed = False
        self._dirty = False
        self._load_succeeded: bool | None = None
        self._last_error: Exception | None = None
        self._successful_save_count = 0
        self._failed_save_count = 0
        self._autosave_blocked = False
        self._last_save_result: ExecutionHistorySaveResult | None = None

    @property
    def history(self) -> ExecutionHistory:
        """Return the active in-memory history, initializing lazily if needed."""
        return self.initialize()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def autosave_enabled(self) -> bool:
        return self._config.autosave

    @property
    def persistence_available(self) -> bool:
        return self._repository is not None

    @property
    def load_succeeded(self) -> bool | None:
        return self._load_succeeded

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    @property
    def successful_save_count(self) -> int:
        return self._successful_save_count

    @property
    def failed_save_count(self) -> int:
        return self._failed_save_count

    @property
    def autosave_blocked(self) -> bool:
        return self._autosave_blocked

    @property
    def last_save_result(self) -> ExecutionHistorySaveResult | None:
        return self._last_save_result

    def initialize(self) -> ExecutionHistory:
        """Initialize the active history once according to the configured policy."""
        if self._initialized:
            return self._require_history()

        if self._history_was_provided:
            self._load_succeeded = None
            self._initialized = True
            return self._require_history()

        if not self._config.autoload or self._repository is None:
            self._history = ExecutionHistory(max_entries=self._config.max_entries)
            self._load_succeeded = None if self._repository is None else False
            self._initialized = True
            return self._history

        try:
            self._history = self._repository.load(max_entries=self._config.max_entries)
        except ExecutionHistoryLoadError as error:
            self._last_error = error
            self._load_succeeded = False
            self._autosave_blocked = True
            self._history = ExecutionHistory(max_entries=self._config.max_entries)
            self._initialized = True
            if self._config.fail_on_load_error:
                raise ExecutionHistoryInitializationError(
                    "Could not initialize execution history from repository."
                ) from error
            return self._history

        self._dirty = False
        self._load_succeeded = True
        self._last_error = None
        self._initialized = True
        return self._history

    start = initialize

    def add(
        self,
        result: PlanExecutionResult,
    ) -> ExecutionHistoryEntry:
        """Record an execution result and autosave according to policy."""
        if self._closed:
            raise ExecutionHistoryClosedError("Execution history manager is closed.")
        entry = self.initialize().add(result)
        self._dirty = True
        self._autosave_after_change()
        return entry

    record = add

    def save(self) -> ExecutionHistorySaveResult:
        """Explicitly save the active history and return a structured outcome."""
        if self._closed:
            return self._save_result(False, True, "closed")
        if self._repository is None:
            return self._save_result(False, True, "no_repository")
        if self._autosave_blocked:
            if self._config.fail_on_save_error:
                raise ExecutionHistoryAutosaveBlockedError(
                    "Execution history autosave is blocked by a previous load error."
                )
            return self._save_result(False, True, "autosave_blocked")
        if self._config.save_only_when_dirty and not self._dirty:
            return self._save_result(False, True, "no_changes")
        return self._save_current()

    def clear_history(self) -> ExecutionHistorySaveResult | None:
        """Clear only the active in-memory history and mark it dirty."""
        if self._closed:
            raise ExecutionHistoryClosedError("Execution history manager is closed.")
        self.initialize().clear()
        self._dirty = True
        return self._autosave_after_change()

    def clear_persistence(self) -> None:
        """Clear only the repository file without mutating the active history."""
        if self._repository is None:
            return
        try:
            self._repository.clear()
        except ExecutionHistoryPersistenceError as error:
            self._last_error = error
            raise ExecutionHistorySaveError(
                "Could not clear execution history persistence."
            ) from error

    def close(self) -> ExecutionHistorySaveResult:
        """Attempt one final save when dirty, then mark the manager closed."""
        if self._closed:
            return self._save_result(False, True, "closed")
        result = self._save_result(False, True, "no_changes")
        try:
            if self._dirty and self._repository is not None:
                result = self.save()
            return result
        finally:
            self._closed = True

    def acknowledge_load_error(self) -> None:
        """Explicitly allow future saves after a failed load preserved evidence."""
        self._autosave_blocked = False

    allow_overwrite_after_load_failure = acknowledge_load_error
    reset_persistence_error = acknowledge_load_error

    def __enter__(self) -> "ExecutionHistoryManager":
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        try:
            self.close()
        except ExecutionHistoryManagerError:
            if exc_type is None:
                raise
        return False

    def _autosave_after_change(self) -> ExecutionHistorySaveResult | None:
        if not self._config.autosave:
            return None
        if self._repository is None:
            return None
        if self._autosave_blocked:
            self._last_save_result = self._save_result(
                False,
                True,
                "autosave_blocked",
            )
            return self._last_save_result
        return self._save_current()

    def _save_current(self) -> ExecutionHistorySaveResult:
        try:
            self._repository.save(self.initialize())
        except ExecutionHistoryPersistenceError as error:
            self._failed_save_count += 1
            self._last_error = error
            result = self._save_result(False, False, "save_failed", error)
            if self._config.fail_on_save_error:
                raise ExecutionHistorySaveError(
                    "Could not save execution history."
                ) from error
            return result

        self._dirty = False
        self._successful_save_count += 1
        self._last_error = None
        return self._save_result(True, False, None)

    def _save_result(
        self,
        saved: bool,
        skipped: bool,
        reason: str | None,
        error: Exception | None = None,
    ) -> ExecutionHistorySaveResult:
        result = ExecutionHistorySaveResult(
            saved=saved,
            skipped=skipped,
            reason=reason,
            entry_count=self._require_history().count()
            if self._history is not None
            else 0,
            error=error,
        )
        self._last_save_result = result
        return result

    def _require_history(self) -> ExecutionHistory:
        if self._history is None:
            self._history = ExecutionHistory(max_entries=self._config.max_entries)
        return self._history

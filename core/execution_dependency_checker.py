"""Explicit execution dependency checks for Atlas plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from core.execution_context import ExecutionContext, ExecutionStepState
from core.planner import ExecutionStep


class ExecutionDependencyError(ValueError):
    """Base error for execution dependency failures."""


class InvalidExecutionDependencyError(ExecutionDependencyError):
    """Raised when a dependency declaration is structurally invalid."""


class ExecutionDependencyNotFoundError(ExecutionDependencyError):
    """Raised when a declared dependency is absent from the plan."""


class ExecutionDependencyOrderError(ExecutionDependencyError):
    """Raised when a step depends on a future step in sequential execution."""


class ExecutionDependencyCycleError(ExecutionDependencyError):
    """Raised when explicit dependencies contain a cycle."""


class ExecutionDependencyNotSatisfiedError(ExecutionDependencyError):
    """Raised when a dependency is not successful at runtime."""


class ImplicitStepDependencyError(ExecutionDependencyError):
    """Raised when a step output reference is not declared as a dependency."""


class TooManyStepDependenciesError(ExecutionDependencyError):
    """Raised when a step exceeds the supported dependency fan-in."""


@dataclass(frozen=True, slots=True)
class ExecutionDependencyCheckResult:
    """Structured result of checking a step's declared dependencies."""

    satisfied: bool
    dependency_ids: tuple[str, ...] = field(default_factory=tuple)
    blocking_dependency_ids: tuple[str, ...] = field(default_factory=tuple)
    blocking_states: Mapping[str, str] = field(default_factory=dict)
    checked_count: int = 0
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependency_ids", tuple(self.dependency_ids))
        object.__setattr__(
            self,
            "blocking_dependency_ids",
            tuple(self.blocking_dependency_ids),
        )
        object.__setattr__(self, "blocking_states", dict(self.blocking_states))


class ExecutionDependencyChecker:
    """Check one step's explicit dependencies against a live context."""

    _SATISFIED_STATE = ExecutionStepState.SUCCESS.value

    def check(
        self,
        step: ExecutionStep,
        context: ExecutionContext,
    ) -> ExecutionDependencyCheckResult:
        """Return whether all dependencies for ``step`` are successful."""
        dependency_ids = tuple(step.depends_on)
        blocking_states: dict[str, str] = {}

        for dependency_id in dependency_ids:
            state = context.state_for_step(dependency_id)
            if state != self._SATISFIED_STATE:
                blocking_states[dependency_id] = state

        blocking_dependency_ids = tuple(
            dependency_id
            for dependency_id in dependency_ids
            if dependency_id in blocking_states
        )
        return ExecutionDependencyCheckResult(
            satisfied=not blocking_dependency_ids,
            dependency_ids=dependency_ids,
            blocking_dependency_ids=blocking_dependency_ids,
            blocking_states=blocking_states,
            checked_count=len(dependency_ids),
            error_code=(
                None
                if not blocking_dependency_ids
                else "EXECUTION_DEPENDENCY_NOT_SATISFIED"
            ),
        )

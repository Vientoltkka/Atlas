"""Controlled execution for validated Atlas execution plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.execution_plan_validator import PlanValidationResult, plan_signature
from core.planner import ExecutionPlan, ExecutionStep
from tools.executor import ToolExecutor
from tools.registry import ToolNotRegisteredError, ToolRegistry
from tools.tool_context import ToolContext


@dataclass(frozen=True, slots=True)
class StepExecutionResult:
    """Structured execution outcome for one plan step."""

    step_id: str
    status: str
    success: bool
    tool_name: str | None
    output: object | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PlanExecutionResult:
    """Structured execution outcome for an execution plan."""

    plan_status: str
    success: bool
    completed_steps: list[str] = field(default_factory=list)
    failed_step: str | None = None
    skipped_steps: list[str] = field(default_factory=list)
    step_results: list[StepExecutionResult] = field(default_factory=list)
    error: str | None = None
    requires_confirmation: bool = False
    interrupted: bool = False


class ExecutionPlanExecutor:
    """Execute validated plans without planning or changing their structure."""

    _EXECUTABLE_PLAN_STATUSES = {"planned"}
    _COMPLETED_STEP_STATUS = "completed"
    _LOGICAL_TOOLS = {None, "direct_response"}

    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor or ToolExecutor(tool_registry)

    def execute(
        self,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult | None,
        *,
        confirmation_granted: bool = False,
    ) -> PlanExecutionResult:
        """Execute a previously validated plan in dependency order."""
        precondition_error = self._precondition_error(plan, validation_result)
        if precondition_error is not None:
            return PlanExecutionResult(
                plan_status="rejected",
                success=False,
                error=precondition_error,
                requires_confirmation=(
                    bool(validation_result.requires_confirmation)
                    if validation_result is not None
                    else plan.requires_confirmation
                ),
                interrupted=True,
            )

        assert validation_result is not None

        if validation_result.requires_confirmation and not confirmation_granted:
            return PlanExecutionResult(
                plan_status="blocked_confirmation",
                success=False,
                error="Plan execution requires explicit confirmation.",
                requires_confirmation=True,
                interrupted=True,
            )

        completed_steps = [
            step.id
            for step in plan.ordered_steps
            if step.status == self._COMPLETED_STEP_STATUS
        ]
        completed: set[str] = set(completed_steps)
        step_results: list[StepExecutionResult] = []

        for index, step in enumerate(plan.ordered_steps):
            if step.id in completed:
                continue

            missing_dependency = self._missing_dependency(step, completed)
            if missing_dependency is not None:
                skipped = self._remaining_step_ids(plan.ordered_steps, index)
                return PlanExecutionResult(
                    plan_status="failed",
                    success=False,
                    completed_steps=completed_steps,
                    failed_step=step.id,
                    skipped_steps=skipped[1:],
                    step_results=step_results
                    + [
                        StepExecutionResult(
                            step_id=step.id,
                            status="failed",
                            success=False,
                            tool_name=step.tool,
                            error=(
                                f"Dependency '{missing_dependency}' is not completed "
                                f"for step '{step.id}'."
                            ),
                        )
                    ],
                    error=(
                        f"Dependency '{missing_dependency}' is not completed "
                        f"for step '{step.id}'."
                    ),
                    requires_confirmation=validation_result.requires_confirmation,
                    interrupted=True,
                )

            outcome = self._execute_step(step)
            step_results.append(outcome)

            if outcome.success:
                completed.add(step.id)
                completed_steps.append(step.id)
                continue

            return PlanExecutionResult(
                plan_status="failed",
                success=False,
                completed_steps=completed_steps,
                failed_step=step.id,
                skipped_steps=self._remaining_step_ids(plan.ordered_steps, index + 1),
                step_results=step_results,
                error=outcome.error,
                requires_confirmation=validation_result.requires_confirmation,
                interrupted=True,
            )

        return PlanExecutionResult(
            plan_status="completed",
            success=True,
            completed_steps=completed_steps,
            failed_step=None,
            skipped_steps=[],
            step_results=step_results,
            error=None,
            requires_confirmation=validation_result.requires_confirmation,
            interrupted=False,
        )

    def _precondition_error(
        self,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult | None,
    ) -> str | None:
        if validation_result is None:
            return "Plan execution requires an explicit PlanValidationResult."

        if not validation_result.is_valid:
            return "Cannot execute an invalid execution plan."

        if (
            validation_result.plan_signature is not None
            and validation_result.plan_signature != plan_signature(plan)
        ):
            return "PlanValidationResult does not match the execution plan."

        if plan.status not in self._EXECUTABLE_PLAN_STATUSES:
            return f"Plan status '{plan.status}' is not executable."

        return None

    def _execute_step(
        self,
        step: ExecutionStep,
    ) -> StepExecutionResult:
        if step.tool in self._LOGICAL_TOOLS:
            return StepExecutionResult(
                step_id=step.id,
                status="completed",
                success=True,
                tool_name=step.tool,
                output=None,
                error=None,
            )

        assert step.tool is not None

        if not self._tool_registry.exists(step.tool):
            return StepExecutionResult(
                step_id=step.id,
                status="failed",
                success=False,
                tool_name=step.tool,
                output=None,
                error=f"Tool '{step.tool}' is not registered.",
            )

        try:
            output = self._tool_executor.execute(
                step.tool,
                ToolContext(),
            )
        except ToolNotRegisteredError as error:
            return StepExecutionResult(
                step_id=step.id,
                status="failed",
                success=False,
                tool_name=step.tool,
                output=None,
                error=str(error),
            )
        except Exception as error:
            return StepExecutionResult(
                step_id=step.id,
                status="failed",
                success=False,
                tool_name=step.tool,
                output=None,
                error=str(error),
            )

        failed_error = self._failed_output_error(output)
        if failed_error is not None:
            return StepExecutionResult(
                step_id=step.id,
                status="failed",
                success=False,
                tool_name=step.tool,
                output=output,
                error=failed_error,
            )

        return StepExecutionResult(
            step_id=step.id,
            status="completed",
            success=True,
            tool_name=step.tool,
            output=output,
            error=None,
        )

    def _missing_dependency(
        self,
        step: ExecutionStep,
        completed: set[str],
    ) -> str | None:
        for dependency in step.dependencies:
            if dependency not in completed:
                return dependency

        return None

    def _remaining_step_ids(
        self,
        steps: tuple[ExecutionStep, ...],
        start_index: int,
    ) -> list[str]:
        return [
            step.id
            for step in steps[start_index:]
            if step.status != self._COMPLETED_STEP_STATUS
        ]

    def _failed_output_error(
        self,
        output: Any,
    ) -> str | None:
        success = getattr(output, "success", None)
        if success is False:
            return str(
                getattr(output, "error_message", None)
                or getattr(output, "error", None)
                or "Tool returned an unsuccessful result."
            )

        if isinstance(output, dict) and output.get("success") is False:
            return str(
                output.get("error")
                or output.get("error_message")
                or "Tool returned an unsuccessful result."
            )

        return None

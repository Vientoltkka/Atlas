"""Bridge: validated ExecutionPlan → AsyncTaskScheduler task specs.

One conceptual frontier between the planner and the background scheduler:
a plan that already passed ``ExecutionPlanValidator`` is converted into
scheduler-ready task specs. Nothing else changes: permissions stay with the
SingleToolRunner, retry with the scheduler, verification with the
DeterministicTaskVerifier and progress with the BackgroundGoalPump.

Conversion rules (conservative by design):

- ``tool`` steps become ``kind: "tool"`` payloads executed by ToolTaskExecutor.
- A reasoning step is only convertible when the plan states it unambiguously:
  ``tool == "direct_response"`` with exactly one ``instruction`` argument.
  Those become ``kind: "transform"`` payloads (model/worker delegation).
- Steps referencing a dependency output (``StepOutputReference``) only chain
  through the existing mechanism: a single top-level ``content`` argument
  becomes ``content_task``. Anything else is rejected, never guessed.
- Subplans, branches, loops and logical steps without a tool are rejected
  with a structured error (no silent flattening).
- ``retry_policy.max_attempts`` maps to ``max_retries = max_attempts - 1``
  (attempts semantics: 3 attempts = 2 retries), respecting the scheduler's
  own bounded-retry limits.
- ``parallel_safe`` / ``priority`` / ``urgency`` / ``criticality`` /
  ``deadline`` are preserved as inert payload metadata; no scheduler field
  is extended and no concurrency is activated.
- ``requires_approval`` is never set from plan data: the planner does not
  decide authority, the tool confirmation policy does.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.execution_arguments import (
    contains_execution_variable_reference,
    contains_step_output_reference,
)
from core.execution_plan_validator import ExecutionPlanValidator
from core.planner import ExecutionPlan, ExecutionStep
from core.step_output_reference import StepOutputReference


TRANSFORM_TOOL = "direct_response"

_ERROR_TAIL = "the plan cannot be converted into background tasks."


class ExecutionPlanTaskBridgeError(Exception):
    """Structured conversion failure; no partial task state is created."""

    def __init__(
        self,
        code: str,
        message: str,
        errors: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.errors = tuple(errors) or (message,)


def execution_plan_to_task_specs(
    plan: ExecutionPlan,
    *,
    validator: ExecutionPlanValidator | None = None,
) -> tuple[dict[str, Any], ...]:
    """Convert a validated plan into scheduler-ready task specs.

    ``VALIDATE → CONVERT`` happens before any ``submit_goal``: an invalid or
    ambiguous plan raises :class:`ExecutionPlanTaskBridgeError` and leaves
    zero persisted tasks behind.
    """
    if not isinstance(plan, ExecutionPlan):
        raise ExecutionPlanTaskBridgeError(
            "INVALID_PLAN",
            "the bridge only accepts ExecutionPlan instances.",
        )
    if validator is not None:
        validation = validator.validate(plan)
        if not validation.is_valid:
            raise ExecutionPlanTaskBridgeError(
                "INVALID_PLAN",
                "the execution plan failed validation; " + _ERROR_TAIL,
                tuple(validation.errors),
            )

    known_ids = {step.id for step in plan.ordered_steps}
    specs: list[dict[str, Any]] = []
    for step in plan.ordered_steps:
        _require_known_dependencies(step, known_ids)
        specs.append(_step_spec(step))
    return tuple(specs)


def _require_known_dependencies(
    step: ExecutionStep,
    known_ids: set[str],
) -> None:
    for dependency in step.dependencies:
        if dependency not in known_ids:
            raise ExecutionPlanTaskBridgeError(
                "UNKNOWN_DEPENDENCY",
                f"step '{step.id}' depends on unknown step '{dependency}'; "
                + _ERROR_TAIL,
            )


def _step_spec(step: ExecutionStep) -> dict[str, Any]:
    if (
        step.subplan is not None
        or getattr(step, "subplan_ref", None) is not None
        or getattr(step, "branch", None) is not None
        or getattr(step, "loop", None) is not None
    ):
        raise ExecutionPlanTaskBridgeError(
            "UNSUPPORTED_STEP_KIND",
            f"step '{step.id}' declares subplan/branch/loop; "
            + _ERROR_TAIL,
        )
    if step.tool is None:
        raise ExecutionPlanTaskBridgeError(
            "AMBIGUOUS_LOGICAL_STEP",
            f"step '{step.id}' is a logical step without an executable tool; "
            + _ERROR_TAIL,
        )
    if step.tool == TRANSFORM_TOOL:
        return _transform_step_spec(step)
    return _tool_step_spec(step)


def _base_spec(step: ExecutionStep, payload: dict[str, Any]) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "task_id": step.id,
        "description": step.description,
        "dependencies": list(step.dependencies),
        "requires_approval": False,
        "payload": payload,
    }
    policy = step.retry_policy
    if policy is not None:
        # attempts semantics: 3 attempts == 2 retries.
        spec["max_retries"] = max(0, int(policy.max_attempts) - 1)
    metadata = _inert_metadata(step)
    if metadata:
        payload["plan_metadata"] = metadata
    return spec


def _inert_metadata(step: ExecutionStep) -> dict[str, Any]:
    """Preserve plan metadata without extending the scheduler's Task model."""
    metadata: dict[str, Any] = {}
    if step.parallel_safe:
        metadata["parallel_safe"] = True
    for name in ("priority", "urgency", "criticality"):
        value = getattr(step, name, 0)
        if value:
            metadata[name] = value
    if step.deadline is not None:
        metadata["deadline"] = step.deadline.isoformat()
    return metadata


def _transform_step_spec(step: ExecutionStep) -> dict[str, Any]:
    arguments = step.arguments.as_dict()
    extra_keys = sorted(key for key in arguments if key != "instruction")
    instruction = arguments.get("instruction")
    if extra_keys or not isinstance(instruction, str) or not instruction.strip():
        raise ExecutionPlanTaskBridgeError(
            "AMBIGUOUS_REASONING_STEP",
            f"step '{step.id}' uses '{TRANSFORM_TOOL}' without an unambiguous "
            "instruction argument; " + _ERROR_TAIL,
        )
    payload: dict[str, Any] = {
        "kind": "transform",
        "instruction": instruction,
    }
    dependencies = tuple(step.dependencies)
    if len(dependencies) == 1:
        payload["input_task"] = dependencies[0]
    elif len(dependencies) > 1:
        payload["input_tasks"] = list(dependencies)
    return _base_spec(step, payload)


def _tool_step_spec(step: ExecutionStep) -> dict[str, Any]:
    arguments = step.arguments.as_dict()
    content_task: str | None = None
    for key, value in arguments.items():
        if isinstance(value, StepOutputReference):
            if value.path:
                raise _reference_error(step, key)
            if key != "content" or content_task is not None:
                raise _reference_error(step, key)
            if value.step_id not in step.dependencies:
                raise _reference_error(step, key)
            content_task = value.step_id
        elif (
            contains_step_output_reference(value)
            or contains_execution_variable_reference(value)
            or _contains_template_reference(value)
        ):
            raise _reference_error(step, key)

    if content_task is not None:
        arguments.pop("content", None)

    payload: dict[str, Any] = {
        "kind": "tool",
        "tool": step.tool,
        "arguments": arguments,
    }
    if content_task is not None:
        payload["content_task"] = content_task
    return _base_spec(step, payload)


def _reference_error(
    step: ExecutionStep,
    key: str,
) -> ExecutionPlanTaskBridgeError:
    return ExecutionPlanTaskBridgeError(
        "UNSUPPORTED_RESULT_REFERENCE",
        f"step '{step.id}' uses an unresolved output reference in argument "
        f"'{key}' that the scheduler cannot chain safely; " + _ERROR_TAIL,
    )


def _contains_template_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "$ref" in value or "$template" in value:
            return True
        return any(_contains_template_reference(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_template_reference(item) for item in value)
    return False

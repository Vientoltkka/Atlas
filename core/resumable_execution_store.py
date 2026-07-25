"""Local persistence for resumable structured executions."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Protocol

from core.execution_context import ExecutionContextSnapshot
from core.goal_verifier import (
    goal_verification_result_from_dict,
    goal_verification_result_to_dict,
)
from core.execution_condition import (
    AllOfCondition,
    AnyOfCondition,
    ExecutionCondition,
    ExecutionConditionNode,
    ExecutionConditionOperator,
    NotCondition,
)
from core.execution_variable_binding import ExecutionVariableBinding
from core.execution_variable_reference import ExecutionVariableReference
from core.execution_plan_output import ExecutionPlanOutput
from core.execution_plan_registry import ExecutionPlanReference, ExecutionPlanRegistryError
from core.execution_plan_executor import ResumableExecutionState
from core.execution_plan_topology import (
    ExecutionPlanTopologicalSorter,
    ExecutionPlanTopologyError,
)
from core.execution_plan_validator import PlanValidationResult, plan_signature
from core.planner import ExecutionBranch, ExecutionLoop, ExecutionPlan, ExecutionStep
from core.step_output_reference import StepOutputReference


SCHEMA_VERSION = 1


class ResumableExecutionStoreError(Exception):
    """Stable structured error raised by resumable execution stores."""

    def __init__(
        self,
        error_code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class ResumableExecutionStore(Protocol):
    """Persistence contract for one resumable structured execution."""

    def save(self, state: ResumableExecutionState) -> None:
        """Persist a resumable execution state."""

    def load(self) -> ResumableExecutionState | None:
        """Load a resumable execution state if present."""

    def delete(self) -> None:
        """Delete a persisted resumable execution state."""

    def exists(self) -> bool:
        """Return whether a persisted state exists."""


class JsonResumableExecutionStore:
    """JSON-file implementation for one local resumable execution state."""

    _SUPPORTED_SCHEMA_VERSIONS = {SCHEMA_VERSION}
    _MAX_FILE_BYTES = 1_000_000
    _TOP_LEVEL_KEYS = {
        "schema_version",
        "created_at",
        "updated_at",
        "objective",
        "original_plan",
        "validation_result",
        "validated_plan_signature",
        "completed_step_ids",
        "pending_step_ids",
        "failed_step_ids",
        "interrupted_step_id",
        "previous_results",
        "resumable",
        "interruption_reason",
        "confirmation_granted",
        "retry_attempts",
        "retry_history",
        "metadata",
        "execution_context_snapshot",
        "goal_verification_result",
    }

    def __init__(
        self,
        path: Path,
        *,
        max_file_bytes: int = _MAX_FILE_BYTES,
    ) -> None:
        self._path = path
        self._max_file_bytes = max_file_bytes

    @property
    def path(self) -> Path:
        """Return the configured state path."""
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def save(
        self,
        state: ResumableExecutionState,
    ) -> None:
        try:
            payload = self._state_to_payload(state)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_SERIALIZATION_FAILED",
                "Execution state is not JSON serializable.",
            ) from error

        if len(encoded.encode("utf-8")) > self._max_file_bytes:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_SERIALIZATION_FAILED",
                "Execution state is larger than the allowed file size.",
            )

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._path.with_name(f"{self._path.name}.tmp")
            with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._path)
        except OSError as error:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_SAVE_FAILED",
                "Could not save resumable execution state.",
            ) from error

    def load(self) -> ResumableExecutionState | None:
        if not self._path.exists():
            return None

        try:
            size = self._path.stat().st_size
        except OSError as error:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_LOAD_FAILED",
                "Could not inspect resumable execution state.",
            ) from error

        if size > self._max_file_bytes:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "Execution state file exceeds the maximum allowed size.",
            )

        try:
            raw = self._path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_CORRUPTED",
                "Execution state file is not valid JSON.",
            ) from error
        except OSError as error:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_LOAD_FAILED",
                "Could not load resumable execution state.",
            ) from error

        return self._payload_to_state(payload)

    def delete(self) -> None:
        if not self._path.exists():
            return

        try:
            self._path.unlink()
        except OSError as error:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_DELETE_FAILED",
                "Could not delete resumable execution state.",
            ) from error

    def _state_to_payload(
        self,
        state: ResumableExecutionState,
    ) -> dict[str, Any]:
        now = _utc_now()
        created_at = str(state.metadata.get("created_at") or now)
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "updated_at": now,
            "objective": state.objective,
            "original_plan": _plan_to_dict(state.original_plan),
            "validation_result": _validation_to_dict(state.validation_result),
            "validated_plan_signature": state.validated_plan_signature,
            "completed_step_ids": list(state.completed_step_ids),
            "pending_step_ids": list(state.pending_step_ids),
            "failed_step_ids": list(state.failed_step_ids),
            "interrupted_step_id": state.interrupted_step_id,
            "previous_results": state.previous_results,
            "resumable": state.resumable,
            "interruption_reason": state.interruption_reason,
            "confirmation_granted": state.confirmation_granted,
            "retry_attempts": state.retry_attempts,
            "retry_history": {
                step_id: list(history)
                for step_id, history in state.retry_history.items()
            },
            "metadata": _safe_metadata(state.metadata, created_at=created_at),
            "execution_context_snapshot": (
                _context_snapshot_to_dict(state.execution_context_snapshot)
                if state.execution_context_snapshot is not None
                else None
            ),
            "goal_verification_result": goal_verification_result_to_dict(
                state.goal_verification_result
            ),
        }

    def _payload_to_state(
        self,
        payload: Any,
    ) -> ResumableExecutionState:
        if not isinstance(payload, dict):
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "Execution state root must be an object.",
            )

        unknown = set(payload) - self._TOP_LEVEL_KEYS
        if unknown:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "Execution state contains unsupported fields.",
            )

        schema_version = payload.get("schema_version")
        if schema_version not in self._SUPPORTED_SCHEMA_VERSIONS:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_SCHEMA_UNSUPPORTED",
                "Execution state schema version is not supported.",
            )

        objective = _required_str(payload, "objective")
        plan = _dict_to_plan(_required_dict(payload, "original_plan"))
        validation = _dict_to_validation(_required_dict(payload, "validation_result"))
        validated_signature = _optional_str(payload, "validated_plan_signature")
        completed = _str_tuple(payload, "completed_step_ids")
        pending = _str_tuple(payload, "pending_step_ids")
        failed = _str_tuple(payload, "failed_step_ids")
        interrupted = _optional_str(payload, "interrupted_step_id")
        previous_results = _required_dict(payload, "previous_results")
        resumable = _required_bool(payload, "resumable")
        confirmation_granted = _required_bool(payload, "confirmation_granted")
        retry_attempts = _int_mapping(payload, "retry_attempts", default={})
        retry_history = _retry_history(payload, "retry_history", default={})
        metadata = _required_dict(payload, "metadata")
        context_snapshot = _optional_context_snapshot(
            payload.get("execution_context_snapshot"),
        )
        try:
            goal_verification_result = goal_verification_result_from_dict(
                payload.get("goal_verification_result"),
            )
        except ValueError as error:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "Goal verification result is invalid.",
            ) from error

        if not resumable:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_NOT_RESUMABLE",
                "Execution state is not resumable.",
            )

        recalculated_signature = plan_signature(plan)
        if validated_signature != recalculated_signature:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_SIGNATURE_MISMATCH",
                "Execution state signature does not match the plan.",
            )

        if validation.plan_signature != validated_signature:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_SIGNATURE_MISMATCH",
                "Execution validation signature does not match the plan.",
            )

        _validate_state_consistency(
            plan=plan,
            completed_step_ids=completed,
            pending_step_ids=pending,
            failed_step_ids=failed,
            interrupted_step_id=interrupted,
            previous_results=previous_results,
            execution_context_snapshot=context_snapshot,
        )

        return ResumableExecutionState(
            objective=objective,
            original_plan=plan,
            validation_result=validation,
            validated_plan_signature=validated_signature,
            completed_step_ids=completed,
            pending_step_ids=pending,
            failed_step_ids=failed,
            interrupted_step_id=interrupted,
            previous_results=previous_results,
            resumable=resumable,
            interruption_reason=_optional_str(payload, "interruption_reason"),
            confirmation_granted=confirmation_granted,
            retry_attempts=retry_attempts,
            retry_history=retry_history,
            metadata=metadata,
            execution_context_snapshot=context_snapshot,
            goal_verification_result=goal_verification_result,
        )


def _plan_to_dict(
    plan: ExecutionPlan,
) -> dict[str, Any]:
    return {
        "goal": plan.goal,
        "ordered_steps": [_step_to_dict(step) for step in plan.ordered_steps],
        "estimated_steps": plan.estimated_steps,
        "required_tools": list(plan.required_tools),
        "detected_risks": list(plan.detected_risks),
        "requires_confirmation": plan.requires_confirmation,
        "status": plan.status,
        "output": _output_to_json(plan.output),
        "required_outputs": list(plan.required_outputs),
        "output_validators": _output_validators_to_json(plan.output_validators),
    }


def _step_to_dict(
    step: ExecutionStep,
) -> dict[str, Any]:
    return {
        "id": step.id,
        "description": step.description,
        "tool": step.tool,
        "subplan": _plan_to_dict(step.subplan) if step.subplan is not None else None,
        "subplan_ref": _plan_reference_to_json(step.subplan_ref),
        "branch": _branch_to_json(step.branch),
        "loop": _loop_to_json(step.loop),
        "depends_on": list(step.depends_on),
        "status": step.status,
        "arguments": _argument_to_json(step.arguments.as_dict()),
        "output_binding": _binding_to_json(step.output_binding),
        "condition": _condition_to_json(step.condition),
    }


def _dict_to_plan(
    payload: dict[str, Any],
) -> ExecutionPlan:
    steps_payload = payload.get("ordered_steps")
    if not isinstance(steps_payload, list):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution plan steps must be a list.",
        )

    steps = tuple(_dict_to_step(step) for step in steps_payload)
    return ExecutionPlan(
        goal=_required_str(payload, "goal"),
        ordered_steps=steps,
        estimated_steps=_required_int(payload, "estimated_steps"),
        required_tools=_str_tuple(payload, "required_tools"),
        detected_risks=_str_tuple(payload, "detected_risks"),
        requires_confirmation=_required_bool(payload, "requires_confirmation"),
        status=_required_str(payload, "status"),
        output=_output_from_json(payload.get("output")),
        required_outputs=_str_tuple(payload, "required_outputs") if "required_outputs" in payload else (),
        output_validators=(
            _output_validators_from_json(payload.get("output_validators"))
            if "output_validators" in payload
            else {}
        ),
    )


def _dict_to_step(
    payload: Any,
) -> ExecutionStep:
    if not isinstance(payload, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution step must be an object.",
        )

    tool = payload.get("tool")
    if tool is not None and not isinstance(tool, str):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution step tool must be a string or null.",
        )
    raw_subplan = payload.get("subplan")
    if raw_subplan is not None and not isinstance(raw_subplan, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution step subplan must be an execution plan object or null.",
        )
    raw_subplan_ref = payload.get("subplan_ref")
    raw_branch = payload.get("branch")
    if raw_branch is not None and not isinstance(raw_branch, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution step branch must be an execution branch object or null.",
        )
    raw_loop = payload.get("loop")
    if raw_loop is not None and not isinstance(raw_loop, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution step loop must be an execution loop object or null.",
        )

    return ExecutionStep(
        id=_required_str(payload, "id"),
        description=_required_str(payload, "description"),
        tool=tool,
        subplan=_dict_to_plan(raw_subplan) if raw_subplan is not None else None,
        subplan_ref=_plan_reference_from_json(raw_subplan_ref),
        branch=_branch_from_json(raw_branch),
        loop=_loop_from_json(raw_loop),
        depends_on=_step_dependencies(payload),
        status=_required_str(payload, "status"),
        arguments=_argument_from_json(_required_dict(payload, "arguments")),
        output_binding=_binding_from_json(payload.get("output_binding")),
        condition=_condition_from_json(payload.get("condition")),
    )


def _branch_to_json(
    branch: ExecutionBranch | None,
) -> dict[str, Any] | None:
    if branch is None:
        return None
    return {
        "$type": "execution_branch",
        "condition": _condition_to_json(branch.condition),
        "then_plan": _plan_to_dict(branch.then_plan),
        "else_plan": _plan_to_dict(branch.else_plan) if branch.else_plan is not None else None,
    }


def _branch_from_json(
    payload: dict[str, Any] | None,
) -> ExecutionBranch | None:
    if payload is None:
        return None
    if payload.get("$type") != "execution_branch":
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution branch type is invalid.",
        )
    raw_then = payload.get("then_plan")
    raw_else = payload.get("else_plan")
    if not isinstance(raw_then, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution branch then_plan must be an execution plan object.",
        )
    if raw_else is not None and not isinstance(raw_else, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution branch else_plan must be an execution plan object or null.",
        )
    condition = _condition_from_json(payload.get("condition"))
    if condition is None:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution branch condition is required.",
        )
    return ExecutionBranch(
        condition=condition,
        then_plan=_dict_to_plan(raw_then),
        else_plan=_dict_to_plan(raw_else) if raw_else is not None else None,
    )


def _loop_to_json(
    loop: ExecutionLoop | None,
) -> dict[str, Any] | None:
    if loop is None:
        return None
    return {
        "$type": "execution_loop",
        "condition": _condition_to_json(loop.condition),
        "body_plan": _plan_to_dict(loop.body_plan),
        "max_iterations": loop.max_iterations,
    }


def _loop_from_json(
    payload: dict[str, Any] | None,
) -> ExecutionLoop | None:
    if payload is None:
        return None
    if payload.get("$type") != "execution_loop":
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution loop type is invalid.",
        )
    raw_body = payload.get("body_plan")
    if not isinstance(raw_body, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution loop body_plan must be an execution plan object.",
        )
    condition = _condition_from_json(payload.get("condition"))
    if condition is None:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution loop condition is required.",
        )
    return ExecutionLoop(
        condition=condition,
        body_plan=_dict_to_plan(raw_body),
        max_iterations=_required_int(payload, "max_iterations"),
    )


def _step_dependencies(
    payload: dict[str, Any],
) -> tuple[str, ...]:
    if "depends_on" in payload:
        return _str_tuple(payload, "depends_on")
    if "dependencies" in payload:
        return _str_tuple(payload, "dependencies")
    return ()


def _condition_to_json(
    condition: ExecutionConditionNode | None,
) -> dict[str, Any] | None:
    if condition is None:
        return None
    if isinstance(condition, AllOfCondition):
        return {
            "$type": "all_of_condition",
            "conditions": [_condition_to_json(item) for item in condition.conditions],
        }
    if isinstance(condition, AnyOfCondition):
        return {
            "$type": "any_of_condition",
            "conditions": [_condition_to_json(item) for item in condition.conditions],
        }
    if isinstance(condition, NotCondition):
        return {
            "$type": "not_condition",
            "condition": _condition_to_json(condition.condition),
        }
    return {
        "$type": "execution_condition",
        "operator": condition.operator.value,
        "left": _argument_to_json(condition.left),
        "right": _argument_to_json(condition.right),
    }


def _condition_from_json(
    payload: Any,
) -> ExecutionConditionNode | None:
    if payload is None:
        return None
    if isinstance(payload, dict) and payload.get("$type") == "all_of_condition":
        if set(payload) != {"$type", "conditions"}:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "AllOf condition must be an explicit object.",
            )
        raw_conditions = payload.get("conditions")
        if not isinstance(raw_conditions, list):
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "AllOf condition children must be a list.",
            )
        try:
            return AllOfCondition(tuple(_condition_from_json(item) for item in raw_conditions))
        except (TypeError, ValueError) as error:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "AllOf condition is invalid.",
            ) from error
    if isinstance(payload, dict) and payload.get("$type") == "any_of_condition":
        if set(payload) != {"$type", "conditions"}:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "AnyOf condition must be an explicit object.",
            )
        raw_conditions = payload.get("conditions")
        if not isinstance(raw_conditions, list):
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "AnyOf condition children must be a list.",
            )
        try:
            return AnyOfCondition(tuple(_condition_from_json(item) for item in raw_conditions))
        except (TypeError, ValueError) as error:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "AnyOf condition is invalid.",
            ) from error
    if isinstance(payload, dict) and payload.get("$type") == "not_condition":
        if set(payload) != {"$type", "condition"}:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "Not condition must be an explicit object.",
            )
        try:
            return NotCondition(_condition_from_json(payload.get("condition")))
        except (TypeError, ValueError) as error:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "Not condition is invalid.",
            ) from error
    if not isinstance(payload, dict) or set(payload) != {
        "$type",
        "operator",
        "left",
        "right",
    }:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution condition must be an explicit object.",
        )
    if payload.get("$type") != "execution_condition":
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution condition type is invalid.",
        )
    try:
        operator = ExecutionConditionOperator(_required_str(payload, "operator"))
        left = _argument_from_json(payload.get("left"))
        if operator.value in {
            "is_none",
            "is_not_none",
            "exists",
            "not_exists",
            "truthy",
            "falsy",
            "is_empty",
            "is_not_empty",
        }:
            if payload.get("right") is not None:
                raise ValueError("unary condition right operand must be null")
            return ExecutionCondition(left=left, operator=operator)
        return ExecutionCondition(
            left=left,
            operator=operator,
            right=_argument_from_json(payload.get("right")),
        )
    except (TypeError, ValueError) as error:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution condition is invalid.",
        ) from error


def _binding_to_json(
    binding: ExecutionVariableBinding | None,
) -> dict[str, Any] | None:
    if binding is None:
        return None
    return {
        "$type": "execution_variable_binding",
        "variable_name": binding.variable_name,
        "path": [_argument_to_json(part) for part in binding.path],
        "overwrite": binding.overwrite,
    }


def _binding_from_json(
    payload: Any,
) -> ExecutionVariableBinding | None:
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "$type",
        "variable_name",
        "path",
        "overwrite",
    }:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution variable binding must be an explicit object.",
        )
    if payload.get("$type") != "execution_variable_binding":
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution variable binding type is invalid.",
        )
    raw_path = payload.get("path")
    if not isinstance(raw_path, list):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution variable binding path must be a list.",
        )
    overwrite = payload.get("overwrite")
    if type(overwrite) is not bool:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution variable binding overwrite must be a boolean.",
        )
    return ExecutionVariableBinding(
        variable_name=_required_str(payload, "variable_name"),
        path=tuple(raw_path),
        overwrite=overwrite,
    )


def _argument_to_json(
    value: Any,
) -> Any:
    if isinstance(value, StepOutputReference):
        return {
            "$type": "step_output_reference",
            "step_id": value.step_id,
            "path": [_argument_to_json(part) for part in value.path],
        }

    if isinstance(value, ExecutionVariableReference):
        return {
            "$type": "execution_variable_reference",
            "name": value.name,
            "path": [_argument_to_json(part) for part in value.path],
        }

    if isinstance(value, dict):
        return {
            key: _argument_to_json(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_argument_to_json(item) for item in value]

    if isinstance(value, tuple):
        return [_argument_to_json(item) for item in value]

    return value


def _output_to_json(
    output: object,
) -> dict[str, Any] | None:
    if output is None:
        return None
    if not isinstance(output, ExecutionPlanOutput):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_SERIALIZATION_FAILED",
            "Execution plan output must be normalized.",
        )
    return {
        "$type": "execution_plan_output",
        "value": _argument_to_json(output.as_definition()),
    }


def _output_from_json(
    payload: Any,
) -> ExecutionPlanOutput | None:
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {"$type", "value"}:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution plan output must be an explicit object or null.",
        )
    if payload.get("$type") != "execution_plan_output":
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution plan output type is invalid.",
        )
    return ExecutionPlanOutput(_argument_from_json(payload.get("value")))


def _output_validators_to_json(
    validators: Any,
) -> dict[str, list[str]]:
    return {
        str(name): list(kinds)
        for name, kinds in dict(validators).items()
    }


def _output_validators_from_json(
    payload: Any,
) -> dict[str, tuple[str, ...]]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution plan output_validators must be an object.",
        )
    result: dict[str, tuple[str, ...]] = {}
    for name, validators in payload.items():
        if not isinstance(name, str):
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "Execution plan output validator names must be strings.",
            )
        if not isinstance(validators, list) or not all(
            isinstance(item, str) for item in validators
        ):
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "Execution plan output validators must be lists of strings.",
            )
        result[name] = tuple(validators)
    return result


def _plan_reference_to_json(
    reference: ExecutionPlanReference | None,
) -> dict[str, Any] | None:
    if reference is None:
        return None
    return {
        "$type": "execution_plan_reference",
        "plan_id": reference.plan_id,
        "version": reference.version,
    }


def _plan_reference_from_json(
    payload: Any,
) -> ExecutionPlanReference | None:
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {"$type", "plan_id", "version"}:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution plan reference must be an explicit object or null.",
        )
    if payload.get("$type") != "execution_plan_reference":
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution plan reference type is invalid.",
        )
    version = payload.get("version")
    if version is not None and not isinstance(version, str):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution plan reference version must be a string or null.",
        )
    try:
        return ExecutionPlanReference(
            plan_id=_required_str(payload, "plan_id"),
            version=version,
        )
    except ExecutionPlanRegistryError as error:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution plan reference is invalid.",
        ) from error


def _argument_from_json(
    value: Any,
) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$type", "step_id", "path"} and value.get("$type") == "step_output_reference":
            raw_path = value.get("path")
            if not isinstance(raw_path, list):
                raise ResumableExecutionStoreError(
                    "EXECUTION_STATE_INVALID",
                    "Step output reference path must be a list.",
                )
            return StepOutputReference(
                step_id=_required_str(value, "step_id"),
                path=tuple(raw_path),
            )
        if set(value) == {"$type", "name", "path"} and value.get("$type") == "execution_variable_reference":
            raw_path = value.get("path")
            if not isinstance(raw_path, list):
                raise ResumableExecutionStoreError(
                    "EXECUTION_STATE_INVALID",
                    "Execution variable reference path must be a list.",
                )
            return ExecutionVariableReference(
                name=_required_str(value, "name"),
                path=tuple(raw_path),
            )
        return {
            key: _argument_from_json(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_argument_from_json(item) for item in value]

    return value


def _context_snapshot_to_dict(
    snapshot: ExecutionContextSnapshot,
) -> dict[str, Any]:
    return {
        "execution_id": snapshot.execution_id,
        "results_by_step_id": _argument_to_json(dict(snapshot.results_by_step_id)),
        "variables": _argument_to_json(dict(snapshot.variables)),
        "step_states": dict(snapshot.step_states),
        "current_step_id": snapshot.current_step_id,
        "current_attempt": snapshot.current_attempt,
        "metadata": dict(snapshot.metadata),
    }


def _optional_context_snapshot(
    payload: Any,
) -> ExecutionContextSnapshot | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution context snapshot must be an object or null.",
        )
    results = _required_dict(payload, "results_by_step_id")
    variables = _required_dict(payload, "variables") if "variables" in payload else {}
    states = _required_dict(payload, "step_states")
    metadata = _required_dict(payload, "metadata")
    return ExecutionContextSnapshot(
        execution_id=_required_str(payload, "execution_id"),
        results_by_step_id=_argument_from_json(results),
        variables=_argument_from_json(variables),
        step_states=states,
        current_step_id=_optional_str(payload, "current_step_id"),
        current_attempt=_optional_int(payload, "current_attempt"),
        metadata=metadata,
    )


def _validation_to_dict(
    validation: PlanValidationResult,
) -> dict[str, Any]:
    return asdict(validation)


def _dict_to_validation(
    payload: dict[str, Any],
) -> PlanValidationResult:
    return PlanValidationResult(
        is_valid=_required_bool(payload, "is_valid"),
        errors=list(_str_tuple(payload, "errors")),
        warnings=list(_str_tuple(payload, "warnings")),
        requires_confirmation=_required_bool(payload, "requires_confirmation"),
        status=_required_str(payload, "status"),
        plan_signature=_optional_str(payload, "plan_signature"),
    )


def _validate_state_consistency(
    *,
    plan: ExecutionPlan,
    completed_step_ids: tuple[str, ...],
    pending_step_ids: tuple[str, ...],
    failed_step_ids: tuple[str, ...],
    interrupted_step_id: str | None,
    previous_results: dict[str, Any],
    execution_context_snapshot: ExecutionContextSnapshot | None,
) -> None:
    ordered_steps = _execution_ordered_steps(plan)
    all_step_ids = tuple(step.id for step in ordered_steps)
    all_step_id_set = set(all_step_ids)
    completed = set(completed_step_ids)
    pending = set(pending_step_ids)
    skipped: set[str] = set()
    blocked: set[str] = set()
    if execution_context_snapshot is not None:
        skipped = {
            step_id
            for step_id, state in execution_context_snapshot.step_states.items()
            if state == "SKIPPED"
        }
        blocked = {
            step_id
            for step_id, state in execution_context_snapshot.step_states.items()
            if state == "BLOCKED"
        }

    if failed_step_ids:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Failed executions cannot be resumed.",
        )

    if not pending_step_ids:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_NOT_RESUMABLE",
            "Completed executions cannot be resumed.",
        )

    if (
        not completed.issubset(all_step_id_set)
        or not pending.issubset(all_step_id_set)
        or not skipped.issubset(all_step_id_set)
        or not blocked.issubset(all_step_id_set)
    ):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution state contains unknown step IDs.",
        )

    if (
        completed & pending
        or completed & skipped
        or completed & blocked
        or pending & skipped
        or pending & blocked
        or skipped & blocked
    ):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution state has contradictory step IDs.",
        )

    expected_pending = tuple(
        step_id
        for step_id in all_step_ids
        if step_id not in completed and step_id not in skipped and step_id not in blocked
    )
    if pending_step_ids != expected_pending:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution state pending steps are inconsistent.",
        )

    if interrupted_step_id not in pending_step_ids:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Interrupted step is not pending.",
        )

    for step in ordered_steps:
        if step.id not in completed:
            continue
        for dependency_id in step.depends_on:
            if dependency_id not in completed:
                raise ResumableExecutionStoreError(
                    "EXECUTION_STATE_INVALID",
                    "A completed step has an unsatisfied dependency.",
                )
        if step.subplan is None and step.tool in {None, "direct_response"}:
            continue
        if step.id not in previous_results:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "A completed step is missing its previous result.",
            )

    if execution_context_snapshot is not None:
        for step in ordered_steps:
            if step.id not in blocked:
                continue
            if step.id in execution_context_snapshot.results_by_step_id:
                raise ResumableExecutionStoreError(
                    "EXECUTION_STATE_INVALID",
                    "A blocked step cannot have a stored result.",
                )
            binding = step.output_binding
            if (
                binding is not None
                and binding.variable_name in execution_context_snapshot.variables
            ):
                raise ResumableExecutionStoreError(
                    "EXECUTION_STATE_INVALID",
                    "A blocked step cannot have an applied output binding.",
                )


def _execution_ordered_steps(
    plan: ExecutionPlan,
) -> tuple[ExecutionStep, ...]:
    try:
        return ExecutionPlanTopologicalSorter().sort(plan).ordered_steps(plan)
    except ExecutionPlanTopologyError as error:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            f"Execution plan topology is invalid: {error}",
        ) from error


def _safe_metadata(
    metadata: dict[str, object],
    *,
    created_at: str,
) -> dict[str, object]:
    safe = {
        key: value
        for key, value in metadata.items()
        if isinstance(key, str)
        and key in {"created_at", "updated_at", "schema_version"}
        and _is_json_safe(value)
    }
    safe["created_at"] = created_at
    safe["schema_version"] = SCHEMA_VERSION
    return safe


def _is_json_safe(
    value: object,
) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return False
    return True


def _required_dict(
    payload: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            f"Execution state field '{key}' must be an object.",
        )
    return value


def _required_str(
    payload: dict[str, Any],
    key: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            f"Execution state field '{key}' must be a string.",
        )
    return value


def _optional_str(
    payload: dict[str, Any],
    key: str,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            f"Execution state field '{key}' must be a string or null.",
        )
    return value


def _required_int(
    payload: dict[str, Any],
    key: str,
) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            f"Execution state field '{key}' must be an integer.",
        )
    return value


def _optional_int(
    payload: dict[str, Any],
    key: str,
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not int:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            f"Execution state field '{key}' must be an integer or null.",
        )
    return value


def _required_bool(
    payload: dict[str, Any],
    key: str,
) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            f"Execution state field '{key}' must be a boolean.",
        )
    return value


def _str_tuple(
    payload: dict[str, Any],
    key: str,
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            f"Execution state field '{key}' must be a list of strings.",
        )
    return tuple(value)


def _int_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    default: dict[str, int],
) -> dict[str, int]:
    value = payload.get(key, default)
    if not isinstance(value, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            f"Execution state field '{key}' must be an object.",
        )
    result: dict[str, int] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str) or not isinstance(item_value, int):
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                f"Execution state field '{key}' must map strings to integers.",
            )
        result[item_key] = item_value
    return result


def _retry_history(
    payload: dict[str, Any],
    key: str,
    *,
    default: dict[str, tuple[dict[str, object], ...]],
) -> dict[str, tuple[dict[str, object], ...]]:
    value = payload.get(key, default)
    if not isinstance(value, dict):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            f"Execution state field '{key}' must be an object.",
        )
    result: dict[str, tuple[dict[str, object], ...]] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str) or not isinstance(item_value, list):
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                f"Execution state field '{key}' must map strings to lists.",
            )
        entries: list[dict[str, object]] = []
        for entry in item_value:
            if not isinstance(entry, dict):
                raise ResumableExecutionStoreError(
                    "EXECUTION_STATE_INVALID",
                    f"Execution state field '{key}' has invalid entries.",
                )
            safe_entry = {
                entry_key: entry_value
                for entry_key, entry_value in entry.items()
                if isinstance(entry_key, str)
                and entry_key in {"attempt_number", "error_code", "error"}
                and _is_json_safe(entry_value)
            }
            entries.append(safe_entry)
        result[item_key] = tuple(entries)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

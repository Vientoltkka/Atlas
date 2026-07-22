"""Local persistence for resumable structured executions."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Protocol

from core.execution_context import ExecutionContextSnapshot
from core.execution_plan_executor import ResumableExecutionState
from core.execution_plan_validator import PlanValidationResult, plan_signature
from core.planner import ExecutionPlan, ExecutionStep
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
    }


def _step_to_dict(
    step: ExecutionStep,
) -> dict[str, Any]:
    return {
        "id": step.id,
        "description": step.description,
        "tool": step.tool,
        "dependencies": list(step.dependencies),
        "status": step.status,
        "arguments": _argument_to_json(step.arguments.as_dict()),
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

    return ExecutionStep(
        id=_required_str(payload, "id"),
        description=_required_str(payload, "description"),
        tool=tool,
        dependencies=_str_tuple(payload, "dependencies"),
        status=_required_str(payload, "status"),
        arguments=_argument_from_json(_required_dict(payload, "arguments")),
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
    states = _required_dict(payload, "step_states")
    metadata = _required_dict(payload, "metadata")
    return ExecutionContextSnapshot(
        execution_id=_required_str(payload, "execution_id"),
        results_by_step_id=_argument_from_json(results),
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
) -> None:
    all_step_ids = tuple(step.id for step in plan.ordered_steps)
    all_step_id_set = set(all_step_ids)
    completed = set(completed_step_ids)
    pending = set(pending_step_ids)

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

    if not completed.issubset(all_step_id_set) or not pending.issubset(all_step_id_set):
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution state contains unknown step IDs.",
        )

    if completed & pending:
        raise ResumableExecutionStoreError(
            "EXECUTION_STATE_INVALID",
            "Execution state has contradictory step IDs.",
        )

    expected_pending = tuple(step_id for step_id in all_step_ids if step_id not in completed)
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

    for step in plan.ordered_steps:
        if step.id not in completed:
            continue
        if step.tool in {None, "direct_response"}:
            continue
        if step.id not in previous_results:
            raise ResumableExecutionStoreError(
                "EXECUTION_STATE_INVALID",
                "A completed step is missing its previous result.",
            )


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

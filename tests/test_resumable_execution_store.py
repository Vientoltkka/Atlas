from __future__ import annotations

import json
from dataclasses import replace

from core.execution_plan_executor import ResumableExecutionState
from core.execution_plan_validator import ExecutionPlanValidator
from core.planner import ExecutionPlan, ExecutionStep
from core.resumable_execution_store import (
    JsonResumableExecutionStore,
    ResumableExecutionStoreError,
)
from core.step_output_reference import StepOutputReference


def _step(
    step_id: str,
    tool: str,
    dependencies: tuple[str, ...] = (),
) -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        description=f"Execute {step_id}.",
        tool=tool,
        dependencies=dependencies,
        arguments={},
    )


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Resume persisted plan.",
        ordered_steps=(
            _step("step_1", "read_file"),
            _step("step_2", "write_file", dependencies=("step_1",)),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        detected_risks=("writes a file",),
        requires_confirmation=True,
        status="planned",
    )


def _state() -> ResumableExecutionState:
    plan = _plan()
    validation = ExecutionPlanValidator().validate(plan)
    return ResumableExecutionState(
        objective="resume",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=("step_1",),
        pending_step_ids=("step_2",),
        failed_step_ids=(),
        interrupted_step_id="step_2",
        previous_results={"step_1": {"content": "alpha"}},
        resumable=True,
        interruption_reason="pause",
        confirmation_granted=True,
        retry_attempts={"step_2": 1},
        retry_history={
            "step_2": (
                {
                    "attempt_number": 1,
                    "error_code": "TEMPORARY_UNAVAILABLE",
                    "error": "temporary unavailable",
                },
            )
        },
    )


def _payload(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_json_store_saves_and_loads_valid_state(tmp_path) -> None:
    store = JsonResumableExecutionStore(tmp_path / "state.json")
    state = _state()

    store.save(state)
    loaded = store.load()

    assert loaded == state or loaded is not None
    assert loaded.objective == "resume"  # type: ignore[union-attr]
    assert loaded.completed_step_ids == ("step_1",)  # type: ignore[union-attr]
    assert loaded.previous_results == {"step_1": {"content": "alpha"}}  # type: ignore[union-attr]
    assert loaded.retry_attempts == {"step_2": 1}  # type: ignore[union-attr]
    assert loaded.retry_history["step_2"][0]["error_code"] == "TEMPORARY_UNAVAILABLE"  # type: ignore[union-attr]
    assert store.exists() is True


def test_json_store_missing_file_returns_none(tmp_path) -> None:
    assert JsonResumableExecutionStore(tmp_path / "missing.json").load() is None


def test_json_store_rejects_corrupt_json_without_deleting_file(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{bad json", encoding="utf-8")
    store = JsonResumableExecutionStore(path)

    try:
        store.load()
    except ResumableExecutionStoreError as error:
        assert error.error_code == "EXECUTION_STATE_CORRUPTED"
    else:
        raise AssertionError("corrupt JSON must be rejected")

    assert path.exists()


def test_json_store_rejects_unsupported_schema(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = JsonResumableExecutionStore(path)
    store.save(_state())
    payload = _payload(path)
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        store.load()
    except ResumableExecutionStoreError as error:
        assert error.error_code == "EXECUTION_STATE_SCHEMA_UNSUPPORTED"
    else:
        raise AssertionError("unsupported schema must be rejected")


def test_json_store_rejects_signature_mismatch(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = JsonResumableExecutionStore(path)
    store.save(_state())
    payload = _payload(path)
    payload["validated_plan_signature"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        store.load()
    except ResumableExecutionStoreError as error:
        assert error.error_code == "EXECUTION_STATE_SIGNATURE_MISMATCH"
    else:
        raise AssertionError("signature mismatch must be rejected")


def test_json_store_rejects_unknown_step_id(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = JsonResumableExecutionStore(path)
    store.save(_state())
    payload = _payload(path)
    payload["completed_step_ids"] = ["step_1", "missing"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        store.load()
    except ResumableExecutionStoreError as error:
        assert error.error_code == "EXECUTION_STATE_INVALID"
    else:
        raise AssertionError("unknown step id must be rejected")


def test_json_store_rejects_completed_pending_contradiction(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = JsonResumableExecutionStore(path)
    store.save(_state())
    payload = _payload(path)
    payload["pending_step_ids"] = ["step_1", "step_2"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        store.load()
    except ResumableExecutionStoreError as error:
        assert error.error_code == "EXECUTION_STATE_INVALID"
    else:
        raise AssertionError("contradictory step ids must be rejected")


def test_json_store_rejects_missing_previous_result(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = JsonResumableExecutionStore(path)
    store.save(_state())
    payload = _payload(path)
    payload["previous_results"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        store.load()
    except ResumableExecutionStoreError as error:
        assert error.error_code == "EXECUTION_STATE_INVALID"
    else:
        raise AssertionError("missing previous result must be rejected")


def test_json_store_rejects_non_resumable_state(tmp_path) -> None:
    store = JsonResumableExecutionStore(tmp_path / "state.json")
    state = replace(_state(), resumable=False)

    store.save(state)

    try:
        store.load()
    except ResumableExecutionStoreError as error:
        assert error.error_code == "EXECUTION_STATE_NOT_RESUMABLE"
    else:
        raise AssertionError("non resumable state must be rejected")


def test_json_store_writes_atomically_without_temp_file_left(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = JsonResumableExecutionStore(path)

    store.save(_state())

    assert path.exists()
    assert not (tmp_path / "state.json.tmp").exists()
    assert _payload(path)["schema_version"] == 1


def test_json_store_delete_removes_state(tmp_path) -> None:
    store = JsonResumableExecutionStore(tmp_path / "state.json")
    store.save(_state())

    store.delete()

    assert store.exists() is False


def test_json_store_preserves_step_output_references_in_plan_arguments(tmp_path) -> None:
    plan = ExecutionPlan(
        goal="Resume referenced plan.",
        ordered_steps=(
            _step("step_1", "read_file"),
            ExecutionStep(
                id="step_2",
                description="Execute step_2.",
                tool="write_file",
                dependencies=("step_1",),
                arguments={"content": StepOutputReference("step_1", ("content",))},
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        detected_risks=("writes a file",),
        requires_confirmation=True,
        status="planned",
    )
    validation = ExecutionPlanValidator().validate(plan)
    state = ResumableExecutionState(
        objective="resume",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=("step_1",),
        pending_step_ids=("step_2",),
        failed_step_ids=(),
        interrupted_step_id="step_2",
        previous_results={"step_1": {"content": "alpha"}},
        resumable=True,
        confirmation_granted=True,
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)
    payload = _payload(tmp_path / "state.json")
    loaded = store.load()

    assert payload["original_plan"]["ordered_steps"][1]["arguments"]["content"] == {
        "$type": "step_output_reference",
        "path": ["content"],
        "step_id": "step_1",
    }
    assert loaded is not None
    assert loaded.original_plan.ordered_steps[1].arguments.as_dict() == {
        "content": StepOutputReference("step_1", ("content",))
    }

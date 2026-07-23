from __future__ import annotations

import json
from dataclasses import replace

from core.execution_context import ExecutionContext
from core.execution_condition import (
    AllOfCondition,
    AnyOfCondition,
    ExecutionCondition,
    ExecutionConditionOperator,
    NotCondition,
)
from core.execution_variable_binding import ExecutionVariableBinding
from core.execution_variable_reference import ExecutionVariableReference
from core.execution_plan_output import ExecutionPlanOutput
from core.execution_plan_registry import ExecutionPlanReference
from core.execution_plan_executor import ResumableExecutionState
from core.execution_plan_validator import ExecutionPlanValidator, PlanValidationResult, plan_signature
from core.planner import ExecutionPlan, ExecutionStep
from core.resumable_execution_store import (
    JsonResumableExecutionStore,
    ResumableExecutionStoreError,
)
from core.step_output_reference import StepOutputReference


def _step(
    step_id: str,
    tool: str | None,
    dependencies: tuple[str, ...] = (),
    condition: ExecutionCondition | None = None,
    subplan: ExecutionPlan | None = None,
    subplan_ref: ExecutionPlanReference | None = None,
) -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        description=f"Execute {step_id}.",
        tool=tool,
        subplan=subplan,
        subplan_ref=subplan_ref,
        dependencies=dependencies,
        arguments={},
        condition=condition,
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
    context = ExecutionContext("exec-store-1")
    context.mark_step_started("step_1", 1)
    context.mark_step_succeeded("step_1", {"content": "alpha"})
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
        execution_context_snapshot=context.snapshot(),
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
    assert loaded.execution_context_snapshot is not None  # type: ignore[union-attr]
    assert loaded.execution_context_snapshot.execution_id == "exec-store-1"  # type: ignore[union-attr]
    assert loaded.execution_context_snapshot.results_by_step_id == {  # type: ignore[union-attr]
        "step_1": {"content": "alpha"}
    }
    assert loaded.execution_context_snapshot.variables == {}  # type: ignore[union-attr]
    assert store.exists() is True


def test_json_store_persists_step_conditions(tmp_path) -> None:
    plan = ExecutionPlan(
        goal="Resume conditional plan.",
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                condition=ExecutionCondition(True, ExecutionConditionOperator.TRUTHY),
            ),
            _step("step_2", "write_file", dependencies=("step_1",)),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        detected_risks=("writes a file",),
        requires_confirmation=True,
        status="planned",
    )
    validation = ExecutionPlanValidator().validate(plan)
    context = ExecutionContext("exec-store-condition")
    context.mark_step_started("step_1", 1)
    context.mark_step_succeeded("step_1", {"content": "alpha"})
    state = replace(
        _state(),
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        execution_context_snapshot=context.snapshot(),
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    condition = loaded.original_plan.ordered_steps[0].condition
    assert condition is not None
    assert condition.operator is ExecutionConditionOperator.TRUTHY
    assert condition.left is True


def test_json_store_loads_legacy_steps_without_condition_as_none(tmp_path) -> None:
    path = tmp_path / "state.json"
    store = JsonResumableExecutionStore(path)
    store.save(_state())
    payload = _payload(path)
    for step in payload["original_plan"]["ordered_steps"]:
        step.pop("condition")
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = store.load()

    assert loaded is not None
    assert all(step.condition is None for step in loaded.original_plan.ordered_steps)


def test_json_store_persists_composite_conditions(tmp_path) -> None:
    plan = ExecutionPlan(
        goal="Resume composite condition plan.",
        ordered_steps=(
            _step(
                "step_1",
                "read_file",
                condition=AllOfCondition(
                    (
                        ExecutionCondition(True, ExecutionConditionOperator.TRUTHY),
                        AnyOfCondition(
                            (
                                ExecutionCondition(False, ExecutionConditionOperator.TRUTHY),
                                NotCondition(
                                    ExecutionCondition(False, ExecutionConditionOperator.TRUTHY)
                                ),
                            )
                        ),
                    )
                ),
            ),
            _step("step_2", "write_file", dependencies=("step_1",)),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        detected_risks=("writes a file",),
        requires_confirmation=True,
        status="planned",
    )
    validation = ExecutionPlanValidator().validate(plan)
    context = ExecutionContext("exec-store-composite")
    context.mark_step_started("step_1", 1)
    context.mark_step_succeeded("step_1", {"content": "alpha"})
    state = replace(
        _state(),
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        execution_context_snapshot=context.snapshot(),
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)
    payload = _payload(store.path)
    loaded = store.load()

    assert payload["original_plan"]["ordered_steps"][0]["condition"]["$type"] == "all_of_condition"
    assert loaded is not None
    condition = loaded.original_plan.ordered_steps[0].condition
    assert isinstance(condition, AllOfCondition)
    assert isinstance(condition.conditions[1], AnyOfCondition)
    assert isinstance(condition.conditions[1].conditions[1], NotCondition)


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


def test_json_store_preserves_execution_variables_and_references(tmp_path) -> None:
    plan = ExecutionPlan(
        goal="Resume variable plan.",
        ordered_steps=(
            ExecutionStep(
                id="step_1",
                description="Execute step_1.",
                tool="read_file",
                arguments={"path": ExecutionVariableReference("workspace_path")},
            ),
        ),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
        status="planned",
    )
    validation = ExecutionPlanValidator().validate(plan)
    context = ExecutionContext(
        "exec-store-variable",
        initial_variables={"workspace_path": "C:/AI/Atlas"},
    )
    state = ResumableExecutionState(
        objective="resume",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=(),
        pending_step_ids=("step_1",),
        failed_step_ids=(),
        interrupted_step_id="step_1",
        previous_results={},
        resumable=True,
        confirmation_granted=True,
        execution_context_snapshot=context.snapshot(),
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)
    payload = _payload(tmp_path / "state.json")
    loaded = store.load()

    assert payload["original_plan"]["ordered_steps"][0]["arguments"]["path"] == {
        "$type": "execution_variable_reference",
        "name": "workspace_path",
        "path": [],
    }
    assert payload["execution_context_snapshot"]["variables"] == {
        "workspace_path": "C:/AI/Atlas"
    }
    assert loaded is not None
    assert loaded.original_plan.ordered_steps[0].arguments.as_dict() == {
        "path": ExecutionVariableReference("workspace_path")
    }
    assert loaded.execution_context_snapshot is not None
    assert loaded.execution_context_snapshot.execution_id == "exec-store-variable"
    assert loaded.execution_context_snapshot.variables == {
        "workspace_path": "C:/AI/Atlas"
    }


def test_json_store_preserves_output_binding_and_loads_old_checkpoints(tmp_path) -> None:
    plan = ExecutionPlan(
        goal="Resume bound plan.",
        ordered_steps=(
            ExecutionStep(
                id="step_1",
                description="Execute step_1.",
                tool="read_file",
                arguments={"path": "README.md"},
                output_binding=ExecutionVariableBinding(
                    "selected_file",
                    ("path",),
                    overwrite=False,
                ),
            ),
        ),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
        status="planned",
    )
    validation = ExecutionPlanValidator().validate(plan)
    state = ResumableExecutionState(
        objective="resume",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=(),
        pending_step_ids=("step_1",),
        failed_step_ids=(),
        interrupted_step_id="step_1",
        previous_results={},
        resumable=True,
        confirmation_granted=True,
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)
    payload = _payload(tmp_path / "state.json")
    loaded = store.load()

    assert payload["original_plan"]["ordered_steps"][0]["output_binding"] == {
        "$type": "execution_variable_binding",
        "variable_name": "selected_file",
        "path": ["path"],
        "overwrite": False,
    }
    assert loaded is not None
    assert loaded.original_plan.ordered_steps[0].output_binding == (
        ExecutionVariableBinding("selected_file", ("path",), overwrite=False)
    )

    payload["original_plan"]["ordered_steps"][0].pop("output_binding")
    old_plan = ExecutionPlan(
        goal=plan.goal,
        ordered_steps=(
            ExecutionStep(
                id="step_1",
                description="Execute step_1.",
                tool="read_file",
                arguments={"path": "README.md"},
            ),
        ),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
        status="planned",
    )
    old_signature = plan_signature(old_plan)
    payload["validated_plan_signature"] = old_signature
    payload["validation_result"]["plan_signature"] = old_signature
    (tmp_path / "state.json").write_text(json.dumps(payload), encoding="utf-8")
    old_loaded = store.load()

    assert old_loaded is not None
    assert old_loaded.original_plan.ordered_steps[0].output_binding is None


def test_json_store_persists_subplan_steps_and_loads_legacy_without_subplan(tmp_path) -> None:
    child_plan = ExecutionPlan(
        goal="Child plan.",
        ordered_steps=(_step("child", "read_file"),),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
        status="planned",
    )
    parent_plan = ExecutionPlan(
        goal="Parent plan.",
        ordered_steps=(
            _step("run_child", None, subplan=child_plan),
            _step("write", "write_file", dependencies=("run_child",)),
        ),
        estimated_steps=2,
        required_tools=("write_file",),
        detected_risks=("writes a file",),
        requires_confirmation=True,
        status="planned",
    )
    validation = PlanValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        requires_confirmation=True,
        status="valid",
        plan_signature=plan_signature(parent_plan),
    )
    context = ExecutionContext("exec-store-subplan")
    context.mark_step_started("run_child", 1)
    context.mark_step_succeeded("run_child", {"child": "output"})
    state = ResumableExecutionState(
        objective="resume subplan",
        original_plan=parent_plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=("run_child",),
        pending_step_ids=("write",),
        failed_step_ids=(),
        interrupted_step_id="write",
        previous_results={"run_child": {"child": "output"}},
        resumable=True,
        confirmation_granted=True,
        execution_context_snapshot=context.snapshot(),
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)
    payload = _payload(tmp_path / "state.json")
    loaded = store.load()

    assert payload["original_plan"]["ordered_steps"][0]["tool"] is None
    assert payload["original_plan"]["ordered_steps"][0]["subplan"]["goal"] == "Child plan."
    assert loaded is not None
    loaded_subplan = loaded.original_plan.ordered_steps[0].subplan
    assert loaded_subplan is not None
    assert loaded_subplan.ordered_steps[0].id == "child"

    payload["original_plan"]["ordered_steps"][1].pop("subplan")
    legacy_plan = ExecutionPlan(
        goal=parent_plan.goal,
        ordered_steps=(
            _step("run_child", None, subplan=child_plan),
            _step("write", "write_file", dependencies=("run_child",)),
        ),
        estimated_steps=2,
        required_tools=("write_file",),
        detected_risks=("writes a file",),
        requires_confirmation=True,
        status="planned",
    )
    signature = plan_signature(legacy_plan)
    payload["validated_plan_signature"] = signature
    payload["validation_result"]["plan_signature"] = signature
    (tmp_path / "state.json").write_text(json.dumps(payload), encoding="utf-8")
    legacy_loaded = store.load()

    assert legacy_loaded is not None
    assert legacy_loaded.original_plan.ordered_steps[1].subplan is None


def test_json_store_persists_subplan_ref_and_loads_legacy_without_reference(tmp_path) -> None:
    parent_plan = ExecutionPlan(
        goal="Parent registered plan.",
        ordered_steps=(
            _step(
                "run_child",
                None,
                subplan_ref=ExecutionPlanReference("project.analysis", "1.0"),
            ),
            _step("write", "write_file", dependencies=("run_child",)),
        ),
        estimated_steps=2,
        required_tools=("write_file",),
        detected_risks=("writes a file",),
        requires_confirmation=True,
        status="planned",
    )
    validation = PlanValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        requires_confirmation=True,
        status="valid",
        plan_signature=plan_signature(parent_plan),
    )
    context = ExecutionContext("exec-store-subplan-ref")
    context.mark_step_started("run_child", 1)
    context.mark_step_succeeded("run_child", {"child": "output"})
    state = ResumableExecutionState(
        objective="resume registered subplan",
        original_plan=parent_plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=("run_child",),
        pending_step_ids=("write",),
        failed_step_ids=(),
        interrupted_step_id="write",
        previous_results={"run_child": {"child": "output"}},
        resumable=True,
        confirmation_granted=True,
        execution_context_snapshot=context.snapshot(),
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)
    payload = _payload(tmp_path / "state.json")
    loaded = store.load()

    reference_payload = payload["original_plan"]["ordered_steps"][0]["subplan_ref"]
    assert reference_payload == {
        "$type": "execution_plan_reference",
        "plan_id": "project.analysis",
        "version": "1.0",
    }
    assert loaded is not None
    assert loaded.original_plan.ordered_steps[0].subplan_ref == ExecutionPlanReference(
        "project.analysis",
        "1.0",
    )

    payload["original_plan"]["ordered_steps"][0].pop("subplan_ref")
    legacy_plan = ExecutionPlan(
        goal=parent_plan.goal,
        ordered_steps=(
            _step("run_child", None),
            _step("write", "write_file", dependencies=("run_child",)),
        ),
        estimated_steps=2,
        required_tools=("write_file",),
        detected_risks=("writes a file",),
        requires_confirmation=True,
        status="planned",
    )
    signature = plan_signature(legacy_plan)
    payload["validated_plan_signature"] = signature
    payload["validation_result"]["plan_signature"] = signature
    (tmp_path / "state.json").write_text(json.dumps(payload), encoding="utf-8")
    legacy_loaded = store.load()

    assert legacy_loaded is not None
    assert legacy_loaded.original_plan.ordered_steps[0].subplan_ref is None


def test_json_store_persists_execution_plan_output_definitions(tmp_path) -> None:
    plan = ExecutionPlan(
        goal="Resume output plan.",
        ordered_steps=(
            _step("step_1", "read_file"),
            _step("step_2", "write_file", dependencies=("step_1",)),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        detected_risks=("writes a file",),
        requires_confirmation=True,
        status="planned",
        output={
            "static": True,
            "content": StepOutputReference("step_1", ("content",)),
            "quality": ExecutionVariableReference("quality"),
            "nested": [StepOutputReference("step_1")],
        },
    )
    validation = ExecutionPlanValidator().validate(plan)
    context = ExecutionContext(
        "exec-store-output",
        initial_variables={"quality": "ok"},
    )
    context.mark_step_started("step_1", 1)
    context.mark_step_succeeded("step_1", {"content": "alpha"})
    state = ResumableExecutionState(
        objective="resume output",
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
        execution_context_snapshot=context.snapshot(),
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)
    payload = _payload(tmp_path / "state.json")
    loaded = store.load()

    assert payload["original_plan"]["output"]["$type"] == "execution_plan_output"
    assert payload["original_plan"]["output"]["value"]["content"] == {
        "$type": "step_output_reference",
        "path": ["content"],
        "step_id": "step_1",
    }
    assert payload["original_plan"]["output"]["value"]["quality"] == {
        "$type": "execution_variable_reference",
        "name": "quality",
        "path": [],
    }
    assert loaded is not None
    assert isinstance(loaded.original_plan.output, ExecutionPlanOutput)
    assert loaded.original_plan.output.as_definition()["content"] == (
        StepOutputReference("step_1", ("content",))
    )

    payload["original_plan"].pop("output")
    legacy_plan = ExecutionPlan(
        goal=plan.goal,
        ordered_steps=plan.ordered_steps,
        estimated_steps=plan.estimated_steps,
        required_tools=plan.required_tools,
        detected_risks=plan.detected_risks,
        requires_confirmation=plan.requires_confirmation,
        status=plan.status,
    )
    signature = plan_signature(legacy_plan)
    payload["validated_plan_signature"] = signature
    payload["validation_result"]["plan_signature"] = signature
    (tmp_path / "state.json").write_text(json.dumps(payload), encoding="utf-8")
    legacy_loaded = store.load()

    assert legacy_loaded is not None
    assert legacy_loaded.original_plan.output is None

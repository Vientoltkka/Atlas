from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.acceptance_criteria import (
    MAX_ACCEPTANCE_CRITERIA,
    AcceptanceCriterion,
    AcceptanceCriterionKind,
)
from core.execution_context import ExecutionContext
from core.execution_plan_executor import (
    ExecutionPlanExecutor,
    PlanExecutionResult,
    ResumableExecutionState,
    StepExecutionResult,
)
from core.execution_plan_validator import ExecutionPlanValidator, plan_signature
from core.goal_verifier import (
    CriterionEvaluationStatus,
    GoalVerificationStatus,
    GoalVerifier,
    goal_verification_result_to_dict,
)
from core.planner import ExecutionPlan, ExecutionStep
from core.resumable_execution_store import JsonResumableExecutionStore
from core.step_output_reference import StepOutputReference
from core.structured_execution import StructuredExecutionCoordinator
from tools.filesystem.read_file_tool import ReadFileTool
from tools.filesystem.write_file_tool import WriteFileTool
from tools.registry import ToolRegistry


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    return registry


def _criteria(destination: Path) -> tuple[AcceptanceCriterion, ...]:
    return (
        AcceptanceCriterion(
            "count",
            AcceptanceCriterionKind.EXPECTED_STEP_COUNT,
            "Three steps complete.",
            expected_count=3,
        ),
        AcceptanceCriterion(
            "read_source",
            AcceptanceCriterionKind.EXPECTED_TOOL_USED,
            "Source read.",
            source_step_id="step_1",
            tool_name="read_file",
        ),
        AcceptanceCriterion(
            "write_destination",
            AcceptanceCriterionKind.EXPECTED_TOOL_USED,
            "Destination written.",
            source_step_id="step_2",
            tool_name="write_file",
        ),
        AcceptanceCriterion(
            "read_destination",
            AcceptanceCriterionKind.EXPECTED_TOOL_USED,
            "Destination read.",
            source_step_id="step_3",
            tool_name="read_file",
        ),
        AcceptanceCriterion(
            "resource_exists",
            AcceptanceCriterionKind.RESOURCE_EXISTS,
            "Destination exists.",
            source_step_id="step_2",
            resource_path=str(destination),
        ),
        AcceptanceCriterion(
            "resource_readable",
            AcceptanceCriterionKind.RESOURCE_READABLE,
            "Destination is readable.",
            source_step_id="step_3",
            resource_path=str(destination),
        ),
        AcceptanceCriterion(
            "resource_matches",
            AcceptanceCriterionKind.RESOURCE_CONTENT_EQUALS,
            "Destination matches source.",
            source_step_id="step_3",
            comparison_step_id="step_1",
            resource_path=str(destination),
        ),
        AcceptanceCriterion(
            "outputs_match",
            AcceptanceCriterionKind.OUTPUT_EQUALS,
            "Read outputs match.",
            source_step_id="step_3",
            comparison_step_id="step_1",
        ),
        AcceptanceCriterion(
            "confirmed",
            AcceptanceCriterionKind.NO_PENDING_CONFIRMATIONS,
            "No confirmation remains.",
        ),
        AcceptanceCriterion(
            "no_critical_failures",
            AcceptanceCriterionKind.NO_CRITICAL_FAILURES,
            "No critical failures.",
        ),
    )


def _plan(
    source: Path,
    destination: Path,
    *,
    content: object = StepOutputReference("step_1"),
    criteria: tuple[AcceptanceCriterion, ...] | None = None,
) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Copy and verify content.",
        ordered_steps=(
            ExecutionStep(
                "step_1",
                "Read source.",
                "read_file",
                arguments={"path": str(source)},
            ),
            ExecutionStep(
                "step_2",
                "Write destination.",
                "write_file",
                dependencies=("step_1",),
                arguments={"path": str(destination), "content": content},
            ),
            ExecutionStep(
                "step_3",
                "Read destination.",
                "read_file",
                dependencies=("step_2",),
                arguments={"path": str(destination)},
            ),
        ),
        estimated_steps=3,
        required_tools=("read_file", "write_file"),
        detected_risks=(),
        requires_confirmation=True,
        acceptance_criteria=(
            _criteria(destination)
            if criteria is None
            else criteria
        ),
    )


def _technical_result(
    outputs: dict[str, object],
    *,
    confirmation_granted: bool = True,
) -> PlanExecutionResult:
    results = [
        StepExecutionResult(
            step_id=step_id,
            status="completed",
            success=True,
            tool_name=(
                "write_file" if step_id == "step_2" else "read_file"
            ),
            output=output,
        )
        for step_id, output in outputs.items()
    ]
    return PlanExecutionResult(
        plan_status="completed",
        success=True,
        completed=True,
        completed_steps=list(outputs),
        step_results=results,
        metadata={"confirmation_granted": confirmation_granted},
    )


def test_real_three_step_objective_is_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "copy.txt"
    source.write_text("real evidence\n", encoding="utf-8")
    plan = _plan(source, destination)
    registry = _registry()

    execution = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
        confirmation_granted=True,
    )

    result = execution.goal_verification_result
    assert execution.success is True
    assert result is not None
    assert result.verification_status is GoalVerificationStatus.VERIFIED
    assert result.satisfied_criteria == 10
    assert result.failed_criteria == 0
    assert result.resources_checked == (str(destination),)
    assert destination.read_text(encoding="utf-8") == "real evidence\n"


def test_real_three_step_different_content_is_not_verified(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "copy.txt"
    source.write_text("expected\n", encoding="utf-8")
    plan = _plan(source, destination, content="altered\n")
    registry = _registry()

    execution = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
        confirmation_granted=True,
    )

    result = execution.goal_verification_result
    assert execution.success is True
    assert result is not None
    assert result.verification_status is GoalVerificationStatus.NOT_VERIFIED
    assert {
        item.criterion_id
        for item in result.criteria
        if item.status is CriterionEvaluationStatus.FAILED
    } == {"resource_matches", "outputs_match"}
    assert "altered" not in json.dumps(
        goal_verification_result_to_dict(result),
        ensure_ascii=False,
    )
    coordinator = object.__new__(StructuredExecutionCoordinator)
    message = coordinator._completed_message(execution)
    assert "objetivo no quedó verificado" in message
    assert "Objetivo verificado" not in message


def test_no_declared_criteria_is_inconclusive(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path / "source.txt",
        tmp_path / "copy.txt",
        criteria=(),
    )
    result = GoalVerifier().verify(
        plan,
        _technical_result(
            {
                "step_1": "available",
                "step_2": "written",
                "step_3": "available",
            }
        ),
    )

    assert result.satisfied is False
    assert result.verification_status is GoalVerificationStatus.INCONCLUSIVE
    assert result.criteria == ()


def test_verification_evidence_sanitizes_sensitive_outputs() -> None:
    plan = ExecutionPlan(
        goal="Verify structured output.",
        ordered_steps=(
            ExecutionStep("step_1", "Produce.", "read_file"),
        ),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
        acceptance_criteria=(
            AcceptanceCriterion(
                "output_exists",
                AcceptanceCriterionKind.OUTPUT_EXISTS,
                "Output exists.",
                source_step_id="step_1",
            ),
        ),
    )

    result = GoalVerifier().verify(
        plan,
        _technical_result(
            {
                "step_1": {
                    "api_token": "sk-sensitive-value",
                    "public": "done",
                }
            }
        ),
    )
    serialized = json.dumps(
        goal_verification_result_to_dict(result),
        ensure_ascii=False,
    ).lower()

    assert result.verification_status is GoalVerificationStatus.VERIFIED
    assert "sk-sensitive-value" not in serialized
    assert "api_token" not in serialized


def test_optional_failure_does_not_prevent_verified() -> None:
    plan = ExecutionPlan(
        goal="Verify required output.",
        ordered_steps=(
            ExecutionStep("step_1", "Produce.", "read_file"),
        ),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
        acceptance_criteria=(
            AcceptanceCriterion(
                "required",
                AcceptanceCriterionKind.STEP_COMPLETED,
                "Required step.",
                source_step_id="step_1",
            ),
            AcceptanceCriterion(
                "optional",
                AcceptanceCriterionKind.OUTPUT_CONTAINS,
                "Optional fragment.",
                required=False,
                source_step_id="step_1",
                expected_value="absent",
            ),
        ),
    )

    result = GoalVerifier().verify(
        plan,
        _technical_result({"step_1": "available"}),
    )

    assert result.verification_status is GoalVerificationStatus.VERIFIED
    assert result.criteria[1].status is CriterionEvaluationStatus.FAILED


def test_mixed_satisfied_and_missing_evidence_is_partially_verified() -> None:
    plan = ExecutionPlan(
        goal="Partial evidence.",
        ordered_steps=(
            ExecutionStep("step_1", "First.", "read_file"),
            ExecutionStep("step_2", "Second.", "read_file"),
        ),
        estimated_steps=2,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
        acceptance_criteria=(
            AcceptanceCriterion(
                "first_done",
                AcceptanceCriterionKind.STEP_COMPLETED,
                "First step done.",
                source_step_id="step_1",
            ),
            AcceptanceCriterion(
                "outputs_match",
                AcceptanceCriterionKind.OUTPUT_EQUALS,
                "Outputs match.",
                source_step_id="step_1",
                comparison_step_id="step_2",
            ),
        ),
    )

    result = GoalVerifier().verify(
        plan,
        _technical_result({"step_1": "available"}),
    )

    assert (
        result.verification_status
        is GoalVerificationStatus.PARTIALLY_VERIFIED
    )
    assert result.unevaluable_criteria == 1


@pytest.mark.parametrize(
    ("right", "expected_status"),
    [
        (
            {"items": [1, {"ok": True}]},
            GoalVerificationStatus.VERIFIED,
        ),
        (
            {"items": [1, {"ok": False}]},
            GoalVerificationStatus.NOT_VERIFIED,
        ),
    ],
)
def test_structured_outputs_are_compared_without_coercion(
    right: object,
    expected_status: GoalVerificationStatus,
) -> None:
    plan = ExecutionPlan(
        goal="Compare structured outputs.",
        ordered_steps=(
            ExecutionStep("step_1", "First.", "read_file"),
            ExecutionStep("step_2", "Second.", "read_file"),
        ),
        estimated_steps=2,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
        acceptance_criteria=(
            AcceptanceCriterion(
                "structured_equal",
                AcceptanceCriterionKind.OUTPUT_EQUALS,
                "Structured outputs match exactly.",
                source_step_id="step_1",
                comparison_step_id="step_2",
            ),
        ),
    )

    result = GoalVerifier().verify(
        plan,
        _technical_result(
            {
                "step_1": {"items": [1, {"ok": True}]},
                "step_2": right,
            }
        ),
    )

    assert result.verification_status is expected_status


def test_pending_confirmation_requires_user_action(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "source.txt", tmp_path / "copy.txt")

    result = GoalVerifier().verify(
        plan,
        _technical_result(
            {"step_1": "a", "step_2": "ok", "step_3": "a"},
            confirmation_granted=False,
        ),
    )

    assert (
        result.verification_status
        is GoalVerificationStatus.USER_ACTION_REQUIRED
    )
    assert result.required_action is not None


@pytest.mark.parametrize(
    ("prepare", "expected_status"),
    [
        (lambda _path: None, CriterionEvaluationStatus.FAILED),
        (
            lambda path: path.write_bytes(b"\xff\xfe\x00"),
            CriterionEvaluationStatus.FAILED,
        ),
    ],
)
def test_resource_missing_or_unreadable_fails_controlled(
    tmp_path: Path,
    prepare,
    expected_status: CriterionEvaluationStatus,
) -> None:
    destination = tmp_path / "resource.txt"
    prepare(destination)
    criterion = AcceptanceCriterion(
        "readable",
        AcceptanceCriterionKind.RESOURCE_READABLE,
        "Resource is readable.",
        source_step_id="step_2",
        resource_path=str(destination),
    )
    plan = _plan(
        tmp_path / "source.txt",
        destination,
        criteria=(criterion,),
    )

    result = GoalVerifier().verify(
        plan,
        _technical_result({"step_1": "a", "step_2": "ok", "step_3": "a"}),
    )

    assert result.verification_status is GoalVerificationStatus.NOT_VERIFIED
    assert result.criteria[0].status is expected_status


def test_criterion_limits_and_plan_validation_are_enforced(
    tmp_path: Path,
) -> None:
    values = tuple(
        AcceptanceCriterion(
            f"criterion_{index}",
            AcceptanceCriterionKind.STEP_COMPLETED,
            "Bounded criterion.",
            source_step_id="step_1",
        )
        for index in range(MAX_ACCEPTANCE_CRITERIA + 1)
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        _plan(
            tmp_path / "source.txt",
            tmp_path / "copy.txt",
            criteria=values,
        )

    invalid = _plan(
        tmp_path / "source.txt",
        tmp_path / "copy.txt",
        criteria=(
            AcceptanceCriterion(
                "unknown",
                AcceptanceCriterionKind.STEP_COMPLETED,
                "Unknown step.",
                source_step_id="missing",
            ),
        ),
    )
    validation = ExecutionPlanValidator(_registry()).validate(invalid)
    assert validation.is_valid is False
    assert "references unknown step 'missing'" in " ".join(validation.errors)

    with_criteria = _plan(
        tmp_path / "source.txt",
        tmp_path / "copy.txt",
    )
    without_criteria = _plan(
        tmp_path / "source.txt",
        tmp_path / "copy.txt",
        criteria=(),
    )
    assert plan_signature(with_criteria) != plan_signature(without_criteria)


def test_criteria_and_verification_round_trip_in_resumable_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "copy.txt"
    source.write_text("persisted\n", encoding="utf-8")
    plan = _plan(source, destination)
    registry = _registry()
    validation = ExecutionPlanValidator(registry).validate(plan)
    execution = ExecutionPlanExecutor(registry).execute(
        plan,
        validation,
        confirmation_granted=True,
    )
    context = ExecutionContext("resume-verification")
    first = execution.step_results[0]
    context.mark_step_started(first.step_id, 1)
    context.mark_step_succeeded(first.step_id, first.output)
    interrupted_verification = GoalVerifier().verify(
        plan,
        PlanExecutionResult(
            plan_status="interrupted",
            success=False,
            interrupted=True,
            resumable=True,
            completed_steps=["step_1"],
            pending_steps=["step_2", "step_3"],
            current_step="step_2",
            step_results=[first],
        ),
    )
    state = ResumableExecutionState(
        objective=plan.goal,
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=("step_1",),
        pending_step_ids=("step_2", "step_3"),
        failed_step_ids=(),
        interrupted_step_id="step_2",
        previous_results={"step_1": first.output},
        resumable=True,
        interruption_reason="test interruption",
        execution_context_snapshot=context.snapshot(),
        goal_verification_result=interrupted_verification,
    )
    store = JsonResumableExecutionStore(tmp_path / "state.json")

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert len(loaded.original_plan.acceptance_criteria) == 10
    assert loaded.goal_verification_result is not None
    assert (
        loaded.goal_verification_result.verification_status
        is GoalVerificationStatus.NOT_VERIFIED
    )
    assert loaded.goal_verification_result.criteria == (
        interrupted_verification.criteria
    )


def test_resume_verifies_without_repeating_completed_read(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "copy.txt"
    expected = "resume evidence\n"
    source.write_text(expected, encoding="utf-8")
    plan = _plan(source, destination)
    registry = _registry()
    validation = ExecutionPlanValidator(registry).validate(plan)
    context = ExecutionContext("resume-objective")
    context.mark_step_started("step_1", 1)
    context.mark_step_succeeded("step_1", expected)
    state = ResumableExecutionState(
        objective=plan.goal,
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=("step_1",),
        pending_step_ids=("step_2", "step_3"),
        failed_step_ids=(),
        interrupted_step_id="step_2",
        previous_results={"step_1": expected},
        resumable=True,
        interruption_reason="controlled test interruption",
        confirmation_granted=True,
        execution_context_snapshot=context.snapshot(),
    )
    source.unlink()

    resumed = ExecutionPlanExecutor(registry).resume(state)

    assert resumed.success is True
    assert resumed.completed_steps == ["step_1", "step_2", "step_3"]
    assert destination.read_text(encoding="utf-8") == expected
    assert resumed.goal_verification_result is not None
    assert (
        resumed.goal_verification_result.verification_status
        is GoalVerificationStatus.VERIFIED
    )
    assert len(resumed.goal_verification_result.evidence) == len(
        set(resumed.goal_verification_result.evidence)
    )

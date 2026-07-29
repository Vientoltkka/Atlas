from __future__ import annotations

import json
from pathlib import Path

from core.acceptance_criteria import AcceptanceCriterion, AcceptanceCriterionKind
from core.execution_plan_executor import (
    ExecutionControl,
    ExecutionPlanExecutor,
    PlanExecutionResult,
    StepExecutionResult,
)
from core.execution_plan_validator import ExecutionPlanValidator, plan_signature
from core.execution_strategy import ExecutionStrategySelector
from core.execution_session_persistence import (
    ExecutionSessionSnapshot,
    snapshot_from_dict,
    snapshot_to_dict,
)
from core.execution_supervisor import ExecutionSupervisor
from core.resumable_execution_store import JsonResumableExecutionStore
from core.goal_verifier import GoalVerificationResult, GoalVerificationStatus, GoalVerifier
from core.objective_correction import (
    CorrectionClassification,
    CorrectionType,
    ObjectiveCorrectionPolicy,
    build_correction_request,
    classify_correction,
    correction_fragment_fingerprint,
)
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult
from core.step_output_reference import StepOutputReference
from core.structured_execution import StructuredExecutionCoordinator
from core.structured_plan_replanner import ExecutionReplanner
from core.execution_variable_reference import ExecutionVariableReference
from tools.filesystem.read_file_tool import ReadFileTool
from tools.filesystem.write_file_tool import WriteFileTool
from tools.registry import ToolRegistry


class _Planner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self.calls = 0

    def generate_execution_plan(self, objective: str, **kwargs: object) -> PlanGenerationResult:
        self.calls += 1
        return PlanGenerationResult(success=True, plan=self.plan)


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
            "source_read",
            AcceptanceCriterionKind.EXPECTED_TOOL_USED,
            "Source read.",
            source_step_id="step_1",
            tool_name="read_file",
        ),
        AcceptanceCriterion(
            "destination_write",
            AcceptanceCriterionKind.EXPECTED_TOOL_USED,
            "Destination written.",
            source_step_id="step_2",
            tool_name="write_file",
        ),
        AcceptanceCriterion(
            "destination_read",
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
            "no_failures",
            AcceptanceCriterionKind.NO_CRITICAL_FAILURES,
            "No critical failures.",
        ),
    )


def _plan(source: Path, destination: Path, *, wrong: bool = True) -> ExecutionPlan:
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
                arguments={
                    "path": str(destination),
                    "content": "wrong\n" if wrong else StepOutputReference("step_1"),
                },
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
        acceptance_criteria=_criteria(destination),
    )


def _execute(plan: ExecutionPlan, registry: ToolRegistry) -> PlanExecutionResult:
    return ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
        confirmation_granted=True,
    )


def _decision(
    tmp_path: Path,
    *,
    previous_attempts: int = 0,
) -> tuple[ExecutionPlan, PlanExecutionResult, object]:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("expected\n", encoding="utf-8")
    plan = _plan(source, destination)
    execution = _execute(plan, _registry())
    planner = _Planner(plan)
    decision = ExecutionReplanner(planner).corrective_fragment(
        plan,
        execution,
        execution.goal_verification_result,
        session_id="session.original",
        previous_attempts=previous_attempts,
    )
    return plan, execution, decision


def test_classifies_correctable_and_builds_serializable_request(tmp_path: Path) -> None:
    plan, execution, decision = _decision(tmp_path)

    assert decision.classification is CorrectionClassification.CORRECTABLE
    assert decision.correction_type is CorrectionType.REWRITE_RESOURCE
    assert decision.request.original_objective == plan.goal
    assert decision.request.completed_step_ids == ("step_1", "step_2", "step_3")
    assert decision.request.remaining_cycles == 1
    assert json.loads(json.dumps(decision.request.to_dict()))["failed_criteria"]
    assert execution.goal_verification_result.verification_status is (
        GoalVerificationStatus.NOT_VERIFIED
    )


def test_fragment_is_minimal_typed_and_preserves_goal(tmp_path: Path) -> None:
    plan, _, decision = _decision(tmp_path)
    fragment = decision.fragment

    assert fragment is not None
    assert fragment.goal == plan.goal
    assert [step.id for step in fragment.ordered_steps] == [
        "corrective_step_1",
        "corrective_step_2",
    ]
    assert fragment.requires_confirmation is True
    assert fragment.ordered_steps[0].arguments["content"] == ExecutionVariableReference(
        "correction_expected_value"
    )
    assert not {"step_1", "step_2", "step_3"}.intersection(
        step.id for step in fragment.ordered_steps
    )
    assert decision.expected_context == {"correction_expected_value": "expected\n"}
    assert decision.fragment_signature == plan_signature(fragment)


def test_fragment_passes_existing_validator(tmp_path: Path) -> None:
    _, _, decision = _decision(tmp_path)

    validation = ExecutionPlanValidator(_registry()).validate(decision.fragment)

    assert validation.is_valid is True
    assert validation.requires_confirmation is True


def test_verified_and_optional_failure_do_not_generate_correction(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("expected\n", encoding="utf-8")
    plan = _plan(source, destination, wrong=False)
    execution = _execute(plan, _registry())

    classification, correction_type, _, _ = classify_correction(
        plan,
        execution,
        execution.goal_verification_result,
        session_id="session.verified",
    )

    assert classification is CorrectionClassification.NOT_APPLICABLE
    assert correction_type is CorrectionType.NO_SAFE_CORRECTION


def test_inconclusive_and_user_action_do_not_execute_speculatively(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    plan = _plan(source, destination)
    technical = PlanExecutionResult(
        plan_status="completed",
        success=True,
        completed=True,
        completed_steps=[],
        step_results=[],
    )
    inconclusive = GoalVerifier().verify(
        ExecutionPlan(
            goal=plan.goal,
            ordered_steps=(),
            estimated_steps=0,
            required_tools=(),
            detected_risks=(),
            requires_confirmation=False,
        ),
        technical,
    )
    user_action = GoalVerificationResult(
        satisfied=False,
        reason="USER_ACTION_REQUIRED",
        verification_status=GoalVerificationStatus.USER_ACTION_REQUIRED,
        message="Action required.",
    )

    first = classify_correction(
        plan, technical, inconclusive, session_id="session.inconclusive"
    )
    second = classify_correction(
        plan, technical, user_action, session_id="session.user"
    )

    assert first[0] is CorrectionClassification.INSUFFICIENT_EVIDENCE
    assert second[0] is CorrectionClassification.USER_INPUT_REQUIRED
    assert second[1] is CorrectionType.REQUEST_USER_ACTION


def test_missing_demonstrated_value_is_insufficient_evidence(tmp_path: Path) -> None:
    plan, execution, _ = _decision(tmp_path)
    missing_source = PlanExecutionResult(
        plan_status="completed",
        success=True,
        completed=True,
        completed_steps=["step_2", "step_3"],
        step_results=[
            item
            for item in execution.step_results
            if item.step_id != "step_1"
        ],
        metadata=execution.metadata,
    )
    verification = GoalVerifier().verify(plan, missing_source)
    decision = ExecutionReplanner(_Planner(plan)).corrective_fragment(
        plan,
        missing_source,
        verification,
        session_id="session.missing",
    )

    assert decision.classification is CorrectionClassification.INSUFFICIENT_EVIDENCE
    assert decision.fragment is None


def test_cycle_limit_blocks_second_correction(tmp_path: Path) -> None:
    _, _, decision = _decision(tmp_path, previous_attempts=1)

    assert decision.classification is CorrectionClassification.LIMIT_REACHED
    assert decision.fragment is None
    assert decision.request.remaining_cycles == 0


def test_real_not_verified_to_verified_requires_two_distinct_confirmations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("expected\n", encoding="utf-8")
    plan = _plan(source, destination)
    planner = _Planner(plan)
    registry = _registry()
    coordinator = StructuredExecutionCoordinator(
        planner=planner,
        validator=ExecutionPlanValidator(registry),
        executor=ExecutionPlanExecutor(registry),
        execution_replanner=ExecutionReplanner(planner),
        execution_strategy_selector=ExecutionStrategySelector(),
    )

    original_pending = coordinator.handle(plan.goal)
    correction_pending = coordinator.confirm(original_pending.confirmation_token or "")

    assert original_pending.status == "confirmation_required"
    assert correction_pending.status == "correction_confirmation_required"
    assert correction_pending.confirmation_token != original_pending.confirmation_token
    assert destination.read_text(encoding="utf-8") == "wrong\n"
    assert correction_pending.objective_correction is not None
    assert correction_pending.objective_correction.classification is (
        CorrectionClassification.CORRECTABLE
    )

    completed = coordinator.confirm(correction_pending.confirmation_token or "")

    assert completed.status == "verified_after_correction"
    assert completed.corrected_verification is not None
    assert completed.corrected_verification.verification_status is (
        GoalVerificationStatus.VERIFIED
    )
    assert destination.read_text(encoding="utf-8") == "expected\n"
    assert completed.dispatch_result is not None
    assert completed.dispatch_result.consumed is True
    assert completed.operational_report is not None
    report = completed.operational_report.to_text()
    assert "Corrección del objetivo:" in report
    assert "Estado inicial: NOT_VERIFIED" in report
    assert "Nueva verificación: VERIFIED" in report
    assert "Ciclos utilizados: 1/1" in report


def test_pending_correction_does_not_write_without_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("expected\n", encoding="utf-8")
    plan = _plan(source, destination)
    planner = _Planner(plan)
    registry = _registry()
    coordinator = StructuredExecutionCoordinator(
        planner=planner,
        validator=ExecutionPlanValidator(registry),
        executor=ExecutionPlanExecutor(registry),
        execution_replanner=ExecutionReplanner(planner),
        execution_strategy_selector=ExecutionStrategySelector(),
    )

    initial = coordinator.handle(plan.goal)
    pending = coordinator.confirm(initial.confirmation_token or "")

    assert pending.status == "correction_confirmation_required"
    assert destination.read_text(encoding="utf-8") == "wrong\n"
    assert pending.authorization_result is not None
    assert pending.authorization_result.decision.value == "CONFIRMATION_PENDING"


def test_no_safe_correction_is_reported_without_fragment(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("expected\n", encoding="utf-8")
    criterion = AcceptanceCriterion(
        "wrong_count",
        AcceptanceCriterionKind.EXPECTED_STEP_COUNT,
        "Wrong count.",
        expected_count=99,
    )
    plan = ExecutionPlan(
        goal="Unsupported correction.",
        ordered_steps=(
            ExecutionStep(
                "read",
                "Read.",
                "read_file",
                arguments={"path": str(source)},
            ),
        ),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
        acceptance_criteria=(criterion,),
    )
    execution = _execute(plan, _registry())
    verification = execution.goal_verification_result

    classification, correction_type, request, reason = classify_correction(
        plan,
        execution,
        verification,
        session_id="session.unsupported",
    )

    assert classification is CorrectionClassification.NOT_CORRECTABLE
    assert correction_type is CorrectionType.NO_SAFE_CORRECTION
    assert request.failed_criteria[0].criterion_id == "wrong_count"
    assert "No single deterministic" in reason


def test_policy_rejects_limits_above_phase_cap() -> None:
    for kwargs in (
        {"max_cycles": 2},
        {"max_steps": 4},
        {"max_resources": 2},
        {"max_new_confirmations": 2},
        {"max_final_verifications": 2},
    ):
        try:
            ObjectiveCorrectionPolicy(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid policy: {kwargs}")


def test_fragment_fingerprint_blocks_same_fragment_even_for_new_request(
    tmp_path: Path,
) -> None:
    _, _, first = _decision(tmp_path)
    _, _, second = _decision(tmp_path)

    assert first.request.correction_request_id != second.request.correction_request_id
    assert correction_fragment_fingerprint(
        first.request.correction_request_id,
        first.fragment,
    ) == correction_fragment_fingerprint(
        second.request.correction_request_id,
        second.fragment,
    )


def test_consumed_correction_confirmation_cannot_dispatch_twice(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("expected\n", encoding="utf-8")
    plan = _plan(source, destination)
    planner = _Planner(plan)
    registry = _registry()
    coordinator = StructuredExecutionCoordinator(
        planner=planner,
        validator=ExecutionPlanValidator(registry),
        executor=ExecutionPlanExecutor(registry),
        execution_replanner=ExecutionReplanner(planner),
        execution_strategy_selector=ExecutionStrategySelector(),
    )

    original = coordinator.handle(plan.goal)
    pending = coordinator.confirm(original.confirmation_token or "")
    token = pending.confirmation_token or ""
    first = coordinator.confirm(token)
    second = coordinator.confirm(token)

    assert first.status == "verified_after_correction"
    assert second.status == "confirmation_not_found"
    assert destination.read_text(encoding="utf-8") == "expected\n"


def test_corrective_fragment_preserves_no_retry_as_one_attempt(
    tmp_path: Path,
) -> None:
    _, _, decision = _decision(tmp_path)

    assert decision.fragment is not None
    assert all(
        step.retry_policy is None
        for step in decision.fragment.ordered_steps
    )


def test_correction_snapshot_survives_session_serialization(
    tmp_path: Path,
) -> None:
    plan, _, decision = _decision(tmp_path)
    supervisor = ExecutionSupervisor()
    session = supervisor.start(plan)
    supervisor.mark_running(session.session_id)
    supervisor.record_objective_correction(
        session.session_id,
        {
            "correction_request_id": decision.request.correction_request_id,
            "classification": decision.classification.value,
            "status": "PENDING_CONFIRMATION",
            "cycle": 1,
            "fragment_signature": decision.fragment_signature,
            "affected_criteria": list(decision.affected_criterion_ids),
        },
    )
    snapshot = ExecutionSessionSnapshot.from_session(
        supervisor.get_session(session.session_id)
    )

    restored = snapshot_from_dict(snapshot_to_dict(snapshot))
    persisted = restored.results["objective_correction"].serializable_value

    assert persisted["classification"] == "CORRECTABLE"
    assert persisted["status"] == "PENDING_CONFIRMATION"
    assert persisted["cycle"] == 1
    assert persisted["fragment_signature"] == decision.fragment_signature


def test_interrupted_correction_reloads_and_does_not_repeat_completed_write(
    tmp_path: Path,
) -> None:
    class _CountingWriteFileTool(WriteFileTool):
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, context):
            self.calls += 1
            return super().execute(context)

    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("expected\n", encoding="utf-8")
    plan = _plan(source, destination)
    planner = _Planner(plan)
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    writer = _CountingWriteFileTool()
    registry.register(writer)
    store = JsonResumableExecutionStore(tmp_path / "correction-state.json")
    coordinator = StructuredExecutionCoordinator(
        planner=planner,
        validator=ExecutionPlanValidator(registry),
        executor=ExecutionPlanExecutor(registry),
        execution_replanner=ExecutionReplanner(planner),
        execution_strategy_selector=ExecutionStrategySelector(),
        resumable_store=store,
    )

    original = coordinator.handle(plan.goal)
    pending = coordinator.confirm(original.confirmation_token or "")
    interrupted = coordinator.confirm(
        pending.confirmation_token or "",
        control=ExecutionControl(should_stop=lambda: writer.calls >= 2),
    )

    assert interrupted.status == "correction_interrupted"
    assert interrupted.resumable_state is not None
    assert interrupted.resumable_state.completed_step_ids == (
        "corrective_step_1",
    )
    assert writer.calls == 2
    assert destination.read_text(encoding="utf-8") == "expected\n"

    coordinator._resumable_execution = None
    loaded = coordinator.load_persisted_resumable_execution()
    assert loaded.resumable_state is not None
    assert "objective_correction_resume" in loaded.resumable_state.metadata
    resumed = coordinator.resume_pending_execution()

    assert loaded.status == "resumable_execution_loaded"
    assert resumed.status == "verified_after_correction"
    assert writer.calls == 2
    assert resumed.corrected_verification is not None
    assert resumed.corrected_verification.verification_status is (
        GoalVerificationStatus.VERIFIED
    )


def test_failure_during_correction_is_recorded_without_recursive_loop(
    tmp_path: Path,
) -> None:
    class _FailSecondWriteTool(WriteFileTool):
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, context):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("controlled correction failure")
            return super().execute(context)

    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("expected\n", encoding="utf-8")
    plan = _plan(source, destination)
    planner = _Planner(plan)
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    writer = _FailSecondWriteTool()
    registry.register(writer)
    coordinator = StructuredExecutionCoordinator(
        planner=planner,
        validator=ExecutionPlanValidator(registry),
        executor=ExecutionPlanExecutor(registry),
        execution_replanner=ExecutionReplanner(planner),
        execution_strategy_selector=ExecutionStrategySelector(),
    )

    original = coordinator.handle(plan.goal)
    pending = coordinator.confirm(original.confirmation_token or "")
    failed = coordinator.confirm(pending.confirmation_token or "")

    assert failed.status == "correction_failed"
    assert writer.calls == 2
    assert coordinator.has_pending_execution() is False
    assert coordinator.confirm_pending().status == "no_pending_execution"
    assert destination.read_text(encoding="utf-8") == "wrong\n"

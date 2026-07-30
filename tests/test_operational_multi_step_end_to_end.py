from __future__ import annotations

from pathlib import Path

from bootstrap.bootstrap import Bootstrap
from core.execution_authorization import (
    DispatchStatus,
    ExecutionAuthorizationDecision,
)
from core.execution_session_persistence import FileExecutionSessionRepository
from core.execution_supervisor import ExecutionState
from core.goal_verifier import GoalVerificationStatus
from core.step_output_reference import StepOutputReference


def _configure_runtime(monkeypatch, tmp_path: Path) -> Path:
    history_path = tmp_path / "sessions"
    monkeypatch.setenv("ATLAS_HYBRID_PLANNING_ENABLED", "true")
    monkeypatch.setenv("ATLAS_STRUCTURED_PLAN_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("ATLAS_EXECUTION_PERSISTENCE_ENABLED", "true")
    monkeypatch.setenv("ATLAS_STRUCTURED_PLAN_PROVIDER_ENABLED", "false")
    monkeypatch.setenv("ATLAS_EXECUTION_HISTORY_PATH", str(history_path))
    monkeypatch.setenv(
        "ATLAS_EXECUTION_STATE_PATH",
        str(tmp_path / "execution_state.json"),
    )
    return history_path


def test_text_read_write_verify_uses_real_tools_references_and_persistence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    history_path = _configure_runtime(monkeypatch, tmp_path)
    source = tmp_path / "source.txt"
    destination = tmp_path / "verified-copy.txt"
    expected = "Atlas multi-step phase 15.2\n"
    source.write_text(expected, encoding="utf-8")
    orchestrator = Bootstrap.build()
    objective = (
        f"Lee {source}, guarda el contenido en {destination} "
        "y comprueba que se escribió correctamente."
    )

    pending_visible = orchestrator.process_prompt(
        objective,
        confirm=lambda _prompt: "",
    )
    pending = orchestrator.last_structured_execution_response

    assert pending is not None
    assert pending.status == "confirmation_required"
    assert pending.plan is not None
    assert [step.tool for step in pending.plan.ordered_steps] == [
        "read_file",
        "write_file",
        "read_file",
    ]
    assert pending.plan.ordered_steps[1].arguments["content"] == (
        StepOutputReference("step_1")
    )
    assert pending.plan.ordered_steps[2].dependencies == ("step_2",)
    assert "Responde 'confirmo'" in pending_visible
    assert "'cancela'" in pending_visible
    assert destination.exists() is False
    assert pending.operational_report is not None
    assert pending.operational_report.steps[1].produced_resource is None
    assert (
        pending.operational_report.goal_verification_status
        == GoalVerificationStatus.USER_ACTION_REQUIRED.value
    )

    visible = orchestrator.process_prompt(
        "confirmo",
        confirm=lambda _prompt: "",
    )
    detail = orchestrator.last_structured_execution_response

    assert detail is not None
    assert detail.status == "completed"
    assert detail.validation_result is not None
    assert detail.validation_result.is_valid is True
    assert detail.strategy_selection is not None
    assert detail.authorization_result is not None
    assert detail.authorization_result.decision is (
        ExecutionAuthorizationDecision.AUTHORIZED
    )
    assert detail.dispatch_result is not None
    assert detail.dispatch_result.status is DispatchStatus.DISPATCHED
    assert detail.execution_result is not None
    assert detail.execution_result.success is True
    assert [result.tool_name for result in detail.execution_result.step_results] == [
        "read_file",
        "write_file",
        "read_file",
    ]
    assert detail.execution_result.step_results[0].output == expected
    assert detail.execution_result.step_results[2].output == expected
    assert destination.read_text(encoding="utf-8") == expected
    assert detail.execution_result.goal_verification_result is not None
    assert (
        detail.execution_result.goal_verification_result.verification_status
        is GoalVerificationStatus.VERIFIED
    )

    write_metadata = detail.execution_result.step_results[1].metadata
    assert write_metadata["parameter_resolution_status"] == "resolved"
    assert write_metadata["resolved_argument_keys"] == ("content", "path")
    assert write_metadata["used_step_ids"] == ("step_1",)
    assert write_metadata["used_references"] == ("steps.step_1.output",)

    report = detail.operational_report
    assert report is not None
    assert report.completed_steps == 3
    assert report.steps[1].tool_name == "write_file"
    assert report.steps[1].resolved_references == (
        "steps.step_1.output",
    )
    assert report.steps[1].produced_resource == str(destination)
    assert "write_file:" in visible
    assert visible.startswith("Plan confirmado. Objetivo verificado.")
    assert "Referencias resueltas:" not in visible
    detailed_report = report.to_text()
    assert "Referencias resueltas: steps.step_1.output" in detailed_report
    assert f"Recurso producido: {destination}" in detailed_report
    assert "Verificación del objetivo:" in detailed_report
    assert "Estado: VERIFIED." in detailed_report
    assert "Criterios satisfechos: 10/10." in detailed_report

    session_id = detail.execution_result.metadata["execution_session_id"]
    restored_snapshot = FileExecutionSessionRepository(history_path).load(
        session_id
    )
    assert restored_snapshot is not None
    restored = restored_snapshot.to_session()
    assert restored.state is ExecutionState.COMPLETED
    assert len(restored.original_plan.ordered_steps) == 3
    assert restored.original_plan.ordered_steps[1].arguments["content"] == (
        StepOutputReference("step_1")
    )
    assert restored.results["step_outputs"]["step_1"] == expected
    assert restored.results["step_outputs"]["step_3"] == expected
    assert restored.results["step_resolution"]["step_2"]["references"] == [
        "steps.step_1.output"
    ]
    assert (
        restored.results["goal_verification"]["verification_status"]
        == GoalVerificationStatus.VERIFIED.value
    )

    history = orchestrator.execution_history
    assert history is not None
    assert session_id in {
        record.id
        for record in history.latest_executions(20)
    }

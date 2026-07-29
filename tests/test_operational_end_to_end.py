from __future__ import annotations

from pathlib import Path

from bootstrap.bootstrap import Bootstrap
from core.execution_authorization import (
    DispatchStatus,
    ExecutionAuthorizationDecision,
)
from core.execution_plan_validator import ExecutionPlanValidator
from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_session_persistence import FileExecutionSessionRepository
from core.execution_supervisor import ExecutionState, ExecutionSupervisor
from core.operational_request_router import RequestRoute
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult
from core.structured_execution import StructuredExecutionCoordinator
from services.file_service import FileService
from tools.filesystem.read_file_tool import ReadFileTool
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry


def _configure_real_runtime(monkeypatch, tmp_path: Path) -> Path:
    history_path = tmp_path / "sessions"
    monkeypatch.delenv("ATLAS_HYBRID_PLANNING_ENABLED", raising=False)
    monkeypatch.delenv(
        "ATLAS_STRUCTURED_PLAN_EXECUTION_ENABLED",
        raising=False,
    )
    monkeypatch.delenv("ATLAS_EXECUTION_PERSISTENCE_ENABLED", raising=False)
    monkeypatch.delenv(
        "ATLAS_STRUCTURED_PLAN_PROVIDER_ENABLED",
        raising=False,
    )
    monkeypatch.setenv("ATLAS_EXECUTION_HISTORY_PATH", str(history_path))
    monkeypatch.setenv(
        "ATLAS_EXECUTION_STATE_PATH",
        str(tmp_path / "execution_state.json"),
    )
    return history_path


def test_bootstrapped_text_flow_is_observable_persistent_and_repeatable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    history_path = _configure_real_runtime(monkeypatch, tmp_path)
    real_read = FileService.read
    read_calls: list[str] = []

    def counted_read(path: str) -> str:
        read_calls.append(str(path))
        return real_read(path)

    monkeypatch.setattr(FileService, "read", staticmethod(counted_read))
    orchestrator = Bootstrap.build()

    visible = orchestrator.process_prompt(
        "Lee README.md",
        confirm=lambda _prompt: "",
    )
    detail = orchestrator.last_structured_execution_response

    assert detail is not None
    assert detail.original_request == "Lee README.md"
    assert detail.route_decision is not None
    assert detail.route_decision.route is RequestRoute.SINGLE_TOOL
    assert detail.route_decision.target_tool_name == "read_file"
    assert detail.plan is not None
    assert detail.validation_result is not None
    assert detail.validation_result.is_valid is True
    assert detail.historical_context is not None
    assert detail.historical_adjustment is not None
    assert detail.strategy_selection is not None
    assert detail.authorization_result is not None
    assert detail.authorization_result.decision is (
        ExecutionAuthorizationDecision.AUTHORIZED
    )
    assert detail.dispatch_result is not None
    assert detail.dispatch_result.status is DispatchStatus.DISPATCHED
    assert detail.execution_result is not None
    assert detail.execution_result.success is True
    assert len(detail.execution_result.step_results) == 1
    assert detail.execution_result.step_results[0].output.lstrip("\ufeff").startswith(
        "# Atlas"
    )
    assert read_calls == ["README.md"]

    report = detail.operational_report
    assert report is not None
    assert report.dispatch_completed is True
    assert report.steps[0].result is not None
    assert report.steps[0].result.startswith("# Atlas")
    assert "Objetivo: Lee README.md" in visible
    assert "Resultado: Ejecución completada" in visible
    assert "Resultado: # Atlas" in visible
    assert "Estrategia:" in visible
    assert "Autorización:" in visible
    assert "Estado: AUTHORIZED" in visible
    assert "Duración:" in visible

    session_id = detail.execution_result.metadata["execution_session_id"]
    repository = FileExecutionSessionRepository(history_path)
    restored_snapshot = repository.load(session_id)
    assert restored_snapshot is not None
    restored = restored_snapshot.to_session()
    assert restored.original_plan.goal == "Lee README.md"
    assert restored.execution_strategy is not None
    assert restored.execution_authorization is not None
    assert restored.results["step_outputs"]["step_1"].lstrip("\ufeff").startswith(
        "# Atlas"
    )

    history = orchestrator.execution_history
    assert history is not None
    first_query = history.latest_executions(20)
    second_query = history.latest_executions(20)
    assert tuple(item.id for item in first_query) == tuple(
        item.id for item in second_query
    )
    assert len({item.id for item in first_query}) == len(first_query)
    assert session_id in {item.id for item in first_query}

    first_authorization_id = detail.authorization_result.authorization_id
    second_visible = orchestrator.process_prompt(
        "Lee README.md",
        confirm=lambda _prompt: "",
    )
    second = orchestrator.last_structured_execution_response
    assert second is not None
    assert second.execution_result is not None
    assert second.authorization_result is not None
    assert second.authorization_result.authorization_id != first_authorization_id
    assert (
        second.execution_result.metadata["execution_session_id"]
        != session_id
    )
    assert "Resultado: # Atlas" in second_visible
    assert read_calls == ["README.md", "README.md"]


def test_real_tool_error_does_not_break_the_next_text_request(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    _configure_real_runtime(monkeypatch, tmp_path)
    real_read = FileService.read
    read_calls: list[str] = []

    def counted_read(path: str) -> str:
        read_calls.append(str(path))
        return real_read(path)

    monkeypatch.setattr(FileService, "read", staticmethod(counted_read))
    orchestrator = Bootstrap.build()
    inputs = iter(
        (
            "Lee __atlas_phase_15_1_missing__.txt",
            "Lee README.md",
            "salir",
        )
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    orchestrator.start()

    output = capsys.readouterr().out
    succeeded_detail = orchestrator.last_structured_execution_response
    assert succeeded_detail is not None
    assert succeeded_detail.execution_result is not None
    assert succeeded_detail.execution_result.success is True
    assert "Ejecución fallida" in output
    assert "Ejecución completada" in output
    assert "Hasta pronto." in output
    assert "Traceback" not in output
    history = orchestrator.execution_history
    assert history is not None
    assert len(history.failed_executions()) == 1
    assert len(history.successful_executions()) == 1
    assert history.failed_executions()[0].retry_count == 0
    assert read_calls == [
        "__atlas_phase_15_1_missing__.txt",
        "README.md",
    ]


def test_real_confirmation_stays_pending_without_tool_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure_real_runtime(monkeypatch, tmp_path)
    real_read = FileService.read
    read_calls: list[str] = []

    def counted_read(path: str) -> str:
        read_calls.append(str(path))
        return real_read(path)

    monkeypatch.setattr(FileService, "read", staticmethod(counted_read))
    orchestrator = Bootstrap.build()

    visible = orchestrator.process_prompt(
        "Lee README.md y copia su contenido en phase_15_1_output.txt",
        confirm=lambda _prompt: "",
    )
    detail = orchestrator.last_structured_execution_response

    assert detail is not None
    assert detail.status == "confirmation_required"
    assert detail.requires_confirmation is True
    assert detail.execution_result is None
    assert detail.authorization_result is not None
    assert detail.authorization_result.decision is (
        ExecutionAuthorizationDecision.CONFIRMATION_PENDING
    )
    assert detail.operational_report is not None
    assert detail.operational_report.pending_user_actions
    assert "Confirma o cancela" in visible
    assert read_calls == []
    assert not (Path.cwd() / "phase_15_1_output.txt").exists()

    cancelled = orchestrator.process_prompt(
        "cancela",
        confirm=lambda _prompt: "",
    )
    assert "cancelado" in cancelled.lower()
    assert read_calls == []


def test_unknown_tool_fails_controlled_and_is_persisted(
    tmp_path: Path,
) -> None:
    plan = ExecutionPlan(
        goal="Usar herramienta inexistente",
        ordered_steps=(
            ExecutionStep(
                id="step_1",
                description="Unavailable operation.",
                tool="missing_tool",
            ),
        ),
        estimated_steps=1,
        required_tools=("missing_tool",),
        detected_risks=(),
        requires_confirmation=False,
    )

    class ExternalPlannerBoundary:
        def generate_execution_plan(self, objective: str, **_kwargs):
            return PlanGenerationResult(success=True, plan=plan)

    repository = FileExecutionSessionRepository(tmp_path / "sessions")
    supervisor = ExecutionSupervisor(session_repository=repository)
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    executor = ExecutionPlanExecutor(registry, ToolExecutor(registry))
    coordinator = StructuredExecutionCoordinator(
        planner=ExternalPlannerBoundary(),  # type: ignore[arg-type]
        validator=ExecutionPlanValidator(registry),
        executor=executor,
        execution_supervisor=supervisor,
    )

    response = coordinator.handle("Usar herramienta inexistente")

    assert response.status == "failed"
    assert response.execution_result is not None
    assert response.execution_result.success is False
    assert response.operational_report is not None
    assert response.operational_report.status.value == "FAILED"
    session_ids = repository.list()
    assert len(session_ids) == 1
    restored_snapshot = repository.load(session_ids[0])
    assert restored_snapshot is not None
    restored = restored_snapshot.to_session()
    assert restored.state is ExecutionState.FAILED
    assert restored.results["plan_status"] == "failed"

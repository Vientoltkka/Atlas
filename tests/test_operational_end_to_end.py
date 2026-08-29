from __future__ import annotations

from pathlib import Path

import pytest

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
from models.prompt_client import PromptClient
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
    assert visible.startswith(
        "Ejecucion completada, pero no hay evidencia suficiente "
        "para verificar el objetivo."
    )
    assert "read_file: # Atlas" in visible
    assert "Estrategia:" not in visible
    assert "Autorización:" not in visible
    assert "Duración:" not in visible
    assert "Estrategia:" in report.to_text()
    assert "Estado: AUTHORIZED" in report.to_text()

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
    assert "read_file: # Atlas" in second_visible
    assert read_calls == ["README.md", "README.md"]


def test_bootstrapped_text_flow_reads_an_absolute_spanish_file_request(
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
    readme_path = str(Path("README.md").resolve())

    prompt = f"Lee el archivo {readme_path}"
    decision = orchestrator.classify_prompt(prompt)
    visible = orchestrator.process_prompt(prompt, confirm=lambda _prompt: "")
    result = orchestrator._execution_conversation.last_result

    assert decision.route is RequestRoute.SINGLE_TOOL
    assert decision.target_tool_name == "read_file"
    assert result is not None
    assert result.execution_result is not None
    assert result.execution_result.tool_name == "read_file"
    assert result.execution_result.success is True
    assert read_calls == [readme_path]
    assert visible.startswith(f"He leído {readme_path}:")
    assert "Archivos encontrados:" not in visible

def test_bootstrapped_project_agent_answers_agent_selection_analysis_without_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure_real_runtime(monkeypatch, tmp_path)
    orchestrator = Bootstrap.build()

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("This project analysis must not invoke the model.")

    monkeypatch.setattr(PromptClient, "ask", unexpected_model_call)
    prompt = (
        "Analiza el proyecto Atlas y dime cuáles son los 3 archivos del código "
        "que consideras más importantes para entender cómo se decide qué agente "
        "debe atender una petición."
    )

    decision = orchestrator.classify_prompt(prompt)
    response = orchestrator.process_prompt(prompt, confirm=lambda _prompt: "")

    assert decision.route is RequestRoute.AGENT_DELEGATION
    assert decision.target_agent_name == "project"
    assert "core/operational_request_router.py" in response
    assert "core/agent_orchestrator.py" in response
    assert "core/orchestrator.py" in response

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
    assert "No pude completar la accion" in output
    assert "Ejecucion completada" in output
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


def test_text_loop_rejects_empty_input_without_closing(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    _configure_real_runtime(monkeypatch, tmp_path)
    orchestrator = Bootstrap.build()
    inputs = iter(("", "salir"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    orchestrator.start()

    output = capsys.readouterr().out
    assert "peticion esta vacia" in output
    assert "Hasta pronto." in output
    assert "Traceback" not in output


def test_text_loop_routes_memory_through_productive_flow_and_rebuild(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    _configure_real_runtime(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "ATLAS_PERSONAL_MEMORY_PATH",
        str(tmp_path / "personal_memory.json"),
    )
    first = Bootstrap.build()
    first_inputs = iter(
        (
            "recuerda que perfil principal profile.main: Smoke Principal",
            "recuerda que profile.main.training.schedule_preference: "
            "prefiero entrenar por la manana",
            "que recuerdas de profile.main.training.schedule_preference",
            "salir",
        )
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(first_inputs))

    first.start()

    first_output = capsys.readouterr().out
    assert "prefiero entrenar por la manana" in first_output
    assert "capacidad necesaria" not in first_output

    second = Bootstrap.build()
    second_inputs = iter(
        (
            "que recuerdas de profile.main.training.schedule_preference",
            "salir",
        )
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(second_inputs))

    second.start()

    second_output = capsys.readouterr().out
    assert "prefiero entrenar por la manana" in second_output
    assert "capacidad necesaria" not in second_output


def test_text_loop_treats_eof_as_clean_close(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    _configure_real_runtime(monkeypatch, tmp_path)
    orchestrator = Bootstrap.build()

    def end_of_input(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", end_of_input)

    orchestrator.start()

    output = capsys.readouterr().out
    assert "Hasta pronto." in output
    assert "Traceback" not in output


def test_text_loop_treats_keyboard_interrupt_as_clean_close(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    _configure_real_runtime(monkeypatch, tmp_path)
    orchestrator = Bootstrap.build()

    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)

    orchestrator.start()

    output = capsys.readouterr().out
    assert "Interrupcion recibida" in output
    assert "Traceback" not in output


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
    assert "Responde 'confirmo'" in visible
    assert "'cancela'" in visible
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


def test_tool_catalog_queries_use_real_registry_without_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure_real_runtime(monkeypatch, tmp_path)
    orchestrator = Bootstrap.build()

    def forbidden_execution(*_args, **_kwargs):
        raise AssertionError("tool execution is forbidden for catalog queries")

    monkeypatch.setattr(ToolExecutor, "execute", forbidden_execution)
    prompts = (
        "\u00bfQu\u00e9 herramientas tienes?",
        "Lista tus herramientas",
        "\u00bfQu\u00e9 puedes ejecutar?",
        "Mu\u00e9strame tus capacidades disponibles",
    )

    for prompt in prompts:
        response = orchestrator.process_prompt(prompt, confirm=lambda _prompt: "")
        assert response.startswith("Herramientas disponibles (")
        assert "El registro activo solo contiene herramientas disponibles." in response
        for tool_name in orchestrator._tool_registry.list():
            assert f"- {tool_name}:" in response

    assert orchestrator.last_structured_execution_response is None


@pytest.mark.parametrize("command", ("exit", "quit"))
def test_text_loop_accepts_english_exit_commands_and_can_restart(
    command: str,
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    _configure_real_runtime(monkeypatch, tmp_path)
    orchestrator = Bootstrap.build()
    inputs = iter((command, "salir"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    orchestrator.start()
    orchestrator.start()

    output = capsys.readouterr().out
    assert output.count("Hasta pronto.") == 2
    assert "Atlas iniciado correctamente" not in output
    assert "Traceback" not in output


def test_execution_history_is_visible_after_restart_without_new_tool_execution(
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
    first_runtime = Bootstrap.build()
    first_runtime._execution_conversation = None
    first_runtime.process_prompt("Lee README.md", confirm=lambda _prompt: "")
    first_detail = first_runtime.last_structured_execution_response

    assert first_detail is not None
    assert first_detail.execution_result is not None
    session_id = first_detail.execution_result.metadata["execution_session_id"]
    assert read_calls == ["README.md"]

    restarted_runtime = Bootstrap.build()
    latest = restarted_runtime.process_prompt(
        "últimas ejecuciones",
        confirm=lambda _prompt: "",
    )
    detail = restarted_runtime.process_prompt(
        f"detalle de ejecución {session_id}",
        confirm=lambda _prompt: "",
    )
    missing = restarted_runtime.process_prompt(
        "detalle de ejecución execution.session.999999",
        confirm=lambda _prompt: "",
    )

    assert "Últimas ejecuciones:" in latest
    assert session_id in latest
    assert "COMPLETED" in latest
    assert "Lee README.md" in latest
    assert read_calls == ["README.md"]
    assert "Objetivo: Lee README.md" in detail
    assert "Resultado:" in detail
    assert "Progreso:" in detail
    assert "Duración:" in detail
    assert "Ejecución:" in detail
    assert "step_outputs" not in detail
    assert "execution_events" not in detail
    assert "No se encontró una ejecución terminal disponible" in missing
    assert read_calls == ["README.md"]
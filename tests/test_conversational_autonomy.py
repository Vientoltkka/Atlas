"""Conversational routing for explicit prolonged-autonomy orders."""

from __future__ import annotations

import time
from types import SimpleNamespace

from bootstrap.bootstrap import Bootstrap
from core.async_task_scheduler import (
    AsyncTaskScheduler,
    GoalBudget,
    JsonGoalTaskStore,
    TaskStatus,
    ToolTaskExecutor,
)
from core.conversational_autonomy import (
    detect_autonomous_goal,
    is_background_goal_cancel_request,
    is_background_goal_status_query,
    parse_requested_minutes,
)
from core.multi_task_goal import detect_multi_task_goal
from core.orchestrator import AtlasOrchestrator
from core.router import Router
from memory.conversation import ConversationMemory
from tools.filesystem.read_file_tool import ReadFileTool
from tools.filesystem.write_file_tool import WriteFileTool
from tools.registry import ToolRegistry


class CountingReadFileTool(ReadFileTool):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, context):
        self.calls.append(str(context.parameters.get("path")))
        return super().execute(context)


def _wait_until(predicate, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


# -- unit detection -------------------------------------------------------------


def test_detects_explicit_duration_order_with_objective():
    request = detect_autonomous_goal(
        "Trabaja durante 30 minutos en este objetivo: "
        "lee README.md, resume el contenido y guarda el resumen en out.txt"
    )
    assert request is not None
    assert request.objective.startswith("lee readme.md")
    assert request.max_duration_seconds == 1800.0
    assert request.duration_requested is True


def test_detects_background_order_without_duration():
    request = detect_autonomous_goal(
        "Trabaja en segundo plano en lee notas.txt"
    )
    assert request is not None
    assert request.objective.startswith("lee notas.txt")
    assert request.max_duration_seconds == 900.0
    assert request.duration_requested is False


def test_detects_continue_until_done_order():
    request = detect_autonomous_goal(
        "Continúa trabajando en este objetivo hasta terminar o bloquearte: "
        "lee README.md"
    )
    assert request is not None
    assert request.objective.startswith("lee readme.md")


def test_normal_conversation_is_not_autonomy():
    assert detect_autonomous_goal("hola, ¿cómo estás?") is None
    assert detect_autonomous_goal("trabajar es importante para mí") is None
    assert detect_autonomous_goal("lee el archivo readme.md") is None


def test_status_and_cancel_detection():
    assert is_background_goal_status_query("¿Cómo va el objetivo?")
    assert is_background_goal_status_query("estado del trabajo")
    assert is_background_goal_status_query("¿Sigues trabajando?")
    assert not is_background_goal_status_query("lee el archivo trabajo.txt")
    assert is_background_goal_cancel_request("Detén el trabajo")
    assert is_background_goal_cancel_request("Cancela el objetivo")
    assert not is_background_goal_cancel_request("cancela")


def test_duration_bounds_are_conservative():
    assert parse_requested_minutes("trabaja durante 30 minutos en X") == 30
    assert parse_requested_minutes("trabaja 2 horas en X") == 120
    assert parse_requested_minutes("trabaja en X") is None
    request = detect_autonomous_goal("Trabaja durante 30 minutos en este objetivo: X")
    budget_cap = 30 * 60
    assert request.max_duration_seconds <= budget_cap


# -- conversational integration --------------------------------------------------


def _build_harness(tmp_path):
    tool_registry = ToolRegistry()
    read_tool = CountingReadFileTool()
    tool_registry.register(read_tool)
    tool_registry.register(WriteFileTool())

    def summarizer(text: str) -> str:
        return f"Resumen breve: {text[:40]}"

    runner = Bootstrap.build_single_tool_runner(tool_registry=tool_registry)
    executor = ToolTaskExecutor(runner, model_transformer=summarizer)
    scheduler = AsyncTaskScheduler(
        executor,
        store=JsonGoalTaskStore(tmp_path / "goal_store"),
    )
    executor.bind_result_lookup(scheduler.task)
    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(
            create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt),
        ),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=SimpleNamespace(get=lambda name: None),
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        async_task_scheduler=scheduler,
    )
    return orchestrator, scheduler, read_tool


ORDER_PROMPT = (
    "Trabaja durante 30 minutos en este objetivo: "
    "lee README.md, resume el contenido y guarda el resumen en autotest_bg.txt"
)


def test_conversational_order_progresses_in_background_without_new_turns(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido para el objetivo", encoding="utf-8")
    orchestrator, scheduler, read_tool = _build_harness(tmp_path)

    response = orchestrator.process_prompt(ORDER_PROMPT, confirm=lambda _p: "")

    assert "Objetivo en segundo plano" in response
    goal_id = orchestrator._background_goal_id
    assert goal_id is not None
    state = scheduler.goal(goal_id)
    budget = state.budget
    assert budget is not None
    assert budget.max_duration_seconds == 1800.0
    assert budget.deadline_epoch <= time.time() + 1800.0

    reached = _wait_until(
        lambda: len(read_tool.calls) == 1
        and scheduler.pending_approvals(goal_id)
        and state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
    )
    assert reached

    status_response = orchestrator.process_prompt(
        "¿Cómo va el objetivo?", confirm=lambda _p: ""
    )
    assert "Estado del trabajo" in status_response
    assert "pendiente de tu confirmación" in status_response
    assert "Presupuesto" in status_response

    resumed = orchestrator.process_prompt("sí", confirm=lambda _p: "")

    assert scheduler.goal_finished(goal_id)
    assert state.tasks["write_target"].status is TaskStatus.DONE
    written = (tmp_path / "autotest_bg.txt").read_text(encoding="utf-8")
    assert written.startswith("Resumen breve:")
    assert "Hecho" in resumed
    assert len(read_tool.calls) == 1


def test_cancellation_blocks_pending_work_and_frees_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido", encoding="utf-8")
    orchestrator, scheduler, read_tool = _build_harness(tmp_path)

    orchestrator.process_prompt(ORDER_PROMPT, confirm=lambda _p: "")
    goal_id = orchestrator._background_goal_id
    state = scheduler.goal(goal_id)
    reached = _wait_until(
        lambda: state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
    )
    assert reached

    cancel_response = orchestrator.process_prompt(
        "Detén el trabajo", confirm=lambda _p: ""
    )
    assert "Trabajo detenido" in cancel_response
    assert scheduler.goal_status(goal_id) is TaskStatus.CANCELLED
    assert state.tasks["read_source"].status is TaskStatus.DONE
    assert state.tasks["write_target"].status is TaskStatus.CANCELLED
    assert not (tmp_path / "autotest_bg.txt").exists()

    stale = orchestrator.process_prompt("sí", confirm=lambda _p: "")
    assert "ya no está disponible" in stale
    assert scheduler.goal_status(goal_id) is TaskStatus.CANCELLED

    later = orchestrator.process_prompt(
        "¿Cómo va el objetivo?", confirm=lambda _p: ""
    )
    assert "cancelado" in later


def test_order_without_structurable_objective_is_rejected_conservatively(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    orchestrator, scheduler, read_tool = _build_harness(tmp_path)

    response = orchestrator.process_prompt(
        "Trabaja durante 30 minutos en este objetivo: piensa sobre el sentido de la vida",
        confirm=lambda _p: "",
    )

    assert "no supe estructurar" in response
    assert read_tool.calls == []
    assert scheduler.persisted_goal_ids() == []


def test_background_pump_lifecycle_via_orchestrator(tmp_path):
    orchestrator, scheduler, _read_tool = _build_harness(tmp_path)
    assert orchestrator._background_pump is not None
    orchestrator.start_background_pump()
    assert orchestrator._background_pump.running
    orchestrator.close()
    assert not orchestrator._background_pump.running

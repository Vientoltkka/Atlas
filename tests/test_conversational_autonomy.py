"""Conversational routing for explicit prolonged-autonomy orders."""

from __future__ import annotations

import time
from types import SimpleNamespace

from bootstrap.bootstrap import Bootstrap
from core.async_task_scheduler import (
    AsyncTaskScheduler,
    GoalBudget,
    InvalidApprovalError,
    JsonGoalTaskStore,
    TaskStatus,
    ToolTaskExecutor,
    _utc_now,
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
from use_cases.execution_conversation import ExecutionConversationController


class CountingReadFileTool(ReadFileTool):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, context):
        self.calls.append(str(context.parameters.get("path")))
        return super().execute(context)


class CountingWriteFileTool(WriteFileTool):
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
    assert is_background_goal_status_query("estado")
    assert is_background_goal_status_query("¿Estado?")
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


def _build_scheduler_stack(tool_registry, runner, store_dir, transform_calls=None):
    if transform_calls is None:
        transform_calls = []

    def summarize(text: str) -> str:
        transform_calls.append(text)
        return f"Resumen breve: {text[:40]}"

    executor = ToolTaskExecutor(runner, model_transformer=summarize)
    scheduler = AsyncTaskScheduler(
        executor,
        store=JsonGoalTaskStore(store_dir),
    )
    executor.bind_result_lookup(scheduler.task)
    return scheduler


def _build_orchestrator(scheduler, tool_registry):
    execution_conversation = ExecutionConversationController(
        Bootstrap.build_execution_coordinator(tool_registry=tool_registry)
    )
    return AtlasOrchestrator(
        planner=SimpleNamespace(
            create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt),
        ),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=SimpleNamespace(get=lambda name: None),
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        async_task_scheduler=scheduler,
        execution_conversation=execution_conversation,
    )


def _build_harness(tmp_path):
    tool_registry = ToolRegistry()
    read_tool = CountingReadFileTool()
    tool_registry.register(read_tool)
    tool_registry.register(WriteFileTool())

    runner = Bootstrap.build_single_tool_runner(tool_registry=tool_registry)
    scheduler = _build_scheduler_stack(tool_registry, runner, tmp_path / "goal_store")
    orchestrator = _build_orchestrator(scheduler, tool_registry)
    return orchestrator, scheduler, read_tool, tool_registry, runner


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
    orchestrator, scheduler, read_tool = _build_harness(tmp_path)[:3]
    try:
        _run_background_goal_and_assert(orchestrator, scheduler, read_tool, tmp_path)
    finally:
        orchestrator.stop_background_pump()


def _run_background_goal_and_assert(orchestrator, scheduler, read_tool, tmp_path):
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

    bare_status_response = orchestrator.process_prompt(
        "estado", confirm=lambda _p: ""
    )
    assert "Estado del trabajo" in bare_status_response
    assert "pendiente de tu confirmación" in bare_status_response

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
    orchestrator, scheduler, read_tool = _build_harness(tmp_path)[:3]
    try:
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
    finally:
        orchestrator.stop_background_pump()
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
    orchestrator, scheduler, read_tool = _build_harness(tmp_path)[:3]

    response = orchestrator.process_prompt(
        "Trabaja durante 30 minutos en este objetivo: piensa sobre el sentido de la vida",
        confirm=lambda _p: "",
    )

    assert "no supe estructurar" in response
    assert read_tool.calls == []
    assert scheduler.persisted_goal_ids() == []


def test_persisted_goal_status_query_survives_restart(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido para el objetivo", encoding="utf-8")
    orchestrator, scheduler, read_tool, tool_registry, runner = _build_harness(tmp_path)
    transform_calls = []
    try:
        orchestrator.process_prompt(ORDER_PROMPT, confirm=lambda _p: "")
        goal_id = orchestrator._background_goal_id
        assert goal_id is not None
        state = scheduler.goal(goal_id)
        reached = _wait_until(
            lambda: state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
        )
        assert reached
        assert state.tasks["read_source"].status is TaskStatus.DONE
    finally:
        orchestrator.stop_background_pump()

    restart_scheduler = _build_scheduler_stack(
        tool_registry,
        runner,
        tmp_path / "goal_store",
        transform_calls=transform_calls,
    )
    restart_orchestrator = _build_orchestrator(restart_scheduler, tool_registry)
    try:
        restart_orchestrator.start_background_pump()
        status_response = restart_orchestrator.process_prompt(
            "estado", confirm=lambda _p: ""
        )
        assert "Estado del trabajo" in status_response
        assert "pendiente de tu confirmación" in status_response

        resumed = restart_orchestrator.process_prompt("sí", confirm=lambda _p: "")

        assert restart_scheduler.goal_finished(goal_id)
        assert "Hecho" in resumed
        written = (tmp_path / "autotest_bg.txt").read_text(encoding="utf-8")
        assert written.startswith("Resumen breve:")
        assert len(read_tool.calls) == 1
        assert transform_calls == []
    finally:
        restart_orchestrator.stop_background_pump()


def test_background_pump_lifecycle_via_orchestrator(tmp_path):
    orchestrator, scheduler, _read_tool = _build_harness(tmp_path)[:3]
    assert orchestrator._background_pump is not None
    orchestrator.start_background_pump()
    assert orchestrator._background_pump.running
    orchestrator.close()
    assert not orchestrator._background_pump.running


def test_pending_approval_survives_restart_with_fresh_confirmation_runtime(
    tmp_path,
    monkeypatch,
):
    """Real restart: a brand-new runner registry must still honor the approval.

    The pre-restart and post-restart confirmation runtimes are distinct
    objects, exactly as in a real process restart, so the runner-side
    pending confirmation is genuinely lost and must be recovered from the
    persisted exact tool and arguments.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido para el objetivo", encoding="utf-8")

    read_tool = CountingReadFileTool()
    write_tool = CountingWriteFileTool()
    tool_registry = ToolRegistry()
    tool_registry.register(read_tool)
    tool_registry.register(write_tool)

    pre_runner = Bootstrap.build_single_tool_runner(tool_registry=tool_registry)
    scheduler = _build_scheduler_stack(tool_registry, pre_runner, tmp_path / "goal_store")
    orchestrator = _build_orchestrator(scheduler, tool_registry)
    try:
        orchestrator.process_prompt(ORDER_PROMPT, confirm=lambda _p: "")
        goal_id = orchestrator._background_goal_id
        assert goal_id is not None
        state = scheduler.goal(goal_id)
        reached = _wait_until(
            lambda: state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
        )
        assert reached
        assert state.tasks["read_source"].status is TaskStatus.DONE
        assert len(write_tool.calls) == 0
    finally:
        orchestrator.stop_background_pump()
        orchestrator.close()

    pre_approval = scheduler.pending_approvals(goal_id)[0]

    # Real restart: fresh scheduler, fresh orchestrator and, critically, a
    # fresh SingleToolRunner with an empty confirmation registry.
    post_runner = Bootstrap.build_single_tool_runner(tool_registry=tool_registry)
    assert post_runner.pending_confirmations == ()
    post_scheduler = _build_scheduler_stack(
        tool_registry, post_runner, tmp_path / "goal_store"
    )
    post_orchestrator = _build_orchestrator(post_scheduler, tool_registry)
    try:
        status_response = post_orchestrator.process_prompt(
            "estado", confirm=lambda _p: ""
        )
        assert "Estado del trabajo" in status_response
        assert "pendiente de tu confirmación" in status_response

        resumed = post_orchestrator.process_prompt("sí", confirm=lambda _p: "")

        assert "Hecho" in resumed
        assert post_scheduler.goal_finished(goal_id)
        assert post_scheduler.task("write_target").status is TaskStatus.DONE
        assert post_scheduler.task("read_source").status is TaskStatus.DONE
        written = (tmp_path / "autotest_bg.txt").read_text(encoding="utf-8")
        assert written.startswith("Resumen breve:")
        assert len(read_tool.calls) == 1
        assert len(write_tool.calls) == 1

        stale = post_orchestrator.process_prompt("sí", confirm=lambda _p: "")
        assert post_scheduler.goal_finished(goal_id)
        assert post_scheduler.task("write_target").status is TaskStatus.DONE
        assert (tmp_path / "autotest_bg.txt").read_text(encoding="utf-8") == written
        assert len(read_tool.calls) == 1
        assert len(write_tool.calls) == 1
        try:
            post_scheduler.approve(pre_approval.confirmation_id)
            raise AssertionError("expected InvalidApprovalError")
        except InvalidApprovalError:
            pass
    finally:
        post_orchestrator.stop_background_pump()
        post_orchestrator.close()


def _seed_stale_goal(store_dir, goal_id, status, *, token=None, write_payload=None):
    """Persist a previous-session goal reusing the same canonical task ids."""
    discardable = ToolTaskExecutor(object())
    scheduler = AsyncTaskScheduler(
        discardable,
        store=JsonGoalTaskStore(store_dir),
    )
    scheduler.submit_goal(
        "objetivo de una sesion anterior",
        [
            {
                "task_id": "write_target",
                "description": "Guardar el resultado anterior",
                "payload": write_payload
                or {"tool": "write_file", "arguments": {"path": "viejo.txt", "content": "viejo"}},
            }
        ],
        goal_id=goal_id,
    )
    task = scheduler.task("write_target")
    task.status = status
    if token is not None:
        task.pending_confirmation_id = token
        task.resumable_payload = dict(write_payload)
    else:
        task.error = "fallida en una sesion anterior"
        task.finished_at = _utc_now()
    scheduler.persist_goal(goal_id)


def _restart_runtime(tool_registry, store_dir, transform_calls):
    runner = Bootstrap.build_single_tool_runner(tool_registry=tool_registry)
    scheduler = _build_scheduler_stack(
        tool_registry,
        runner,
        store_dir,
        transform_calls=transform_calls,
    )
    return _build_orchestrator(scheduler, tool_registry), scheduler


def test_confirmation_after_restart_survives_store_with_stale_goals(
    tmp_path,
    monkeypatch,
):
    """Literal UI repro: 'si' after restart with other goals in the store.

    The real store accumulates goals across sessions and every goal reuses
    the same task ids, so the approval used to be validated against another
    goal's task and the UI answered 'Esa operación pendiente ya no está
    disponible.' instead of resuming the write.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido para el objetivo", encoding="utf-8")
    store_dir = tmp_path / "goal_store"
    _seed_stale_goal(store_dir, "0000stale0000", TaskStatus.FAILED)

    read_tool = CountingReadFileTool()
    write_tool = CountingWriteFileTool()
    tool_registry = ToolRegistry()
    tool_registry.register(read_tool)
    tool_registry.register(write_tool)

    runner_a = Bootstrap.build_single_tool_runner(tool_registry=tool_registry)
    scheduler_a = _build_scheduler_stack(tool_registry, runner_a, store_dir)
    orchestrator_a = _build_orchestrator(scheduler_a, tool_registry)
    try:
        orchestrator_a.process_prompt(ORDER_PROMPT, confirm=lambda _p: "")
        goal_id = orchestrator_a._background_goal_id
        assert goal_id is not None
        state = scheduler_a.goal(goal_id)
        assert _wait_until(
            lambda: state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
        )
    finally:
        orchestrator_a.stop_background_pump()
        orchestrator_a.close()

    restart_transform_calls: list[str] = []
    orchestrator_b, scheduler_b = _restart_runtime(
        tool_registry,
        store_dir,
        restart_transform_calls,
    )
    try:
        orchestrator_b.start_background_pump()
        status_response = orchestrator_b.process_prompt("estado", confirm=lambda _p: "")
        assert "Estado del trabajo" in status_response
        assert "pendiente de tu confirmación" in status_response

        resumed = orchestrator_b.process_prompt("si", confirm=lambda _p: "")

        assert "ya no está disponible" not in resumed
        assert "Hecho" in resumed
        assert scheduler_b.goal_finished(goal_id)
        written = (tmp_path / "autotest_bg.txt").read_text(encoding="utf-8")
        assert written.startswith("Resumen breve:")
        assert len(read_tool.calls) == 1
        assert len(write_tool.calls) == 1
        assert restart_transform_calls == []

        repeated = orchestrator_b.process_prompt("si", confirm=lambda _p: "")
        assert scheduler_b.goal_finished(goal_id)
        assert (tmp_path / "autotest_bg.txt").read_text(encoding="utf-8") == written
        assert len(read_tool.calls) == 1
        assert len(write_tool.calls) == 1
    finally:
        orchestrator_b.stop_background_pump()
        orchestrator_b.close()


def test_bare_confirmation_resumes_active_goal_not_a_stale_token(
    tmp_path,
    monkeypatch,
):
    """'estado' describes the newest active goal; 'si' must resume that goal.

    Older goals may still hold unused approval tokens; a bare confirmation
    must never consume one of those and run another goal's write.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido para el objetivo", encoding="utf-8")
    store_dir = tmp_path / "goal_store"
    stale_payload = {
        "tool": "write_file",
        "arguments": {"path": "resultado_viejo.txt", "content": "contenido viejo"},
    }
    _seed_stale_goal(
        store_dir,
        "0000stale0000",
        TaskStatus.WAITING_APPROVAL,
        token="stale-token-0001",
        write_payload=stale_payload,
    )

    read_tool = CountingReadFileTool()
    write_tool = CountingWriteFileTool()
    tool_registry = ToolRegistry()
    tool_registry.register(read_tool)
    tool_registry.register(write_tool)

    runner_a = Bootstrap.build_single_tool_runner(tool_registry=tool_registry)
    scheduler_a = _build_scheduler_stack(tool_registry, runner_a, store_dir)
    orchestrator_a = _build_orchestrator(scheduler_a, tool_registry)
    try:
        orchestrator_a.process_prompt(ORDER_PROMPT, confirm=lambda _p: "")
        goal_id = orchestrator_a._background_goal_id
        assert goal_id is not None
        state = scheduler_a.goal(goal_id)
        assert _wait_until(
            lambda: state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
        )
    finally:
        orchestrator_a.stop_background_pump()
        orchestrator_a.close()

    orchestrator_b, scheduler_b = _restart_runtime(tool_registry, store_dir, [])
    try:
        orchestrator_b.start_background_pump()
        status_response = orchestrator_b.process_prompt("estado", confirm=lambda _p: "")
        assert "autotest_bg.txt" in status_response

        resumed = orchestrator_b.process_prompt("si", confirm=lambda _p: "")

        assert scheduler_b.goal_finished(goal_id)
        written = (tmp_path / "autotest_bg.txt").read_text(encoding="utf-8")
        assert written.startswith("Resumen breve:")
        assert not (tmp_path / "resultado_viejo.txt").exists()
        assert len(write_tool.calls) == 1
    finally:
        orchestrator_b.stop_background_pump()
        orchestrator_b.close()


ACEPTA2_PROMPT = (
    "Lee acepta2_fuente.txt, resume el contenido y guarda el resumen "
    "en acepta2_resumen.txt, y además lee acepta2_notas.txt"
)


def test_status_prefers_active_goal_over_finished_older_goal(tmp_path, monkeypatch):
    """'estado' must describe the active WAITING_APPROVAL goal.

    A fully DONE older goal that still owns the in-memory background id
    must never hide the newest non-terminal conversational goal, and a
    bare confirmation must keep acting on that same active goal.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido para el objetivo", encoding="utf-8")
    (tmp_path / "acepta2_fuente.txt").write_text("fuente acepta2", encoding="utf-8")
    (tmp_path / "acepta2_notas.txt").write_text("notas acepta2", encoding="utf-8")
    orchestrator, scheduler, read_tool = _build_harness(tmp_path)[:3]
    try:
        orchestrator.process_prompt(ORDER_PROMPT, confirm=lambda _p: "")
        old_goal_id = orchestrator._background_goal_id
        old_state = scheduler.goal(old_goal_id)
        assert _wait_until(
            lambda: old_state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
        )
        orchestrator.process_prompt("sí", confirm=lambda _p: "")
        assert scheduler.goal_finished(old_goal_id)
        old_written = (tmp_path / "autotest_bg.txt").read_text(encoding="utf-8")

        new_response = orchestrator.process_prompt(
            ACEPTA2_PROMPT, confirm=lambda _p: ""
        )
        assert "Objetivo en marcha" in new_response
        new_goal_id = next(
            goal_id
            for goal_id in scheduler.persisted_goal_ids()
            if goal_id != old_goal_id
        )
        new_state = scheduler.goal(new_goal_id)
        assert _wait_until(
            lambda: new_state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
        )

        status_response = orchestrator.process_prompt("estado", confirm=lambda _p: "")
        assert "acepta2_resumen.txt" in status_response
        assert "pendiente de tu confirmación" in status_response

        resumed = orchestrator.process_prompt("sí", confirm=lambda _p: "")

        assert "Hecho" in resumed
        assert scheduler.goal_finished(new_goal_id)
        assert scheduler.goal_status(old_goal_id) is TaskStatus.DONE
        assert (tmp_path / "acepta2_resumen.txt").read_text(encoding="utf-8").startswith(
            "Resumen breve:"
        )
        assert (tmp_path / "autotest_bg.txt").read_text(encoding="utf-8") == old_written
    finally:
        orchestrator.stop_background_pump()
        orchestrator.close()


def _stale_waiting_goal(store_dir, goal_id="0000stale0000", token="stale-token-0001"):
    stale_payload = {
        "tool": "write_file",
        "arguments": {"path": "resultado_viejo.txt", "content": "contenido viejo"},
    }
    _seed_stale_goal(
        store_dir,
        goal_id,
        TaskStatus.WAITING_APPROVAL,
        token=token,
        write_payload=stale_payload,
    )
    return stale_payload


def test_finished_current_goal_keeps_status_and_never_runs_old_write(
    tmp_path,
    monkeypatch,
):
    """Finishing the current goal must not resurface an older pending one.

    The runtime may have adopted an older WAITING_APPROVAL goal before the
    current work started. Once the current goal is DONE, 'estado' must keep
    describing it and a bare 'sí' must never consume the older goal's
    unused approval token.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido para el objetivo", encoding="utf-8")
    store_dir = tmp_path / "goal_store"
    _stale_waiting_goal(store_dir)

    read_tool = CountingReadFileTool()
    write_tool = CountingWriteFileTool()
    tool_registry = ToolRegistry()
    tool_registry.register(read_tool)
    tool_registry.register(write_tool)

    runner = Bootstrap.build_single_tool_runner(tool_registry=tool_registry)
    scheduler = _build_scheduler_stack(tool_registry, runner, store_dir)
    orchestrator = _build_orchestrator(scheduler, tool_registry)
    try:
        initial = orchestrator.process_prompt("estado", confirm=lambda _p: "")
        assert "sesion anterior" in initial
        assert scheduler.pending_approvals()

        orchestrator.process_prompt(ORDER_PROMPT, confirm=lambda _p: "")
        new_goal_id = orchestrator._background_goal_id
        assert new_goal_id is not None
        new_state = scheduler.goal(new_goal_id)
        assert _wait_until(
            lambda: new_state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
        )
        resumed = orchestrator.process_prompt("sí", confirm=lambda _p: "")
        assert "Hecho" in resumed
        assert scheduler.goal_finished(new_goal_id)

        status_response = orchestrator.process_prompt("estado", confirm=lambda _p: "")
        assert "autotest_bg.txt" in status_response
        assert "sesion anterior" not in status_response

        orchestrator.process_prompt("sí", confirm=lambda _p: "")

        assert scheduler.goal_status(new_goal_id) is TaskStatus.DONE
        assert scheduler.goal_status("0000stale0000") is TaskStatus.WAITING_APPROVAL
        assert not (tmp_path / "resultado_viejo.txt").exists()
        assert len(write_tool.calls) == 1
    finally:
        orchestrator.stop_background_pump()
        orchestrator.close()


def test_restart_status_prefers_recent_finished_goal_over_stale_pending(
    tmp_path,
    monkeypatch,
):
    """After restart, recency wins even when the newest goal already finished.

    A store holding an older WAITING_APPROVAL goal and a newer DONE goal
    must answer 'estado' with the recent finished goal, and a bare 'sí'
    must not execute the older goal's pending write.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido para el objetivo", encoding="utf-8")
    store_dir = tmp_path / "goal_store"
    _stale_waiting_goal(store_dir)

    read_tool = CountingReadFileTool()
    write_tool = CountingWriteFileTool()
    tool_registry = ToolRegistry()
    tool_registry.register(read_tool)
    tool_registry.register(write_tool)

    orchestrator_a = _build_orchestrator(
        _build_scheduler_stack(
            tool_registry,
            Bootstrap.build_single_tool_runner(tool_registry=tool_registry),
            store_dir,
        ),
        tool_registry,
    )
    try:
        orchestrator_a.process_prompt(ORDER_PROMPT, confirm=lambda _p: "")
        new_goal_id = orchestrator_a._background_goal_id
        assert new_goal_id is not None
        state = orchestrator_a._async_task_scheduler.goal(new_goal_id)
        assert _wait_until(
            lambda: state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
        )
        orchestrator_a.process_prompt("sí", confirm=lambda _p: "")
        assert orchestrator_a._async_task_scheduler.goal_finished(new_goal_id)
    finally:
        orchestrator_a.stop_background_pump()
        orchestrator_a.close()

    restart_write = CountingWriteFileTool()
    restart_read = CountingReadFileTool()
    restart_registry = ToolRegistry()
    restart_registry.register(restart_read)
    restart_registry.register(restart_write)
    orchestrator_b, scheduler_b = _restart_runtime(restart_registry, store_dir, [])
    try:
        status_response = orchestrator_b.process_prompt(
            "estado", confirm=lambda _p: ""
        )
        assert "autotest_bg.txt" in status_response
        assert "sesion anterior" not in status_response

        orchestrator_b.process_prompt("sí", confirm=lambda _p: "")

        assert scheduler_b.goal_status("0000stale0000") is TaskStatus.WAITING_APPROVAL
        assert not (tmp_path / "resultado_viejo.txt").exists()
        assert len(restart_write.calls) == 0
    finally:
        orchestrator_b.stop_background_pump()
        orchestrator_b.close()

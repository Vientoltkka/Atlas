"""Canonical tests: structured ExecutionPlan → scheduler → background pump.

Covers the whole bridge without any manual run_ready(): an explicit
"planifica y ejecuta" order produces a real ExecutionPlan (existing planner
classes), the bridge converts it into persistent scheduler tasks, and the
BackgroundGoalPump drives it until the write step asks for approval.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from bootstrap.bootstrap import Bootstrap
from core.async_task_scheduler import (
    AsyncTaskScheduler,
    JsonGoalTaskStore,
    TaskOutcome,
    TaskStatus,
    ToolTaskExecutor,
)
from core.background_goal_pump import BackgroundGoalPump  # noqa: F401 - pump contract
from core.conversational_autonomy import detect_structured_plan_objective
from core.execution_plan_validator import ExecutionPlanValidator
from core.orchestrator import AtlasOrchestrator
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult, Plan
from core.router import Router
from core.step_output_reference import StepOutputReference
from memory.conversation import ConversationMemory
from tools.filesystem.read_file_tool import ReadFileTool
from tools.filesystem.write_file_tool import WriteFileTool
from tools.registry import ToolRegistry


GOAL_PROMPT = (
    "Planifica y ejecuta este objetivo: compara nota1.txt y nota2.txt "
    "y guarda las diferencias en resultado.txt"
)


class ChatAgentFake:
    name = "chat"
    description = "fake chat"

    def run(self, model, messages):
        return f"respuesta de chat: {messages[-1]['content']}"


class CountingReadFileTool(ReadFileTool):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, context):
        self.calls.append(str(context.parameters.get("path")))
        return super().execute(context)


class StructuredPlannerStub:
    """Returns one fixed, real ExecutionPlan built with existing classes."""

    def __init__(self, plan: ExecutionPlan | None, success: bool = True) -> None:
        self._plan = plan
        self._success = success
        self.requested_objectives: list[str] = []
        self.requested_kwargs: list[dict] = []

    def create_plan(self, prompt: str) -> Plan:
        return Plan(task="chat", objective=prompt)

    def generate_execution_plan(self, objective: str, **kwargs) -> PlanGenerationResult:
        self.requested_objectives.append(objective)
        self.requested_kwargs.append(dict(kwargs))
        return PlanGenerationResult(
            success=self._success,
            plan=self._plan,
            generation_attempted=True,
        )


def _compare_plan() -> ExecutionPlan:
    read_first = ExecutionStep(
        "read_first",
        "Leer nota1.txt",
        "read_file",
        arguments={"path": "nota1.txt"},
    )
    read_second = ExecutionStep(
        "read_second",
        "Leer nota2.txt",
        "read_file",
        arguments={"path": "nota2.txt"},
    )
    compare = ExecutionStep(
        "compare",
        "Comparar los contenidos leidos",
        "direct_response",
        dependencies=("read_first", "read_second"),
        arguments={
            "instruction": (
                "Compara los siguientes contenidos y resume sus diferencias: {input}"
            )
        },
    )
    write_result = ExecutionStep(
        "write_result",
        "Guardar el resultado",
        "write_file",
        dependencies=("compare",),
        arguments={
            "path": "resultado.txt",
            "content": StepOutputReference("compare"),
        },
    )
    return ExecutionPlan(
        goal="comparar las dos notas y guardar las diferencias",
        ordered_steps=(read_first, read_second, compare, write_result),
        estimated_steps=4,
        required_tools=("read_file", "write_file"),
        detected_risks=("write_file",),
        requires_confirmation=True,
        status="planned",
    )


def _invalid_plan() -> ExecutionPlan:
    duplicate = ExecutionStep(
        "dup",
        "Paso duplicado",
        "read_file",
        arguments={"path": "a.txt"},
    )
    return ExecutionPlan(
        goal="plan invalido",
        ordered_steps=(duplicate, duplicate),
        estimated_steps=2,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
        status="planned",
    )


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        plan: ExecutionPlan | None,
        success=True,
        planner=None,
    ):
        self.tmp_path = tmp_path
        self.store = JsonGoalTaskStore(tmp_path / "goal_store")
        tool_registry = ToolRegistry()
        self.read_tool = CountingReadFileTool()
        tool_registry.register(self.read_tool)
        tool_registry.register(WriteFileTool())
        self.transform_calls: list[str] = []

        def transformer(text: str) -> str:
            self.transform_calls.append(text)
            return f"DIFERENCIAS: {text[:80]}"

        self.runner = Bootstrap.build_single_tool_runner(tool_registry=tool_registry)
        self.executor = ToolTaskExecutor(self.runner, model_transformer=transformer)
        self.scheduler = AsyncTaskScheduler(self.executor, store=self.store)
        self.executor.bind_result_lookup(self.scheduler.task)
        self.planner = planner or StructuredPlannerStub(plan, success=success)
        self.orchestrator = AtlasOrchestrator(
            planner=self.planner,
            router=Router(),
            model_manager=SimpleNamespace(choose_model=lambda name: f"model:{name}"),
            memory=ConversationMemory(),
            registry=SimpleNamespace(
                get=lambda name: ChatAgentFake() if name == "chat" else None
            ),
            write_file=SimpleNamespace(execute=lambda *_a: "written"),
            async_task_scheduler=self.scheduler,
            background_pump_interval_seconds=0.05,
        )

    def stop(self) -> None:
        self.orchestrator.stop_background_pump()

    def wait_until(self, predicate, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return bool(predicate())


def _wait_write_approval(harness: Harness):
    assert harness.wait_until(
        lambda: harness.scheduler.pending_approvals()
        and harness.scheduler.pending_approvals()[0].task_id == "write_result"
    )
    approval = harness.scheduler.pending_approvals()[0]
    return approval, harness.scheduler.goal(approval.goal_id)


REAL_ACCEPTANCE_PROMPT = """Planifica y ejecuta este objetivo:

Lee los archivos {path_a} y
{path_b}.

Compara su contenido y crea un resumen conjunto
explicando qué principio aporta cada archivo,
cómo se complementan y cuáles son las tres reglas
principales que debería seguir Atlas.

Guarda el resultado en
{path_result}.

Trabaja en segundo plano y utiliza un modelo local
cuando sea suficiente."""


def _real_production_planner():
    """Planner assembled exactly like production (deterministic routes)."""
    from bootstrap.bootstrap import Bootstrap
    from core.deterministic_multi_tool_planner import DeterministicMultiToolPlanner
    from core.planner import Planner

    tool_registry = ToolRegistry()
    tool_registry.register(ReadFileTool())
    tool_registry.register(WriteFileTool())
    tool_selector = Bootstrap.build_tool_selector(tool_registry)
    schema_registry = Bootstrap.build_argument_schema_registry()
    return Planner(
        tool_registry=tool_registry,
        tool_selector=tool_selector,
        schema_registry=schema_registry,
        argument_validator=Bootstrap.build_argument_validator(schema_registry),
        semantic_tool_catalog=Bootstrap.build_semantic_tool_catalog(
            tool_registry,
            tool_selector,
            schema_registry,
        ),
        multi_tool_planner=DeterministicMultiToolPlanner(),
    )


def test_real_acceptance_prompt_becomes_expected_task_dag(tmp_path, monkeypatch) -> None:
    """The literal E2E prompt must reach the bridge as the canonical DAG."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prueba_atlas_A.txt").write_text("principio A\ncomun\n", encoding="utf-8")
    (tmp_path / "prueba_atlas_B.txt").write_text("principio B\ncomun\n", encoding="utf-8")
    result_path = tmp_path / "resultado_prueba_autonomia.txt"

    prompt = REAL_ACCEPTANCE_PROMPT.format(
        path_a=tmp_path / "prueba_atlas_A.txt",
        path_b=tmp_path / "prueba_atlas_B.txt",
        path_result=result_path,
    )
    objective = detect_structured_plan_objective(prompt)
    assert objective is not None, "el objetivo estructurado no fue detectado"

    planner = _real_production_planner()
    generation = planner.generate_execution_plan(objective, structured_planning=True)
    assert generation.success, generation.errors
    plan = generation.plan
    assert [step.tool for step in plan.ordered_steps] == [
        "read_file",
        "read_file",
        "direct_response",
        "write_file",
    ]
    from core.execution_plan_task_adapter import execution_plan_to_task_specs
    from core.execution_plan_validator import ExecutionPlanValidator

    specs = execution_plan_to_task_specs(
        plan,
        validator=ExecutionPlanValidator(),
    )
    assert [spec["task_id"] for spec in specs] == [
        "step_1",
        "step_2",
        "transform_content",
        "write_target",
    ]
    assert specs[2]["payload"]["kind"] == "transform"
    assert specs[2]["payload"]["input_tasks"] == ["step_1", "step_2"]
    assert specs[3]["payload"]["content_task"] == "transform_content"

    harness = Harness(tmp_path, None, planner=planner)
    try:
        response = harness.orchestrator.process_prompt(prompt, confirm=lambda _p: "")

        assert "Plan estructurado en marcha" in response
        assert harness.wait_until(
            lambda: harness.scheduler.pending_approvals()
            and harness.scheduler.pending_approvals()[0].task_id == "write_target"
        )
        goal = harness.scheduler.goal(
            harness.scheduler.pending_approvals()[0].goal_id
        )
        assert goal.tasks["step_1"].status is TaskStatus.DONE
        assert goal.tasks["step_2"].status is TaskStatus.DONE
        assert goal.tasks["transform_content"].status is TaskStatus.DONE
        assert goal.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
        assert sorted(harness.read_tool.calls) == sorted(
            [
                str(tmp_path / "prueba_atlas_A.txt"),
                str(tmp_path / "prueba_atlas_B.txt"),
            ]
        )
        assert len(harness.transform_calls) == 1
        assert "principio A" in harness.transform_calls[0]
        assert "principio B" in harness.transform_calls[0]
        assert not result_path.exists()

        resumed = harness.orchestrator.process_prompt("sí", confirm=lambda _p: "")

        assert harness.scheduler.goal_finished(goal.goal_id)
        assert goal.tasks["write_target"].status is TaskStatus.DONE
        assert result_path.exists()
        assert result_path.read_text(encoding="utf-8").startswith("DIFERENCIAS:")
        assert "Hecho" in resumed
    finally:
        harness.stop()


def test_structured_plan_runs_in_background_until_write_approval(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nota1.txt").write_text("alpha\ncomun\n", encoding="utf-8")
    (tmp_path / "nota2.txt").write_text("beta\ncomun\n", encoding="utf-8")
    harness = Harness(tmp_path, _compare_plan())
    try:
        response = harness.orchestrator.process_prompt(
            GOAL_PROMPT,
            confirm=lambda _p: "",
        )

        assert harness.planner.requested_objectives, "el planificador no fue invocado"
        assert harness.planner.requested_kwargs
        assert harness.planner.requested_kwargs[0].get("structured_planning") is True
        assert "Plan estructurado en marcha" in response

        # No manual run_ready(): the pump drives every READY task.
        _approval, state = _wait_write_approval(harness)
        assert state.tasks["read_first"].status is TaskStatus.DONE
        assert state.tasks["read_second"].status is TaskStatus.DONE
        assert state.tasks["compare"].status is TaskStatus.DONE
        assert state.tasks["write_result"].status is TaskStatus.WAITING_APPROVAL

        # A/B once each; C once, fed with the real A/B outputs (DAG order).
        assert sorted(harness.read_tool.calls) == ["nota1.txt", "nota2.txt"]
        assert len(harness.transform_calls) == 1
        assert "[1] read_first" in harness.transform_calls[0]
        assert "alpha" in harness.transform_calls[0]
        assert "[2] read_second" in harness.transform_calls[0]
        assert "beta" in harness.transform_calls[0]
        # Permissions intact: no write happens without explicit confirmation.
        assert not (tmp_path / "resultado.txt").exists()

        resumed = harness.orchestrator.process_prompt("sí", confirm=lambda _p: "")

        assert harness.scheduler.goal_finished(state.goal_id)
        assert state.tasks["write_result"].status is TaskStatus.DONE
        assert sorted(harness.read_tool.calls) == ["nota1.txt", "nota2.txt"]
        assert len(harness.transform_calls) == 1
        written = (tmp_path / "resultado.txt").read_text(encoding="utf-8")
        assert written.startswith("DIFERENCIAS:")
        assert "Hecho" in resumed
    finally:
        harness.stop()


def test_invalid_plan_creates_zero_tasks_and_executes_nothing(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nota1.txt").write_text("contenido\n", encoding="utf-8")
    harness = Harness(tmp_path, _invalid_plan())
    try:
        response = harness.orchestrator.process_prompt(
            GOAL_PROMPT,
            confirm=lambda _p: "",
        )

        assert "no puede convertirse" in response
        assert harness.scheduler.pending_approvals() == []
        assert not harness.scheduler._goals
        assert list((tmp_path / "goal_store").glob("*.json")) == []
        assert harness.read_tool.calls == []
        assert (tmp_path / "nota1.txt").read_text(encoding="utf-8") == "contenido\n"
    finally:
        harness.stop()


def test_failed_planning_does_not_create_tasks(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    harness = Harness(tmp_path, plan=None, success=False)
    try:
        response = harness.orchestrator.process_prompt(
            GOAL_PROMPT,
            confirm=lambda _p: "",
        )
        assert "No pude estructurar" in response
        assert not harness.scheduler._goals
        assert harness.read_tool.calls == []
    finally:
        harness.stop()


def test_denial_blocks_only_the_write_and_writes_nothing(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nota1.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "nota2.txt").write_text("beta\n", encoding="utf-8")
    harness = Harness(tmp_path, _compare_plan())
    try:
        harness.orchestrator.process_prompt(GOAL_PROMPT, confirm=lambda _p: "")
        _approval, state = _wait_write_approval(harness)

        response = harness.orchestrator.process_prompt("no", confirm=lambda _p: "")

        assert state.tasks["write_result"].status is TaskStatus.BLOCKED
        assert state.tasks["read_first"].status is TaskStatus.DONE
        assert state.tasks["compare"].status is TaskStatus.DONE
        assert not (tmp_path / "resultado.txt").exists()
        assert "cancelada" in response
    finally:
        harness.stop()


def test_restart_recovers_persisted_tasks_without_repeating_done_work(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nota1.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "nota2.txt").write_text("beta\n", encoding="utf-8")
    harness = Harness(tmp_path, _compare_plan())
    try:
        harness.orchestrator.process_prompt(GOAL_PROMPT, confirm=lambda _p: "")
        approval, _state = _wait_write_approval(harness)
        goal_id = approval.goal_id
        harness.stop()  # simulated process restart

        restart_executor = ToolTaskExecutor(
            harness.runner,
            model_transformer=lambda text: f"DIFERENCIAS: {text[:80]}",
        )
        restart_scheduler = AsyncTaskScheduler(
            restart_executor,
            store=harness.store,
        )
        restart_executor.bind_result_lookup(restart_scheduler.task)
        restart_orchestrator = AtlasOrchestrator(
            planner=harness.planner,
            router=Router(),
            model_manager=SimpleNamespace(choose_model=lambda name: f"model:{name}"),
            memory=ConversationMemory(),
            registry=SimpleNamespace(
                get=lambda name: ChatAgentFake() if name == "chat" else None
            ),
            write_file=SimpleNamespace(execute=lambda *_a: "written"),
            async_task_scheduler=restart_scheduler,
            background_pump_interval_seconds=0.05,
        )
        # Recovery path used by BackgroundGoalPump.start(): persisted state,
        # never a replan of the same goal.
        restart_scheduler.load_goal(goal_id)
        try:
            state = restart_scheduler.goal(goal_id)
            assert state.tasks["read_first"].status is TaskStatus.DONE
            assert state.tasks["read_second"].status is TaskStatus.DONE
            assert state.tasks["compare"].status is TaskStatus.DONE
            assert state.tasks["write_result"].status is TaskStatus.WAITING_APPROVAL
            assert sorted(harness.read_tool.calls) == ["nota1.txt", "nota2.txt"]
            assert len(harness.transform_calls) == 1

            restart_orchestrator.process_prompt("sí", confirm=lambda _p: "")

            assert restart_scheduler.goal_finished(goal_id)
            written = (tmp_path / "resultado.txt").read_text(encoding="utf-8")
            assert written.startswith("DIFERENCIAS:")
            assert sorted(harness.read_tool.calls) == ["nota1.txt", "nota2.txt"]
            assert len(harness.transform_calls) == 1
        finally:
            restart_orchestrator.stop_background_pump()
    finally:
        harness.stop()


def test_cancel_stops_the_structured_goal(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nota1.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "nota2.txt").write_text("beta\n", encoding="utf-8")
    harness = Harness(tmp_path, _compare_plan())
    try:
        harness.orchestrator.process_prompt(GOAL_PROMPT, confirm=lambda _p: "")
        approval, _state = _wait_write_approval(harness)
        executed_before = harness.scheduler.goal(approval.goal_id).executed_count

        response = harness.orchestrator.process_prompt(
            "detén el trabajo",
            confirm=lambda _p: "",
        )

        assert harness.scheduler.goal_status(approval.goal_id) is TaskStatus.CANCELLED
        state = harness.scheduler.goal(approval.goal_id)
        assert state.tasks["write_result"].status is TaskStatus.CANCELLED
        assert state.tasks["compare"].status is TaskStatus.DONE
        assert harness.scheduler.goal(approval.goal_id).executed_count == executed_before
        assert not (tmp_path / "resultado.txt").exists()
        assert response
    finally:
        harness.stop()


def test_existing_paths_keep_working(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido\n", encoding="utf-8")
    harness = Harness(tmp_path, _compare_plan())
    try:
        chat = harness.orchestrator.process_prompt("hola", confirm=lambda _p: "")
        multi_task = harness.orchestrator.process_prompt(
            "Lee README.md y guarda el contenido en copia.txt",
            confirm=lambda _p: "",
        )

        assert chat.startswith("respuesta de chat")
        # The legacy multi-task routing stays untouched (no planner call).
        assert harness.planner.requested_objectives == []
        assert "Objetivo en marcha" in multi_task
        assert len(harness.scheduler.pending_approvals()) == 1
    finally:
        harness.stop()


def test_structured_entry_detection_is_explicit_only() -> None:
    assert (
        detect_structured_plan_objective(
            "Planifica y ejecuta este objetivo: lee notas.txt"
        )
        == "lee notas.txt"
    )
    assert (
        detect_structured_plan_objective(
            "Trabaja en este objetivo: compara a.txt y b.txt"
        )
        == "compara a.txt y b.txt"
    )
    assert detect_structured_plan_objective("hola") is None
    assert (
        detect_structured_plan_objective("trabaja en este objetivo hasta terminar")
        is None
    )
    assert (
        detect_structured_plan_objective(
            "trabaja durante 10 minutos en este objetivo: lee notas.txt"
        )
        is None
    )


def test_structured_entry_detection_captures_multiline_objective() -> None:
    """The typed E2E objective spans several lines and ends with a trailing
    "trabaja en segundo plano" order that must not derail the routing."""
    prompt = (
        "Planifica y ejecuta este objetivo:\n\n"
        "Lee los archivos C:\\AI\\Atlas\\prueba_atlas_A.txt y\n"
        "C:\\AI\\Atlas\\prueba_atlas_B.txt.\n\n"
        "Compara su contenido y crea un resumen conjunto\n"
        "explicando qué principio aporta cada archivo,\n"
        "cómo se complementan y cuáles son las tres reglas\n"
        "principales que debería seguir Atlas.\n\n"
        "Guarda el resultado en\n"
        "C:\\AI\\Atlas\\resultado_prueba_autonomia.txt.\n\n"
        "Trabaja en segundo plano y utiliza un modelo local\n"
        "cuando sea suficiente."
    )

    objective = detect_structured_plan_objective(prompt)

    assert objective is not None
    assert "prueba_atlas_A.txt" in objective
    assert "prueba_atlas_B.txt" in objective
    assert "resultado_prueba_autonomia.txt" in objective
    assert "Compara su contenido" in objective
    # The structured entry wins over the trailing background-work order.
    from core.conversational_autonomy import detect_autonomous_goal

    request = detect_autonomous_goal(prompt)
    assert request is None or "Lee los archivos" in request.objective


def test_converted_step_retries_through_scheduler_reliability() -> None:
    from core.execution_plan_task_adapter import execution_plan_to_task_specs
    from core.execution_retry import RetryPolicy

    flaky = ExecutionStep(
        "flaky",
        "Lectura inestable",
        "read_file",
        arguments={"path": "a.txt"},
        retry_policy=RetryPolicy(max_attempts=3),
    )
    plan = ExecutionPlan(
        goal="objetivo con reintento",
        ordered_steps=(flaky,),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
        status="planned",
    )
    specs = execution_plan_to_task_specs(plan, validator=ExecutionPlanValidator())

    calls: list[str] = []

    def executor(task, payload):
        calls.append(task.task_id)
        if len(calls) < 3:
            return TaskOutcome.fail("TRANSIENT_ERROR: backend no disponible")
        return TaskOutcome.succeed("contenido")

    class FakeClock:
        def __init__(self) -> None:
            self.now = 1_000_000.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

    clock = FakeClock()
    scheduler = AsyncTaskScheduler(
        executor,
        clock=clock,
        retry_backoff_seconds=(5.0, 5.0),
    )
    scheduler.submit_goal("objetivo con reintento", list(specs))
    scheduler.run_ready()
    assert calls == ["flaky"]
    clock.advance(6.0)
    scheduler.run_ready()
    assert calls == ["flaky", "flaky"]
    clock.advance(6.0)
    scheduler.run_ready()

    assert calls == ["flaky", "flaky", "flaky"]
    assert scheduler.task("flaky").status is TaskStatus.DONE

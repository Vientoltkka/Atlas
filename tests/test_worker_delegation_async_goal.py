"""Async goal E2E with dynamic worker delegation (fake deterministic workers)."""

from __future__ import annotations

from types import SimpleNamespace

from bootstrap.bootstrap import Bootstrap
from core.async_task_scheduler import (
    AsyncTaskScheduler,
    JsonGoalTaskStore,
    TaskStatus,
    ToolTaskExecutor,
)
from core.multi_task_goal import detect_multi_task_goal
from core.orchestrator import AtlasOrchestrator
from core.router import Router
from core.worker_delegation import DynamicWorkerDelegator
from memory.conversation import ConversationMemory
from tools.filesystem.read_file_tool import ReadFileTool
from tools.filesystem.write_file_tool import WriteFileTool
from tools.registry import ToolRegistry


GOAL_PROMPT = (
    "Lee a.txt, resume el contenido y guarda el resumen en b.txt, "
    "y además lee notas.txt"
)


class CountingReadFileTool(ReadFileTool):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, context):
        self.calls.append(str(context.parameters.get("path")))
        return super().execute(context)


class FakeWorker:
    def __init__(
        self,
        worker_id: str,
        *,
        tier: int,
        available: bool = True,
        error: str | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.tier = tier
        self._available = available
        self._error = error
        self.calls: list[str] = []

    @property
    def attempt_count(self) -> int:
        return len(self.calls)

    def available(self) -> bool:
        return self._available

    def supports(self, task_kind: str) -> bool:
        return task_kind == "transform"

    def execute(self, instruction: str) -> str:
        self.calls.append(instruction)
        if self._error is not None:
            raise RuntimeError(self._error)
        return f"Resumen por {self.worker_id}: {instruction[-40:]}"


class ChatAgentFake:
    name = "chat"
    description = "fake chat"

    def run(self, model, messages):
        return f"respuesta de chat: {messages[-1]['content']}"


def _scheduler(tmp_path, local: FakeWorker, gemini: FakeWorker):
    tool_registry = ToolRegistry()
    read_tool = CountingReadFileTool()
    tool_registry.register(read_tool)
    tool_registry.register(WriteFileTool())
    runner = Bootstrap.build_single_tool_runner(tool_registry=tool_registry)
    delegator = DynamicWorkerDelegator((local, gemini))
    executor = ToolTaskExecutor(runner, worker_delegator=delegator)
    scheduler = AsyncTaskScheduler(
        executor,
        store=JsonGoalTaskStore(tmp_path / "goal_store"),
    )
    executor.bind_result_lookup(scheduler.task)
    return scheduler, read_tool


def _orchestrator(scheduler) -> AtlasOrchestrator:
    return AtlasOrchestrator(
        planner=SimpleNamespace(
            create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt),
        ),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=SimpleNamespace(get=lambda name: ChatAgentFake() if name == "chat" else None),
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        async_task_scheduler=scheduler,
    )


def test_goal_delegates_transform_to_local_and_waits_only_for_write(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("contenido fuente", encoding="utf-8")
    (tmp_path / "notas.txt").write_text("notas independientes", encoding="utf-8")
    local = FakeWorker("LOCAL", tier=0)
    gemini = FakeWorker("GEMINI", tier=1)
    scheduler, read_tool = _scheduler(tmp_path, local, gemini)
    orchestrator = _orchestrator(scheduler)

    assert detect_multi_task_goal(GOAL_PROMPT) is not None
    orchestrator.process_prompt(GOAL_PROMPT, confirm=lambda _p: "")

    pending = scheduler.pending_approvals()
    assert len(pending) == 1
    approval = pending[0]
    state = scheduler.goal(approval.goal_id)
    assert state.tasks["read_source"].status is TaskStatus.DONE
    assert state.tasks["summarize_source"].status is TaskStatus.DONE
    assert state.tasks["read_extra_1"].status is TaskStatus.DONE
    assert state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
    assert sorted(read_tool.calls) == ["a.txt", "notas.txt"]
    assert local.attempt_count == 1
    assert gemini.calls == []
    transform_metadata = state.tasks["summarize_source"].metadata
    assert transform_metadata["final_worker"] == "LOCAL"
    assert transform_metadata["preferred_worker"] == "LOCAL"
    assert not (tmp_path / "b.txt").exists()

    orchestrator.process_prompt("sí", confirm=lambda _p: "")

    assert scheduler.goal_finished(approval.goal_id)
    assert state.tasks["write_target"].status is TaskStatus.DONE
    assert sorted(read_tool.calls) == ["a.txt", "notas.txt"]
    assert local.attempt_count == 1
    assert gemini.calls == []
    written = (tmp_path / "b.txt").read_text(encoding="utf-8")
    assert written.startswith("Resumen por LOCAL:")


def test_worker_fallback_keeps_the_goal_alive_in_a_real_goal(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("contenido fuente", encoding="utf-8")
    local = FakeWorker("LOCAL", tier=0, error="ollama no responde")
    gemini = FakeWorker("GEMINI", tier=1)
    scheduler, _read_tool = _scheduler(tmp_path, local, gemini)
    orchestrator = _orchestrator(scheduler)

    orchestrator.process_prompt(
        "Lee a.txt, resume el contenido y guarda el resumen en b.txt",
        confirm=lambda _p: "",
    )
    approval = scheduler.pending_approvals()[0]
    state = scheduler.goal(approval.goal_id)

    assert state.tasks["summarize_source"].status is TaskStatus.DONE
    assert local.attempt_count == 1
    assert gemini.attempt_count == 1
    metadata = state.tasks["summarize_source"].metadata
    assert metadata["final_worker"] == "GEMINI"
    assert metadata["preferred_worker"] == "LOCAL"
    assert [
        record["worker_id"] for record in metadata["attempted_workers"]
    ] == ["LOCAL", "GEMINI"]
    assert "ollama no responde" in metadata["attempted_workers"][0]["error"]

    orchestrator.process_prompt("sí", confirm=lambda _p: "")
    assert scheduler.goal_finished(approval.goal_id)
    assert (tmp_path / "b.txt").read_text(encoding="utf-8").startswith(
        "Resumen por GEMINI:"
    )


def test_tool_tasks_never_reach_the_delegator(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("contenido fuente", encoding="utf-8")
    local = FakeWorker("LOCAL", tier=0)
    gemini = FakeWorker("GEMINI", tier=1)
    scheduler, read_tool = _scheduler(tmp_path, local, gemini)

    goal_id = scheduler.submit_goal(
        "escritura protegida",
        [
            {
                "task_id": "write_target",
                "description": "guardar",
                "payload": {
                    "tool": "write_file",
                    "arguments": {"path": "b.txt", "content": "texto"},
                },
            },
            {
                "task_id": "read_source",
                "description": "leer",
                "payload": {"tool": "read_file", "arguments": {"path": "a.txt"}},
            },
        ],
    )

    scheduler.run_ready()
    state = scheduler.goal(goal_id)
    assert state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
    assert state.tasks["read_source"].status is TaskStatus.DONE
    assert local.calls == []
    assert gemini.calls == []
    assert state.tasks["read_source"].metadata is None
    assert state.tasks["write_target"].metadata is None

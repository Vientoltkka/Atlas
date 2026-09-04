"""Conversational async goal execution: multi-task routing E2E coverage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
from memory.conversation import ConversationMemory
from tools.filesystem.read_file_tool import ReadFileTool
from tools.filesystem.write_file_tool import WriteFileTool
from tools.registry import ToolRegistry


CANONICAL_PROMPT = (
    "Lee README.md, resume el contenido y guarda el resumen en "
    "atlas_async_test_summary.txt"
)


class CountingReadFileTool(ReadFileTool):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, context):
        self.calls.append(str(context.parameters.get("path")))
        return super().execute(context)


class ChatAgentFake:
    name = "chat"
    description = "fake chat"

    def run(self, model, messages):
        return f"respuesta de chat: {messages[-1]['content']}"


def _build_scheduler(
    tmp_path,
    *,
    summarizer=None,
) -> tuple[AsyncTaskScheduler, CountingReadFileTool, list[str]]:
    """Real SingleToolRunner + scheduler, mirroring the production wiring."""
    tool_registry = ToolRegistry()
    read_tool = CountingReadFileTool()
    tool_registry.register(read_tool)
    tool_registry.register(WriteFileTool())

    transform_calls: list[str] = []

    def default_summarizer(text: str) -> str:
        transform_calls.append(text)
        return f"Resumen breve: {text[:40]}"

    runner = Bootstrap.build_single_tool_runner(tool_registry=tool_registry)
    executor = ToolTaskExecutor(
        runner,
        model_transformer=summarizer or default_summarizer,
    )
    scheduler = AsyncTaskScheduler(
        executor,
        store=JsonGoalTaskStore(tmp_path / "goal_store"),
    )
    executor.bind_result_lookup(scheduler.task)
    return scheduler, read_tool, transform_calls


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


def test_multi_task_goal_executes_and_pauses_only_write_until_confirmation(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text(
        "Atlas planifica objetivos en tareas independientes.",
        encoding="utf-8",
    )
    scheduler, read_tool, transform_calls = _build_scheduler(tmp_path)
    orchestrator = _orchestrator(scheduler)

    response = orchestrator.process_prompt(CANONICAL_PROMPT, confirm=lambda _p: "")

    pending = scheduler.pending_approvals()
    assert len(pending) == 1
    approval = pending[0]
    state = scheduler.goal(approval.goal_id)
    assert state.tasks["read_source"].status is TaskStatus.DONE
    assert state.tasks["summarize_source"].status is TaskStatus.DONE
    assert state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
    assert read_tool.calls == ["README.md"]
    assert len(transform_calls) == 1
    assert "atlas_async_test_summary.txt" in response
    assert "pendiente de tu confirmación" in response
    assert "Responde sí" in response
    assert not (tmp_path / "atlas_async_test_summary.txt").exists()

    resumed = orchestrator.process_prompt("sí", confirm=lambda _p: "")

    assert scheduler.goal_finished(approval.goal_id)
    assert state.tasks["write_target"].status is TaskStatus.DONE
    assert read_tool.calls == ["README.md"]
    assert len(transform_calls) == 1
    written = (tmp_path / "atlas_async_test_summary.txt").read_text(encoding="utf-8")
    assert written.startswith("Resumen breve:")
    assert "Hecho" in resumed
    assert "README.md" not in resumed or "hecho" in resumed
    assert scheduler.pending_approvals() == []


def test_independent_task_completes_while_write_waits_for_approval(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido principal", encoding="utf-8")
    (tmp_path / "notas.txt").write_text("notas independientes", encoding="utf-8")
    scheduler, read_tool, transform_calls = _build_scheduler(tmp_path)
    orchestrator = _orchestrator(scheduler)

    response = orchestrator.process_prompt(
        "Lee README.md, resume el contenido y guarda el resumen en resumen.txt, "
        "y además lee notas.txt",
        confirm=lambda _p: "",
    )

    pending = scheduler.pending_approvals()
    assert len(pending) == 1
    approval = pending[0]
    state = scheduler.goal(approval.goal_id)
    assert state.tasks["read_source"].status is TaskStatus.DONE
    assert state.tasks["summarize_source"].status is TaskStatus.DONE
    assert state.tasks["read_extra_1"].status is TaskStatus.DONE
    assert state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
    assert sorted(read_tool.calls) == ["README.md", "notas.txt"]
    assert "Leer notas.txt: hecho" in response

    orchestrator.process_prompt("sí", confirm=lambda _p: "")

    assert scheduler.goal_finished(approval.goal_id)
    assert (tmp_path / "resumen.txt").exists()
    assert sorted(read_tool.calls) == ["README.md", "notas.txt"]
    assert len(transform_calls) == 1


def test_denial_blocks_only_the_write_task_and_writes_nothing(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido", encoding="utf-8")
    scheduler, _read_tool, transform_calls = _build_scheduler(tmp_path)
    orchestrator = _orchestrator(scheduler)

    orchestrator.process_prompt(CANONICAL_PROMPT, confirm=lambda _p: "")
    approval = scheduler.pending_approvals()[0]

    response = orchestrator.process_prompt("no", confirm=lambda _p: "")

    state = scheduler.goal(approval.goal_id)
    assert state.tasks["write_target"].status is TaskStatus.BLOCKED
    assert state.tasks["read_source"].status is TaskStatus.DONE
    assert not (tmp_path / "atlas_async_test_summary.txt").exists()
    assert scheduler.pending_approvals() == []
    assert "cancelada" in response


def test_simple_prompts_keep_the_existing_conversational_flow(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    scheduler, _read_tool, transform_calls = _build_scheduler(tmp_path)
    orchestrator = _orchestrator(scheduler)

    chat = orchestrator.process_prompt("hola", confirm=lambda _p: "")
    single_read = orchestrator.process_prompt(
        "Lee README.md",
        confirm=lambda _p: "",
    )

    assert chat.startswith("respuesta de chat")
    assert single_read.startswith("respuesta de chat")
    assert "Objetivo en marcha" not in chat
    assert "Objetivo en marcha" not in single_read
    assert transform_calls == []
    assert scheduler.pending_approvals() == []


def test_confirmation_without_pending_approvals_falls_back_to_normal_flow(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    scheduler, _read_tool, _transform_calls = _build_scheduler(tmp_path)
    orchestrator = _orchestrator(scheduler)

    response = orchestrator.process_prompt("sí", confirm=lambda _p: "")

    assert "Objetivo" not in response


@pytest.mark.parametrize(
    "prompt",
    [
        "¿Cuál es la capital de Francia?",
        "Abre calculadora",
        "Lee README.md",
        "Explícame qué es Docker",
        "¿Puedes leer a.txt y guardarlo en b.txt?",
        "no leas notas.txt ni guardes nada en salida.txt",
        r"Lee C:\proyecto\README.md y guarda el resumen en salida.txt",
        "Lee README.md y guárdalo en README.md",
        "Lee a.txt y b.txt, compáralos",
        "Compara a.txt y b.txt y guarda el resultado en comparacion.txt",
    ],
)
def test_detector_rejects_non_multi_task_prompts(prompt: str) -> None:
    assert detect_multi_task_goal(prompt) is None


def test_detector_builds_chained_tool_and_transform_tasks() -> None:
    goal = detect_multi_task_goal(
        "Lee README.md, resume el contenido y guarda el resumen en resumen.txt",
    )

    assert goal is not None
    task_ids = [task["task_id"] for task in goal.tasks]
    assert task_ids == ["read_source", "summarize_source", "write_target"]
    read, summarize, write = goal.tasks
    assert read["dependencies"] == []
    assert read["payload"] == {
        "kind": "tool",
        "tool": "read_file",
        "arguments": {"path": "README.md"},
    }
    assert summarize["dependencies"] == ["read_source"]
    assert summarize["payload"]["kind"] == "transform"
    assert summarize["payload"]["input_task"] == "read_source"
    assert write["dependencies"] == ["summarize_source"]
    assert write["payload"]["content_task"] == "summarize_source"
    assert all(task["requires_approval"] is False for task in goal.tasks)


def test_detector_keeps_policy_ownership_of_write_without_summary() -> None:
    goal = detect_multi_task_goal(
        "Lee README.md y guarda el contenido en copia.txt",
    )

    assert goal is not None
    assert [task["task_id"] for task in goal.tasks] == ["read_source", "write_target"]
    write = goal.tasks[-1]
    assert write["payload"]["content_task"] == "read_source"


def test_detector_builds_multi_source_transform_chain() -> None:
    goal = detect_multi_task_goal(
        "Lee a.txt y b.txt, compáralos y guarda las diferencias en resultado.txt",
    )

    assert goal is not None
    assert [task["task_id"] for task in goal.tasks] == [
        "read_source",
        "read_extra_1",
        "transform_content",
        "write_target",
    ]
    transform = goal.tasks[2]
    assert transform["dependencies"] == ["read_source", "read_extra_1"]
    assert transform["payload"]["kind"] == "transform"
    assert transform["payload"]["input_tasks"] == ["read_source", "read_extra_1"]
    assert "{input}" in transform["payload"]["instruction"]
    write = goal.tasks[3]
    assert write["dependencies"] == ["transform_content"]
    assert write["payload"]["content_task"] == "transform_content"


def test_detector_builds_single_file_transform_chain() -> None:
    goal = detect_multi_task_goal(
        "Lee notas.txt, ordénalo y guarda el resultado en ordenado.txt",
    )

    assert goal is not None
    assert [task["task_id"] for task in goal.tasks] == [
        "read_source",
        "transform_content",
        "write_target",
    ]
    transform = goal.tasks[1]
    assert transform["dependencies"] == ["read_source"]
    assert transform["payload"]["input_task"] == "read_source"
    assert "ordena" in transform["payload"]["instruction"] or "Ordena" in transform["payload"]["instruction"]
    write = goal.tasks[2]
    assert write["dependencies"] == ["transform_content"]


def test_detector_builds_independent_safe_read_bundle_without_write() -> None:
    goal = detect_multi_task_goal(
        "Lee README.md y también lee notas.txt",
    )

    assert goal is not None
    assert [task["task_id"] for task in goal.tasks] == ["read_source", "read_extra_1"]
    assert all(task["payload"]["tool"] == "read_file" for task in goal.tasks)
    assert all(task["dependencies"] == [] for task in goal.tasks)


def test_multi_source_compare_goal_executes_and_waits_only_for_write(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("alpha\nshared\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\nshared\n", encoding="utf-8")
    scheduler, read_tool, transform_calls = _build_scheduler(tmp_path)
    orchestrator = _orchestrator(scheduler)

    response = orchestrator.process_prompt(
        "Lee a.txt y b.txt, compáralos y guarda las diferencias en resultado.txt",
        confirm=lambda _p: "",
    )

    pending = scheduler.pending_approvals()
    assert len(pending) == 1
    approval = pending[0]
    state = scheduler.goal(approval.goal_id)
    assert state.tasks["read_source"].status is TaskStatus.DONE
    assert state.tasks["read_extra_1"].status is TaskStatus.DONE
    assert state.tasks["transform_content"].status is TaskStatus.DONE
    assert state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
    assert sorted(read_tool.calls) == ["a.txt", "b.txt"]
    assert len(transform_calls) == 1
    assert "[1] read_source" in transform_calls[0]
    assert "alpha" in transform_calls[0]
    assert "[2] read_extra_1" in transform_calls[0]
    assert "beta" in transform_calls[0]
    assert "resultado.txt" in response
    assert not (tmp_path / "resultado.txt").exists()

    orchestrator.process_prompt("sí", confirm=lambda _p: "")

    assert scheduler.goal_finished(approval.goal_id)
    assert state.tasks["write_target"].status is TaskStatus.DONE
    assert sorted(read_tool.calls) == ["a.txt", "b.txt"]
    assert len(transform_calls) == 1
    written = (tmp_path / "resultado.txt").read_text(encoding="utf-8")
    assert written.startswith("Resumen breve: Compara")
    assert scheduler.pending_approvals() == []


def test_single_file_transform_goal_waits_for_write_approval(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notas.txt").write_text("c\na\nb\n", encoding="utf-8")
    scheduler, read_tool, transform_calls = _build_scheduler(tmp_path)
    orchestrator = _orchestrator(scheduler)

    orchestrator.process_prompt(
        "Lee notas.txt, ordénalo y guarda el resultado en ordenado.txt",
        confirm=lambda _p: "",
    )

    pending = scheduler.pending_approvals()
    assert len(pending) == 1
    approval = pending[0]
    state = scheduler.goal(approval.goal_id)
    assert state.tasks["read_source"].status is TaskStatus.DONE
    assert state.tasks["transform_content"].status is TaskStatus.DONE
    assert state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
    assert read_tool.calls == ["notas.txt"]
    assert len(transform_calls) == 1
    assert not (tmp_path / "ordenado.txt").exists()

    orchestrator.process_prompt("sí", confirm=lambda _p: "")

    assert scheduler.goal_finished(approval.goal_id)
    assert state.tasks["write_target"].status is TaskStatus.DONE
    assert read_tool.calls == ["notas.txt"]
    assert len(transform_calls) == 1
    assert (tmp_path / "ordenado.txt").exists()


def test_independent_read_completes_before_multi_source_write_approval(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("gamma independiente\n", encoding="utf-8")
    scheduler, read_tool, transform_calls = _build_scheduler(tmp_path)
    orchestrator = _orchestrator(scheduler)

    response = orchestrator.process_prompt(
        "Lee a.txt y b.txt, compáralos y guarda las diferencias en resultado.txt, "
        "y además lee c.txt",
        confirm=lambda _p: "",
    )

    pending = scheduler.pending_approvals()
    assert len(pending) == 1
    approval = pending[0]
    state = scheduler.goal(approval.goal_id)
    assert state.tasks["read_source"].status is TaskStatus.DONE
    assert state.tasks["read_extra_1"].status is TaskStatus.DONE
    assert state.tasks["read_extra_2"].status is TaskStatus.DONE
    assert state.tasks["transform_content"].status is TaskStatus.DONE
    assert state.tasks["write_target"].status is TaskStatus.WAITING_APPROVAL
    assert sorted(read_tool.calls) == ["a.txt", "b.txt", "c.txt"]
    assert "Leer c.txt: hecho" in response
    assert len(transform_calls) == 1
    assert "[1] read_source" in transform_calls[0]
    assert "[2] read_extra_1" in transform_calls[0]
    assert "c.txt" not in transform_calls[0]

    orchestrator.process_prompt("sí", confirm=lambda _p: "")

    assert scheduler.goal_finished(approval.goal_id)
    assert (tmp_path / "resultado.txt").exists()
    assert sorted(read_tool.calls) == ["a.txt", "b.txt", "c.txt"]
    assert len(transform_calls) == 1


def test_two_independent_safe_reads_complete_without_any_approval(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("contenido principal", encoding="utf-8")
    (tmp_path / "notas.txt").write_text("notas sueltas", encoding="utf-8")
    scheduler, read_tool, transform_calls = _build_scheduler(tmp_path)
    orchestrator = _orchestrator(scheduler)

    response = orchestrator.process_prompt(
        "Lee README.md y también lee notas.txt",
        confirm=lambda _p: "",
    )

    assert scheduler.pending_approvals() == []
    assert scheduler._goals
    goal_id = next(iter(scheduler._goals))
    assert scheduler.goal_finished(goal_id)
    state = scheduler.goal(goal_id)
    assert state.tasks["read_source"].status is TaskStatus.DONE
    assert state.tasks["read_extra_1"].status is TaskStatus.DONE
    assert sorted(read_tool.calls) == ["README.md", "notas.txt"]
    assert transform_calls == []
    assert "Leer README.md: hecho" in response
    assert "Leer notas.txt: hecho" in response

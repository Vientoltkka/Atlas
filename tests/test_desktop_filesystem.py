from bootstrap.bootstrap import Bootstrap
from core.orchestrator import AtlasOrchestrator
from use_cases.execution_conversation import ExecutionConversationController
from pathlib import Path

import pytest

from bootstrap.bootstrap import Bootstrap
from core.orchestrator import AtlasOrchestrator
from use_cases.execution_conversation import ExecutionConversationController

from tools.desktop.desktop_tools import (
    CopyPathTool,
    CreateFolderTool,
    DeletePathTool,
    MovePathTool,
    OpenFileTool,
    OpenFolderTool,
    RenamePathTool,
)
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext
from use_cases.desktop_interaction import DesktopInteractionUseCase


class FileSystemController:
    def create_folder(self, path: Path) -> bool:
        if path.exists():
            return False
        path.mkdir()
        return True

    def copy_path(self, source: Path, destination: Path) -> None:
        if source.is_dir():
            import shutil
            shutil.copytree(source, destination)
        else:
            import shutil
            shutil.copy2(source, destination)

    def move_path(self, source: Path, destination: Path) -> None:
        import shutil
        shutil.move(str(source), str(destination))

    def rename_path(self, source: Path, destination: Path) -> None:
        source.rename(destination)

    def delete_path(self, path: Path) -> None:
        import shutil
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    def open_file(self, path: Path, application: str | None = None) -> None:
        self.opened = ("file", path, application)

    def open_folder(self, path: Path) -> None:
        self.opened = ("folder", path)


def run(tool, parameters):
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry)
    authorization = executor.authorize(tool.name)
    return executor.execute(tool.name, ToolContext(parameters=parameters), authorization=authorization)


def test_create_folder_and_existing_folder(tmp_path: Path) -> None:
    controller = FileSystemController()
    tool = CreateFolderTool(controller)
    path = tmp_path / "new folder"
    assert run(tool, {"path": str(path)}) == {"operation": "create_folder", "path": str(path), "created": True}
    assert run(tool, {"path": str(path)})["created"] is False


def test_copy_file_and_folder_with_spaces(tmp_path: Path) -> None:
    controller = FileSystemController()
    source = tmp_path / "source file.txt"
    source.write_text("Atlas", encoding="utf-8")
    copied = tmp_path / "copied file.txt"
    result = run(CopyPathTool(controller), {"source_path": str(source), "destination_path": str(copied)})
    assert result["kind"] == "file" and copied.read_text(encoding="utf-8") == "Atlas"
    folder = tmp_path / "source folder"
    folder.mkdir()
    (folder / "nested.txt").write_text("nested", encoding="utf-8")
    destination = tmp_path / "copied folder"
    assert run(CopyPathTool(controller), {"source_path": str(folder), "destination_path": str(destination)})["kind"] == "folder"
    assert (destination / "nested.txt").read_text(encoding="utf-8") == "nested"


def test_copy_rejects_missing_source_and_existing_destination(tmp_path: Path) -> None:
    controller = FileSystemController()
    tool = CopyPathTool(controller)
    with pytest.raises(FileNotFoundError):
        run(tool, {"source_path": str(tmp_path / "missing"), "destination_path": str(tmp_path / "new")})
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("source", encoding="utf-8")
    destination.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(tool, {"source_path": str(source), "destination_path": str(destination)})
    assert destination.read_text(encoding="utf-8") == "keep"


def test_move_rename_and_delete(tmp_path: Path) -> None:
    controller = FileSystemController()
    source = tmp_path / "source.txt"
    source.write_text("Atlas", encoding="utf-8")
    moved = tmp_path / "moved.txt"
    run(MovePathTool(controller), {"source_path": str(source), "destination_path": str(moved)})
    assert not source.exists() and moved.exists()
    renamed = run(RenamePathTool(controller), {"source_path": str(moved), "new_name": "renamed.txt"})
    path = Path(renamed["destination"])
    assert path.exists()
    assert run(DeletePathTool(controller), {"path": str(path)})["operation"] == "delete"
    assert not path.exists()


class RoutingExecutor:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, name, context, **_kwargs):
        self.calls.append((name, context.parameters))
        if name == "desktop.create_folder":
            return {"created": True, "path": context.parameters["path"]}
        if name == "desktop.delete_path":
            return {"kind": "file", "path": context.parameters["path"]}
        return {"kind": "file", "source": context.parameters["source_path"], "destination": context.parameters.get("destination_path", context.parameters["source_path"])}


@pytest.mark.parametrize(("prompt", "tool"), [
    (r"crea la carpeta C:\Temp\folder", "desktop.create_folder"),
    (r"copia C:\Temp\a.txt a C:\Temp\b.txt", "desktop.copy_path"),
    (r"mueve C:\Temp\a.txt a C:\Temp\b.txt", "desktop.move_path"),
    (r"renombra C:\Temp\a.txt a b.txt", "desktop.rename_path"),
    (r"elimina C:\Temp\a.txt", "desktop.delete_path"),
])
def test_conversation_routes_every_filesystem_operation(prompt: str, tool: str) -> None:
    executor = RoutingExecutor()
    DesktopInteractionUseCase(executor).execute(prompt, confirm=lambda _: "s")
    assert executor.calls[0][0] == tool


def test_conversation_cancellation_makes_no_filesystem_call() -> None:
    executor = RoutingExecutor()
    assert DesktopInteractionUseCase(executor).execute(r"crea la carpeta C:\Temp\folder", confirm=lambda _: "n") == "Acción cancelada."
    assert executor.calls == []


def test_open_file_and_folder_regression(tmp_path: Path) -> None:
    controller = FileSystemController()
    folder = tmp_path / "folder"
    folder.mkdir()
    file = folder / "file.txt"
    file.write_text("Atlas", encoding="utf-8")
    OpenFolderTool(controller).execute(ToolContext(parameters={"path": str(folder)}))

def _filesystem_orchestrator(controller: FileSystemController) -> tuple[AtlasOrchestrator, ExecutionConversationController]:
    registry = ToolRegistry()
    registry.register(CreateFolderTool(controller))
    executor = ToolExecutor(registry)
    conversation = ExecutionConversationController(Bootstrap.build_execution_coordinator(tool_registry=registry, executor=executor))
    return AtlasOrchestrator(planner=None, router=None, model_manager=None, memory=None, registry=None, write_file=None, desktop_interaction=DesktopInteractionUseCase(executor), execution_conversation=conversation), conversation


def test_filesystem_conversation_keeps_create_pending_then_confirms(tmp_path: Path) -> None:
    controller = FileSystemController()
    orchestrator, conversation = _filesystem_orchestrator(controller)
    path = tmp_path / "pending folder"
    prompt = f"crea la carpeta {path}"
    pending = orchestrator.process_prompt(prompt, confirm=lambda _: "")
    assert "Voy a crear la carpeta" in pending
    assert str(path) in pending
    assert not path.exists()
    assert conversation.pending_confirmation_id is not None
    assert conversation.pending_confirmation_context is not None
    assert conversation.pending_confirmation_context.original_text == prompt
    assert "'created': True" in orchestrator.process_prompt("sí", confirm=lambda _: "")
    assert path.is_dir()
    assert conversation.pending_confirmation_id is None


def test_filesystem_conversation_rejects_without_changes(tmp_path: Path) -> None:
    controller = FileSystemController()
    orchestrator, conversation = _filesystem_orchestrator(controller)
    path = tmp_path / "cancelled folder"
    orchestrator.process_prompt(f"crea la carpeta {path}", confirm=lambda _: "")
    assert "cancelada" in orchestrator.process_prompt("no", confirm=lambda _: "").lower()
    assert not path.exists()
    assert conversation.pending_confirmation_id is None
    assert controller.opened == ("folder", folder)
    OpenFileTool(controller).execute(ToolContext(parameters={"path": str(file)}))
    assert controller.opened == ("file", file, None)

def _filesystem_orchestrator(controller: FileSystemController) -> tuple[AtlasOrchestrator, ExecutionConversationController]:
    registry = ToolRegistry()
    registry.register(CreateFolderTool(controller))
    executor = ToolExecutor(registry)
    conversation = ExecutionConversationController(Bootstrap.build_execution_coordinator(tool_registry=registry, executor=executor))
    return AtlasOrchestrator(planner=None, router=None, model_manager=None, memory=None, registry=None, write_file=None, desktop_interaction=DesktopInteractionUseCase(executor), execution_conversation=conversation), conversation


def test_filesystem_conversation_keeps_create_pending_then_confirms(tmp_path: Path) -> None:
    controller = FileSystemController()
    orchestrator, conversation = _filesystem_orchestrator(controller)
    path = tmp_path / "pending folder"
    prompt = f"crea la carpeta {path}"

    pending = orchestrator.process_prompt(prompt, confirm=lambda _: "")

    assert "Voy a crear la carpeta" in pending
    assert str(path) in pending
    assert not path.exists()
    assert conversation.pending_confirmation_id is not None
    assert conversation.pending_confirmation_context is not None
    assert conversation.pending_confirmation_context.original_text == prompt
    assert "'created': True" in orchestrator.process_prompt("sí", confirm=lambda _: "")
    assert path.is_dir()
    assert conversation.pending_confirmation_id is None


def test_filesystem_conversation_rejects_without_changes(tmp_path: Path) -> None:
    controller = FileSystemController()
    orchestrator, conversation = _filesystem_orchestrator(controller)
    path = tmp_path / "cancelled folder"

    orchestrator.process_prompt(f"crea la carpeta {path}", confirm=lambda _: "")
    assert "cancelada" in orchestrator.process_prompt("no", confirm=lambda _: "").lower()
    assert not path.exists()
    assert conversation.pending_confirmation_id is None

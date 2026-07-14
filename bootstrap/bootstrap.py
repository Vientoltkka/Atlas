"""Bootstrap module for Atlas."""

from agents.chat_agent import ChatAgent
from agents.coding_agent import CodingAgent
from agents.project_agent import ProjectAgent
from agents.registry import AgentRegistry

from core.model_manager import ModelManager
from core.orchestrator import AtlasOrchestrator
from core.planner import Planner
from core.router import Router

from memory.conversation import ConversationMemory

from models.prompt_client import PromptClient

from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

from tools.filesystem.read_file_tool import ReadFileTool
from tools.filesystem.write_file_tool import WriteFileTool
from tools.filesystem.list_directory_tool import ListDirectoryTool
from tools.project.tree_tool import TreeTool
from tools.desktop.desktop_tools import (
    ActivateWindowTool,
    BringWindowToFrontTool,
    CaptureScreenshotTool,
    CloseWindowTool,
    DoubleClickTool,
    GetCursorPositionTool,
    GetScreenSizeTool,
    GetWindowRectTool,
    LeftClickTool,
    ListWindowsTool,
    MaximizeWindowTool,
    MinimizeWindowTool,
    MoveCursorTool,
    MoveResizeWindowTool,
    MoveWindowTool,
    OpenApplicationTool,
    OpenFileTool,
    OpenFolderTool,
    PressHotkeyTool,
    ResizeWindowTool,
    RestoreWindowTool,
    RightClickTool,
    SaveFileTool,
    ScrollVerticalTool,
    TypeTextTool,
)

from use_cases.read_file import ReadFileUseCase
from use_cases.write_file import WriteFileUseCase
from use_cases.list_python_files import ListPythonFilesUseCase
from use_cases.read_project import ReadProjectUseCase
from use_cases.read_project_index import ReadProjectIndexUseCase
from use_cases.find_project_file import FindProjectFileUseCase
from use_cases.resolve_project_dependencies import ResolveProjectDependenciesUseCase
from use_cases.build_architecture_graph import BuildArchitectureGraphUseCase
from use_cases.query_architecture_graph import QueryArchitectureGraphUseCase
from use_cases.correction_interaction import CorrectionInteractionUseCase
from use_cases.plan_refactoring import PlanRefactoringUseCase
from use_cases.rename_symbol import RenameSymbolUseCase
from use_cases.refactoring_interaction import RefactoringInteractionUseCase
from use_cases.desktop_interaction import DesktopInteractionUseCase


class Bootstrap:
    """Build the Atlas application."""

    @staticmethod
    def build() -> AtlasOrchestrator:

        # -----------------------
        # Core
        # -----------------------

        planner = Planner()
        router = Router()
        model_manager = ModelManager()
        memory = ConversationMemory()

        prompt_client = PromptClient()

        # -----------------------
        # Tools
        # -----------------------

        tool_registry = ToolRegistry()

        tool_registry.register(ReadFileTool())
        tool_registry.register(WriteFileTool())
        tool_registry.register(ListDirectoryTool())
        tool_registry.register(TreeTool())
        tool_registry.register(OpenApplicationTool())
        tool_registry.register(OpenFolderTool())
        tool_registry.register(OpenFileTool())
        tool_registry.register(TypeTextTool())
        tool_registry.register(PressHotkeyTool())
        tool_registry.register(SaveFileTool())
        tool_registry.register(ActivateWindowTool())
        tool_registry.register(GetScreenSizeTool())
        tool_registry.register(GetCursorPositionTool())
        tool_registry.register(MoveCursorTool())
        tool_registry.register(LeftClickTool())
        tool_registry.register(DoubleClickTool())
        tool_registry.register(RightClickTool())
        tool_registry.register(ScrollVerticalTool())
        tool_registry.register(CaptureScreenshotTool())
        tool_registry.register(ListWindowsTool())
        tool_registry.register(GetWindowRectTool())
        tool_registry.register(BringWindowToFrontTool())
        tool_registry.register(MaximizeWindowTool())
        tool_registry.register(MinimizeWindowTool())
        tool_registry.register(RestoreWindowTool())
        tool_registry.register(MoveWindowTool())
        tool_registry.register(ResizeWindowTool())
        tool_registry.register(MoveResizeWindowTool())
        tool_registry.register(CloseWindowTool())

        tool_executor = ToolExecutor(tool_registry)

        # -----------------------
        # Use Cases
        # -----------------------

        read_file = ReadFileUseCase(
            tool_executor,
        )

        write_file = WriteFileUseCase(
            tool_executor,
        )

        list_python_files = ListPythonFilesUseCase()

        read_project = ReadProjectUseCase(
            list_python_files,
            read_file,
        )

        read_project_index = ReadProjectIndexUseCase()

        find_project_file = FindProjectFileUseCase(
            read_project_index,
        )

        resolve_project_dependencies = ResolveProjectDependenciesUseCase()

        architecture_index = read_project_index.execute(".")
        architecture_graph = BuildArchitectureGraphUseCase().execute(
            architecture_index,
        )
        query_architecture_graph = QueryArchitectureGraphUseCase(
            architecture_graph,
        )
        plan_refactoring = PlanRefactoringUseCase(
            architecture_graph,
            query_architecture_graph,
        )
        rename_symbol = RenameSymbolUseCase(
            plan_refactoring,
        )
        refactoring_interaction = RefactoringInteractionUseCase(
            plan_refactoring,
            rename_symbol,
        )
        correction_interaction = CorrectionInteractionUseCase(
            read_file,
            write_file,
            query_architecture_graph,
            prompt_client,
        )
        desktop_interaction = DesktopInteractionUseCase(
            tool_executor,
        )

        # -----------------------
        # Agents
        # -----------------------

        registry = AgentRegistry()

        registry.register(
            ChatAgent(
                prompt_client,
            )
        )

        registry.register(
            CodingAgent(
                prompt_client,
                read_file,
                write_file,
            )
        )

        registry.register(
            ProjectAgent(
                prompt_client,
                read_project,
                read_file,
                read_project_index,
                find_project_file,
                resolve_project_dependencies,
                query_architecture_graph,
            )
        )

        # -----------------------
        # Orchestrator
        # -----------------------

        return AtlasOrchestrator(
            planner=planner,
            router=router,
            model_manager=model_manager,
            memory=memory,
            registry=registry,
            write_file=write_file,
            refactoring_interaction=refactoring_interaction,
            correction_interaction=correction_interaction,
            desktop_interaction=desktop_interaction,
        )

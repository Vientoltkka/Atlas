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

from use_cases.analyze_project import AnalyzeProjectUseCase
from use_cases.read_file import ReadFileUseCase
from use_cases.write_file import WriteFileUseCase

class Bootstrap:
    """Build the Atlas application."""

    @staticmethod
    def build() -> AtlasOrchestrator:

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

        tool_executor = ToolExecutor(tool_registry)

        # -----------------------
        # Use Cases
        # -----------------------

        analyze_project = AnalyzeProjectUseCase(
            tool_executor,
        )

        read_file = ReadFileUseCase(
            tool_executor,
        )

        write_file = WriteFileUseCase(
            tool_executor,
)

        # -----------------------
        # Agents
        # -----------------------

        registry = AgentRegistry()

        registry.register(
            ChatAgent(prompt_client)
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
                analyze_project,
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
        )
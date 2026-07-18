"""Bootstrap module for Atlas."""

import os
from pathlib import Path

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
from tools.argument_schema import (
    ArgumentField,
    ArgumentSchema,
    ArgumentSchemaRegistry,
    ArgumentValidator,
    require_non_empty,
)
from tools.execution_decision import ExecutionDecisionEngine
from tools.intent_selector import ToolIntentRegistry, ToolSelector
from tools.registry import ToolRegistry
from tools.tool_chain_runner import ToolChainRunner
from tools.tool_proposal_builder import ToolProposalBuilder
from tools.single_tool_runner import SingleToolRunner

from tools.filesystem.read_file_tool import ReadFileTool
from tools.filesystem.write_file_tool import WriteFileTool
from tools.filesystem.list_directory_tool import ListDirectoryTool
from tools.project.tree_tool import TreeTool
from tools.desktop.desktop_tools import (
    ActivateWindowTool,
    BringWindowToFrontTool,
    CaptureScreenshotTool,
    ClearClipboardTool,
    ClipboardHasTextTool,
    CloseApplicationTool,
    CloseWindowTool,
    CopyClipboardTextTool,
    DoubleClickTool,
    GetCursorPositionTool,
    GetForegroundWindowTool,
    GetProcessTool,
    GetScreenSizeTool,
    GetWindowRectTool,
    LeftClickTool,
    IsProcessRunningTool,
    ListProcessesTool,
    ListWindowsTool,
    MaximizeWindowTool,
    MinimizeWindowTool,
    MoveCursorTool,
    MoveResizeWindowTool,
    MoveWindowTool,
    OpenApplicationTool,
    OpenFileTool,
    OpenFolderTool,
    PasteClipboardTool,
    PressHotkeyTool,
    ReadClipboardTextTool,
    ResizeWindowTool,
    RestoreWindowTool,
    RightClickTool,
    SaveFileTool,
    ScrollVerticalTool,
    TerminateProcessTool,
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
from use_cases.permanent_assistant import PermanentAssistantUseCase
from use_cases.action_engine import (
    ActionEngineUseCase,
    PrepareAtlasWorkspaceUseCase,
    RestartApplicationUseCase,
)
from use_cases.verified_text_file import CreateVerifiedTextFileUseCase
from use_cases.wait_engine import WaitEngine
from use_cases.speech_engine import (
    FasterWhisperSpeechToTextProvider,
    SoundDeviceAudioCapture,
    SpeechCaptureSettings,
    SpeechEngineUseCase,
    SpeechInteractionUseCase,
)
from use_cases.speech_output_engine import Pyttsx3SpeechOutputEngine
from use_cases.stt_wake_word_engine import SttWakeWordEngine
from use_cases.voice_conversation import VoiceConversationUseCase
from use_cases.wake_word_engine import (
    OpenWakeWordProvider,
    WakeWordEngine,
    WakeWordInteractionUseCase,
)


class Bootstrap:
    """Build the Atlas application."""

    @staticmethod
    def build_tool_registry() -> ToolRegistry:
        """Build the central registry with the tools available in Atlas."""
        tool_registry = ToolRegistry()

        tool_registry.register(ReadFileTool())
        tool_registry.register(WriteFileTool())
        tool_registry.register(ListDirectoryTool())
        tool_registry.register(TreeTool())
        tool_registry.register(OpenApplicationTool())
        tool_registry.register(ListProcessesTool())
        tool_registry.register(IsProcessRunningTool())
        tool_registry.register(GetProcessTool())
        tool_registry.register(CloseApplicationTool())
        tool_registry.register(TerminateProcessTool())
        tool_registry.register(OpenFolderTool())
        tool_registry.register(OpenFileTool())
        tool_registry.register(TypeTextTool())
        tool_registry.register(CopyClipboardTextTool())
        tool_registry.register(ReadClipboardTextTool())
        tool_registry.register(ClearClipboardTool())
        tool_registry.register(ClipboardHasTextTool())
        tool_registry.register(PasteClipboardTool())
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
        tool_registry.register(GetForegroundWindowTool())
        tool_registry.register(BringWindowToFrontTool())
        tool_registry.register(MaximizeWindowTool())
        tool_registry.register(MinimizeWindowTool())
        tool_registry.register(RestoreWindowTool())
        tool_registry.register(MoveWindowTool())
        tool_registry.register(ResizeWindowTool())
        tool_registry.register(MoveResizeWindowTool())
        tool_registry.register(CloseWindowTool())

        return tool_registry

    @staticmethod
    def build_tool_selector(
        tool_registry: ToolRegistry | None = None,
    ) -> ToolSelector:
        """Build the deterministic selector for supported tool intents."""
        registry = tool_registry or Bootstrap.build_tool_registry()
        intent_registry = ToolIntentRegistry()

        for action, tool_name in (
            ("file.read", "read_file"),
            ("file.write", "write_file"),
            ("directory.list", "list_directory"),
            ("project.tree", "project_tree"),
            ("desktop.application.open", "desktop.open_application"),
            ("desktop.file.open", "desktop.open_file"),
            ("desktop.text.type", "desktop.type_text"),
            ("desktop.hotkey.press", "desktop.press_hotkey"),
            ("desktop.windows.list", "desktop.list_windows"),
        ):
            if registry.exists(tool_name):
                intent_registry.register(action, tool_name)

        return ToolSelector(registry, intent_registry)

    @staticmethod
    def build_argument_schema_registry() -> ArgumentSchemaRegistry:
        """Build argument schemas for supported tool intents."""
        schema_registry = ArgumentSchemaRegistry()

        schema_registry.register(
            ArgumentSchema(
                "file.read",
                (
                    ArgumentField("path", str, required=True, description="File path."),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "file.write",
                (
                    ArgumentField("path", str, required=True, description="File path."),
                    ArgumentField("content", str, required=True, description="File content."),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "directory.list",
                (
                    ArgumentField(
                        "path",
                        str,
                        default=".",
                        description="Directory path.",
                    ),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "project.tree",
                (
                    ArgumentField(
                        "path",
                        str,
                        default=".",
                        description="Project root path.",
                    ),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "desktop.application.open",
                (
                    ArgumentField(
                        "application",
                        str,
                        required=True,
                        description="Application name.",
                    ),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "desktop.file.open",
                (
                    ArgumentField("path", str, required=True, description="File path."),
                    ArgumentField(
                        "application",
                        str,
                        description="Optional application name.",
                    ),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "desktop.text.type",
                (
                    ArgumentField("text", str, required=True, description="Text to type."),
                    ArgumentField(
                        "window_title",
                        str,
                        required=True,
                        description="Target window title.",
                    ),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "desktop.hotkey.press",
                (
                    ArgumentField(
                        "keys",
                        list,
                        required=True,
                        description="Keyboard shortcut keys.",
                        validator=require_non_empty,
                    ),
                    ArgumentField(
                        "window_title",
                        str,
                        required=True,
                        description="Target window title.",
                    ),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "desktop.windows.list",
                (
                    ArgumentField(
                        "title",
                        str,
                        required=True,
                        description="Window title query.",
                    ),
                ),
            )
        )

        return schema_registry

    @staticmethod
    def build_argument_validator(
        schema_registry: ArgumentSchemaRegistry | None = None,
    ) -> ArgumentValidator:
        """Build the validator for selected tool intent arguments."""
        return ArgumentValidator(
            schema_registry or Bootstrap.build_argument_schema_registry()
        )

    @staticmethod
    def build_tool_proposal_builder(
        tool_registry: ToolRegistry | None = None,
        selector: ToolSelector | None = None,
        schema_registry: ArgumentSchemaRegistry | None = None,
        validator: ArgumentValidator | None = None,
    ) -> ToolProposalBuilder:
        """Build the natural-language to structured-tool proposal mapper."""
        registry = tool_registry or Bootstrap.build_tool_registry()
        active_selector = selector or Bootstrap.build_tool_selector(registry)
        active_schema_registry = schema_registry or Bootstrap.build_argument_schema_registry()

        return ToolProposalBuilder(
            registry,
            active_selector,
            active_schema_registry,
            validator or Bootstrap.build_argument_validator(active_schema_registry),
        )

    @staticmethod
    def build_single_tool_runner(
        tool_registry: ToolRegistry | None = None,
        selector: ToolSelector | None = None,
        validator: ArgumentValidator | None = None,
        executor: ToolExecutor | None = None,
    ) -> SingleToolRunner:
        """Build the coordinator for one selected, validated tool execution."""
        registry = tool_registry or Bootstrap.build_tool_registry()

        return SingleToolRunner(
            selector or Bootstrap.build_tool_selector(registry),
            validator or Bootstrap.build_argument_validator(),
            executor or ToolExecutor(registry),
        )

    @staticmethod
    def build_tool_chain_runner(
        single_tool_runner: SingleToolRunner | None = None,
    ) -> ToolChainRunner:
        """Build the coordinator for deterministic linear tool chains."""
        return ToolChainRunner(
            single_tool_runner or Bootstrap.build_single_tool_runner()
        )

    @staticmethod
    def build_execution_decision_engine(
        selector: ToolSelector | None = None,
    ) -> ExecutionDecisionEngine:
        """Build the deterministic execution-mode classifier."""
        tool_selector = selector or Bootstrap.build_tool_selector()

        return ExecutionDecisionEngine(
            tool_selector.supported_intents()
        )

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

        tool_registry = Bootstrap.build_tool_registry()

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
        action_engine = ActionEngineUseCase()
        wait_engine = WaitEngine(tool_executor)
        prepare_atlas_workspace = PrepareAtlasWorkspaceUseCase(
            tool_executor,
            action_engine,
            wait_engine,
        )
        restart_application = RestartApplicationUseCase(
            tool_executor,
            action_engine,
            wait_engine,
        )
        create_verified_text_file = CreateVerifiedTextFileUseCase(
            tool_executor,
            action_engine,
            wait_engine,
        )
        speech_engine = SpeechEngineUseCase(
            SoundDeviceAudioCapture(),
            FasterWhisperSpeechToTextProvider(),
        )
        speech_output_engine = Pyttsx3SpeechOutputEngine.from_environment()
        speech_interaction = SpeechInteractionUseCase(speech_engine)
        wake_word_interaction = WakeWordInteractionUseCase(
            WakeWordEngine(
                speech_engine,
                provider=OpenWakeWordProvider.from_environment(),
                wake_word="Atlas",
                timeout_seconds=30.0,
            )
        )
        voice_conversation = VoiceConversationUseCase(
            speech_engine=speech_engine,
            wake_word_engine=WakeWordEngine(
                speech_engine,
                provider=OpenWakeWordProvider.from_environment(),
                wake_word="Atlas",
                timeout_seconds=30.0,
                capture_phrase_after_detection=False,
            ),
            speech_output_engine=speech_output_engine,
            conversation_idle_timeout=25.0,
            max_session_duration=600.0,
            max_turns=20,
            max_consecutive_no_speech=_read_int(
                "ATLAS_VOICE_MAX_CONSECUTIVE_TIMEOUTS",
                3,
                minimum=1,
                maximum=10,
            ),
            diagnostics_enabled=_read_bool("ATLAS_VOICE_DIAGNOSTICS", False),
        )
        permanent_assistant = PermanentAssistantUseCase(
            wake_word_engine=_assistant_wake_word_engine(speech_engine),
            voice_conversation=voice_conversation,
            max_consecutive_errors=_read_int(
                "ATLAS_ASSISTANT_MAX_CONSECUTIVE_ERRORS",
                5,
                minimum=1,
                maximum=20,
            ),
        )
        desktop_interaction = DesktopInteractionUseCase(
            tool_executor,
            project_root=Path(".").resolve(),
            prepare_atlas_workspace=prepare_atlas_workspace,
            restart_application=restart_application,
            create_verified_text_file=create_verified_text_file,
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
            speech_interaction=speech_interaction,
            wake_word_interaction=wake_word_interaction,
            voice_conversation=voice_conversation,
            permanent_assistant=permanent_assistant,
        )


def _read_int(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(name, "").strip()

    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    return min(max(value, minimum), maximum)


def _read_bool(
    name: str,
    default: bool,
) -> bool:
    raw = os.getenv(name, "").strip().lower()

    if not raw:
        return default

    return raw in {"1", "true", "yes", "s", "si", "sí"}


def _assistant_wake_word_engine(
    speech_engine: SpeechEngineUseCase,
):
    model_path = os.getenv("ATLAS_WAKE_WORD_MODEL_PATH", "").strip()

    if model_path:
        resolved_model_path = Path(model_path).expanduser()

        if resolved_model_path.suffix.lower() == ".onnx" and resolved_model_path.is_file():
            return WakeWordEngine(
                speech_engine,
                provider=OpenWakeWordProvider.from_environment(),
                wake_word="Atlas",
                timeout_seconds=30.0,
                capture_phrase_after_detection=False,
            )

    return SttWakeWordEngine(
        speech_engine,
        wake_word="Atlas",
        capture_settings=SpeechCaptureSettings(
            max_duration=_read_float(
                "ATLAS_ASSISTANT_WAKE_STT_MAX_DURATION",
                2.2,
                minimum=0.8,
                maximum=6.0,
            ),
            initial_silence_timeout=_read_float(
                "ATLAS_ASSISTANT_WAKE_STT_INITIAL_SILENCE",
                1.4,
                minimum=0.3,
                maximum=4.0,
            ),
            trailing_silence=_read_float(
                "ATLAS_ASSISTANT_WAKE_STT_TRAILING_SILENCE",
                0.45,
                minimum=0.2,
                maximum=2.0,
            ),
            chunk_duration=0.1,
            speech_threshold=_read_float(
                "ATLAS_ASSISTANT_WAKE_STT_RMS_THRESHOLD",
                0.004,
                minimum=0.001,
                maximum=0.05,
            ),
            minimum_audio_duration=0.25,
        ),
    )


def _read_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.getenv(name, "").strip()

    if not raw:
        return default

    try:
        value = float(raw)
    except ValueError:
        return default

    return min(max(value, minimum), maximum)

"""Bootstrap module for Atlas."""

import math
import os
import sys
from pathlib import Path

from agents.chat_agent import ChatAgent
from agents.code_agent import CodeAgent
from agents.coding_agent import CodingAgent
from agents.finance_agent import FinanceAgent
from agents.legal_agent import LegalAgent
from agents.medical_agent import MedicalAgent
from agents.nutrition_agent import NutritionAgent
from agents.project_agent import ProjectAgent
from agents.training_agent import TrainingAgent
from agents.registry import AgentRegistry

from bootstrap.atlas_request_classifier import build_core_atlas_request_classifier
from bootstrap.atlas_request_adapter import build_core_atlas_request_adapter
from bootstrap.atlas_request_normalizer import build_core_atlas_request_normalizer
from bootstrap.atlas_router import build_core_atlas_router
from bootstrap.agent_system import build_core_agent_system
from bootstrap.skill_system import (
    build_builtin_skill_handler_registry,
    register_builtin_skills,
    register_desktop_skills,
)
from bootstrap.capability_execution_service import build_capability_execution_service
from bootstrap.capability_orchestrator import build_core_capability_orchestrator
from bootstrap.capability_planner import build_core_capability_planner
from bootstrap.capability_resolver import build_core_capability_resolver
from bootstrap.execution_plan_library import build_core_execution_plan_library
from bootstrap.workflow_selector import build_core_workflow_selector
from core.capability_execution_service import CapabilityExecutionService
from core.supervised_capability_gap import SupervisedCapabilityGapDetector
from core.model_health import ModelHealthChecker, OllamaModelHealthChecker
from core.model_inference import ModelInferenceRunner, ModelSelectionError
from core.model_manager import ModelManager
from core.model_registry import load_model_descriptors_from_environment
from core.model_selection_policy import ModelSelectionPolicy
from core.multi_capability_planner import MultiCapabilityPlanner
from core.agent_orchestrator import AgentOrchestrator
from core.orchestrator import AtlasOrchestrator
from core.operational_request_router import OperationalRequestRouter
from core.operational_route_executor import (
    OperationalRouteExecutor,
    RouteExecutionPresenter,
    build_default_route_handlers,
)
from core.operational_context import OperationalContextBuilder
from core.execution_memory_recorder import ExecutionMemoryRecorder
from core.execution_history import ExecutionSessionHistory
from core.execution_history_advisor import ExecutionHistoryAdvisor
from core.historical_plan_adjustment import HistoricalPlanAdjuster
from core.execution_strategy import ExecutionStrategySelector
from core.execution_authorization import (
    ExecutionAuthorizationGate,
    ExecutionDispatcher,
)
from core.execution_session_persistence import FileExecutionSessionRepository
from core.autonomous_execution import AutonomousExecutionOrchestrator
from core.execution_supervisor import ExecutionSupervisor
from core.planner import Planner
from core.router import Router
from core.deterministic_multi_tool_planner import DeterministicMultiToolPlanner
from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_plan_registry import ExecutionPlanRegistry
from core.execution_plan_validator import ExecutionPlanValidator
from core.execution_retry import RetryPolicy
from core.hybrid_execution_planner import (
    HybridExecutionPlanner,
    PromptClientStructuredPlanProvider,
    StructuredPlanProviderConfig,
)
from core.resumable_execution_store import JsonResumableExecutionStore
from core.structured_execution import StructuredExecutionCoordinator
from core.structured_plan_replanner import ExecutionReplanner, ReplanPolicy

from memory.conversation import ConversationMemory
from memory.operational import MemoryPolicy
from memory.repository import FileMemoryEntryRepository

from models.prompt_client import PromptClient

from tools.executor import ToolExecutor
from tools.execution_coordinator import ExecutionCoordinator
from tools.argument_schema import (
    ArgumentField,
    ArgumentSchema,
    ArgumentSchemaRegistry,
    ArgumentValidator,
    require_non_empty,
)
from tools.execution_decision import ExecutionDecisionEngine

from core.async_task_scheduler import (
    AsyncTaskScheduler,
    JsonGoalTaskStore,
    ToolTaskExecutor,
)
from tools.intent_selector import ToolIntentRegistry, ToolSelector
from tools.registry import ToolRegistry
from tools.tool_chain_runner import ToolChainRunner
from tools.tool_chain_proposal_builder import ToolChainProposalBuilder
from tools.tool_proposal_builder import ToolProposalBuilder
from tools.single_tool_runner import SingleToolRunner
from tools.semantic_catalog import SemanticToolCatalog
from tools.tool_schema import ToolArgumentsSchema, ToolParameterSchema

from tools.filesystem.read_file_tool import ReadFileTool
from tools.temperature_conversion import TemperatureConversionTool
from tools.filesystem.write_file_tool import WriteFileTool
from tools.documents.create_training_pdf_tool import CreateTrainingPdfTool
from tools.filesystem.list_directory_tool import ListDirectoryTool
from tools.project.tree_tool import TreeTool
from tools.web_search import WebSearchTool
from tools.calendar.calendar_list_events_tool import (
    CALENDAR_LIST_EVENTS_ARGUMENTS_SCHEMA,
    CalendarListEventsTool,
)
from tools.calendar.calendar_request_parser import (
    require_calendar_max_results,
    require_rfc3339_timestamp,
)
from tools.calendar.calendar_create_event_tool import (
    CALENDAR_CREATE_EVENT_ARGUMENTS_SCHEMA,
    CalendarCreateEventTool,
)
from tools.gmail.gmail_request_parser import extract_gmail_arguments
from tools.gmail.gmail_service import (
    require_email_address,
    require_gmail_body,
    require_gmail_max_results,
    require_gmail_message_id,
    require_gmail_sender,
    require_gmail_subject,
)
from tools.gmail.gmail_tools import (
    GmailListTool,
    GmailReadTool,
    GmailSendTool,
)
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
    CreateFolderTool,
    CopyPathTool,
    MovePathTool,
    RenamePathTool,
    DeletePathTool,

)
from use_cases.read_file import ReadFileUseCase
from use_cases.write_file import WriteFileUseCase
from use_cases.create_training_pdf import CreateTrainingPdfUseCase
from services.pdf_service import PdfService
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
from use_cases.execution_conversation import ExecutionConversationController
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

        tool_registry.register(
            ReadFileTool(),
            arguments_schema=ToolArgumentsSchema(
                parameters=(
                    ToolParameterSchema("path", str, required=True),
                    ToolParameterSchema("limit", int, minimum=1),
                ),
            ),
        )
        tool_registry.register(TemperatureConversionTool())
        tool_registry.register(
            WebSearchTool(),
            arguments_schema=ToolArgumentsSchema(
                parameters=(
                    ToolParameterSchema("query", str, required=True),
                    ToolParameterSchema("max_results", int, default=5),
                ),
            ),
        )
        tool_registry.register(
            WriteFileTool(),
            arguments_schema=ToolArgumentsSchema(
                parameters=(
                    ToolParameterSchema("path", str, required=True),
                    ToolParameterSchema("content", str, required=True),
                ),
            ),
        )
        tool_registry.register(
            ListDirectoryTool(),
            arguments_schema=ToolArgumentsSchema(
                parameters=(
                    ToolParameterSchema("path", str, default="."),
                ),
            ),
        )
        tool_registry.register(
            TreeTool(),
            arguments_schema=ToolArgumentsSchema(
                parameters=(
                    ToolParameterSchema("path", str, default="."),
                ),
            ),
        )
        tool_registry.register(
            CalendarListEventsTool(),
            arguments_schema=CALENDAR_LIST_EVENTS_ARGUMENTS_SCHEMA,
        )
        tool_registry.register(
            CalendarCreateEventTool(),
            arguments_schema=CALENDAR_CREATE_EVENT_ARGUMENTS_SCHEMA,
        )
        tool_registry.register(
            GmailListTool(),
            arguments_schema=ToolArgumentsSchema(
                parameters=(
                    ToolParameterSchema(
                        "max_results",
                        int,
                        default=5,
                        minimum=1,
                        maximum=20,
                    ),
                ),
            ),
        )
        tool_registry.register(
            GmailReadTool(),
            arguments_schema=ToolArgumentsSchema(
                parameters=(
                    ToolParameterSchema("message_id", str, required=False),
                    ToolParameterSchema("sender", str, required=False),
                ),
            ),
        )
        tool_registry.register(
            GmailSendTool(),
            arguments_schema=ToolArgumentsSchema(
                parameters=(
                    ToolParameterSchema("to", str, required=True),
                    ToolParameterSchema("subject", str, required=True),
                    ToolParameterSchema("body", str, required=True),
                ),
            ),
        )
        tool_registry.register(OpenApplicationTool())
        tool_registry.register(ListProcessesTool())
        tool_registry.register(IsProcessRunningTool())
        tool_registry.register(GetProcessTool())
        tool_registry.register(CloseApplicationTool())
        tool_registry.register(TerminateProcessTool())
        tool_registry.register(OpenFolderTool())
        open_file_tool = OpenFileTool()
        tool_registry.register(CreateFolderTool())
        tool_registry.register(CopyPathTool())
        tool_registry.register(MovePathTool())
        tool_registry.register(RenamePathTool())
        tool_registry.register(DeletePathTool())
        tool_registry.register(open_file_tool)
        tool_registry.register(
            CreateTrainingPdfTool(CreateTrainingPdfUseCase(PdfService(), open_file_tool)),
            arguments_schema=ToolArgumentsSchema(
                parameters=(
                    ToolParameterSchema("content", str, required=True),
                    ToolParameterSchema("output_dir", str, default="artifacts/documents"),
                ),
            ),
        )
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
            ("training.pdf.create", "training.create_pdf"),
            ("directory.list", "list_directory"),
            ("calendar.events.list", "calendar_list_events"),
            ("calendar.events.create", "calendar_create_event"),
            ("gmail.messages.list", "gmail_list"),
            ("gmail.messages.read", "gmail_read"),
            ("gmail.messages.send", "gmail_send"),
            ("project.tree", "project_tree"),
            ("web.search", "web_search"),
            ("desktop.application.open", "desktop.open_application"),
            ("desktop.application.close", "desktop.close_application"),
            ("desktop.filesystem.create_folder", "desktop.create_folder"),
            ("desktop.filesystem.copy", "desktop.copy_path"),
            ("desktop.filesystem.move", "desktop.move_path"),
            ("desktop.filesystem.rename", "desktop.rename_path"),
            ("desktop.filesystem.delete", "desktop.delete_path"),
            ("desktop.file.open", "desktop.open_file"),
            ("desktop.text.type", "desktop.type_text"),
            ("desktop.clipboard.copy", "desktop.copy_clipboard_text"),
            ("desktop.clipboard.paste", "desktop.paste_clipboard"),
            ("desktop.hotkey.press", "desktop.press_hotkey"),
            ("desktop.windows.list", "desktop.list_windows"),
            ("desktop.window.bring_to_front", "desktop.bring_window_to_front"),
            ("desktop.window.close", "desktop.close_window"),
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
                "training.pdf.create",
                (
                    ArgumentField("content", str, required=True, description="Training content."),
                    ArgumentField("output_dir", str, default="artifacts/documents", description="PDF output directory."),
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
                "calendar.events.list",
                (
                    ArgumentField(
                        "time_min",
                        str,
                        required=True,
                        description="RFC3339 inclusive range start.",
                        validator=require_rfc3339_timestamp,
                    ),
                    ArgumentField(
                        "time_max",
                        str,
                        required=True,
                        description="RFC3339 exclusive range end.",
                        validator=require_rfc3339_timestamp,
                    ),
                    ArgumentField(
                        "max_results",
                        int,
                        default=5,
                        description="Maximum number of events.",
                        validator=require_calendar_max_results,
                    ),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "gmail.messages.list",
                (
                    ArgumentField(
                        "max_results",
                        int,
                        default=5,
                        description="Maximum number of messages.",
                        validator=require_gmail_max_results,
                    ),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "gmail.messages.read",
                (
                    ArgumentField(
                        "message_id",
                        str,
                        description="Gmail message id (UID).",
                        validator=require_gmail_message_id,
                    ),
                    ArgumentField(
                        "sender",
                        str,
                        description="Sender address or name.",
                        validator=require_gmail_sender,
                    ),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "gmail.messages.send",
                (
                    ArgumentField(
                        "to",
                        str,
                        required=True,
                        description="Recipient email address.",
                        validator=require_email_address,
                    ),
                    ArgumentField(
                        "subject",
                        str,
                        required=True,
                        description="Email subject.",
                        validator=require_gmail_subject,
                    ),
                    ArgumentField(
                        "body",
                        str,
                        required=True,
                        description="Plain text email body.",
                        validator=require_gmail_body,
                    ),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "calendar.events.create",
                (
                    ArgumentField(
                        "title",
                        str,
                        required=True,
                        description="Event title.",
                    ),
                    ArgumentField(
                        "start_time",
                        str,
                        required=True,
                        description="RFC3339 event start.",
                        validator=require_rfc3339_timestamp,
                    ),
                    ArgumentField(
                        "end_time",
                        str,
                        required=True,
                        description="RFC3339 event end.",
                        validator=require_rfc3339_timestamp,
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
        schema_registry.register(ArgumentSchema("desktop.filesystem.create_folder", (ArgumentField("path", str, required=True, description="Absolute folder path."),)))
        schema_registry.register(ArgumentSchema("desktop.filesystem.copy", (
            ArgumentField("source_path", str, required=True, description="Absolute source path."),
            ArgumentField("destination_path", str, required=True, description="Absolute destination path."),
        )))
        schema_registry.register(ArgumentSchema("desktop.filesystem.move", (
            ArgumentField("source_path", str, required=True, description="Absolute source path."),
            ArgumentField("destination_path", str, required=True, description="Absolute destination path."),
        )))
        schema_registry.register(ArgumentSchema("desktop.filesystem.rename", (
            ArgumentField("source_path", str, required=True, description="Absolute source path."),
            ArgumentField("new_name", str, required=True, description="New leaf name."),
        )))
        schema_registry.register(ArgumentSchema("desktop.filesystem.delete", (ArgumentField("path", str, required=True, description="Absolute path to delete."),)))

        schema_registry.register(
            ArgumentSchema(
                "desktop.application.close",
                (
                    ArgumentField(
                        "pid",
                        int,
                        required=True,
                        description="Target process id.",
                    ),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "desktop.text.type",
                (
                    ArgumentField("text", str, required=True, description="Text to type."),
                ),
            )
        )
        schema_registry.register(
            ArgumentSchema(
                "desktop.clipboard.copy",
                (
                    ArgumentField("text", str, required=True, description="Text to copy."),
                ),
            )
        )
        schema_registry.register(ArgumentSchema("desktop.clipboard.paste", ()))
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

        schema_registry.register(
            ArgumentSchema(
                "desktop.window.bring_to_front",
                (
                    ArgumentField(
                        "handle",
                        int,
                        required=True,
                        description="Exact target window handle.",
                    ),
                ),
            )
        )

        schema_registry.register(
            ArgumentSchema(
                "desktop.window.close",
                (
                    ArgumentField(
                        "handle",
                        int,
                        required=True,
                        description="Exact target window handle.",
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
    def build_semantic_tool_catalog(
        tool_registry: ToolRegistry | None = None,
        selector: ToolSelector | None = None,
        schema_registry: ArgumentSchemaRegistry | None = None,
    ) -> SemanticToolCatalog:
        """Build the passive semantic catalog for registered tools."""
        registry = tool_registry or Bootstrap.build_tool_registry()
        return SemanticToolCatalog.build_from_registry(
            registry,
            tool_selector=selector or Bootstrap.build_tool_selector(registry),
            schema_registry=schema_registry or Bootstrap.build_argument_schema_registry(),
        )

    @staticmethod
    def build_hybrid_execution_planner(
        tool_registry: ToolRegistry | None = None,
        schema_registry: ArgumentSchemaRegistry | None = None,
        *,
        hybrid_planning_enabled: bool | None = None,
    ) -> HybridExecutionPlanner:
        """Build the passive hybrid planner without invoking any provider."""
        registry = tool_registry or Bootstrap.build_tool_registry()
        return HybridExecutionPlanner(
            tool_registry=registry,
            schema_registry=schema_registry or Bootstrap.build_argument_schema_registry(),
            hybrid_planning_enabled=(
                _read_bool("ATLAS_HYBRID_PLANNING_ENABLED", False)
                if hybrid_planning_enabled is None
                else hybrid_planning_enabled
            ),
        )

    @staticmethod
    def build_model_selection_policy() -> ModelSelectionPolicy:
        """Build the immutable runtime preferences used by model requests."""
        return ModelSelectionPolicy(
            preferred_provider=_read_text("ATLAS_MODEL_PREFERRED_PROVIDER"),
            prefer_local=_read_optional_bool("ATLAS_MODEL_PREFER_LOCAL"),
            max_cost=_read_optional_float("ATLAS_MODEL_MAX_COST"),
            max_latency=_read_optional_float("ATLAS_MODEL_MAX_LATENCY"),
            allow_fallback=_read_optional_bool(
                "ATLAS_MODEL_ALLOW_FALLBACK",
                default=True,
            ),
        )

    @staticmethod
    def build_structured_plan_provider(
        prompt_client: PromptClient | None = None,
        model_manager: ModelManager | None = None,
        *,
        health_checker: ModelHealthChecker | None = None,
        model_selection_policy: ModelSelectionPolicy | None = None,
        structured_plan_provider_enabled: bool | None = None,
        structured_plan_model: str | None = None,
        structured_plan_streaming_enabled: bool | None = None,
        config: StructuredPlanProviderConfig | None = None,
        diagnostic_sink=None,
    ) -> PromptClientStructuredPlanProvider | None:
        """Build the real PromptClient-backed provider only when explicitly enabled."""
        if config is None:
            enabled = (
                _read_bool("ATLAS_STRUCTURED_PLAN_PROVIDER_ENABLED", False)
                if structured_plan_provider_enabled is None
                else structured_plan_provider_enabled
            )
            config = StructuredPlanProviderConfig(
                enabled=enabled,
                model_name=(
                    structured_plan_model
                    if structured_plan_model is not None
                    else _read_text("ATLAS_STRUCTURED_PLAN_MODEL")
                ),
                max_objective_chars=_read_int(
                    "ATLAS_STRUCTURED_PLAN_MAX_OBJECTIVE_CHARS",
                    4000,
                    minimum=1,
                    maximum=20000,
                ),
                max_catalog_chars=_read_int(
                    "ATLAS_STRUCTURED_PLAN_MAX_CATALOG_CHARS",
                    50000,
                    minimum=1000,
                    maximum=200000,
                ),
                max_response_chars=_read_int(
                    "ATLAS_STRUCTURED_PLAN_MAX_RESPONSE_CHARS",
                    30000,
                    minimum=1000,
                    maximum=100000,
                ),
                max_steps=_read_int(
                    "ATLAS_STRUCTURED_PLAN_MAX_STEPS",
                    12,
                    minimum=1,
                    maximum=50,
                ),
                streaming_enabled=_read_bool(
                    "ATLAS_STRUCTURED_PLAN_STREAMING_ENABLED",
                    False,
                )
                if structured_plan_streaming_enabled is None
                else structured_plan_streaming_enabled,
            )

        if not config.enabled:
            return None

        resolved_prompt_client = prompt_client or PromptClient()
        resolved_health_checker = health_checker
        if resolved_health_checker is None and callable(
            getattr(resolved_prompt_client, "check_model_health", None)
        ):
            resolved_health_checker = OllamaModelHealthChecker(resolved_prompt_client)
        return PromptClientStructuredPlanProvider.from_config(
            resolved_prompt_client,
            config,
            model_manager=model_manager or ModelManager(),
            model_selection_policy=model_selection_policy,
            health_checker=resolved_health_checker,
            diagnostic_sink=diagnostic_sink,
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
    def build_tool_chain_proposal_builder(
        tool_registry: ToolRegistry | None = None,
        selector: ToolSelector | None = None,
        schema_registry: ArgumentSchemaRegistry | None = None,
        validator: ArgumentValidator | None = None,
        proposal_builder: ToolProposalBuilder | None = None,
    ) -> ToolChainProposalBuilder:
        """Build the natural-language to structured-chain proposal mapper."""
        registry = tool_registry or Bootstrap.build_tool_registry()
        active_selector = selector or Bootstrap.build_tool_selector(registry)
        active_schema_registry = schema_registry or Bootstrap.build_argument_schema_registry()
        active_validator = validator or Bootstrap.build_argument_validator(
            active_schema_registry
        )

        return ToolChainProposalBuilder(
            proposal_builder
            or Bootstrap.build_tool_proposal_builder(
                registry,
                active_selector,
                active_schema_registry,
                active_validator,
            ),
            active_selector,
            active_validator,
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
    def build_execution_coordinator(
        tool_registry: ToolRegistry | None = None,
        selector: ToolSelector | None = None,
        schema_registry: ArgumentSchemaRegistry | None = None,
        validator: ArgumentValidator | None = None,
        executor: ToolExecutor | None = None,
        single_tool_runner: SingleToolRunner | None = None,
    ) -> ExecutionCoordinator:
        """Build the coordinator for decision, proposal and runner execution."""
        registry = tool_registry or Bootstrap.build_tool_registry()
        active_selector = selector or Bootstrap.build_tool_selector(registry)
        active_schema_registry = schema_registry or Bootstrap.build_argument_schema_registry()
        active_validator = validator or Bootstrap.build_argument_validator(
            active_schema_registry
        )
        active_executor = executor or ToolExecutor(registry)
        single_runner = single_tool_runner or Bootstrap.build_single_tool_runner(
            registry,
            active_selector,
            active_validator,
            active_executor,
        )

        return ExecutionCoordinator(
            Bootstrap.build_execution_decision_engine(active_selector),
            Bootstrap.build_tool_proposal_builder(
                registry,
                active_selector,
                active_schema_registry,
                active_validator,
            ),
            Bootstrap.build_tool_chain_proposal_builder(
                registry,
                active_selector,
                active_schema_registry,
                active_validator,
            ),
            single_runner,
            Bootstrap.build_tool_chain_runner(single_runner),
        )

    @staticmethod
    def build_capability_execution_service(
        *,
        tool_registry: ToolRegistry,
        execution_plan_validator: ExecutionPlanValidator,
        execution_plan_executor: ExecutionPlanExecutor,
        execution_plan_libraries=None,
        execution_plan_registry: ExecutionPlanRegistry | None = None,
    ) -> CapabilityExecutionService:
        """Build capability execution from explicitly shared runtime objects."""
        if execution_plan_libraries is None:
            core_library = build_core_execution_plan_library()
            active_libraries = () if core_library is None else (core_library,)
        else:
            active_libraries = tuple(execution_plan_libraries)
        active_registry = execution_plan_registry or ExecutionPlanRegistry()
        if execution_plan_registry is None:
            for library in active_libraries:
                library.install(active_registry)
        capability_resolver = build_core_capability_resolver(
            tool_registry=tool_registry,
            execution_plan_libraries=active_libraries,
        )
        workflow_selector, _policy = build_core_workflow_selector()
        capability_planner = build_core_capability_planner(
            capability_resolver=capability_resolver,
            workflow_selector=workflow_selector,
            execution_plan_libraries=active_libraries,
            execution_plan_registry=active_registry,
        )
        multi_capability_planner = MultiCapabilityPlanner(execution_plan_libraries=active_libraries)
        capability_orchestrator = build_core_capability_orchestrator(
            capability_planner,
            execution_plan_validator,
            execution_plan_executor,
        )
        return build_capability_execution_service(
            capability_orchestrator,
            multi_capability_planner=multi_capability_planner,
        )

    @staticmethod
    def build() -> AtlasOrchestrator:

        # -----------------------
        # Core
        # -----------------------

        tool_registry = Bootstrap.build_tool_registry()
        web_search_tool = tool_registry.get("web_search")
        tool_executor = ToolExecutor(tool_registry)
        tool_selector = Bootstrap.build_tool_selector(tool_registry)
        schema_registry = Bootstrap.build_argument_schema_registry()
        argument_validator = Bootstrap.build_argument_validator(schema_registry)
        model_manager = ModelManager(
            descriptors=load_model_descriptors_from_environment(
                reserved_logical_ids=(item.logical_id for item in ModelManager._DEFAULT_DESCRIPTORS),
            ),
        )
        model_selection_policy = Bootstrap.build_model_selection_policy()
        prompt_client = PromptClient()
        model_health_checker = OllamaModelHealthChecker(prompt_client)
        hybrid_planning_enabled = _read_bool("ATLAS_HYBRID_PLANNING_ENABLED", True)
        provider_enabled = _read_bool("ATLAS_STRUCTURED_PLAN_PROVIDER_ENABLED", False)
        structured_plan_streaming_enabled = _read_bool(
            "ATLAS_STRUCTURED_PLAN_STREAMING_ENABLED",
            False,
        )
        structured_plan_execution_enabled = _read_bool(
            "ATLAS_STRUCTURED_PLAN_EXECUTION_ENABLED",
            True,
        )
        execution_persistence_enabled = _read_bool(
            "ATLAS_EXECUTION_PERSISTENCE_ENABLED",
            True,
        )
        execution_retry_enabled = _read_bool(
            "ATLAS_EXECUTION_RETRY_ENABLED",
            False,
        )
        execution_max_attempts = _read_int(
            "ATLAS_EXECUTION_MAX_ATTEMPTS",
            2,
            minimum=1,
            maximum=5,
        )
        execution_retry_delay_ms = _read_int(
            "ATLAS_EXECUTION_RETRY_DELAY_MS",
            0,
            minimum=0,
            maximum=10_000,
        )
        execution_retry_policy = RetryPolicy(
            max_attempts=(
                execution_max_attempts
                if execution_retry_enabled
                else 1
            ),
            delay_ms=(
                execution_retry_delay_ms
                if execution_retry_enabled
                else 0
            ),
        )
        semantic_catalog = None
        hybrid_execution_planner = None
        structured_plan_provider = None
        multi_tool_planner = None
        if hybrid_planning_enabled or provider_enabled:
            semantic_catalog = Bootstrap.build_semantic_tool_catalog(
                tool_registry,
                tool_selector,
                schema_registry,
            )
            multi_tool_planner = DeterministicMultiToolPlanner()
            hybrid_execution_planner = Bootstrap.build_hybrid_execution_planner(
                tool_registry,
                schema_registry,
                hybrid_planning_enabled=hybrid_planning_enabled,
            )
            structured_plan_provider = Bootstrap.build_structured_plan_provider(
                prompt_client,
                model_manager,
                health_checker=model_health_checker,
                model_selection_policy=model_selection_policy,
                structured_plan_provider_enabled=provider_enabled,
                structured_plan_streaming_enabled=structured_plan_streaming_enabled,
            )

        planner = Planner(
            tool_registry=tool_registry,
            tool_selector=tool_selector,
            schema_registry=schema_registry,
            argument_validator=argument_validator,
            semantic_tool_catalog=semantic_catalog,
            multi_tool_planner=multi_tool_planner,
            hybrid_execution_planner=hybrid_execution_planner,
            structured_plan_provider=structured_plan_provider,
        )
        router = Router()
        memory_policy = MemoryPolicy()
        memory = ConversationMemory(
            policy=memory_policy,
            repository=FileMemoryEntryRepository(_personal_memory_path()),
        )

        # -----------------------
        # Tools
        # -----------------------

        core_execution_plan_library = build_core_execution_plan_library()
        execution_plan_libraries = () if core_execution_plan_library is None else (core_execution_plan_library,)
        execution_plan_registry = ExecutionPlanRegistry()
        for library in execution_plan_libraries:
            library.install(execution_plan_registry)
        execution_plan_validator = ExecutionPlanValidator(tool_registry, plan_registry=execution_plan_registry)
        execution_plan_executor = ExecutionPlanExecutor(
            tool_registry,
            tool_executor,
            retry_policy=execution_retry_policy,
            plan_registry=execution_plan_registry,
        )
        capability_execution_service = Bootstrap.build_capability_execution_service(
            tool_registry=tool_registry,
            execution_plan_validator=execution_plan_validator,
            execution_plan_executor=execution_plan_executor,
            execution_plan_libraries=execution_plan_libraries,
            execution_plan_registry=execution_plan_registry,
        )
        skill_handler_registry = build_builtin_skill_handler_registry(tool_executor)
        agent_system_result = build_core_agent_system(
            tool_executor=tool_executor,
            capability_execution_service=capability_execution_service,
            skill_handler_registry=skill_handler_registry,
        )
        agent_system = agent_system_result.system if agent_system_result.system is not None else None
        if agent_system is not None:
            register_builtin_skills(agent_system.skill_system)
            register_desktop_skills(agent_system.skill_system)
        capability_gap_detector = SupervisedCapabilityGapDetector.from_registries(
            tool_registry=tool_registry,
            execution_plan_libraries=execution_plan_libraries,
            skill_system=(agent_system.skill_system if agent_system is not None else None),
            agent_registry=(agent_system.agent_registry if agent_system is not None else None),
        )
        atlas_router = build_core_atlas_router(
            capability_execution_service=capability_execution_service,
            agent_system=agent_system,
        )
        atlas_request_adapter = build_core_atlas_request_adapter()
        atlas_request_classifier = build_core_atlas_request_classifier()
        atlas_request_normalizer = build_core_atlas_request_normalizer()

        execution_session_repository = (
            FileExecutionSessionRepository(_execution_history_path())
            if execution_persistence_enabled
            else None
        )
        execution_supervisor = ExecutionSupervisor(
            session_repository=execution_session_repository,
        )
        execution_history = ExecutionSessionHistory(
            session_source=execution_supervisor,
            session_repository=execution_session_repository,
        )
        execution_history_advisor = ExecutionHistoryAdvisor(execution_history)
        historical_plan_adjuster = HistoricalPlanAdjuster(
            execution_plan_validator,
        )
        execution_strategy_selector = ExecutionStrategySelector()
        execution_dispatcher = ExecutionDispatcher()
        execution_authorization_gate = ExecutionAuthorizationGate(
            already_dispatched=execution_dispatcher.has_dispatched,
            confirmation_consumed=execution_dispatcher.confirmation_consumed,
        )
        structured_execution = StructuredExecutionCoordinator(
            planner=planner,
            validator=execution_plan_validator,
            executor=execution_plan_executor,
            execution_supervisor=execution_supervisor,
            execution_replanner=ExecutionReplanner(planner),
            replan_policy=ReplanPolicy(max_replans_per_session=1),
            execution_strategy_selector=execution_strategy_selector,
            execution_authorization_gate=execution_authorization_gate,
            execution_dispatcher=execution_dispatcher,
            execution_history_advisor=execution_history_advisor,
            historical_plan_adjuster=historical_plan_adjuster,
            resumable_store=(
                JsonResumableExecutionStore(_execution_state_path())
                if execution_persistence_enabled
                else None
            ),
        )
        autonomous_execution = AutonomousExecutionOrchestrator(
            planner=planner,
            validator=execution_plan_validator,
            executor=execution_plan_executor,
            supervisor=execution_supervisor,
            execution_strategy_selector=execution_strategy_selector,
            execution_authorization_gate=execution_authorization_gate,
            execution_dispatcher=execution_dispatcher,
            execution_history_advisor=execution_history_advisor,
            historical_plan_adjuster=historical_plan_adjuster,
        )

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
            SoundDeviceAudioCapture(
                sample_rate=_read_int(
                    "ATLAS_VOICE_SAMPLE_RATE",
                    16_000,
                    minimum=8_000,
                    maximum=48_000,
                )
            ),
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
        single_tool_runner = Bootstrap.build_single_tool_runner(
            tool_registry,
            tool_selector,
            argument_validator,
            tool_executor,
        )
        execution_conversation = ExecutionConversationController(
            Bootstrap.build_execution_coordinator(
                tool_registry=tool_registry,
                executor=tool_executor,
                selector=tool_selector,
                validator=argument_validator,
                single_tool_runner=single_tool_runner,
            ),
            restore_pending_target=desktop_interaction.restore_external_foreground_handle,
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
            CodeAgent(
                prompt_client,
            )
        )

        registry.register(
            TrainingAgent(
                prompt_client,
            )
        )
        registry.register(
            NutritionAgent(
                prompt_client,
            )
        )
        registry.register(
            MedicalAgent(
                prompt_client,
            )
        )
        registry.register(
            LegalAgent(
                prompt_client,
            )
        )
        registry.register(
            FinanceAgent(
                prompt_client,
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
        router = Router(
            operational_router=OperationalRequestRouter(
                tool_registry=tool_registry,
                agent_registry=registry,
            )
        )
        def direct_messages(request, context):
            messages = [
                {
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                }
                for message in context.recent_messages
            ]
            prompt_context = context.prompt_context()
            if prompt_context:
                messages.append(
                    {
                        "role": "system",
                        "content": "Contexto operativo limitado:\n" + prompt_context,
                    }
                )
            messages.append({"role": "user", "content": request.content})
            return messages

        direct_inference_runner = ModelInferenceRunner(
            model_manager,
            health_checker=model_health_checker,
        )
        direct_selection_request = model_selection_policy.create_request(task="chat")

        def direct_provider_id(selected_model: str) -> str:
            descriptor = model_manager.resolve_model(selected_model)
            if descriptor is None:
                raise RuntimeError(f"Selected chat model '{selected_model}' is not registered.")
            return descriptor.provider_id

        def direct_responder(request, context):
            chat_agent = registry.get("chat")
            if chat_agent is None:
                raise RuntimeError("Agent 'chat' is not registered.")
            messages = direct_messages(request, context)
            try:
                return direct_inference_runner.run(
                    direct_selection_request,
                    lambda selected_model: chat_agent.run(
                        model=selected_model,
                        messages=messages,
                        provider_id=direct_provider_id(selected_model),
                    ),
                )
            except ModelSelectionError as error:
                if model_selection_policy != ModelSelectionPolicy():
                    raise
                selected_model = model_manager.choose_model(
                    "chat",
                    selection_result=error.result,
                )
                return chat_agent.run(
                    model=selected_model,
                    messages=messages,
                    provider_id=direct_provider_id(selected_model),
                )

        def direct_streaming_responder(request, context):
            chat_agent = registry.get("chat")
            if chat_agent is None:
                raise RuntimeError("Agent 'chat' is not registered.")
            stream = getattr(chat_agent, "stream", None)
            if not callable(stream):
                raise RuntimeError("Agent 'chat' does not support streaming.")
            messages = direct_messages(request, context)
            try:
                yield from direct_inference_runner.stream(
                    direct_selection_request,
                    lambda selected_model: stream(
                        model=selected_model,
                        messages=messages,
                        provider_id=direct_provider_id(selected_model),
                    ),
                )
            except ModelSelectionError as error:
                if model_selection_policy != ModelSelectionPolicy():
                    raise
                selected_model = model_manager.choose_model(
                    "chat",
                    selection_result=error.result,
                )
                yield from stream(
                    model=selected_model,
                    messages=messages,
                    provider_id=direct_provider_id(selected_model),
                )

        operational_route_executor = OperationalRouteExecutor(
            build_default_route_handlers(
                direct_responder=direct_responder,
                direct_streaming_responder=direct_streaming_responder,
                memory=memory,
                tool_registry=tool_registry,
                tool_executor=tool_executor,
                single_tool_runner=single_tool_runner,
                agent_registry=registry,
                model_selector=model_manager.choose_model,
                autonomous_orchestrator=autonomous_execution,
                execution_supervisor=execution_supervisor,
                execution_history=execution_history,
                diagnostics=lambda: {
                    "tool_count": len(tool_registry.list()),
                    "agent_count": len(registry.list()),
                    "execution_count": len(execution_supervisor.list_sessions()),
                },
            ),
            context_builder=OperationalContextBuilder(
                memory,
                policy=memory_policy,
            ),
            execution_memory_recorder=ExecutionMemoryRecorder(
                memory,
                policy=memory_policy,
            ),
        )

        # -----------------------
        # Orchestrator
        # -----------------------

        tool_task_executor = ToolTaskExecutor(
            single_tool_runner,
            model_transformer=_build_async_goal_model_transformer(
                registry,
                model_manager,
            ),
        )
        async_task_scheduler = AsyncTaskScheduler(
            tool_task_executor,
            store=JsonGoalTaskStore(
                _execution_state_path().parent / "task_scheduler"
            ),
        )
        tool_task_executor.bind_result_lookup(async_task_scheduler.task)
        orchestrator = AtlasOrchestrator(
            planner=planner,
            router=router,
            model_manager=model_manager,
            model_selection_policy=model_selection_policy,
            model_health_checker=model_health_checker,
            memory=memory,
            registry=registry,
            write_file=write_file,
            refactoring_interaction=refactoring_interaction,
            correction_interaction=correction_interaction,
            desktop_interaction=desktop_interaction,
            execution_conversation=execution_conversation,
            speech_interaction=speech_interaction,
            wake_word_interaction=wake_word_interaction,
            voice_conversation=voice_conversation,
            permanent_assistant=permanent_assistant,
            structured_execution_coordinator=structured_execution,
            capability_execution_service=capability_execution_service,
            atlas_router=atlas_router,
            atlas_request_adapter=atlas_request_adapter,
            atlas_request_classifier=atlas_request_classifier,
            atlas_request_normalizer=atlas_request_normalizer,
            operational_route_executor=operational_route_executor,
            route_execution_presenter=RouteExecutionPresenter(),
            execution_history=execution_history,
            execution_history_advisor=execution_history_advisor,
            historical_plan_adjuster=historical_plan_adjuster,
            execution_strategy_selector=execution_strategy_selector,
            execution_authorization_gate=execution_authorization_gate,
            execution_dispatcher=execution_dispatcher,
            tool_registry=tool_registry,
            web_search_tool=web_search_tool,
            skill_system=(
                agent_system.skill_system
                if agent_system is not None
                else None
            ),
            capability_gap_detector=capability_gap_detector,
            structured_execution_enabled=hybrid_planning_enabled or provider_enabled,
            structured_plan_streaming_enabled=structured_plan_streaming_enabled,
            structured_plan_execution_enabled=structured_plan_execution_enabled,
            async_task_scheduler=async_task_scheduler,
        )
        orchestrator.agent_orchestrator = AgentOrchestrator(registry)
        return orchestrator


def _build_async_goal_model_transformer(registry, model_manager):
    """Reuse the existing chat agent and model selection for text transforms."""

    def transform(instruction: str) -> str:
        chat_agent = registry.get("chat")
        if chat_agent is None:
            raise RuntimeError("Agent 'chat' is not registered.")
        model = model_manager.choose_model("chat")
        return chat_agent.run(
            model=model,
            messages=[{"role": "user", "content": instruction}],
        )

    return transform


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
        print(
            f"Warning: invalid integer value for {name}; using {default}.",
            file=sys.stderr,
        )
        return default

    return min(max(value, minimum), maximum)


def _read_bool(
    name: str,
    default: bool,
) -> bool:
    raw = os.getenv(name, "").strip().lower()

    if not raw:
        return default

    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False

    print(
        f"Warning: invalid boolean value for {name}; using false.",
        file=sys.stderr,
    )
    return False


def _read_optional_bool(
    name: str,
    default: bool | None = None,
) -> bool | None:
    raw = os.getenv(name, "").strip().lower()

    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False

    print(
        f"Warning: invalid boolean value for {name}; using {default}.",
        file=sys.stderr,
    )
    return default


def _read_optional_float(
    name: str,
) -> float | None:
    raw = os.getenv(name, "").strip()

    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0:
        print(
            f"Warning: invalid non-negative number for {name}; using None.",
            file=sys.stderr,
        )
        return None
    return value


def _read_text(
    name: str,
) -> str | None:
    raw = os.getenv(name, "").strip()
    return raw or None


def _execution_state_path() -> Path:
    configured = _read_text("ATLAS_EXECUTION_STATE_PATH")
    if configured is not None:
        return Path(configured).expanduser()

    return Path(".atlas") / "execution_state.json"


def _personal_memory_path() -> Path:
    configured = _read_text("ATLAS_PERSONAL_MEMORY_PATH")
    if configured is not None:
        return Path(configured).expanduser()

    return Path(".atlas") / "personal_memory.json"


def _execution_history_path() -> Path:
    configured = _read_text("ATLAS_EXECUTION_HISTORY_PATH")
    if configured is not None:
        return Path(configured).expanduser()

    return _execution_state_path().parent / "execution_sessions"


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


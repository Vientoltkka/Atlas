"""Planner for Atlas."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import json
import re
from typing import Any, Callable, Mapping
import unicodedata

from core.execution_arguments import ExecutionArguments, InvalidExecutionArgumentError
from core.execution_variable_binding import (
    ExecutionVariableBinding,
    copy_execution_variable_binding,
)
from tools.argument_schema import (
    ArgumentSchemaRegistry,
    ArgumentValidator,
)
from tools.execution_decision import ExecutionDecision, ExecutionDecisionEngine, ExecutionMode
from tools.intent_selector import ToolIntent, ToolSelector
from tools.intent_selector import ToolSelectionResult
from tools.registry import ToolDescriptor, ToolNotRegisteredError, ToolRegistry
from tools.tool_proposal_builder import (
    StructuredToolProposal,
    ToolProposalBuilder,
    ToolProposalStatus,
)
from tools.semantic_catalog import SemanticToolCatalog


@dataclass
class Plan:
    task: str
    objective: str


@dataclass(frozen=True, slots=True)
class ExecutionStep:
    """One pending step in an execution plan."""

    id: str
    description: str
    tool: str | None
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    status: str = "pending"
    arguments: ExecutionArguments | Mapping[str, Any] = field(default_factory=ExecutionArguments.empty)
    output_binding: ExecutionVariableBinding | None = None

    def __post_init__(self) -> None:
        if isinstance(self.arguments, ExecutionArguments):
            arguments = self.arguments
        else:
            arguments = ExecutionArguments(self.arguments)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(
            self,
            "output_binding",
            copy_execution_variable_binding(self.output_binding),
        )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Structured execution plan generated before any tool execution."""

    goal: str
    ordered_steps: tuple[ExecutionStep, ...]
    estimated_steps: int
    required_tools: tuple[str, ...]
    detected_risks: tuple[str, ...]
    requires_confirmation: bool
    status: str = "planned"


class PlannerErrorCode(str, Enum):
    """Stable error codes for execution-plan generation."""

    EMPTY_OBJECTIVE = "EMPTY_OBJECTIVE"
    PLAN_GENERATION_FAILED = "PLAN_GENERATION_FAILED"
    INVALID_PLAN_RESPONSE = "INVALID_PLAN_RESPONSE"
    PLAN_PARSE_ERROR = "PLAN_PARSE_ERROR"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    MISSING_REQUIRED_ARGUMENT = "MISSING_REQUIRED_ARGUMENT"
    INVALID_STEP_REFERENCE = "INVALID_STEP_REFERENCE"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    INTERNAL_PLANNER_ERROR = "INTERNAL_PLANNER_ERROR"


@dataclass(frozen=True, slots=True)
class PlanGenerationResult:
    """Structured result for advanced execution-plan generation."""

    success: bool
    plan: ExecutionPlan | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_response: str | None = None
    generation_attempted: bool = False
    error_code: str | None = None


class Planner:
    """Creates an execution plan from the user's request."""

    _INTENT_TOOL_MAP = {
        "file.read": "read_file",
        "file.write": "write_file",
        "directory.list": "list_directory",
        "project.tree": "project_tree",
        "desktop.application.open": "desktop.open_application",
        "desktop.file.open": "desktop.open_file",
        "desktop.text.type": "desktop.type_text",
        "desktop.hotkey.press": "desktop.press_hotkey",
        "desktop.windows.list": "desktop.list_windows",
    }
    _CONFIRMATION_TOOLS = {
        "write_file",
        "desktop.type_text",
        "desktop.press_hotkey",
    }
    _INTENT_DESCRIPTIONS = {
        "file.read": "Read requested file content.",
        "file.write": "Write requested file content.",
        "directory.list": "List requested directory content.",
        "project.tree": "Inspect requested project tree.",
        "desktop.application.open": "Open requested desktop application.",
        "desktop.file.open": "Open requested file on the desktop.",
        "desktop.text.type": "Type requested text in the target window.",
        "desktop.hotkey.press": "Send requested keyboard shortcut.",
        "desktop.windows.list": "List matching desktop windows.",
    }
    _OUTPUT_DESCRIPTIONS = {
        "read_file": "UTF-8 file content as a string.",
        "write_file": "Human-readable write confirmation message.",
        "list_directory": "Directory entries.",
        "project_tree": "Project tree text.",
        "desktop.open_application": "Whether the application open action succeeded.",
        "desktop.open_file": "Whether the file open action succeeded.",
        "desktop.type_text": "Whether text typing succeeded.",
        "desktop.press_hotkey": "Whether hotkey execution succeeded.",
        "desktop.list_windows": "List of matching windows.",
    }

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        tool_selector: ToolSelector | None = None,
        schema_registry: ArgumentSchemaRegistry | None = None,
        argument_validator: ArgumentValidator | None = None,
        semantic_tool_catalog: SemanticToolCatalog | None = None,
        tool_selection_result: ToolSelectionResult | None = None,
        multi_tool_planner: Any | None = None,
        hybrid_execution_planner: Any | None = None,
        structured_plan_provider: Any | None = None,
        plan_response_provider: Callable[[str, tuple[ToolDescriptor, ...]], str] | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._tool_selector = tool_selector
        self._schema_registry = schema_registry
        self._argument_validator = (
            argument_validator
            or (ArgumentValidator(schema_registry) if schema_registry is not None else None)
        )
        self._semantic_tool_catalog = semantic_tool_catalog
        self._tool_selection_result = tool_selection_result
        self._multi_tool_planner = multi_tool_planner
        self._hybrid_execution_planner = hybrid_execution_planner
        self._structured_plan_provider = structured_plan_provider
        self._plan_response_provider = plan_response_provider

    def create_plan(self, prompt: str) -> Plan:
        text = prompt.lower()
        normalized_text = self._normalize_text(prompt)

        if self._is_architecture_query(normalized_text):
            return Plan(task="project", objective=prompt)

        if any(
            word in text
            for word in (
                "analiza este proyecto",
                "analiza el proyecto",
                "proyecto",
                "repositorio",
                "arquitectura",
                "estructura",
            )
        ):
            return Plan(task="project", objective=prompt)

        if any(
            word in text
            for word in (
                "lee ",
                "abrir ",
                "abre ",
                "mostrar ",
                "muestra ",
                "corrige ",
                "modifica ",
                "editar ",
                "edita ",
            )
        ):
            return Plan(task="coding", objective=prompt)

        if any(
            word in text
            for word in (
                "programa",
                "python",
                "codigo",
                "script",
                "funcion",
            )
        ):
            return Plan(task="coding", objective=prompt)

        if any(word in text for word in ("investiga", "buscar", "busca", "resume")):
            return Plan(task="research", objective=prompt)

        return Plan(task="chat", objective=prompt)

    def create_execution_plan(
        self,
        prompt: str,
    ) -> ExecutionPlan:
        """Create a structured execution plan without executing tools."""
        result = self.generate_execution_plan(prompt)
        if result.plan is not None:
            return result.plan

        return self._direct_response_plan(prompt.strip(), result.error_code or "Planning failed.")

    def generate_execution_plan(
        self,
        prompt: str,
        *,
        on_planning_progress: Callable[[Any], None] | None = None,
        planning_control: Any | None = None,
    ) -> PlanGenerationResult:
        """Create a structured execution plan with generation diagnostics."""
        goal = prompt.strip()
        if not goal:
            plan = self._direct_response_plan("", "Empty objective cannot be planned.")
            return PlanGenerationResult(
                success=False,
                plan=plan,
                errors=["Execution objective cannot be empty."],
                generation_attempted=True,
                error_code=PlannerErrorCode.EMPTY_OBJECTIVE.value,
            )

        catalog = self.tool_catalog()
        if self._plan_response_provider is not None:
            raw_response = self._plan_response_provider(goal, catalog)
            return self._parse_plan_response(goal, raw_response, catalog)

        multi_tool_result = self._try_multi_tool_plan(goal)
        if multi_tool_result is not None:
            return multi_tool_result

        hybrid_result = self._try_hybrid_plan(
            goal,
            on_planning_progress=on_planning_progress,
            planning_control=planning_control,
        )
        if hybrid_result is not None:
            return hybrid_result

        decision = ExecutionDecisionEngine(self._supported_intents()).decide(goal)
        if decision.mode == ExecutionMode.DIRECT_RESPONSE:
            return PlanGenerationResult(
                success=True,
                plan=self._direct_response_plan(goal, decision.reason),
                generation_attempted=True,
            )

        proposal_by_intent = self._build_step_proposals(goal, decision)
        ordered_steps = self._build_execution_steps(
            goal,
            decision.candidate_tools,
            proposal_by_intent,
        )
        required_tools = self._required_tools(ordered_steps)
        risks = self._detect_risks(ordered_steps, decision.mode)
        warnings = self._proposal_warnings(proposal_by_intent)
        errors = self._proposal_errors(proposal_by_intent)
        plan = ExecutionPlan(
            goal=goal,
            ordered_steps=ordered_steps,
            estimated_steps=len(ordered_steps),
            required_tools=required_tools,
            detected_risks=risks + tuple(warnings),
            requires_confirmation=self._requires_confirmation(required_tools),
            status="planned",
        )

        return PlanGenerationResult(
            success=not errors,
            plan=plan,
            errors=errors,
            warnings=warnings,
            generation_attempted=True,
            error_code=PlannerErrorCode.INSUFFICIENT_INFORMATION.value if errors else None,
        )

    def _try_hybrid_plan(
        self,
        goal: str,
        *,
        on_planning_progress: Callable[[Any], None] | None = None,
        planning_control: Any | None = None,
    ) -> PlanGenerationResult | None:
        if (
            self._hybrid_execution_planner is None
            or self._semantic_tool_catalog is None
            or self._tool_selector is None
        ):
            return None

        result = self._hybrid_execution_planner.plan(
            goal,
            deterministic_planner=None,
            catalog=self._semantic_tool_catalog,
            selector=self._tool_selector,
            plan_provider=self._structured_plan_provider,
            on_planning_progress=on_planning_progress,
            planning_control=planning_control,
        )
        if not result.handled:
            return None

        return PlanGenerationResult(
            success=result.success,
            plan=result.plan,
            errors=list(result.errors),
            warnings=list(result.warnings),
            raw_response=result.raw_response,
            generation_attempted=True,
            error_code=result.error_code,
        )

    def _try_multi_tool_plan(
        self,
        goal: str,
    ) -> PlanGenerationResult | None:
        if (
            self._multi_tool_planner is None
            or self._semantic_tool_catalog is None
            or self._tool_selector is None
        ):
            return None

        result = self._multi_tool_planner.plan(
            goal,
            self._semantic_tool_catalog,
            self._tool_selector,
        )
        if not result.handled:
            return None

        return PlanGenerationResult(
            success=result.success,
            plan=result.plan,
            errors=list(result.errors),
            warnings=list(result.warnings),
            generation_attempted=True,
            error_code=result.error_code,
        )

    def tool_catalog(self) -> tuple[ToolDescriptor, ...]:
        """Return deterministic tool metadata without mutating registries."""
        if self._tool_registry is None:
            return tuple(
                ToolDescriptor(
                    name=tool_name,
                    description=self._INTENT_DESCRIPTIONS.get(intent, f"Tool for intent '{intent}'."),
                    tool=_CatalogOnlyTool(tool_name),
                    requires_confirmation=tool_name in self._CONFIRMATION_TOOLS,
                    dangerous=tool_name in self._CONFIRMATION_TOOLS,
                    output_description=self._OUTPUT_DESCRIPTIONS.get(tool_name),
                )
                for intent, tool_name in self._INTENT_TOOL_MAP.items()
            )

        return tuple(self._enriched_descriptor(descriptor) for descriptor in self._tool_registry.descriptors())

    def _direct_response_plan(
        self,
        goal: str,
        reason: str,
    ) -> ExecutionPlan:
        """Create a safe plan for requests that do not map to tools."""
        step = ExecutionStep(
            id="step_1",
            description="Respond directly without tool execution.",
            tool="direct_response",
        )

        return ExecutionPlan(
            goal=goal,
            ordered_steps=(step,),
            estimated_steps=1,
            required_tools=(),
            detected_risks=(reason,),
            requires_confirmation=False,
            status="planned",
        )

    def _build_execution_steps(
        self,
        goal: str,
        candidate_tools: tuple[str, ...],
        proposals: Mapping[str, StructuredToolProposal],
    ) -> tuple[ExecutionStep, ...]:
        """Build ordered pending steps from detected tool intents."""
        steps: list[ExecutionStep] = []

        for index, intent in enumerate(candidate_tools, start=1):
            step_id = f"step_{index}"
            dependencies = self._dependencies_for_step(intent, steps)
            tool = self._INTENT_TOOL_MAP[intent]
            arguments = self._arguments_for_step(
                goal=goal,
                intent=intent,
                proposal=proposals.get(intent),
                previous_steps=tuple(steps),
            )

            steps.append(
                ExecutionStep(
                    id=step_id,
                    description=self._INTENT_DESCRIPTIONS.get(intent, f"Run tool intent '{intent}'."),
                    tool=tool,
                    dependencies=dependencies,
                    status="pending",
                    arguments=arguments,
                )
            )

        return tuple(steps)

    def _required_tools(
        self,
        ordered_steps: tuple[ExecutionStep, ...],
    ) -> tuple[str, ...]:
        """Return required tools in first-use order."""
        tools: list[str] = []

        for step in ordered_steps:
            if step.tool is not None and step.tool != "direct_response" and step.tool not in tools:
                tools.append(step.tool)

        return tuple(tools)

    def _detect_risks(
        self,
        ordered_steps: tuple[ExecutionStep, ...],
        mode: ExecutionMode,
    ) -> tuple[str, ...]:
        """Detect plan risks before execution."""
        risks: list[str] = []

        if mode == ExecutionMode.TOOL_CHAIN:
            risks.append("Multi-step plan must preserve dependency order.")

        for step in ordered_steps:
            if step.tool in self._CONFIRMATION_TOOLS:
                risks.append(f"Step {step.id} uses confirmation-gated tool '{step.tool}'.")

        return tuple(risks)

    def _requires_confirmation(
        self,
        required_tools: tuple[str, ...],
    ) -> bool:
        catalog = {descriptor.name: descriptor for descriptor in self.tool_catalog()}
        return any(
            tool in self._CONFIRMATION_TOOLS
            or (tool in catalog and catalog[tool].requires_confirmation)
            or (tool in catalog and catalog[tool].dangerous)
            for tool in required_tools
        )

    def _supported_intents(self) -> tuple[str, ...]:
        if self._tool_selector is not None:
            return self._tool_selector.supported_intents()
        return tuple(self._INTENT_TOOL_MAP)

    def _build_step_proposals(
        self,
        goal: str,
        decision: ExecutionDecision,
    ) -> dict[str, StructuredToolProposal]:
        if (
            self._tool_registry is None
            or self._tool_selector is None
            or self._schema_registry is None
            or self._argument_validator is None
        ):
            return {}

        builder = ToolProposalBuilder(
            self._tool_registry,
            self._tool_selector,
            self._schema_registry,
            self._argument_validator,
        )
        proposals: dict[str, StructuredToolProposal] = {}
        segments = _split_segments(goal)

        for index, intent in enumerate(decision.candidate_tools):
            segment = _segment_for_intent(intent, segments, index, goal)
            single_decision = replace(
                decision,
                mode=ExecutionMode.SINGLE_TOOL,
                candidate_tools=(intent,),
            )
            proposals[intent] = builder.build(segment, single_decision, candidate_tools=(intent,))

        return proposals

    def _arguments_for_step(
        self,
        *,
        goal: str,
        intent: str,
        proposal: StructuredToolProposal | None,
        previous_steps: tuple[ExecutionStep, ...],
    ) -> dict[str, Any]:
        if proposal is not None:
            arguments = dict(proposal.arguments)
        else:
            segment = _segment_for_intent(intent, _split_segments(goal), len(previous_steps), goal)
            arguments = self._fallback_arguments(segment, intent)

        if intent == "file.write":
            return self._write_arguments_with_references(goal, arguments, previous_steps)

        return arguments

    def _fallback_arguments(
        self,
        text: str,
        intent: str,
    ) -> dict[str, Any]:
        normalized = self._normalize_text(text)
        if intent in {"file.read", "desktop.file.open"}:
            path = _extract_path(text)
            return {"path": path} if path is not None else {}

        if intent == "file.write":
            path = _extract_last_path(text)
            content = _extract_literal_write_content(text, path)
            arguments: dict[str, Any] = {}
            if path is not None:
                arguments["path"] = path
            if content is not None:
                arguments["content"] = content
            return arguments

        if intent == "directory.list":
            return {"path": _extract_directory_path(text) or "."}

        if intent == "project.tree":
            return {"path": "."}

        if intent == "desktop.application.open":
            application = _extract_application(text, normalized)
            return {"application": application} if application is not None else {}

        if intent == "desktop.hotkey.press":
            keys = _extract_hotkey(normalized)
            return {"keys": keys} if keys is not None else {}

        if intent == "desktop.text.type":
            text_argument = _extract_type_text(text)
            return {"text": text_argument} if text_argument is not None else {}

        return {}

    def _write_arguments_with_references(
        self,
        goal: str,
        arguments: dict[str, Any],
        previous_steps: tuple[ExecutionStep, ...],
    ) -> dict[str, Any]:
        read_step = next((step for step in reversed(previous_steps) if step.tool == "read_file"), None)
        if read_step is None:
            return arguments

        segment = _segment_for_intent("file.write", _split_segments(goal), len(previous_steps), goal)
        prefix = _extract_template_prefix(segment)
        if prefix is not None:
            arguments["content"] = {"$template": f"{prefix}{{{{steps.{read_step.id}.output}}}}"}
        elif (
            "content" not in arguments
            or not arguments["content"]
            or _looks_like_previous_output_reference(str(arguments["content"]))
        ):
            arguments["content"] = {"$ref": f"steps.{read_step.id}.output"}

        return arguments

    def _dependencies_for_step(
        self,
        intent: str,
        previous_steps: list[ExecutionStep],
    ) -> tuple[str, ...]:
        if not previous_steps:
            return ()
        if intent == "file.write" and any(step.tool == "read_file" for step in previous_steps):
            return (previous_steps[-1].id,)
        return ()

    def _proposal_warnings(
        self,
        proposals: Mapping[str, StructuredToolProposal],
    ) -> list[str]:
        warnings: list[str] = []
        for intent, proposal in proposals.items():
            if proposal.status is ToolProposalStatus.COMPLETE:
                continue
            warnings.append(f"Intent '{intent}' generated {proposal.status.value} arguments: {proposal.reason}")
        return warnings

    def _proposal_errors(
        self,
        proposals: Mapping[str, StructuredToolProposal],
    ) -> list[str]:
        errors: list[str] = []
        for intent, proposal in proposals.items():
            if proposal.status is ToolProposalStatus.COMPLETE:
                continue
            for missing in proposal.missing_arguments:
                errors.append(f"{PlannerErrorCode.MISSING_REQUIRED_ARGUMENT.value}: {intent}.{missing}")
            for ambiguous in proposal.ambiguous_arguments:
                errors.append(f"{PlannerErrorCode.INSUFFICIENT_INFORMATION.value}: {intent}.{ambiguous}")
            for validation_error in proposal.validation_errors:
                errors.append(f"{PlannerErrorCode.INVALID_PLAN_RESPONSE.value}: {validation_error}")
        return errors

    def _enriched_descriptor(
        self,
        descriptor: ToolDescriptor,
    ) -> ToolDescriptor:
        if self._tool_selector is None or self._schema_registry is None:
            return descriptor

        argument_names: tuple[str, ...] = ()
        required: tuple[str, ...] = ()
        optional: tuple[str, ...] = ()
        for intent in self._tool_selector.supported_intents():
            try:
                selection = self._tool_selector.select(ToolIntent(intent))
            except ToolNotRegisteredError:
                continue
            if selection.tool_name != descriptor.name or not self._schema_registry.exists(intent):
                continue
            schema = self._schema_registry.get(intent)
            argument_names = tuple(field.name for field in schema.fields)
            required = tuple(field.name for field in schema.fields if field.required)
            optional = tuple(field.name for field in schema.fields if not field.required)
            break

        return replace(
            descriptor,
            argument_names=argument_names,
            required_arguments=required,
            optional_arguments=optional,
            dangerous=descriptor.requires_confirmation,
            output_description=self._OUTPUT_DESCRIPTIONS.get(descriptor.name),
        )

    def _parse_plan_response(
        self,
        goal: str,
        raw_response: str,
        catalog: tuple[ToolDescriptor, ...],
    ) -> PlanGenerationResult:
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as error:
            return PlanGenerationResult(
                success=False,
                errors=[f"Invalid JSON plan response: {error.msg}."],
                raw_response=raw_response,
                generation_attempted=True,
                error_code=PlannerErrorCode.PLAN_PARSE_ERROR.value,
            )

        try:
            plan = self._plan_from_payload(goal, payload, catalog)
        except PlannerPayloadError as error:
            return PlanGenerationResult(
                success=False,
                errors=[str(error)],
                raw_response=raw_response,
                generation_attempted=True,
                error_code=error.code,
            )

        return PlanGenerationResult(
            success=True,
            plan=plan,
            raw_response=raw_response,
            generation_attempted=True,
        )

    def _plan_from_payload(
        self,
        fallback_goal: str,
        payload: Any,
        catalog: tuple[ToolDescriptor, ...],
    ) -> ExecutionPlan:
        if not isinstance(payload, Mapping):
            raise PlannerPayloadError(PlannerErrorCode.INVALID_PLAN_RESPONSE.value, "Plan response must be a JSON object.")

        goal = payload.get("goal", fallback_goal)
        steps_payload = payload.get("steps")
        if not isinstance(goal, str) or not goal.strip():
            raise PlannerPayloadError(PlannerErrorCode.INVALID_PLAN_RESPONSE.value, "Plan response goal must be a non-empty string.")
        if not isinstance(steps_payload, list) or not steps_payload:
            raise PlannerPayloadError(PlannerErrorCode.INVALID_PLAN_RESPONSE.value, "Plan response must include a non-empty steps list.")

        available_tools = {descriptor.name: descriptor for descriptor in catalog}
        steps = tuple(
            self._step_from_payload(index, item, available_tools)
            for index, item in enumerate(steps_payload, start=1)
        )

        risks = payload.get("risks", [])
        if not isinstance(risks, list) or not all(isinstance(item, str) for item in risks):
            raise PlannerPayloadError(PlannerErrorCode.INVALID_PLAN_RESPONSE.value, "Plan risks must be a list of strings.")

        requires_confirmation = payload.get("requires_confirmation", None)
        if requires_confirmation is not None and not isinstance(requires_confirmation, bool):
            raise PlannerPayloadError(PlannerErrorCode.INVALID_PLAN_RESPONSE.value, "requires_confirmation must be a boolean when provided.")

        required_tools = self._required_tools(steps)
        return ExecutionPlan(
            goal=goal.strip(),
            ordered_steps=steps,
            estimated_steps=len(steps),
            required_tools=required_tools,
            detected_risks=tuple(risks) + self._detect_risks(steps, ExecutionMode.TOOL_CHAIN),
            requires_confirmation=bool(requires_confirmation) or self._requires_confirmation(required_tools),
            status="planned",
        )

    def _step_from_payload(
        self,
        index: int,
        payload: Any,
        available_tools: Mapping[str, ToolDescriptor],
    ) -> ExecutionStep:
        if not isinstance(payload, Mapping):
            raise PlannerPayloadError(PlannerErrorCode.INVALID_PLAN_RESPONSE.value, "Each plan step must be an object.")

        raw_id = payload.get("id", f"step_{index}")
        if isinstance(raw_id, int):
            step_id = f"step_{raw_id}"
        elif isinstance(raw_id, str) and raw_id.strip():
            step_id = raw_id.strip()
        else:
            raise PlannerPayloadError(PlannerErrorCode.INVALID_PLAN_RESPONSE.value, "Step id must be a non-empty string or integer.")

        description = payload.get("description")
        tool = payload.get("tool")
        arguments = payload.get("arguments", {})
        dependencies = payload.get("dependencies", [])
        if not isinstance(description, str) or not description.strip():
            raise PlannerPayloadError(PlannerErrorCode.INVALID_PLAN_RESPONSE.value, f"Step '{step_id}' description must be a non-empty string.")
        if tool is not None and not isinstance(tool, str):
            raise PlannerPayloadError(PlannerErrorCode.INVALID_PLAN_RESPONSE.value, f"Step '{step_id}' tool must be a string or null.")
        if tool is not None and tool not in available_tools:
            raise PlannerPayloadError(PlannerErrorCode.UNKNOWN_TOOL.value, f"Step '{step_id}' uses unknown tool '{tool}'.")
        if not isinstance(arguments, Mapping):
            raise PlannerPayloadError(PlannerErrorCode.INVALID_PLAN_RESPONSE.value, f"Step '{step_id}' arguments must be an object.")
        try:
            normalized_arguments = ExecutionArguments(arguments)
        except InvalidExecutionArgumentError as error:
            raise PlannerPayloadError(
                PlannerErrorCode.INVALID_PLAN_RESPONSE.value,
                f"Step '{step_id}' arguments are invalid: {error}.",
            ) from error
        if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
            raise PlannerPayloadError(PlannerErrorCode.INVALID_STEP_REFERENCE.value, f"Step '{step_id}' dependencies must be a list of strings.")

        return ExecutionStep(
            id=step_id,
            description=description.strip(),
            tool=tool,
            dependencies=tuple(dependencies),
            status="pending",
            arguments=normalized_arguments,
        )

    def _is_architecture_query(
        self,
        text: str,
    ) -> bool:
        """Detect deterministic architecture graph queries."""
        return (
            "quien usa" in text
            or "modulos dependen de" in text
            or "clases importa" in text
            or ("archivos" in text and "afectados" in text and "modifico" in text)
        )

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        """Normalize accents and punctuation for query detection."""
        normalized = unicodedata.normalize("NFKD", text.lower())
        without_accents = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        return without_accents.translate(
            {
                ord("?"): " ",
                ord("!"): " ",
                ord("¿"): " ",
                ord("¡"): " ",
            }
        )


class PlannerPayloadError(ValueError):
    """Raised when a structured plan response cannot become an ExecutionPlan."""

    def __init__(
        self,
        code: str,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


class _CatalogOnlyTool:
    """Minimal non-executable object for compatibility-only descriptors."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Catalog-only descriptor for {self._name}."

    @property
    def requires_confirmation(self) -> bool:
        return self._name in Planner._CONFIRMATION_TOOLS


def _split_segments(source_text: str) -> tuple[str, ...]:
    parts = re.split(
        r"\s*(?:,|\by\b|\bdespues\b|\bdespu.s\b|\bluego\b|\bentonces\b)\s*",
        source_text,
        flags=re.IGNORECASE,
    )
    return tuple(part.strip() for part in parts if part.strip())


def _segment_for_intent(
    intent: str,
    segments: tuple[str, ...],
    index: int,
    fallback: str,
) -> str:
    for segment in segments:
        if _segment_matches_intent(intent, _normalize(segment)):
            return segment
    if index < len(segments):
        return segments[index]
    return fallback


def _segment_matches_intent(intent: str, normalized: str) -> bool:
    patterns = {
        "file.read": (r"\blee", r"\bmuestra"),
        "directory.list": (r"\blista", r"\blistar"),
        "file.write": (r"\bescribe", r"\bcopia", r"\bguarda", r"\bguardalo", r"\bguardala"),
        "desktop.application.open": (r"\babre", r"\babrir"),
        "desktop.file.open": (r"\babre", r"\babrir"),
        "desktop.text.type": (r"\bescribe", r"\bteclea"),
        "desktop.hotkey.press": (r"\bpulsa", r"\batajo"),
    }
    return any(re.search(pattern, normalized) for pattern in patterns.get(intent, ()))


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(re.sub(r"[^\w\s./+:-]", " ", without_accents).split())


def _extract_path(source_text: str) -> str | None:
    quoted = _extract_quoted(source_text)
    if quoted and _looks_like_path(quoted):
        return quoted

    keyword_match = re.search(
        r"\b(?:archivo|fichero|ruta)\s+(?P<path>(?:[A-Za-z]:[\\/])?(?:[\w.-]+[\\/])*[\w.-]+\.(?:md|txt|py|json|csv|yaml|yml|toml))\b",
        source_text,
        flags=re.IGNORECASE,
    )
    if keyword_match:
        return keyword_match.group("path").strip()

    match = re.search(
        r"(?P<path>(?:[A-Za-z]:[\\/])?(?:[\w.-]+[\\/])*[\w.-]+\.(?:md|txt|py|json|csv|yaml|yml|toml))\b",
        source_text,
        flags=re.IGNORECASE,
    )
    return match.group("path").strip() if match else None


def _extract_last_path(source_text: str) -> str | None:
    matches = list(
        re.finditer(
            r"(?P<path>(?:[A-Za-z]:[\\/])?(?:[\w.-]+[\\/])*[\w.-]+\.(?:md|txt|py|json|csv|yaml|yml|toml))\b",
            source_text,
            flags=re.IGNORECASE,
        )
    )
    return matches[-1].group("path").strip() if matches else None


def _extract_directory_path(source_text: str) -> str | None:
    quoted = _extract_quoted(source_text)
    if quoted:
        return quoted
    match = re.search(
        r"\b(?:carpeta|directorio|ruta)\s+(?P<path>(?!esta\b|este\b|un\b|una\b)(?:[A-Za-z]:[\\/])?[\w./\\-]+)",
        source_text,
        flags=re.IGNORECASE,
    )
    return match.group("path").strip() if match else None


def _extract_literal_write_content(source_text: str, path: str | None) -> str | None:
    quoted = _extract_quoted(source_text)
    if quoted and quoted != path and not _looks_like_path(quoted):
        return quoted
    match = re.search(
        r"\b(?:escribe|guarda|copia)\s+(?P<content>.+?)\s+\ben\b",
        source_text,
        flags=re.IGNORECASE,
    )
    if match:
        content = match.group("content").strip()
        if content and "contenido" not in _normalize(content):
            return content
    return None


def _extract_template_prefix(source_text: str) -> str | None:
    match = re.search(r"\b(?:prefijo|mensaje)\s+[\"'](?P<prefix>.+?)[\"']", source_text, flags=re.IGNORECASE)
    if match:
        return match.group("prefix")
    match = re.search(
        r"\bmensaje\s+(?P<prefix>.+?)\s+(?:con|usando)\s+(?:su\s+)?contenido\b",
        source_text,
        flags=re.IGNORECASE,
    )
    return match.group("prefix").strip() if match else None


def _looks_like_previous_output_reference(value: str) -> bool:
    normalized = _normalize(value)
    return normalized in {
        "contenido",
        "su contenido",
        "el contenido",
        "resultado",
        "su resultado",
        "salida",
        "su salida",
    }


def _extract_application(source_text: str, normalized: str) -> str | None:
    aliases = {
        "vs code": "VS Code",
        "vscode": "VS Code",
        "visual studio code": "VS Code",
        "bloc de notas": "notepad",
        "notepad": "notepad",
    }
    for alias, value in aliases.items():
        if alias in normalized:
            return value
    return _extract_quoted(source_text)


def _extract_hotkey(normalized: str) -> list[str] | None:
    match = re.search(
        r"\b(?P<keys>(?:ctrl|control|alt|shift|mayus|win|windows)(?:\s*[+]\s*|\s+mas\s+|\s+)(?:[\w]+)(?:(?:\s*[+]\s*|\s+mas\s+|\s+)(?:[\w]+))*)\b",
        normalized,
    )
    if not match:
        return None
    return [key for key in re.split(r"\s*(?:[+]|\bmas\b)\s*|\s+", match.group("keys")) if key]


def _extract_type_text(source_text: str) -> str | None:
    quoted = _extract_quoted(source_text)
    if quoted:
        return quoted
    match = re.search(r"\b(?:escribe|teclea)\s+(?P<text>.+)$", source_text, flags=re.IGNORECASE)
    return match.group("text").strip() if match else None


def _extract_quoted(source_text: str) -> str | None:
    match = re.search(r"[\"'â€œâ€â€˜â€™](?P<value>.+?)[\"'â€œâ€â€˜â€™]", source_text)
    if not match:
        return None
    value = match.group("value").strip()
    return value or None


def _looks_like_path(value: str) -> bool:
    return bool(re.search(r"\.[A-Za-z0-9]{1,8}\b", value) or "/" in value or "\\" in value)

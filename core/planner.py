"""Planner for Atlas."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
import unicodedata

from tools.execution_decision import ExecutionDecisionEngine, ExecutionMode


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
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
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

    def create_plan(self, prompt: str) -> Plan:

        text = prompt.lower()
        normalized_text = self._normalize_text(prompt)

        if self._is_architecture_query(normalized_text):
            return Plan(
                task="project",
                objective=prompt,
            )

        # ---------------------------------
        # Project analysis
        # ---------------------------------

        if any(word in text for word in (
            "analiza este proyecto",
            "analiza el proyecto",
            "proyecto",
            "repositorio",
            "arquitectura",
            "estructura",
        )):
            return Plan(
                task="project",
                objective=prompt,
            )

        # ---------------------------------
        # File operations
        # ---------------------------------

        if any(word in text for word in (
            "lee ",
            "abrir ",
            "abre ",
            "mostrar ",
            "muestra ",
            "corrige ",
            "modifica ",
            "editar ",
            "edita ",
        )):
            return Plan(
                task="coding",
                objective=prompt,
            )

        # ---------------------------------
        # Coding
        # ---------------------------------

        if any(word in text for word in (
            "programa",
            "python",
            "código",
            "codigo",
            "script",
            "función",
            "funcion",
        )):
            return Plan(
                task="coding",
                objective=prompt,
            )

        # ---------------------------------
        # Research
        # ---------------------------------

        if any(word in text for word in (
            "investiga",
            "buscar",
            "busca",
            "resume",
        )):
            return Plan(
                task="research",
                objective=prompt,
            )

        # ---------------------------------
        # Default
        # ---------------------------------

        return Plan(
            task="chat",
            objective=prompt,
        )

    def create_execution_plan(
        self,
        prompt: str,
    ) -> ExecutionPlan:
        """Create a structured execution plan without executing tools."""
        goal = prompt.strip()
        decision = ExecutionDecisionEngine(
            tuple(self._INTENT_TOOL_MAP)
        ).decide(goal)

        if decision.mode == ExecutionMode.DIRECT_RESPONSE:
            return self._direct_response_plan(goal, decision.reason)

        ordered_steps = self._build_execution_steps(decision.candidate_tools)
        required_tools = self._required_tools(ordered_steps)
        risks = self._detect_risks(ordered_steps, decision.mode)

        return ExecutionPlan(
            goal=goal,
            ordered_steps=ordered_steps,
            estimated_steps=len(ordered_steps),
            required_tools=required_tools,
            detected_risks=risks,
            requires_confirmation=any(
                tool in self._CONFIRMATION_TOOLS
                for tool in required_tools
            ),
            status="planned",
        )

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
        candidate_tools: tuple[str, ...],
    ) -> tuple[ExecutionStep, ...]:
        """Build ordered pending steps from detected tool intents."""
        steps: list[ExecutionStep] = []

        for index, intent in enumerate(candidate_tools, start=1):
            step_id = f"step_{index}"
            dependencies = (steps[-1].id,) if steps else ()
            tool = self._INTENT_TOOL_MAP[intent]

            steps.append(
                ExecutionStep(
                    id=step_id,
                    description=self._INTENT_DESCRIPTIONS.get(
                        intent,
                        f"Run tool intent '{intent}'.",
                    ),
                    tool=tool,
                    dependencies=dependencies,
                    status="pending",
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
            if step.tool not in tools:
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
                risks.append(
                    f"Step {step.id} uses confirmation-gated tool '{step.tool}'."
                )

        return tuple(risks)

    def _is_architecture_query(
        self,
        text: str,
    ) -> bool:
        """Detect deterministic architecture graph queries."""
        return (
            "quien usa" in text
            or "modulos dependen de" in text
            or "clases importa" in text
            or (
                "archivos" in text
                and "afectados" in text
                and "modifico" in text
            )
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
            str.maketrans("¿?¡!", "    ")
        )

"""Deterministic operational classification for Atlas requests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
from types import MappingProxyType
from typing import Any
import unicodedata

from core.agent_orchestrator import AgentOrchestrator
from agents.registry import AgentRegistry
from core.request_gateway import AtlasRequest, RequestSource
from tools.registry import ToolDescriptor, ToolRegistry


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RequestRoute(str, Enum):
    DIRECT_RESPONSE = "direct_response"
    MEMORY_QUERY = "memory_query"
    SINGLE_TOOL = "single_tool"
    AGENT_DELEGATION = "agent_delegation"
    AUTONOMOUS_EXECUTION = "autonomous_execution"
    RESUME_EXECUTION = "resume_execution"
    SYSTEM_COMMAND = "system_command"
    CLARIFICATION_REQUIRED = "clarification_required"
    UNSUPPORTED = "unsupported"


class SystemCommand(str, Enum):
    EXIT = "exit"
    HELP = "help"
    STATUS = "status"
    CANCEL_EXECUTION = "cancel_execution"
    LIST_EXECUTIONS = "list_executions"
    EXECUTION_DETAIL = "execution_detail"
    DIAGNOSTICS = "diagnostics"
    VOICE_MODE = "voice_mode"
    STOP_LISTENING = "stop_listening"


class MemoryOperation(str, Enum):
    STORE = "store"
    RETRIEVE = "retrieve"
    FORGET = "forget"
    LIST = "list"
    UPDATE = "update"


class RequestRoutingError(RuntimeError):
    """Base operational routing error."""


class InvalidRoutingRuleError(ValueError):
    """Raised when a routing rule definition is invalid."""


class RoutingConfigurationError(ValueError):
    """Raised when router configuration is invalid."""


class RouteClassificationError(RequestRoutingError):
    """Raised for unexpected classifier failures."""


@dataclass(frozen=True, slots=True)
class OperationalRouterConfig:
    confidence_threshold: float = 0.35
    complexity_threshold: float = 0.65
    clarification_threshold: float = 0.5
    enabled_routes: frozenset[RequestRoute] = field(
        default_factory=lambda: frozenset(RequestRoute)
    )
    direct_response_enabled: bool = True
    autonomous_execution_enabled: bool = True
    strict_tool_matching: bool = True
    strict_agent_matching: bool = True
    router_version: str = "operational-router/14.2"

    def __post_init__(self) -> None:
        for name in (
            "confidence_threshold",
            "complexity_threshold",
            "clarification_threshold",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise RoutingConfigurationError(f"{name} must be between 0.0 and 1.0.")
            object.__setattr__(self, name, float(value))
        for name in (
            "direct_response_enabled",
            "autonomous_execution_enabled",
            "strict_tool_matching",
            "strict_agent_matching",
        ):
            if type(getattr(self, name)) is not bool:
                raise RoutingConfigurationError(f"{name} must be a bool.")
        object.__setattr__(
            self,
            "enabled_routes",
            frozenset(
                route if isinstance(route, RequestRoute) else RequestRoute(route)
                for route in self.enabled_routes
            ),
        )


@dataclass(frozen=True, slots=True)
class ComplexityAssessment:
    action_count: int = 0
    dependency_markers: int = 0
    domain_count: int = 0
    tool_count_estimate: int = 0
    agent_count_estimate: int = 0
    ambiguity_score: float = 0.0
    requires_planning: bool = False
    final_complexity: float = 0.0


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    route: RequestRoute
    score: float
    rule_id: str
    reason: str
    target: str | None = None
    missing_information: tuple[str, ...] = ()
    safety_flags: tuple[str, ...] = ()
    rule_priority: int = 0
    clarification_question: str | None = None
    confidence: float | None = None
    system_command: SystemCommand | None = None
    memory_operation: MemoryOperation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route, RequestRoute):
            object.__setattr__(self, "route", RequestRoute(self.route))
        if not 0.0 <= float(self.score) <= 1.0:
            raise InvalidRoutingRuleError("candidate score must be between 0.0 and 1.0.")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "missing_information", tuple(self.missing_information))
        object.__setattr__(self, "safety_flags", tuple(self.safety_flags))
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise InvalidRoutingRuleError("candidate confidence must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class RouteDecision:
    request_id: str
    route: RequestRoute
    confidence: float
    reason: str
    matched_rules: tuple[str, ...]
    target_tool_name: str | None = None
    target_agent_name: str | None = None
    target_session_id: str | None = None
    requires_confirmation: bool = False
    requires_clarification: bool = False
    clarification_question: str | None = None
    safety_flags: tuple[str, ...] = ()
    fallback_route: RequestRoute | None = None
    created_at: datetime = field(default_factory=_utc_now)
    router_version: str = "operational-router/14.2"
    system_command: SystemCommand | None = None
    memory_operation: MemoryOperation | None = None
    complexity: ComplexityAssessment | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route, RequestRoute):
            object.__setattr__(self, "route", RequestRoute(self.route))
        if self.fallback_route is not None and not isinstance(self.fallback_route, RequestRoute):
            object.__setattr__(self, "fallback_route", RequestRoute(self.fallback_route))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware.")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "matched_rules", tuple(self.matched_rules))
        object.__setattr__(self, "safety_flags", tuple(self.safety_flags))


@dataclass(frozen=True, slots=True)
class RoutingRule:
    rule_id: str
    priority: int
    supported_sources: frozenset[RequestSource]
    route: RequestRoute
    reason: str

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise InvalidRoutingRuleError("rule_id must be non-empty.")
        object.__setattr__(
            self,
            "supported_sources",
            frozenset(
                source if isinstance(source, RequestSource) else RequestSource(source)
                for source in self.supported_sources
            ),
        )
        if not isinstance(self.route, RequestRoute):
            object.__setattr__(self, "route", RequestRoute(self.route))


@dataclass(frozen=True, slots=True)
class RouterEvent:
    event_type: str
    request_id: str
    route: RequestRoute | None
    rule_id: str | None
    score: float | None
    target_name: str | None
    reason_code: str
    source: RequestSource
    timestamp: datetime


class OperationalRequestRouter:
    """Classify AtlasRequest values without executing any downstream route."""

    _PRECEDENCE: Mapping[RequestRoute, int] = MappingProxyType(
        {
            RequestRoute.RESUME_EXECUTION: 0,
            RequestRoute.SYSTEM_COMMAND: 1,
            RequestRoute.MEMORY_QUERY: 2,
            RequestRoute.SINGLE_TOOL: 3,
            RequestRoute.AGENT_DELEGATION: 4,
            RequestRoute.AUTONOMOUS_EXECUTION: 5,
            RequestRoute.DIRECT_RESPONSE: 6,
            RequestRoute.CLARIFICATION_REQUIRED: 7,
            RequestRoute.UNSUPPORTED: 8,
        }
    )

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry | None = None,
        agent_registry: AgentRegistry | None = None,
        config: OperationalRouterConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._agent_registry = agent_registry
        self._config = config or OperationalRouterConfig()
        self._clock = clock or _utc_now
        self._events: list[RouterEvent] = []

    @property
    def events(self) -> tuple[RouterEvent, ...]:
        return tuple(self._events)

    @property
    def config(self) -> OperationalRouterConfig:
        return self._config

    def classify(self, request: AtlasRequest) -> RouteDecision:
        if not isinstance(request, AtlasRequest):
            raise RouteClassificationError("classify expects an AtlasRequest.")
        self._record("request_classification_started", request, None, "started")
        normalized = _normalize_for_matching(request.content)
        complexity = self._assess_complexity(normalized)
        candidates = self._generate_candidates(request, normalized, complexity)
        for candidate in candidates:
            self._record(
                "route_candidate_generated",
                request,
                candidate,
                candidate.rule_id,
            )
        selected = self._select_candidate(candidates, request)
        for candidate in candidates:
            if candidate is not selected:
                self._record(
                    "route_candidate_rejected",
                    request,
                    candidate,
                    candidate.rule_id,
                )
        decision = self._decision_from_candidate(request, selected, complexity)
        event_type = "route_selected"
        if decision.route is RequestRoute.CLARIFICATION_REQUIRED:
            event_type = "clarification_required"
        elif decision.route is RequestRoute.UNSUPPORTED:
            event_type = "unsupported_request_detected"
        self._record(event_type, request, selected, selected.rule_id)
        return decision

    def _generate_candidates(
        self,
        request: AtlasRequest,
        normalized: str,
        complexity: ComplexityAssessment,
    ) -> tuple[RouteCandidate, ...]:
        candidates: list[RouteCandidate] = []
        candidates.extend(self._resume_candidates(request, normalized))
        candidates.extend(self._system_candidates(normalized))
        candidates.extend(self._memory_candidates(normalized))
        candidates.extend(self._tool_candidates(request, normalized))
        candidates.extend(self._agent_candidates(normalized))
        candidates.extend(self._autonomous_candidates(normalized, complexity))
        candidates.extend(self._direct_candidates(normalized, complexity))
        candidates.extend(self._clarification_candidates(normalized))
        if not candidates:
            candidates.append(
                RouteCandidate(
                    RequestRoute.UNSUPPORTED,
                    0.4,
                    "unsupported.no_capability",
                    "No registered route can handle the request.",
                    rule_priority=1,
                )
            )
        return tuple(
            candidate
            for candidate in candidates
            if candidate.route in self._config.enabled_routes
        ) or (
            RouteCandidate(
                RequestRoute.UNSUPPORTED,
                0.35,
                "unsupported.disabled_routes",
                "Matching routes are disabled by configuration.",
            ),
        )

    def _resume_candidates(
        self,
        request: AtlasRequest,
        normalized: str,
    ) -> tuple[RouteCandidate, ...]:
        context = request.execution_context
        if request.source is RequestSource.RESUME and context is not None and context.session_id:
            return (
                RouteCandidate(
                    RequestRoute.RESUME_EXECUTION,
                    0.98,
                    "resume.source_context",
                    "Request source and execution context identify a resumable session.",
                    target=context.session_id,
                    rule_priority=100,
                ),
            )
        if context is not None and context.session_id and (
            context.confirmation_response is not None
            or context.recovery_authorization is not None
        ):
            return (
                RouteCandidate(
                    RequestRoute.RESUME_EXECUTION,
                    0.96,
                    "resume.context_confirmation",
                    "Execution context carries a confirmation or recovery response.",
                    target=context.session_id,
                    rule_priority=95,
                ),
            )
        if _contains_any(normalized, ("continua", "reanuda", "recupera", "confirma", "rechaza")):
            session_id = _extract_session_id(normalized)
            if session_id is not None:
                return (
                    RouteCandidate(
                        RequestRoute.RESUME_EXECUTION,
                        0.9,
                        "resume.text_session",
                        "Text requests continuation of a named execution session.",
                        target=session_id,
                        rule_priority=90,
                    ),
                )
            return (
                RouteCandidate(
                    RequestRoute.CLARIFICATION_REQUIRED,
                    0.86,
                    "resume.missing_session",
                    "Resume intent is clear but no session_id is available.",
                    missing_information=("session_id",),
                    rule_priority=90,
                    clarification_question="Que ejecucion quieres continuar? Indica el session_id.",
                ),
            )
        return ()

    def _system_candidates(self, normalized: str) -> tuple[RouteCandidate, ...]:
        command = _system_command(normalized)
        if command is None:
            return ()
        return (
            RouteCandidate(
                RequestRoute.SYSTEM_COMMAND,
                0.95,
                f"system.{command.value}",
                "Request is an explicit internal Atlas command.",
                target=command.value,
                rule_priority=80,
                system_command=command,
            ),
        )

    def _memory_candidates(self, normalized: str) -> tuple[RouteCandidate, ...]:
        operation = _memory_operation(normalized)
        if operation is None:
            return ()
        return (
            RouteCandidate(
                RequestRoute.MEMORY_QUERY,
                0.9,
                f"memory.{operation.value}",
                "Request explicitly addresses Atlas memory.",
                target=operation.value,
                rule_priority=70,
                memory_operation=operation,
            ),
        )

    def _tool_candidates(
        self,
        request: AtlasRequest,
        normalized: str,
    ) -> tuple[RouteCandidate, ...]:
        if any(marker in normalized for marker in (" y despues ", " luego ", " despues ")):
            return ()
        action = _single_tool_action(normalized)
        if action is None:
            return ()
        if _ambiguous_missing_object(normalized):
            return (
                RouteCandidate(
                    RequestRoute.CLARIFICATION_REQUIRED,
                    0.85,
                    "tool.missing_target",
                    "A concrete tool action is missing its target.",
                    missing_information=("target",),
                    rule_priority=65,
                    clarification_question="Que archivo, aplicacion u objetivo quieres usar?",
                ),
            )
        descriptors = self._tool_descriptors()
        matches = tuple(
            descriptor
            for descriptor in descriptors
            if _tool_matches(descriptor, normalized, action, self._config.strict_tool_matching)
        )
        if not matches:
            if action in {"open", "read", "write", "list", "copy", "status", "send"}:
                return (
                    RouteCandidate(
                        RequestRoute.UNSUPPORTED,
                        0.65,
                        f"tool.{action}.unavailable",
                        "No registered tool matches the requested action.",
                        rule_priority=20,
                    ),
                )
            return ()
        ordered = sorted(matches, key=lambda item: item.name)
        best_score = max(
            _tool_match_score(descriptor, normalized, action)
            for descriptor in ordered
        )
        tied = tuple(
            descriptor
            for descriptor in ordered
            if _tool_match_score(descriptor, normalized, action) == best_score
        )
        if len(tied) > 1:
            return (
                RouteCandidate(
                    RequestRoute.CLARIFICATION_REQUIRED,
                    0.82,
                    "tool.ambiguous",
                    "More than one registered tool matches the request.",
                    target=",".join(item.name for item in tied),
                    missing_information=("tool_name",),
                    rule_priority=64,
                    clarification_question="Que herramienta quieres usar?",
                ),
            )
        descriptor = tied[0]
        flags = _safety_flags_for_tool(descriptor, request)
        return (
            RouteCandidate(
                RequestRoute.SINGLE_TOOL,
                best_score,
                f"tool.{action}",
                "A single registered tool clearly matches the request.",
                target=descriptor.name,
                safety_flags=flags,
                rule_priority=60,
            ),
        )

    def _agent_candidates(self, normalized: str) -> tuple[RouteCandidate, ...]:
        descriptors = self._agent_descriptors()

        if self._agent_registry is not None:
            specialist_selection = AgentOrchestrator(self._agent_registry).select(normalized)
            if specialist_selection.primary_agent is not None:
                return (
                    RouteCandidate(
                        RequestRoute.AGENT_DELEGATION,
                        0.9,
                        f"agent.{specialist_selection.primary_agent}",
                        "A registered specialized agent matches the request.",
                        target=specialist_selection.primary_agent,
                        rule_priority=50,
                    ),
                )
        matches = tuple(
            (name, description, _agent_match_score(name, description, normalized))
            for name, description in descriptors
            if name != "chat"
        )
        matches = tuple(item for item in matches if item[2] >= 0.65)
        if not matches:
            return ()
        best = max(score for _name, _description, score in matches)
        tied = tuple(item for item in matches if item[2] == best)
        if len(tied) > 1:
            return (
                RouteCandidate(
                    RequestRoute.CLARIFICATION_REQUIRED,
                    0.75,
                    "agent.ambiguous",
                    "Multiple registered agents match the request.",
                    target=",".join(name for name, _description, _score in tied),
                    missing_information=("agent_name",),
                    rule_priority=50,
                    clarification_question="Que agente quieres usar?",
                ),
            )
        name, _description, score = tied[0]
        return (
            RouteCandidate(
                RequestRoute.AGENT_DELEGATION,
                score,
                f"agent.{name}",
                "A registered specialized agent matches the request.",
                target=name,
                rule_priority=50,
            ),
        )

    def _autonomous_candidates(
        self,
        normalized: str,
        complexity: ComplexityAssessment,
    ) -> tuple[RouteCandidate, ...]:
        if not self._config.autonomous_execution_enabled:
            return ()
        if complexity.final_complexity < self._config.complexity_threshold:
            return ()
        return (
            RouteCandidate(
                RequestRoute.AUTONOMOUS_EXECUTION,
                min(0.95, complexity.final_complexity),
                "autonomous.complexity",
                "Request contains multiple actions, dependencies, or planning markers.",
                rule_priority=40,
            ),
        )

    def _direct_candidates(
        self,
        normalized: str,
        complexity: ComplexityAssessment,
    ) -> tuple[RouteCandidate, ...]:
        if (
            _single_tool_action(normalized) is not None
            or _contains_any(normalized, ("continua", "reanuda", "recupera", "confirma", "rechaza"))
            or normalized in {"abre el archivo", "lee el archivo", "envialo", "usa el mejor agente"}
        ):
            return ()
        if not self._config.direct_response_enabled:
            return (
                RouteCandidate(
                    RequestRoute.UNSUPPORTED,
                    0.4,
                    "direct.disabled",
                    "Direct responses are disabled by configuration.",
                    rule_priority=5,
                ),
            )
        if complexity.final_complexity >= self._config.complexity_threshold:
            return ()
        if _looks_like_unsupported_action(normalized):
            return (
                RouteCandidate(
                    RequestRoute.UNSUPPORTED,
                    0.55,
                    "unsupported.action",
                    "The requested action has no compatible current route.",
                    rule_priority=5,
                ),
            )
        return (
            RouteCandidate(
                RequestRoute.DIRECT_RESPONSE,
                0.6,
                "direct.simple_request",
                "Request can be handled by the existing conversational flow.",
                rule_priority=1,
            ),
        )

    def _clarification_candidates(self, normalized: str) -> tuple[RouteCandidate, ...]:
        if normalized in {"abre el archivo", "lee el archivo", "envialo", "usa el mejor agente"}:
            return (
                RouteCandidate(
                    RequestRoute.CLARIFICATION_REQUIRED,
                    0.7,
                    "clarification.missing_reference",
                    "Request lacks an essential reference.",
                    missing_information=("reference",),
                    rule_priority=10,
                    clarification_question="Que elemento concreto quieres usar?",
                ),
            )
        return ()

    def _select_candidate(
        self,
        candidates: tuple[RouteCandidate, ...],
        request: AtlasRequest,
    ) -> RouteCandidate:
        if not candidates:
            return RouteCandidate(
                RequestRoute.UNSUPPORTED,
                0.35,
                "unsupported.no_candidate",
                "No route candidate was generated.",
            )
        ordered = tuple(sorted(candidates, key=_candidate_sort_key))
        first = ordered[0]
        if first.route is RequestRoute.DIRECT_RESPONSE:
            operational = tuple(
                candidate
                for candidate in ordered
                if candidate.route in {
                    RequestRoute.CLARIFICATION_REQUIRED,
                    RequestRoute.UNSUPPORTED,
                }
                and candidate.score >= self._config.clarification_threshold
            )
            if operational:
                return operational[0]
        semantic_tie = tuple(
            candidate
            for candidate in ordered
            if _candidate_sort_key(candidate)[:3] == _candidate_sort_key(first)[:3]
        )
        if len(semantic_tie) > 1 and first.route not in {
            RequestRoute.DIRECT_RESPONSE,
            RequestRoute.UNSUPPORTED,
        }:
            self._record("route_ambiguous", request, first, "ambiguous_tie")
            return RouteCandidate(
                RequestRoute.CLARIFICATION_REQUIRED,
                0.75,
                "clarification.ambiguous_tie",
                "Multiple route candidates remain equivalent after deterministic sorting.",
                target=",".join(candidate.rule_id for candidate in semantic_tie),
                missing_information=("route",),
                clarification_question="Que tipo de accion quieres realizar?",
            )
        return first

    def _decision_from_candidate(
        self,
        request: AtlasRequest,
        candidate: RouteCandidate,
        complexity: ComplexityAssessment,
    ) -> RouteDecision:
        safety_flags = tuple(dict.fromkeys(candidate.safety_flags))
        requires_confirmation = (
            "side_effects_disabled" in safety_flags
            or "tool_requires_confirmation" in safety_flags
            or request.safety_context.requires_confirmation_hint
        )
        return RouteDecision(
            request_id=request.request_id,
            route=candidate.route,
            confidence=candidate.confidence if candidate.confidence is not None else candidate.score,
            reason=candidate.reason,
            matched_rules=(candidate.rule_id,),
            target_tool_name=candidate.target
            if candidate.route is RequestRoute.SINGLE_TOOL
            else None,
            target_agent_name=candidate.target
            if candidate.route is RequestRoute.AGENT_DELEGATION
            else None,
            target_session_id=candidate.target
            if candidate.route is RequestRoute.RESUME_EXECUTION
            else None,
            requires_confirmation=requires_confirmation,
            requires_clarification=candidate.route is RequestRoute.CLARIFICATION_REQUIRED,
            clarification_question=candidate.clarification_question,
            safety_flags=safety_flags,
            fallback_route=RequestRoute.DIRECT_RESPONSE
            if candidate.route is RequestRoute.UNSUPPORTED and self._config.direct_response_enabled
            else None,
            created_at=self._clock(),
            router_version=self._config.router_version,
            system_command=candidate.system_command,
            memory_operation=candidate.memory_operation,
            complexity=complexity,
        )

    def _assess_complexity(self, normalized: str) -> ComplexityAssessment:
        actions = _count_actions(normalized)
        dependency_markers = sum(
            1
            for marker in (
                " y despues ",
                " luego ",
                " antes de ",
                " cuando termine ",
                " despues ",
                " tras ",
            )
            if marker in f" {normalized} "
        )
        domains = sum(
            1
            for markers in (
                ("archivo", "fichero", "codigo", ".py"),
                ("test", "prueba", "pytest"),
                ("investiga", "busca", "analiza"),
                ("abre", "aplicacion", "vscode", "bloc"),
            )
            if any(marker in normalized for marker in markers)
        )
        requires_planning = any(
            marker in normalized
            for marker in (
                "prepara",
                "coordina",
                "organiza",
                "construye",
                "implementa",
                "planifica",
                "varios archivos",
            )
        )
        score = min(
            1.0,
            actions * 0.25
            + dependency_markers * 0.25
            + max(0, domains - 1) * 0.15
            + (0.35 if requires_planning else 0.0),
        )
        return ComplexityAssessment(
            action_count=actions,
            dependency_markers=dependency_markers,
            domain_count=domains,
            tool_count_estimate=max(0, actions),
            agent_count_estimate=1 if requires_planning else 0,
            ambiguity_score=0.0,
            requires_planning=requires_planning,
            final_complexity=score,
        )

    def _tool_descriptors(self) -> tuple[ToolDescriptor, ...]:
        if self._tool_registry is None:
            return ()
        descriptors = getattr(self._tool_registry, "descriptors", None)
        if callable(descriptors):
            return tuple(descriptors())
        names = getattr(self._tool_registry, "list", lambda: ())()
        return tuple(self._tool_registry.descriptor(name) for name in names)

    def _agent_descriptors(self) -> tuple[tuple[str, str], ...]:
        if self._agent_registry is None:
            return ()
        names = tuple(sorted(self._agent_registry.list()))
        result = []
        for name in names:
            agent = self._agent_registry.get(name)
            if agent is None:
                continue
            result.append((name, str(getattr(agent, "description", ""))))
        return tuple(result)

    def _record(
        self,
        event_type: str,
        request: AtlasRequest,
        candidate: RouteCandidate | None,
        reason_code: str,
    ) -> None:
        self._events.append(
            RouterEvent(
                event_type=event_type,
                request_id=request.request_id,
                route=candidate.route if candidate is not None else None,
                rule_id=candidate.rule_id if candidate is not None else None,
                score=candidate.score if candidate is not None else None,
                target_name=candidate.target if candidate is not None else None,
                reason_code=reason_code,
                source=request.source,
                timestamp=self._clock(),
            )
        )


def _candidate_sort_key(candidate: RouteCandidate) -> tuple[int, int, float, str, str]:
    return (
        OperationalRequestRouter._PRECEDENCE[candidate.route],
        -candidate.rule_priority,
        -candidate.score,
        candidate.target or "",
        candidate.rule_id,
    )


def _normalize_for_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold().strip())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    without_punctuation = re.sub(r"[^\w\s.:_\\/-]", " ", without_accents)
    return " ".join(without_punctuation.split())


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _extract_session_id(text: str) -> str | None:
    match = re.search(r"execution\.session\.\d{6}|session[-_.:][a-z0-9_.:-]+", text)
    return match.group(0) if match else None


def _system_command(text: str) -> SystemCommand | None:
    exact = {
        "salir": SystemCommand.EXIT,
        "exit": SystemCommand.EXIT,
        "quit": SystemCommand.EXIT,
        "ayuda": SystemCommand.HELP,
        "help": SystemCommand.HELP,
        "estado": SystemCommand.STATUS,
        "status": SystemCommand.STATUS,
        "diagnostico": SystemCommand.DIAGNOSTICS,
        "diagnosticos": SystemCommand.DIAGNOSTICS,
        "listar ejecuciones": SystemCommand.LIST_EXECUTIONS,
        "lista ejecuciones": SystemCommand.LIST_EXECUTIONS,
        "ultimas ejecuciones": SystemCommand.LIST_EXECUTIONS,
        "modo voz": SystemCommand.VOICE_MODE,
        "detener escucha": SystemCommand.STOP_LISTENING,
    }
    if text in exact:
        return exact[text]
    if text.startswith("cancelar ejecucion") or text.startswith("cancela ejecucion"):
        return SystemCommand.CANCEL_EXECUTION
    if re.fullmatch(r"detalle de ejecucion\s+\S+", text):
        return SystemCommand.EXECUTION_DETAIL
    return None


def _memory_operation(text: str) -> MemoryOperation | None:
    if text.startswith(("recuerda que ", "guarda esto", "guarda que ")):
        return MemoryOperation.STORE
    if text.startswith(
        (
            "que recuerdas sobre",
            "que recuerdas de ",
            "recuerdame lo que",
            "cual fue mi ultimo",
        )
    ):
        return MemoryOperation.RETRIEVE
    if text.startswith(("olvida ", "borra de tu memoria", "elimina de tu memoria")):
        return MemoryOperation.FORGET
    if text in {"que recuerdas", "lista memoria", "muestra memoria"}:
        return MemoryOperation.LIST
    if text.startswith(
        (
            "actualiza el recuerdo",
            "cambia lo que recuerdas",
            "actualiza mi preferencia",
            "actualiza la preferencia",
            "cambia mi preferencia",
            "cambia la preferencia",
        )
    ):
        return MemoryOperation.UPDATE
    return None


def _single_tool_action(text: str) -> str | None:
    if _is_calendar_list_request(text):
        return "calendar_list"
    if _is_gmail_list_request(text):
        return "gmail_list"
    if text.startswith(("abre ", "abrir ", "open ")):
        return "open"
    if text.startswith(("lee ", "leer ", "read ")):
        if _mentions_email(text):
            return "gmail_read"
        return "read"
    if text.startswith(("escribe ", "guarda ", "write ")):
        return "write"
    if text.startswith(("envia ", "enviame ", "manda ", "mandame ", "send ")):
        if _mentions_email(text):
            return "gmail_send"
        return "send"
    if text.startswith(("lista ", "listar ", "muestra carpeta")):
        return "list"
    if "portapapeles" in text or "clipboard" in text:
        return "copy"
    if text.startswith(("estado de ", "comprueba estado")):
        return "status"
    if text in {"que hora es", "dime la hora", "hora actual"}:
        return "time"
    return None


def _is_gmail_list_request(text: str) -> bool:
    email_markers = ("correo", "correos", "email", "emails", "mail", "mails", "bandeja", "mensajes")
    action_markers = (
        "muestra ",
        "muestrame ",
        "lista ",
        "listame ",
        "ensename ",
        "ultimos ",
        "ultimo ",
        "mis ",
        "recientes ",
        "nuevos ",
        "show ",
        "list ",
    )
    return _contains_any(text, email_markers) and _contains_any(text, action_markers)


def _mentions_email(text: str) -> bool:
    return any(
        marker in text
        for marker in ("correo", "correos", "email", "emails", "mail", "mails", "gmail")
    )


def _is_calendar_list_request(text: str) -> bool:
    calendar_markers = ("calendario", "calendar")
    action_markers = (
        "lista ",
        "listar ",
        "muestra ",
        "consulta ",
        "consultar ",
        "busca ",
        "buscar ",
        "list ",
        "show ",
        "search ",
    )
    return _contains_any(text, calendar_markers) and _contains_any(text, action_markers)


def _ambiguous_missing_object(text: str) -> bool:
    return text in {
        "abre el archivo",
        "abre archivo",
        "lee el archivo",
        "leer archivo",
        "envialo",
        "guardalo",
    }


def _tool_matches(
    descriptor: ToolDescriptor,
    text: str,
    action: str,
    strict: bool,
) -> bool:
    if action == "calendar_list":
        return descriptor.name == "calendar_list_events"
    if action == "gmail_list":
        return descriptor.name == "gmail_list"
    if action == "gmail_read":
        return descriptor.name == "gmail_read"
    if action == "gmail_send":
        return descriptor.name == "gmail_send"
    haystack = _normalize_for_matching(f"{descriptor.name} {descriptor.description}")
    action_markers = {
        "open": ("open", "abre", "abrir", "desktop", "application", "aplicacion", "vscode", "notepad"),
        "read": ("read", "lee", "leer", "file", "archivo"),
        "write": ("write", "escribe", "guardar", "file", "archivo"),
        "list": ("list", "listar", "folder", "carpeta"),
        "calendar_list": ("calendar", "events", "calendario", "eventos"),
        "gmail_list": ("gmail", "email", "correo", "mail"),
        "gmail_read": ("gmail", "email", "correo", "mail"),
        "gmail_send": ("gmail", "email", "correo", "mail", "send", "enviar"),
        "copy": ("clipboard", "portapapeles", "copy"),
        "send": ("send", "enviar", "mensaje", "message", "whatsapp", "email", "correo", "mail"),
        "status": ("status", "estado", "process", "proceso"),
        "time": ("time", "hora", "date", "fecha"),
    }[action]
    if not any(marker in haystack for marker in action_markers):
        return False
    request_tokens = set(text.split())
    if action == "send" and "whatsapp" in request_tokens and "whatsapp" not in haystack:
        return False
    tool_tokens = (
        set(haystack.split())
        | set(haystack.replace(".", " ").replace("_", " ").split())
    )
    meaningful = {
        token
        for token in request_tokens
        - {"abre", "abrir", "lee", "leer", "el", "la", "los", "las", "de", "un", "una"}
        if not _looks_like_tool_argument(token)
    }
    if action in {"read", "write"}:
        meaningful.difference_update({"archivo", "fichero"})
    if "aplicacion" in meaningful and "application" in tool_tokens:
        meaningful = set(meaningful)
        meaningful.remove("aplicacion")
    if strict and meaningful and meaningful.isdisjoint(tool_tokens):
        if not any(token in haystack for token in meaningful):
            return False
    return True


def _looks_like_tool_argument(token: str) -> bool:
    """Keep concrete file/path arguments out of semantic tool matching."""
    return "." in token or "/" in token or "\\" in token


def _tool_match_score(descriptor: ToolDescriptor, text: str, action: str) -> float:
    haystack = _normalize_for_matching(f"{descriptor.name} {descriptor.description}")
    overlap = len(
        set(text.split()).intersection(
            set(haystack.replace(".", " ").replace("_", " ").split())
        )
    )
    score = 0.7 + min(0.2, overlap * 0.05)
    preferred_tool = {
        "read": (
            "gmail_read"
            if _mentions_email(text)
            else "read_file"
        ),
        "write": "write_file",
        "list": "list_directory",
        "calendar_list": "calendar_list_events",
        "gmail_list": "gmail_list",
        "gmail_read": "gmail_read",
        "gmail_send": "gmail_send",
    }.get(action)
    if descriptor.name == preferred_tool:
        score += 0.15
    if descriptor.requires_confirmation:
        score -= 0.02
    return round(max(0.0, min(1.0, score)), 3)


def _safety_flags_for_tool(
    descriptor: ToolDescriptor,
    request: AtlasRequest,
) -> tuple[str, ...]:
    flags = []
    if descriptor.requires_confirmation:
        flags.append("tool_requires_confirmation")
    if descriptor.dangerous:
        flags.append("dangerous_tool")
    if not request.safety_context.allow_side_effects:
        flags.append("side_effects_disabled")
    return tuple(flags)


def _agent_match_score(name: str, description: str, text: str) -> float:
    haystack = _normalize_for_matching(f"{name} {description}")
    markers = {
        "coding": ("codigo", "programa", "python", "bug", "refactor", "implementa"),
        "project": ("proyecto", "arquitectura", "router.py", "archivo", "clase"),
        "research": ("investiga", "busca", "research"),
        "chat": ("explica", "que es", "define"),
        "coach": ("entrenamiento", "crossfit", "hyrox", "halterofilia"),
    }
    score = 0.0
    for key, values in markers.items():
        if key in haystack and any(value in text for value in values):
            score = max(score, 0.75)
    overlap = len(set(text.split()).intersection(set(haystack.split())))
    if overlap:
        score = max(score, min(0.9, 0.6 + overlap * 0.05))
    return round(score, 3)


def _count_actions(text: str) -> int:
    action_markers = (
        "abre ",
        "lee ",
        "escribe ",
        "ejecuta ",
        "analiza ",
        "implementa ",
        "prepara ",
        "busca ",
        "comprueba ",
        "guarda ",
    )
    count = sum(1 for marker in action_markers if marker in f"{text} ")
    if any(marker in text for marker in (" y despues ", " luego ", " despues ")):
        count = max(count, 2)
    return count


def _looks_like_unsupported_action(text: str) -> bool:
    return text.startswith(
        (
            "envia un email",
            "manda un email",
            "publica en",
            "compra ",
            "reserva ",
            "paga ",
        )
    )

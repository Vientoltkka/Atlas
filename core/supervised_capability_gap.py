"""Deterministic, approval-gated detection of missing reusable capabilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re
import unicodedata

from core.agent_registry import AgentRegistry
from core.capability_resolver import CapabilityResolutionRequest, CapabilityResolver, ToolCapabilityProvider, WorkflowCapabilityProvider
from core.execution_plan_library import ExecutionPlanLibrary
from core.skill_system import SkillSystem
from tools.registry import ToolRegistry

_CELSIUS_TO_FAHRENHEIT = re.compile(r"\bcon(?:vert(?:ir|e|eis|imos|en)?|viert(?:e|es|en)?)\s+[-+]?\d+(?:[.,]\d+)?\s+grados?\s+celsius\s+a\s+fahrenheit\b", re.IGNORECASE)
_TERMS = ("convert", "temperature", "celsius", "fahrenheit")

@dataclass(frozen=True, slots=True)
class MissingCapabilityProposal:
    capability_id: str
    minimum_scope: str

    def present(self) -> str:
        return ("No dispongo de una capacidad registrada para realizar esa acción. "
                "He comprobado tools, skills y agentes disponibles y no encuentro una capacidad reutilizable. "
                "Puedo preparar una mejora mínima para añadir esta capacidad. La mejora requeriría "
                f"{self.minimum_scope}. ¿Quieres que prepare esta mejora para tu aprobación?")

class SupervisedCapabilityGapDetector:
    """Recognize this bounded request and inspect registries without execution."""
    def __init__(self, resolver: CapabilityResolver, *, skill_system: SkillSystem | None = None, agent_registry: AgentRegistry | None = None) -> None:
        self._resolver, self._skill_system, self._agent_registry = resolver, skill_system, agent_registry

    @classmethod
    def from_registries(cls, *, tool_registry: ToolRegistry, execution_plan_libraries: Iterable[ExecutionPlanLibrary] = (), skill_system: SkillSystem | None = None, agent_registry: AgentRegistry | None = None) -> "SupervisedCapabilityGapDetector":
        providers = [ToolCapabilityProvider(tool_registry)]
        libraries = tuple(execution_plan_libraries)
        if libraries:
            providers.append(WorkflowCapabilityProvider(libraries))
        return cls(CapabilityResolver(tuple(providers)), skill_system=skill_system, agent_registry=agent_registry)

    def proposal_for(self, prompt: str) -> MissingCapabilityProposal | None:
        if not isinstance(prompt, str) or not _CELSIUS_TO_FAHRENHEIT.search(_normal(prompt)) or self._exists():
            return None
        return MissingCapabilityProposal("unit.temperature-conversion", "una tool determinista de conversión Celsius–Fahrenheit, su registro y pruebas focalizadas")

    def _exists(self) -> bool:
        resolution = self._resolver.resolve(CapabilityResolutionRequest(enabled_only=True))
        for candidate in resolution.candidates:
            item = candidate.capability
            if _matches(" ".join((item.capability_id, item.title, item.description, *item.categories, *item.tags, *item.input_names, *item.output_names))):
                return True
        if self._skill_system is not None:
            for skill in self._skill_system.skill_registry.list_skills():
                if _matches(" ".join(map(str, (skill.skill_id, skill.name, skill.description, *skill.required_capability_ids, *skill.tags)))):
                    return True
        if self._agent_registry is not None:
            for agent in self._agent_registry.list_agents(enabled_only=True):
                if _matches(" ".join((agent.agent_id, agent.name, agent.description, *agent.capabilities.capabilities, *agent.capabilities.tags))):
                    return True
        return False

def _matches(text: str) -> bool:
    return all(term in _normal(text) for term in _TERMS)

def _normal(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(character for character in decomposed if unicodedata.category(character) != "Mn").casefold()

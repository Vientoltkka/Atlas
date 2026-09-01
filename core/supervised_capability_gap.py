"""Deterministic, approval-gated detection of missing reusable capabilities."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
import hmac
import re
import unicodedata

from core.agent_registry import AgentRegistry
from core.capability_resolver import CapabilityResolutionRequest, CapabilityResolver, ToolCapabilityProvider, WorkflowCapabilityProvider
from core.execution_plan_library import ExecutionPlanLibrary
from core.skill_system import SkillSystem
from tools.registry import ToolRegistry

_CELSIUS_TO_FAHRENHEIT = re.compile(r"\bcon(?:vert(?:ir|e|eis|imos|en)?|viert(?:e|es|en)?)\s+[-+]?\d+(?:[.,]\d+)?\s+grados?\s+celsius\s+a\s+fahrenheit\b", re.IGNORECASE)
_TERMS = ("convert", "temperature", "celsius", "fahrenheit")
_SKILL_CREATION_INTENT = re.compile(r"\b(?:crea|crear|necesito)\s+(?:una\s+)?skill\s+(?:que|para)\s*(?P<request>.*)$", re.IGNORECASE)
_SKILL_CREATION_BARE_INTENT = re.compile(r"\b(?:crea|crear|necesito)\s+(?:una\s+)?skill\b", re.IGNORECASE)
_STOP_TERMS = frozenset({"a", "al", "atlas", "con", "de", "del", "el", "en", "la", "las", "lo", "los", "para", "por", "que", "una", "un", "y"})
_TERM_ALIASES = {"convertir": "convert", "convierta": "convert", "convierte": "convert", "mayuscula": "uppercase", "mayusculas": "uppercase", "texto": "text"}

@dataclass(frozen=True, slots=True)
class MissingCapabilityProposal:
    capability_id: str
    minimum_scope: str

    @property
    def planned_files(self) -> tuple[str, ...]:
        """Return the smallest expected implementation surface."""
        return (
            "tools/temperature_conversion.py",
            "bootstrap/bootstrap.py",
            "tests/test_temperature_conversion.py",
        )

    @property
    def focused_tests(self) -> tuple[str, ...]:
        """Return the focused tests needed before applying a future change."""
        return (
            "conversión correcta de Celsius a Fahrenheit",
            "registro de la tool y ruta normal",
        )

    @property
    def risk(self) -> str:
        """Describe the bounded impact of the future implementation."""
        return "Añade una capacidad nueva; debe registrarse sin alterar rutas existentes."
    def present(self) -> str:
        return ("No dispongo de una capacidad registrada para realizar esa acción. "
                "He comprobado tools, skills y agentes disponibles y no encuentro una capacidad reutilizable. "
                "Puedo preparar una mejora mínima para añadir esta capacidad. La mejora requeriría "
                f"{self.minimum_scope}. ¿Quieres que prepare esta mejora para tu aprobación?")

@dataclass(frozen=True, slots=True)
class SkillCreationResponse:
    """Read-only result for a natural-language request to create a Skill."""
    status: str
    requested_capability: str = ""
    existing_name: str | None = None
    existing_type: str | None = None
    reason: str = ""
    skill_id: str | None = None
    reusable_capabilities: tuple[str, ...] = ()
    handler_id: str | None = None
    authorization_token: str | None = None

    def present(self) -> str:
        if self.status == "REUSE":
            return "\n".join(("REUSE", f"- Capacidad solicitada: {self.requested_capability}", f"- Capacidad existente: {self.existing_name}", f"- Tipo: {self.existing_type}", f"- Motivo: {self.reason}", "- Recomendación: reutilizar la capacidad existente; no se creará una Skill duplicada."))
        if self.status == "CLARIFICATION_REQUIRED":
            return "CLARIFICATION_REQUIRED\n- Falta describir qué debe hacer la Skill."
        authorization = f"AUTORIZAR {self.skill_id} {self.authorization_token}"
        return "\n".join(("CREATE_PROPOSAL", f"- Capacidad solicitada: {self.requested_capability}", f"- Skill propuesta: {self.skill_id}", f"- Propósito: {self.requested_capability}", "- Alcance: manifiesto declarativo conectado exclusivamente a un handler registrado.", "- Capacidades reutilizables: " + (", ".join(self.reusable_capabilities) or "ninguna identificada"), "- Archivo previsto: skills/builtin/<skill_id>/skill.json.", "- Riesgo: medio; requiere validar el target antes de cualquier escritura.", f"- Acción requerida: {authorization}"))
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

    def skill_creation_response_for(self, prompt: str) -> SkillCreationResponse | None:
        """Inspect registered capabilities before proposing a new Skill, without effects."""
        if not isinstance(prompt, str):
            return None
        match = _SKILL_CREATION_INTENT.search(prompt)
        if match is None:
            return SkillCreationResponse("CLARIFICATION_REQUIRED") if _SKILL_CREATION_BARE_INTENT.search(prompt) else None
        requested = _request_description(match.group("request"))
        if not requested:
            return SkillCreationResponse("CLARIFICATION_REQUIRED")
        existing = self._find_equivalent(requested)
        if existing is not None:
            name, kind = existing
            return SkillCreationResponse("REUSE", requested, name, kind, "La descripción coincide de forma suficiente con una capacidad registrada.")
        reusable = tuple(candidate.capability.capability_id for candidate in self._resolver.resolve(CapabilityResolutionRequest(enabled_only=True)).candidates[:3])
        skill_id = _proposed_skill_id(requested)
        handler_id = _declared_handler_id(requested)
        return SkillCreationResponse(
            "CREATE_PROPOSAL",
            requested,
            skill_id=skill_id,
            reusable_capabilities=reusable,
            handler_id=handler_id,
            authorization_token=_authorization_token(skill_id, handler_id, requested),
        )

    def apply_declarative_skill(
        self,
        proposal: SkillCreationResponse,
        authorization: str,
        project_root: Path,
    ) -> str:
        """Apply one approved manifest-only Skill and roll back exactly on failure."""
        if (
            proposal.status != "CREATE_PROPOSAL"
            or not proposal.skill_id
            or not proposal.handler_id
            or not proposal.authorization_token
            or not hmac.compare_digest(
                authorization,
                _authorization_text(proposal.skill_id, proposal.authorization_token),
            )
        ):
            return "UNSUPPORTED_FOR_SAFE_CREATION: requiere un handler existente explícito. No se han realizado cambios."
        if self._skill_system is None or self._skill_system.skill_registry.contains(proposal.skill_id):
            return "UNSUPPORTED_FOR_SAFE_CREATION: skill_id ocupado o runtime no disponible. No se han realizado cambios."
        source = next((item for item in self._skill_system.skill_registry.list_skills(enabled_only=True) if item.execution_target_type.value == "handler" and item.handler_id == proposal.handler_id), None)
        handler_registry = getattr(self._skill_system.skill_executor, "_handler_registry", None)
        try:
            handler_registry.get(proposal.handler_id)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            handler_registry = None
        if source is None or handler_registry is None:
            return "UNSUPPORTED_FOR_SAFE_CREATION: handler no registrado. No se han realizado cambios."
        before = self._skill_system.skill_registry.list_skills(enabled_only=False)
        root = project_root.resolve()
        builtin_root = (root / "skills" / "builtin").resolve()
        target = (builtin_root / proposal.skill_id.removeprefix("skill.")).resolve()
        manifest_path = target / "skill.json"
        if builtin_root not in target.parents or target.exists():
            return "UNSUPPORTED_FOR_SAFE_CREATION: ruta no permitida o existente. No se han realizado cambios."
        manifest = {"schema_version": "1.0", "skill_id": proposal.skill_id, "name": proposal.requested_capability.title(), "version": "1.0", "description": proposal.requested_capability, "enabled": True, "input_fields": [{"name": field.name, "type": field.type_name, "required": field.required} for field in source.input_fields], "output_fields": [{"name": field.name, "type": field.type_name, "required": field.required} for field in source.output_fields], "execution_target": source.execution_target, "execution_target_type": "handler", "handler_id": source.handler_id}
        content = json.dumps(manifest, ensure_ascii=True, indent=2) + "\n"
        created_paths = tuple(path for path in (target, target.parent, target.parent.parent) if not path.exists())
        try:
            target.mkdir(parents=True)
            manifest_path.write_text(content, encoding="utf-8")
            if manifest_path.read_text(encoding="utf-8") != content:
                raise RuntimeError("read-back fallido")
            loaded = self._skill_system.skill_manifest_loader.load(content)
            if not loaded.valid or loaded.definition is None:
                raise RuntimeError("manifest inválido")
            self._skill_system.skill_registry.register(loaded.definition)
            return f"SKILL_CREATED: {proposal.skill_id} registrada y disponible."
        except Exception as error:
            if self._skill_system.skill_registry.contains(proposal.skill_id):
                self._skill_system.skill_registry.unregister(proposal.skill_id)
            if manifest_path.exists():
                manifest_path.unlink()
            for path in created_paths:
                if path.exists():
                    path.rmdir()
            if self._skill_system.skill_registry.list_skills(enabled_only=False) != before:
                return "SKILL_CREATION_ROLLED_BACK: rollback del registry no preservó el snapshot"
            return f"SKILL_CREATION_ROLLED_BACK: {error}"
    def _find_equivalent(self, requested: str) -> tuple[str, str] | None:
        request_terms = _meaningful_terms(requested)
        for candidate in self._resolver.resolve(CapabilityResolutionRequest(enabled_only=True)).candidates:
            item = candidate.capability
            text = " ".join((item.capability_id, item.title, item.description, *item.categories, *item.tags, *item.input_names, *item.output_names))
            if _is_clear_match(request_terms, text):
                return item.capability_id, item.capability_type.value
        if self._skill_system is not None:
            for skill in self._skill_system.skill_registry.list_skills(enabled_only=True):
                if _is_clear_match(request_terms, " ".join((skill.skill_id, skill.name, skill.description, *skill.required_capability_ids, *skill.tags))):
                    return skill.skill_id, "skill"
        if self._agent_registry is not None:
            for agent in self._agent_registry.list_agents(enabled_only=True):
                if _is_clear_match(request_terms, " ".join((agent.agent_id, agent.name, agent.description, *agent.capabilities.capabilities, *agent.capabilities.tags))):
                    return agent.agent_id, "agent"
        return None

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

def _request_description(value: str) -> str:
    return " ".join(value.split()).strip(" .,:;")

def _meaningful_terms(text: str) -> frozenset[str]:
    return frozenset(_TERM_ALIASES.get(term, term) for term in re.findall(r"[a-z0-9]+", _normal(text)) if term not in _STOP_TERMS)

def _is_clear_match(request_terms: frozenset[str], candidate_text: str) -> bool:
    if not request_terms:
        return False
    matched = request_terms.intersection(_meaningful_terms(candidate_text))
    return len(matched) >= 2 and len(matched) / len(request_terms) >= 0.66

def _proposed_skill_id(requested: str) -> str:
    terms = tuple(sorted(_meaningful_terms(requested)))[:4]
    return "skill." + ("-".join(terms) or "proposed-capability")


def _declared_handler_id(requested: str) -> str | None:
    match = re.search(r"\b(handler\.[a-z0-9_.-]+)\b", _normal(requested))
    return match.group(1) if match else None


def _authorization_token(skill_id: str, handler_id: str | None, requested: str) -> str:
    payload = "\0".join((skill_id, handler_id or "", requested)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _authorization_text(skill_id: str, token: str) -> str:
    return f"AUTORIZAR {skill_id} {token}"

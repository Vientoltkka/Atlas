"""Bounded, approval-gated self-improvement turns for the Atlas dialogue."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Protocol, runtime_checkable
import unicodedata

from core.supervised_repair import (
    ImprovementClassification,
    RepairProposal,
    RepairState,
    RepairValidation,
    SupervisedRepairWorkflow,
)


_AFFIRMATIVE = frozenset({"si", "s", "vale", "ok", "de acuerdo", "adelante"})
_NEGATIVE = frozenset({"no", "n", "cancelar", "cancela"})
_IMPROVEMENT = re.compile(
    r"\b(?:mejora|mejorar|corrige|corregir|repara|reparar|optimiza|optimizar|haz que puedas|crea la capacidad para)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ImprovementDiagnosis:
    """Read-only result of a narrow self-improvement diagnosis."""

    classification: ImprovementClassification
    objective: str
    scope: tuple[str, ...]
    focused_tests: tuple[str, ...]
    metrics: tuple[str, ...]
    risk: str
    finding: str


ProposalBuilder = Callable[[ImprovementDiagnosis, str], RepairProposal | None]
ValidatorFactory = Callable[[RepairProposal], Callable[[RepairProposal], RepairValidation]]


@runtime_checkable
class SupervisedRepairBuilder(Protocol):
    """Contract every supervised repair domain must satisfy.

    A builder owns one bounded domain: it recognizes the prompts it
    understands (diagnose), decides whether it can handle a diagnosis,
    produces one exact reviewed proposal and provides the trusted validator.
    It cannot bypass workflow scope, authorization, validation, acceptance or
    rollback controls.
    """

    def diagnose(self, prompt: str) -> ImprovementDiagnosis | None:
        """Return this builder's own domain diagnosis, or None if unrecognized."""
        ...

    def can_handle(self, diagnosis: ImprovementDiagnosis, prompt: str) -> bool: ...

    def build(self, diagnosis: ImprovementDiagnosis, prompt: str) -> RepairProposal | None: ...

    def validator(self, proposal: RepairProposal) -> RepairValidation: ...


@dataclass(frozen=True, slots=True)
class BuilderDiagnosis:
    """One builder's compatible diagnosis, in explicit registry order."""

    builder: SupervisedRepairBuilder
    diagnosis: ImprovementDiagnosis


class SupervisedRepairBuilderRegistry:
    """Small, explicit first-match registry. No dynamic discovery."""

    def __init__(self, builders: Sequence[SupervisedRepairBuilder]) -> None:
        self._builders = tuple(builders)

    def diagnose(self, prompt: str) -> tuple[BuilderDiagnosis, ...]:
        """All compatible builder diagnoses, deterministic in declared order."""
        return tuple(
            BuilderDiagnosis(builder, diagnosis)
            for builder in self._builders
            if (diagnosis := builder.diagnose(prompt)) is not None
        )

    def builder_for(self, diagnosis: ImprovementDiagnosis, prompt: str) -> SupervisedRepairBuilder | None:
        return next((builder for builder in self._builders if builder.can_handle(diagnosis, prompt)), None)


class _CallableRepairBuilder:
    """Adapter preserving the injected proposal_builder/validator_factory path."""

    def __init__(self, proposal_builder: ProposalBuilder, validator_factory: ValidatorFactory | None) -> None:
        self._build, self._validator_factory = proposal_builder, validator_factory

    def diagnose(self, prompt: str) -> ImprovementDiagnosis | None:
        return _legacy_domain_diagnosis(prompt)

    def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
        return diagnosis.classification is ImprovementClassification.CODE_REPAIR

    def build(self, diagnosis: ImprovementDiagnosis, prompt: str) -> RepairProposal | None:
        return self._build(diagnosis, prompt)

    def validator(self, proposal: RepairProposal) -> RepairValidation:
        factory = self._validator_factory or (lambda _proposal: _unavailable_validator)
        return factory(proposal)(proposal)


class SelfImprovementConversation:
    """Keeps a single active repair proposal in the normal conversation session."""

    def __init__(
        self,
        project_root: Path,
        *,
        builders: Sequence[SupervisedRepairBuilder] | None = None,
        proposal_builder: ProposalBuilder | None = None,
        validator_factory: ValidatorFactory | None = None,
    ) -> None:
        self._root = project_root
        if builders is not None:
            registry = SupervisedRepairBuilderRegistry(builders)
        elif proposal_builder is not None:
            registry = SupervisedRepairBuilderRegistry((_CallableRepairBuilder(proposal_builder, validator_factory),))
        else:
            # Built-in deterministic repairs and improvements only; no model output and no discovery.
            from core.voice_repair_builder import VoiceCodeRepairBuilder
            from core.routing_repair_builder import RoutingRepairBuilder
            from core.file_read_capability_builder import FileReadCapabilityImprovementBuilder
            from core.desktop_capability_builder import DesktopCapabilityImprovementBuilder

            registry = SupervisedRepairBuilderRegistry(
                (
                    VoiceCodeRepairBuilder(project_root),
                    RoutingRepairBuilder(project_root),
                    FileReadCapabilityImprovementBuilder(project_root),
                    DesktopCapabilityImprovementBuilder(project_root),
                )
            )
        self._builders = registry
        self._workflow: SupervisedRepairWorkflow | None = None
        self._diagnosis: ImprovementDiagnosis | None = None

    @property
    def active(self) -> bool:
        return self._workflow is not None

    @property
    def proposal(self) -> RepairProposal | None:
        """Expose the active immutable proposal without exposing workflow state."""
        return self._workflow.proposal if self._workflow is not None else None

    def handle(self, prompt: str) -> str | None:
        """Handle one pending authorization/finalization or detect a new request."""
        pending = self._handle_pending(prompt)
        if pending is not None:
            return pending
        if not self.is_self_improvement_request(prompt):
            return None
        diagnosis = self.diagnose(prompt)
        if diagnosis.classification is ImprovementClassification.CAPABILITY_GAP:
            return self._present_stop(diagnosis)
        if diagnosis.classification not in (ImprovementClassification.CODE_REPAIR, ImprovementClassification.CAPABILITY_IMPROVEMENT):
            return self._present_stop(diagnosis)
        builder = self._builders.builder_for(diagnosis, prompt)
        proposal = builder.build(diagnosis, prompt) if builder is not None else None
        if proposal is None:
            return self._present_stop(
                ImprovementDiagnosis(
                    ImprovementClassification.CLARIFICATION_REQUIRED,
                    diagnosis.objective,
                    diagnosis.scope,
                    diagnosis.focused_tests,
                    diagnosis.metrics,
                    diagnosis.risk,
                    "El diagnóstico está acotado, pero aún no hay un cambio exacto y revisable que sea seguro aplicar.",
                )
            )
        workflow = SupervisedRepairWorkflow(self._root, validator=builder.validator if builder is not None else _unavailable_validator)
        workflow.propose(proposal)
        self._workflow, self._diagnosis = workflow, diagnosis
        return self._present_proposal(diagnosis, proposal)

    @staticmethod
    def is_self_improvement_request(prompt: str) -> bool:
        if not isinstance(prompt, str) or not _IMPROVEMENT.search(prompt):
            return False
        text = _normal(prompt)
        # A normal editing request has neither Atlas nor one of its own surfaces.
        return any(term in text for term in ("atlas", "voz", "voice", "control pc", "desktop", "puedas", "capacidad"))

    def diagnose(self, prompt: str) -> ImprovementDiagnosis:
        """Classify a prompt without domain knowledge; builders own their domains."""
        text = _normal(prompt)
        if any(term in text for term in ("proveedor", "infraestructura", "integracion externa", "api externa")):
            return ImprovementDiagnosis(ImprovementClassification.CAPABILITY_GAP, prompt, (), (), (), "Requiere infraestructura o un proveedor externo no autorizado.", "La capacidad solicitada no puede resolverse sólo con cambios locales.")
        if "skill" in text:
            return ImprovementDiagnosis(ImprovementClassification.SKILL_GAP, prompt, (), (), (), "Puede duplicar una Skill existente.", "Hay que comprobar Skills registradas antes de proponer una nueva.")
        if "usa" in text and "existente" in text:
            return ImprovementDiagnosis(ImprovementClassification.REUSE, prompt, (), (), (), "Una duplicación puede romper rutas ya registradas.", "La petición indica reutilizar una capacidad existente.")
        matches = self._builders.diagnose(prompt)
        if len(matches) > 1:
            # Never pick between compatible builders silently.
            return _clarification_required(prompt)
        if len(matches) == 1:
            return matches[0].diagnosis
        if "capacidad" in text or "puedas" in text:
            return ImprovementDiagnosis(ImprovementClassification.CAPABILITY_GAP, prompt, (), (), (), "La capacidad concreta no está suficientemente definida.", "Indica qué entrada, salida y entorno necesita la nueva capacidad.")
        return _clarification_required(prompt)

    def _handle_pending(self, prompt: str) -> str | None:
        workflow = self._workflow
        if workflow is None:
            return None
        answer = _normal(prompt)
        if workflow.state is RepairState.PROPOSED:
            if answer in _NEGATIVE:
                self._clear()
                return "Reparación cancelada. No se han realizado cambios."
            if answer not in _AFFIRMATIVE:
                return "Hay una propuesta de reparación activa. Responde sí para autorizarla o no para cancelarla."
            proposal = self.proposal
            assert proposal is not None
            if not workflow.authorize_and_apply(proposal.authorization):
                self._clear()
                return "La autorización de la propuesta activa fue rechazada. No se han realizado cambios."
            validation = workflow.validate()
            if workflow.state is RepairState.ROLLED_BACK:
                self._clear()
                return "La validación falló; se restauró exactamente el alcance aprobado. " + validation.detail
            return self._present_validation(validation)
        if workflow.state is RepairState.VALIDATED:
            if answer not in _AFFIRMATIVE | _NEGATIVE:
                return "La reparación está validada. Responde sí para conservarla o no para restaurar el estado anterior."
            accepted = answer in _AFFIRMATIVE
            workflow.finalize(accepted=accepted)
            if accepted:
                self._clear()
                return "Reparación aceptada. Se conserva el cambio validado."
            self._clear()
            return "Reparación rechazada. Se restauró exactamente el estado anterior."
        return None

    def _clear(self) -> None:
        self._workflow, self._diagnosis = None, None

    @staticmethod
    def _present_proposal(diagnosis: ImprovementDiagnosis, proposal: RepairProposal) -> str:
        lines = [f"Detecto una posible reparación de Atlas ({diagnosis.classification.value})."]
        if diagnosis.classification is ImprovementClassification.CAPABILITY_IMPROVEMENT:
            lines.append("Esto añade/mejora una capacidad; no es una reparación de bug.")
        lines.extend((
            "No he modificado nada.",
            f"proposal_id: {proposal.proposal_id}",
            "", "Objetivo:", diagnosis.objective,
            "", "Qué creo que falla/falta:", diagnosis.finding,
            "", "Alcance previsto:", *(f"- {path}" for path in proposal.files),
            "", "Validación:", *(f"- {test}" for test in proposal.focused_tests),
            "", "Métricas:", *(f"- {metric}" for metric in diagnosis.metrics),
            "", "Riesgo:", diagnosis.risk,
            "", "¿Autorizas que aplique esta reparación? [sí/No]",
        ))
        return "\n".join(lines)

    @staticmethod
    def _present_stop(diagnosis: ImprovementDiagnosis) -> str:
        return "\n".join((diagnosis.classification.value, "No he modificado nada.", f"Diagnóstico: {diagnosis.finding}", f"Riesgo: {diagnosis.risk}"))

    @staticmethod
    def _present_validation(validation: RepairValidation) -> str:
        metrics = ", ".join(f"{name}: {validation.before_metrics.get(name)} -> {validation.after_metrics.get(name)}" for name in validation.after_metrics) or "sin métricas declaradas"
        return f"Validación completada. Antes/después: {metrics}. {validation.detail}\n¿Aceptas conservar esta reparación? [sí/No]"


def _normal(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn").casefold().strip()


def normalize_prompt(value: str) -> str:
    """Accent-insensitive, case-folded prompt text shared with domain builders."""
    return _normal(value)


def _clarification_required(prompt: str) -> ImprovementDiagnosis:
    return ImprovementDiagnosis(ImprovementClassification.CLARIFICATION_REQUIRED, prompt, (), (), (), "El alcance no es verificable.", "Indica la capacidad propia de Atlas que quieres mejorar.")


def _legacy_domain_diagnosis(prompt: str) -> ImprovementDiagnosis | None:
    """Legacy static domain recognition kept only for the proposal_builder path."""
    text = _normal(prompt)
    if any(term in text for term in ("voz", "voice")):
        return ImprovementDiagnosis(
            ImprovementClassification.CODE_REPAIR,
            "Evitar que un timeout de modelo bloquee indefinidamente el siguiente turno de voz.",
            ("use_cases/voice_conversation.py", "tests/test_voice_conversation.py"),
            ("tests/test_voice_conversation.py",),
            ("latencia de espera post-timeout del modelo",),
            "Puede afectar la interacción de voz; no se tocarán proveedores, secretos ni dependencias.",
            "Un worker de modelo expirado puede dejar la siguiente interacción esperando sin límite si ignora la cancelación.",
        )
    if any(term in text for term in ("control pc", "desktop")):
        return ImprovementDiagnosis(
            ImprovementClassification.CODE_REPAIR,
            "Mejorar Control PC dentro del comportamiento solicitado.",
            ("use_cases/desktop_interaction.py", "tests/test_desktop_interaction.py"),
            ("tests/test_desktop_interaction.py",),
            ("éxito de la operación solicitada",),
            "Puede afectar acciones locales; se mantiene la confirmación existente.",
            "El alcance se limita a Control PC y sus pruebas.",
        )
    if any(term in text for term in ("routing", "router", "rutas")):
        return ImprovementDiagnosis(
            ImprovementClassification.CODE_REPAIR,
            "Evitar que el enrutado de tareas dependa de mayúsculas accidentales.",
            ("core/router.py", "tests/test_router.py"),
            ("tests/test_router.py",),
            ("rutas de tarea sensibles a mayusculas correctas",),
            "Puede afectar qué agente ejecuta cada plan; alcance limitado al router y sus pruebas.",
            "Un nombre de tarea con mayúsculas cae al agente chat en lugar del agente correcto.",
        )
    return None


def _unavailable_validator(_proposal: RepairProposal) -> RepairValidation:
    return RepairValidation(False, detail="No hay un validador confiable configurado para esta reparación.")

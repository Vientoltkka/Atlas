"""Bounded, approval-gated self-improvement turns for the Atlas dialogue."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
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


class SelfImprovementConversation:
    """Keeps a single active repair proposal in the normal conversation session."""

    def __init__(
        self,
        project_root: Path,
        *,
        proposal_builder: ProposalBuilder | None = None,
        validator_factory: ValidatorFactory | None = None,
    ) -> None:
        self._root = project_root
        if proposal_builder is None:
            # The only built-in repair is a deterministic voice patch, not model output.
            from core.voice_repair_builder import VoiceCodeRepairBuilder

            voice_repair = VoiceCodeRepairBuilder(project_root)
            proposal_builder = voice_repair.build
            validator_factory = lambda _proposal: voice_repair.validator
        self._proposal_builder = proposal_builder
        self._validator_factory = validator_factory or (lambda _proposal: _unavailable_validator)
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
        if diagnosis.classification is not ImprovementClassification.CODE_REPAIR:
            return self._present_stop(diagnosis)
        proposal = self._proposal_builder(diagnosis, prompt) if self._proposal_builder else None
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
        workflow = SupervisedRepairWorkflow(self._root, validator=self._validator_factory(proposal))
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

    @staticmethod
    def diagnose(prompt: str) -> ImprovementDiagnosis:
        text = _normal(prompt)
        if any(term in text for term in ("proveedor", "infraestructura", "integracion externa", "api externa")):
            return ImprovementDiagnosis(ImprovementClassification.CAPABILITY_GAP, prompt, (), (), (), "Requiere infraestructura o un proveedor externo no autorizado.", "La capacidad solicitada no puede resolverse sólo con cambios locales.")
        if "skill" in text:
            return ImprovementDiagnosis(ImprovementClassification.SKILL_GAP, prompt, (), (), (), "Puede duplicar una Skill existente.", "Hay que comprobar Skills registradas antes de proponer una nueva.")
        if "usa" in text and "existente" in text:
            return ImprovementDiagnosis(ImprovementClassification.REUSE, prompt, (), (), (), "Una duplicación puede romper rutas ya registradas.", "La petición indica reutilizar una capacidad existente.")
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
        if "capacidad" in text or "puedas" in text:
            return ImprovementDiagnosis(ImprovementClassification.CAPABILITY_GAP, prompt, (), (), (), "La capacidad concreta no está suficientemente definida.", "Indica qué entrada, salida y entorno necesita la nueva capacidad.")
        return ImprovementDiagnosis(ImprovementClassification.CLARIFICATION_REQUIRED, prompt, (), (), (), "El alcance no es verificable.", "Indica la capacidad propia de Atlas que quieres mejorar.")

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
        return "\n".join((
            f"Detecto una posible reparación de Atlas ({diagnosis.classification.value}).",
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


def _unavailable_validator(_proposal: RepairProposal) -> RepairValidation:
    return RepairValidation(False, detail="No hay un validador confiable configurado para esta reparación.")

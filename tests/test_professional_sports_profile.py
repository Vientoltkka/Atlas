from __future__ import annotations

from agents.registry import AgentRegistry
from core.agent_orchestrator import AgentOrchestrator
from profiles.professional_sports_profile import ProfessionalSportsProfile


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name


def _orchestrator() -> AgentOrchestrator:
    registry = AgentRegistry()
    registry.register(Agent("training"))  # type: ignore[arg-type]
    registry.register(Agent("medical"))  # type: ignore[arg-type]
    return AgentOrchestrator(registry)


def test_professional_sports_profile_contains_required_expertise_and_guidance() -> None:
    profile = ProfessionalSportsProfile()

    for capability in ("CrossFit", "HYROX", "Halterofilia", "Gimnasia aplicada al CrossFit", "Hipertrofia", "Fuerza", "Movilidad", "Programacion y periodizacion", "Diseno de clases y sesiones", "Adaptacion por nivel del atleta", "Preparacion de competiciones", "Prevencion de lesiones y retorno progresivo"):
        assert capability in profile.expertise
    assert "periodizacion" in profile.context
    assert "progresion" in profile.context
    assert "seguridad" in profile.context


def test_orchestrator_reuses_sports_profile_for_training_requests_only() -> None:
    orchestrator = _orchestrator()

    training = orchestrator.select("Programa una sesion HYROX con fuerza y movilidad")
    medical = orchestrator.select("Tengo dolor lumbar persistente")

    assert training.agents == ("training",)
    assert training.sports_profile is not None
    assert training.sports_profile.name == "professional_sports"
    assert training.sports_profile is orchestrator.select("Mejora mi tecnica de halterofilia").sports_profile
    assert medical.agents == ("medical",)
    assert medical.sports_profile is None
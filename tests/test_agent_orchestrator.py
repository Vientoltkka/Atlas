from __future__ import annotations

from agents.registry import AgentRegistry
from bootstrap.bootstrap import Bootstrap
from core.agent_orchestrator import AgentOrchestrator


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name


def _orchestrator() -> AgentOrchestrator:
    registry = AgentRegistry()
    for name in ("training", "nutrition", "medical", "legal", "finance", "code"):
        registry.register(Agent(name))  # type: ignore[arg-type]
    return AgentOrchestrator(registry)


def test_agent_orchestrator_selects_expected_specialists() -> None:
    orchestrator = _orchestrator()

    assert orchestrator.select("Quiero preparar un HYROX").agents == ("training",)
    assert orchestrator.select("Hazme una dieta para perder grasa").agents == ("nutrition",)
    assert orchestrator.select("Tengo dolor lumbar entrenando").agents == ("medical", "training")
    assert orchestrator.select("Quiero crear una app para mis clientes del box").agents == ("code", "training", "nutrition")
    assert orchestrator.select("Quiero invertir parte de mis ingresos").agents == ("finance",)
    assert orchestrator.select("Necesito revisar un contrato laboral").agents == ("legal",)


def test_agent_orchestrator_prioritizes_available_domains_and_bootstrap_exposes_it() -> None:
    orchestrator = _orchestrator()

    selection = orchestrator.select("Tengo dolor y quiero entrenar fuerza")

    assert selection.primary_agent == "medical"
    assert selection.agents == ("medical", "training")
    assert _orchestrator().select("texto sin dominio").agents == ()

    bootstrapped = Bootstrap.build()

    assert isinstance(bootstrapped.agent_orchestrator, AgentOrchestrator)
    assert bootstrapped.agent_orchestrator.select("Necesito invertir en ETFs").agents == ("finance",)
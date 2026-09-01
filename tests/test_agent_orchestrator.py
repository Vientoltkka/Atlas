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
    assert orchestrator.select("Calcula mi 1RM estimado de back squat con 5 repeticiones").agents == ("training",)
    assert orchestrator.select("Hazme una dieta para perder grasa").agents == ("nutrition",)
    assert orchestrator.select("Tengo dolor lumbar entrenando").agents == ("medical", "training")
    assert orchestrator.select("Quiero crear una app para mis clientes del box").agents == ("code", "training", "nutrition")
    assert orchestrator.select("Quiero invertir parte de mis ingresos").agents == ("finance",)
    assert orchestrator.select("Necesito revisar un contrato laboral").agents == ("legal",)


def test_agent_orchestrator_selects_training_for_coach_requests_only() -> None:
    orchestrator = _orchestrator()

    for prompt in (
        "hazme un entrenamiento de CrossFit de 60 minutos",
        "créame un WOD de HYROX para 20 personas",
        "quiero mejorar mi clean",
        "programa una sesión de hipertrofia de tren superior",
        "haz una sesión de fuerza de sentadilla",
        "qué ejercicio me recomiendas para mejorar los glúteos",
    ):
        assert orchestrator.select(prompt).primary_agent == "training"

    for prompt in (
        "abre VS Code",
        "qué eventos tengo hoy",
        "qué es la fotosíntesis",
    ):
        assert orchestrator.select(prompt).primary_agent != "training"

    assert (
        orchestrator.select("me duele mucho la rodilla y está hinchada").primary_agent
        == "medical"
    )


def test_agent_orchestrator_routes_clear_medical_requests_without_capturing_coach_or_nutrition() -> None:
    orchestrator = _orchestrator()

    for prompt in (
        "Tengo dolor e inflamacion en la rodilla.",
        "Tengo fiebre y tos desde anoche.",
        "Me he desmayado y tengo dolor de pecho.",
    ):
        assert orchestrator.select(prompt).primary_agent == "medical"

    for prompt in (
        "Hazme un WOD de CrossFit de 60 minutos.",
        "Prepara una sesion de HYROX.",
        "Calcula mis macros para perder grasa.",
        "Hazme una dieta con suplementacion basica.",
        "Quiero invertir en ETFs.",
        "Necesito revisar un contrato legal.",
        "Abre VS Code.",
        "Que eventos tengo hoy en el calendario?",
    ):
        assert orchestrator.select(prompt).primary_agent != "medical"


def test_agent_orchestrator_routes_finance_without_capturing_other_domains() -> None:
    orchestrator = _orchestrator()

    for prompt in (
        "Crea un presupuesto mensual con mis gastos.",
        "Calcula mi ahorro con 1000 euros al 5% de interes anual.",
        "Explica que es un fondo indexado.",
    ):
        assert orchestrator.select(prompt).primary_agent == "finance"

    for prompt in (
        "Necesito revisar un contrato legal.",
        "Tengo fiebre y dolor de garganta.",
        "Calcula mis macros para perder grasa.",
        "Hazme un WOD de CrossFit de 60 minutos.",
        "Abre VS Code.",
        "Que eventos tengo hoy en el calendario?",
    ):
        assert orchestrator.select(prompt).primary_agent != "finance"


def test_agent_orchestrator_routes_legal_without_capturing_other_domains() -> None:
    orchestrator = _orchestrator()

    for prompt in (
        "Puedes revisar esta clausula contractual?",
        "Cuales son mis derechos laborales basicos?",
        "Como presento una reclamacion de consumo por un producto defectuoso?",
    ):
        assert orchestrator.select(prompt).primary_agent == "legal"

    for prompt in (
        "Crea un presupuesto mensual con mis gastos.",
        "Tengo fiebre y dolor de garganta.",
        "Calcula mis macros para perder grasa.",
        "Hazme un WOD de CrossFit de 60 minutos.",
        "Abre VS Code.",
        "Que eventos tengo hoy en el calendario?",
    ):
        assert orchestrator.select(prompt).primary_agent != "legal"


def test_agent_orchestrator_routes_nutrition_without_capturing_other_domains() -> None:
    orchestrator = _orchestrator()

    for prompt in (
        "Calcula mis calorias y macros con mis datos.",
        "Que suplementacion basica tiene evidencia para ganar masa?",
        "Necesito nutricion para CrossFit.",
    ):
        assert orchestrator.select(prompt).primary_agent == "nutrition"

    for prompt in (
        "Hazme un WOD de CrossFit de 60 minutos.",
        "Tengo nauseas y dolor abdominal despues de comer.",
        "Tengo una enfermedad digestiva y necesito una dieta.",
        "Necesito revisar un contrato legal.",
        "Quiero invertir en ETFs.",
        "Abre VS Code.",
        "Que eventos tengo hoy en el calendario?",
    ):
        assert orchestrator.select(prompt).primary_agent != "nutrition"


def test_agent_orchestrator_prioritizes_available_domains_and_bootstrap_exposes_it() -> None:
    orchestrator = _orchestrator()

    selection = orchestrator.select("Tengo dolor y quiero entrenar fuerza")

    assert selection.primary_agent == "medical"
    assert selection.agents == ("medical", "training")
    assert _orchestrator().select("texto sin dominio").agents == ()

    bootstrapped = Bootstrap.build()

    assert isinstance(bootstrapped.agent_orchestrator, AgentOrchestrator)
    assert bootstrapped.agent_orchestrator.select("Necesito invertir en ETFs").agents == ("finance",)

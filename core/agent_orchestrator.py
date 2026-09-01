"""Deterministic specialist selection for Atlas requests."""

from __future__ import annotations

from dataclasses import dataclass
from unicodedata import normalize

from agents.registry import AgentRegistry
from profiles.professional_sports_profile import ProfessionalSportsProfile


@dataclass(frozen=True)
class AgentSelection:
    """Specialists selected for one request, ordered by primary domain."""

    agents: tuple[str, ...]
    primary_agent: str | None
    sports_profile: ProfessionalSportsProfile | None = None


class AgentOrchestrator:
    """Select registered specialists from explicit, bounded domain markers."""

    _DOMAIN_ORDER = ("medical", "legal", "finance", "project", "code", "training", "nutrition")
    _MARKERS = {
        "training": (
            "crossfit",
            "hyrox",
            "halterofilia",
            "weightlifting",
            "powerlifting",
            "gimnasia",
            "entren",
            "programacion",
            "fuerza",
            "hipertrofia",
            "gimnasio",
            "movilidad",
            "acondicionamiento",
            "cardio",
            "wod",
            "tecnica",
            "levantamiento",
            "mi clean",
            "power clean",
            "squat clean",
            "clean and jerk",
            "gluteos",
            "box",
            "1rm",
            "back squat",
            "sentadilla",
        ),
        "nutrition": (
            "dieta",
            "nutric",
            "perder grasa",
            "ganar masa",
            "masa muscular",
            "ganancia muscular",
            "menu",
            "aliment",
            "calori",
            "macro",
            "calcula mis calorias",
            "calcula mis macros",
            "proteina",
            "carbohidr",
            "comidas",
            "preentreno",
            "postentreno",
            "pre entrenamiento",
            "post entrenamiento",
            "hidrat",
            "suplement",
            "creatina",
            "nutricion para crossfit",
            "nutricion para hyrox",
            "nutricion para hipertrofia",
            "box",
        ),
        "medical": (
            "dolor",
            "duele",
            "hinch",
            "lesion",
            "sintoma",
            "enfermed",
            "rehabilit",
            "medic",
            "fiebre",
            "nause",
            "vomit",
            "diarrea",
            "mareo",
            "desmayo",
            "erupcion",
            "palpit",
        ),
        "legal": (
            "contrato",
            "clausula",
            "laboral",
            "despido",
            "indemnizacion laboral",
            "jurid",
            "abogado",
            "legal",
            "consumo",
            "reclamacion",
            "devolucion",
            "garantia",
            "derechos laborales",
            "derechos del consumidor",
            "obligaciones contractuales",
        ),
        "finance": (
            "invert",
            "acciones",
            "etf",
            "bolsa",
            "cripto",
            "defi",
            "ingresos",
            "presupuesto",
            "gastos",
            "ahorro",
            "interes",
            "rentabilidad",
            "fondo indexado",
            "finanzas personales",
        ),
        "project": ("analiza el proyecto", "analisis del proyecto", "arquitectura del proyecto"),
        "code": ("crear una app", "aplicacion", "app", "react", "next", "vite", "flutter", "api", "software", "codigo"),
    }

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry
        self._sports_profile = ProfessionalSportsProfile()

    def select(self, text: str) -> AgentSelection:
        """Select available specialists while preserving deterministic priority."""
        normalized = _normalize(text)
        scores = {
            domain: sum(marker in normalized for marker in markers)
            for domain, markers in self._MARKERS.items()
        }
        selected = [domain for domain, score in scores.items() if score]
        selected.sort(key=lambda domain: (domain != "medical", -scores[domain], self._DOMAIN_ORDER.index(domain)))
        available = tuple(domain for domain in selected if self._registry.get(domain) is not None)
        profile = self._sports_profile if "training" in available and self._sports_profile.applies_to(text) else None
        return AgentSelection(
            agents=available,
            primary_agent=available[0] if available else None,
            sports_profile=profile,
        )


def _normalize(text: str) -> str:
    return "".join(
        character for character in normalize("NFD", text.casefold()) if ord(character) < 0x300 or ord(character) > 0x36F
    )

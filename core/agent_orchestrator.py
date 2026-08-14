"""Deterministic specialist selection for Atlas requests."""

from __future__ import annotations

from dataclasses import dataclass
from unicodedata import normalize

from agents.registry import AgentRegistry


@dataclass(frozen=True)
class AgentSelection:
    """Specialists selected for one request, ordered by primary domain."""

    agents: tuple[str, ...]
    primary_agent: str | None


class AgentOrchestrator:
    """Select registered specialists from explicit, bounded domain markers."""

    _DOMAIN_ORDER = ("medical", "legal", "finance", "code", "training", "nutrition")
    _MARKERS = {
        "training": ("crossfit", "hyrox", "halterofilia", "entren", "fuerza", "box"),
        "nutrition": ("dieta", "nutric", "perder grasa", "menu", "aliment", "box"),
        "medical": ("dolor", "lesion", "sintoma", "rehabilit", "medic"),
        "legal": ("contrato", "laboral", "jurid", "abogado", "legal"),
        "finance": ("invert", "acciones", "etf", "bolsa", "cripto", "defi", "ingresos"),
        "code": ("crear una app", "aplicacion", "app", "react", "next", "vite", "flutter", "api", "software", "codigo"),
    }

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

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
        return AgentSelection(agents=available, primary_agent=available[0] if available else None)


def _normalize(text: str) -> str:
    return "".join(
        character for character in normalize("NFD", text.casefold()) if ord(character) < 0x300 or ord(character) > 0x36F
    )
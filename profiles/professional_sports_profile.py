"""Reusable professional profile for sports coaching requests."""

from __future__ import annotations

from dataclasses import dataclass
from unicodedata import normalize


@dataclass(frozen=True)
class ProfessionalSportsProfile:
    """Immutable coaching profile reused for training-related selections."""

    name: str = "professional_sports"
    expertise: tuple[str, ...] = (
        "CrossFit",
        "HYROX",
        "Halterofilia",
        "Gimnasia aplicada al CrossFit",
        "Hipertrofia",
        "Fuerza",
        "Movilidad",
        "Programacion y periodizacion",
        "Diseno de clases y sesiones",
        "Adaptacion por nivel del atleta",
        "Preparacion de competiciones",
        "Prevencion de lesiones y retorno progresivo",
    )
    context: str = (
        "Responde como entrenador profesional: aumenta la profundidad tecnica; "
        "prioriza periodizacion, progresion, adaptacion por nivel y seguridad."
    )

    def applies_to(self, text: str) -> bool:
        """Return whether the profile applies to a training request."""
        normalized = _normalize(text)
        return any(
            marker in normalized
            for marker in (
                "entren",
                "programacion",
                "fuerza",
                "movilidad",
                "tecnica",
                "hyrox",
                "crossfit",
                "halterofilia",
                "gimnasia",
            )
        )


def _normalize(text: str) -> str:
    return "".join(
        character
        for character in normalize("NFD", text.casefold())
        if ord(character) < 0x300 or ord(character) > 0x36F
    )
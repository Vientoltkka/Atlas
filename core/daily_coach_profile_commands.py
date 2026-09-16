"""Natural-language registration of the persistent Daily Coach profile."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from core.daily_coach_profile import DailyCoachProfile, DailyCoachProfileStore


@dataclass(frozen=True, slots=True)
class DailyCoachProfileCommandResult:
    handled: bool
    message: str = ""


class DailyCoachProfileCommandHandler:
    def __init__(self, store: DailyCoachProfileStore) -> None:
        self._store = store

    def handle(self, text: str) -> DailyCoachProfileCommandResult:
        raw = " ".join(text.split())
        folded = _fold(raw)
        if not _looks_like_profile_registration(folded):
            return DailyCoachProfileCommandResult(False)

        sex = _sex(folded)
        age = _number_before(folded, r"anos?")
        height = _number_before(folded, r"cm")
        weights = [float(value.replace(",", ".")) for value in re.findall(r"(\d+(?:[.,]\d+)?)\s*kg\b", folded)]
        if sex is None or age is None or height is None or not weights:
            return DailyCoachProfileCommandResult(
                True,
                "Perfil Daily Coach incompleto. Indica sexo, edad, altura en cm y peso actual en kg.",
            )

        target = _target_weight(folded)
        objective = _objective(raw)
        priorities = _priorities(raw)
        profile = DailyCoachProfile(
            sex=sex,
            age=int(age),
            height_cm=height,
            weight_kg=weights[0],
            target_weight_kg=target,
            objective=objective,
            priorities=priorities,
        )
        self._store.save(profile)
        target_text = "" if target is None else f", objetivo {target:g} kg"
        return DailyCoachProfileCommandResult(
            True,
            f"Perfil Daily Coach guardado: {profile.age} años, {profile.height_cm:g} cm, {profile.weight_kg:g} kg{target_text}.",
        )


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _looks_like_profile_registration(text: str) -> bool:
    return any(marker in text for marker in ("mi perfil", "perfil:", "perfil daily coach"))


def _sex(text: str) -> str | None:
    if re.search(r"\b(?:hombre|varon|masculino)\b", text):
        return "hombre"
    if re.search(r"\b(?:mujer|femenino)\b", text):
        return "mujer"
    return None


def _number_before(text: str, unit_pattern: str) -> float | None:
    match = re.search(rf"(\d+(?:[.,]\d+)?)\s*{unit_pattern}\b", text)
    return None if match is None else float(match.group(1).replace(",", "."))


def _target_weight(text: str) -> float | None:
    patterns = (
        r"(?:llegar|subir|bajar)\s+a\s+(\d+(?:[.,]\d+)?)\s*kg",
        r"(?:peso\s+)?objetivo\s*(?:de|:)?\s*(\d+(?:[.,]\d+)?)\s*kg",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def _objective(raw: str) -> str:
    match = re.search(r"\bobjetivo\s*:\s*(.+?)(?=\bprioridades?\s*:|$)", raw, re.IGNORECASE)
    if match:
        return " ".join(match.group(1).strip(" .,").split())
    match = re.search(r"\b(?:llegar|subir|bajar)\s+a\s+\d+(?:[.,]\d+)?\s*kg\s*(.+)$", raw, re.IGNORECASE)
    return "" if match is None else " ".join(match.group(1).strip(" .,").split())


def _priorities(raw: str) -> tuple[str, ...]:
    match = re.search(r"\bprioridades?\s*:\s*(.+)$", raw, re.IGNORECASE)
    if match is None:
        marker = re.search(r"\bpriorizando\s+(.+)$", raw, re.IGNORECASE)
        if marker is None:
            return ()
        value = marker.group(1)
    else:
        value = match.group(1)
    parts = re.split(r"\s*,\s*|\s+y\s+", value.strip(" .,"), flags=re.IGNORECASE)
    return tuple(" ".join(part.split()) for part in parts if part.strip())

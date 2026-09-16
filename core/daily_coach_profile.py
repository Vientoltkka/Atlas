"""Small persistent profile used by the Daily Coach planning context."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


SCHEMA_VERSION = 1


class DailyCoachProfileError(RuntimeError):
    """Base profile persistence error."""


class InvalidDailyCoachProfileError(DailyCoachProfileError):
    """Raised when profile data is outside the supported bounds."""


@dataclass(frozen=True, slots=True)
class DailyCoachProfile:
    sex: str
    age: int
    height_cm: float
    weight_kg: float
    target_weight_kg: float | None = None
    objective: str = ""
    priorities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        sex = self.sex.strip().casefold()
        if sex not in {"male", "female", "hombre", "mujer"}:
            raise InvalidDailyCoachProfileError("Unsupported sex value.")
        if not 14 <= self.age <= 100:
            raise InvalidDailyCoachProfileError("Age is outside supported bounds.")
        if not 100 <= self.height_cm <= 230:
            raise InvalidDailyCoachProfileError("Height is outside supported bounds.")
        if not 30 <= self.weight_kg <= 300:
            raise InvalidDailyCoachProfileError("Weight is outside supported bounds.")
        if self.target_weight_kg is not None and not 30 <= self.target_weight_kg <= 300:
            raise InvalidDailyCoachProfileError("Target weight is outside supported bounds.")
        objective = " ".join(self.objective.split())
        priorities = tuple(" ".join(item.split()) for item in self.priorities if item.strip())
        object.__setattr__(self, "sex", sex)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "priorities", priorities)


class DailyCoachProfileStore:
    """Atomic JSON persistence kept separate from generic personal memory."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> DailyCoachProfile | None:
        if not self._path.exists():
            return None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DailyCoachProfileError("Could not load Daily Coach profile.") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise InvalidDailyCoachProfileError("Invalid Daily Coach profile payload.")
        raw = payload.get("profile")
        if not isinstance(raw, dict):
            raise InvalidDailyCoachProfileError("Invalid Daily Coach profile payload.")
        try:
            return DailyCoachProfile(
                sex=raw["sex"],
                age=raw["age"],
                height_cm=raw["height_cm"],
                weight_kg=raw["weight_kg"],
                target_weight_kg=raw.get("target_weight_kg"),
                objective=raw.get("objective", ""),
                priorities=tuple(raw.get("priorities", ())),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidDailyCoachProfileError("Invalid Daily Coach profile payload.") from error

    def save(self, profile: DailyCoachProfile) -> None:
        if not isinstance(profile, DailyCoachProfile):
            raise InvalidDailyCoachProfileError("Unsupported Daily Coach profile value.")
        payload = {"schema_version": SCHEMA_VERSION, "profile": asdict(profile)}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        temp_path: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w", encoding="utf-8", newline="\n", dir=self._path.parent,
                prefix=f".{self._path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._path)
        except OSError as error:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise DailyCoachProfileError("Could not save Daily Coach profile.") from error


def render_daily_coach_profile(profile: DailyCoachProfile | None) -> str:
    if profile is None:
        return "Perfil Daily Coach: no registrado."
    lines = [
        "Perfil Daily Coach:",
        f"- sexo: {profile.sex}",
        f"- edad: {profile.age} años",
        f"- altura: {profile.height_cm:g} cm",
        f"- peso actual: {profile.weight_kg:g} kg",
    ]
    if profile.target_weight_kg is not None:
        lines.append(f"- peso objetivo: {profile.target_weight_kg:g} kg")
    if profile.objective:
        lines.append(f"- objetivo: {profile.objective}")
    if profile.priorities:
        lines.append(f"- prioridades: {', '.join(profile.priorities)}")
    return "\n".join(lines)

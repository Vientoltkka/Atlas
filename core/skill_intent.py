"""Skill intent detection, input extraction and safe output presentation.

Extracted verbatim from core/orchestrator.py (V3.2.1-1) so that the
skill request path lives next to the rest of the skill subsystem.
Semantics are preserved exactly; orchestrator.py only delegates here.
"""

from __future__ import annotations

import re

from core.skill_system import SkillSystem


def requested_skill_id(prompt: str, skill_system: SkillSystem) -> str | None:
    normalized = " ".join(prompt.casefold().split())
    if "skill" not in normalized:
        return None

    for skill in skill_system.skill_registry.list_skills(enabled_only=True):
        identifiers = (skill.skill_id.casefold(), skill.name.casefold())
        if any(identifier in normalized for identifier in identifiers):
            return skill.skill_id
    return None


def skill_inputs_from_text(prompt: str, skill) -> dict[str, object]:
    input_names = tuple(field.name for field in skill.input_fields) or tuple(
        skill.input_names
    )
    if input_names == ("text",):
        quoted = re.findall(r'''["']([^"']+)["']''', prompt)
        if quoted:
            return {"text": quoted[-1]}

        match = re.search(
            r"(?:con\s+el\s+texto|texto)\s*[:=]?\s*(.+)$",
            prompt,
            re.IGNORECASE,
        )
        return {"text": match.group(1).strip()} if match else {}

    return _layout_inputs_from_text(prompt, input_names)


_LAYOUT_FIELD_NAMES = ("window_title", "secondary_title", "width", "height", "position")


def _layout_inputs_from_text(prompt: str, input_names: tuple[str, ...]) -> dict[str, object]:
    """Extract typed inputs for parametrized layout skills from natural text.

    Recognized patterns (all optional per skill contract):
    - quoted titles: "..." or '...' map in order to window_title, secondary_title
    - size: "800x700" (also 800 x 700) maps to width/height
    - position: "a la derecha|izquierda" or "en el centro" maps to position
    """

    if not set(input_names) & set(_LAYOUT_FIELD_NAMES):
        return {}

    inputs: dict[str, object] = {}
    quoted = re.findall(r'''["']([^"']+)["']''', prompt)
    string_fields = [name for name in ("window_title", "secondary_title") if name in input_names]
    for name, value in zip(string_fields, quoted):
        inputs[name] = value.strip()

    if {"width", "height"} <= set(input_names):
        size = re.search(r"\b(\d{1,5})\s*[xX×]\s*(\d{1,5})\b", prompt)
        if size:
            inputs["width"] = int(size.group(1))
            inputs["height"] = int(size.group(2))

    if "position" in input_names:
        position = re.search(
            r"(?:a la|al)?\s*(izquierda|derecha|centro)\b",
            prompt,
            re.IGNORECASE,
        )
        if position:
            inputs["position"] = position.group(1).casefold()

    return inputs


def present_skill_output(output) -> str:
    values = dict(output or {})
    if "result" in values:
        return str(values["result"])
    if len(values) == 1:
        return str(next(iter(values.values())))
    return "\n".join(f"{key}: {value}" for key, value in values.items())

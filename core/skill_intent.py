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
    if input_names != ("text",):
        return {}

    quoted = re.findall(r'''["']([^"']+)["']''', prompt)
    if quoted:
        return {"text": quoted[-1]}

    match = re.search(
        r"(?:con\s+el\s+texto|texto)\s*[:=]?\s*(.+)$",
        prompt,
        re.IGNORECASE,
    )
    return {"text": match.group(1).strip()} if match else {}


def present_skill_output(output) -> str:
    values = dict(output or {})
    if "result" in values:
        return str(values["result"])
    if len(values) == 1:
        return str(next(iter(values.values())))
    return "\n".join(f"{key}: {value}" for key, value in values.items())

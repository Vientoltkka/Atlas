"""Safe, allowlisted catalog view of the registered skills (V3.2.1-2).

The catalog exposes only public descriptive fields. It never includes
metadata, permissions, capability requirements, execution targets,
handlers, workflow references, limits or any configuration detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from core.skill_system import SkillSystem


@dataclass(frozen=True, slots=True)
class SkillCatalogEntry:
    """Public, safe description of one registered skill."""

    id: str
    name: str
    description: str
    version: str
    enabled: bool


def build_skill_catalog(
    skill_system: SkillSystem,
    *,
    enabled_only: bool = False,
) -> tuple[SkillCatalogEntry, ...]:
    """Build the safe catalog view of registered skills.

    Mirrors ``SkillRegistry.list_skills`` semantics: by default every
    registered skill is listed (including disabled ones, flagged with
    ``enabled=False``); pass ``enabled_only=True`` to exclude them.
    """
    if not isinstance(skill_system, SkillSystem):
        raise TypeError("skill_system must be SkillSystem.")
    return tuple(
        _entry(definition)
        for definition in skill_system.skill_registry.list_skills(enabled_only=enabled_only)
    )


def _entry(definition) -> SkillCatalogEntry:
    return SkillCatalogEntry(
        id=definition.skill_id,
        name=definition.name,
        description=definition.description,
        version=definition.version,
        enabled=definition.enabled,
    )


def catalog_to_mappings(entries: tuple[SkillCatalogEntry, ...]) -> tuple[Mapping[str, object], ...]:
    """Serializable form with exactly the allowlisted keys."""
    return tuple(
        {
            "id": entry.id,
            "name": entry.name,
            "description": entry.description,
            "version": entry.version,
            "enabled": entry.enabled,
        }
        for entry in entries
    )

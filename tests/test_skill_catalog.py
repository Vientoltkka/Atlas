"""Tests for the safe skill catalog view (V3.2.1-2)."""

from __future__ import annotations

import pytest

from core.skill_catalog import (
    SkillCatalogEntry,
    build_skill_catalog,
    catalog_to_mappings,
)
from core.skill_registry import SkillDefinition, SkillRegistry
from core.skill_system import build_skill_system


def make_skill(skill_id: str, enabled: bool = True) -> SkillDefinition:
    return SkillDefinition(
        skill_id=skill_id,
        name=f"Skill {skill_id}",
        version="1.0",
        description=f"Description of {skill_id}.",
        enabled=enabled,
        execution_target="tool.something",
        handler_id="handler.something",
        metadata={"internal": "value"},
        required_permission_ids=("perm.filesystem",),
    )


@pytest.fixture()
def skill_system():
    registry = SkillRegistry(
        (
            make_skill("skill.alpha"),
            make_skill("skill.beta", enabled=False),
        )
    )
    return build_skill_system(skill_registry=registry)


def test_catalog_lists_all_skills_by_default(skill_system) -> None:
    entries = build_skill_catalog(skill_system)
    assert [entry.id for entry in entries] == ["skill.alpha", "skill.beta"]


def test_disabled_skills_are_flagged_not_excluded(skill_system) -> None:
    entries = build_skill_catalog(skill_system)
    by_id = {entry.id: entry for entry in entries}
    assert by_id["skill.alpha"].enabled is True
    assert by_id["skill.beta"].enabled is False


def test_enabled_only_policy_excludes_disabled(skill_system) -> None:
    entries = build_skill_catalog(skill_system, enabled_only=True)
    assert [entry.id for entry in entries] == ["skill.alpha"]


def test_entries_contain_only_allowlisted_fields(skill_system) -> None:
    for entry in build_skill_catalog(skill_system):
        assert isinstance(entry, SkillCatalogEntry)
        assert entry.name == f"Skill {entry.id}"
        assert entry.version == "1.0"


def test_mappings_expose_exactly_the_allowlist(skill_system) -> None:
    mappings = catalog_to_mappings(build_skill_catalog(skill_system))
    expected_keys = {"id", "name", "description", "version", "enabled"}
    for mapping in mappings:
        assert set(mapping.keys()) == expected_keys


def test_no_internal_details_leak_into_catalog(skill_system) -> None:
    serialized = repr(build_skill_catalog(skill_system)) + repr(
        catalog_to_mappings(build_skill_catalog(skill_system))
    )
    assert "tool.something" not in serialized
    assert "handler.something" not in serialized
    assert "perm.filesystem" not in serialized
    assert "internal" not in serialized


def test_rejects_non_skill_system_argument() -> None:
    with pytest.raises(TypeError):
        build_skill_catalog(object())

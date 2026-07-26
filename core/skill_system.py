"""Composable declarative skill subsystem for Atlas."""

from __future__ import annotations

from dataclasses import dataclass

from core.skill_discovery import SkillDiscovery
from core.skill_executor import SkillExecutor
from core.skill_manifest import SkillManifestLoader
from core.skill_registration import SkillRegistrationService
from core.skill_registry import SkillRegistry
from core.skill_resolver import SkillResolver


@dataclass(frozen=True, slots=True)
class SkillSystem:
    """Shared skill infrastructure graph."""

    skill_registry: SkillRegistry
    skill_manifest_loader: SkillManifestLoader
    skill_discovery: SkillDiscovery
    skill_registration_service: SkillRegistrationService
    skill_resolver: SkillResolver
    skill_executor: SkillExecutor


def build_skill_system(
    *,
    skill_registry: SkillRegistry | None = None,
    skill_manifest_loader: SkillManifestLoader | None = None,
    skill_discovery: SkillDiscovery | None = None,
    skill_registration_service: SkillRegistrationService | None = None,
    skill_resolver: SkillResolver | None = None,
    skill_executor: SkillExecutor | None = None,
) -> SkillSystem:
    registry = skill_registry or SkillRegistry()
    loader = skill_manifest_loader or SkillManifestLoader()
    discovery = skill_discovery or SkillDiscovery()
    registration = skill_registration_service or SkillRegistrationService(discovery, loader, registry)
    resolver = skill_resolver or SkillResolver(registry)
    executor = skill_executor or SkillExecutor()
    return SkillSystem(registry, loader, discovery, registration, resolver, executor)

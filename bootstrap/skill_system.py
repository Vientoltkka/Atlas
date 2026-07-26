"""Factory for Atlas declarative skill system."""

from __future__ import annotations

from core.skill_discovery import SkillDiscovery
from core.skill_executor import SkillExecutor
from core.skill_manifest import SkillManifestLoader
from core.skill_registration import SkillRegistrationService
from core.skill_registry import SkillRegistry
from core.skill_resolver import SkillResolver
from core.skill_system import SkillSystem, build_skill_system


def build_core_skill_system(
    *,
    skill_registry: SkillRegistry | None = None,
    skill_manifest_loader: SkillManifestLoader | None = None,
    skill_discovery: SkillDiscovery | None = None,
    skill_registration_service: SkillRegistrationService | None = None,
    skill_resolver: SkillResolver | None = None,
    skill_executor: SkillExecutor | None = None,
) -> SkillSystem:
    return build_skill_system(
        skill_registry=skill_registry,
        skill_manifest_loader=skill_manifest_loader,
        skill_discovery=skill_discovery,
        skill_registration_service=skill_registration_service,
        skill_resolver=skill_resolver,
        skill_executor=skill_executor,
    )

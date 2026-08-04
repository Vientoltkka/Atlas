"""Composable declarative skill subsystem for Atlas."""

from __future__ import annotations

from dataclasses import dataclass

from core.agent_executor import AgentExecutor
from core.capability_execution_service import CapabilityExecutionService
from core.skill_discovery import SkillDiscovery
from core.skill_executor import SkillExecutor, SkillHandlerRegistry
from core.skill_manifest import SkillManifestLoader
from core.skill_registration import SkillRegistrationService
from core.skill_registry import SkillRegistry
from core.skill_resolver import SkillResolver
from tools.executor import ToolExecutor


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
    tool_executor: ToolExecutor | None = None,
    capability_execution_service: CapabilityExecutionService | None = None,
    agent_executor: AgentExecutor | None = None,
    skill_handler_registry: SkillHandlerRegistry | None = None,
) -> SkillSystem:
    if skill_registry is None and skill_executor is not None:
        registry = skill_executor.skill_registry
    else:
        registry = skill_registry if skill_registry is not None else SkillRegistry()
    loader = skill_manifest_loader or SkillManifestLoader()
    discovery = skill_discovery or SkillDiscovery()
    registration = skill_registration_service or SkillRegistrationService(discovery, loader, registry)
    resolver = skill_resolver or SkillResolver(registry)
    executor = skill_executor or SkillExecutor(
        skill_registry=registry,
        tool_executor=tool_executor,
        capability_execution_service=capability_execution_service,
        agent_executor=agent_executor,
        handler_registry=skill_handler_registry,
    )
    if executor.skill_registry is not registry:
        raise ValueError("skill_executor must use the central skill_registry.")
    return SkillSystem(registry, loader, discovery, registration, resolver, executor)

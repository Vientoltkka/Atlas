"""Factory for controlled specialized-agent registration."""

from __future__ import annotations

from core.agent_discovery import AgentDiscovery
from core.agent_manifest import AgentManifestLoader
from core.agent_registration import AgentRegistrationService
from core.agent_registry import AgentRegistry


def build_core_agent_registration_service(
    agent_discovery: AgentDiscovery,
    manifest_loader: AgentManifestLoader,
    agent_registry: AgentRegistry,
) -> AgentRegistrationService:
    """Build AgentRegistrationService from explicit local collaborators."""

    return AgentRegistrationService(
        agent_discovery=agent_discovery,
        manifest_loader=manifest_loader,
        agent_registry=agent_registry,
    )

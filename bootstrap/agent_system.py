"""Factory for the composed specialized-agent system."""

from __future__ import annotations

from core.agent_context import AgentContextBuilder
from core.agent_discovery import AgentDiscovery
from core.agent_executor import AgentExecutor, AgentHandlerRegistry
from core.agent_handler_registration import AgentHandlerRegistrationService
from core.agent_manifest import AgentManifestLoader
from core.agent_registration import AgentRegistrationService
from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolver
from core.agent_system import AgentSystemBuildRequest, AgentSystemBuildResult, AgentSystemBuilder


def build_core_agent_system(
    request: AgentSystemBuildRequest | None = None,
    *,
    agent_registry: AgentRegistry | None = None,
    agent_handler_registry: AgentHandlerRegistry | None = None,
    agent_manifest_loader: AgentManifestLoader | None = None,
    agent_discovery: AgentDiscovery | None = None,
    agent_registration_service: AgentRegistrationService | None = None,
    agent_handler_registration_service: AgentHandlerRegistrationService | None = None,
    agent_resolver: AgentResolver | None = None,
    agent_context_builder: AgentContextBuilder | None = None,
    agent_executor: AgentExecutor | None = None,
) -> AgentSystemBuildResult:
    """Build a fully composed AgentSystem with explicit optional injections."""

    return AgentSystemBuilder(
        agent_registry=agent_registry,
        agent_handler_registry=agent_handler_registry,
        agent_manifest_loader=agent_manifest_loader,
        agent_discovery=agent_discovery,
        agent_registration_service=agent_registration_service,
        agent_handler_registration_service=agent_handler_registration_service,
        agent_resolver=agent_resolver,
        agent_context_builder=agent_context_builder,
        agent_executor=agent_executor,
    ).build(request)

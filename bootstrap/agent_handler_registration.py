"""Factory for controlled specialized-agent handler registration."""

from __future__ import annotations

from core.agent_executor import AgentHandlerRegistry
from core.agent_handler_registration import AgentHandlerRegistrationService
from core.agent_registry import AgentRegistry


def build_core_agent_handler_registration_service(
    agent_registry: AgentRegistry,
    agent_handler_registry: AgentHandlerRegistry,
) -> AgentHandlerRegistrationService:
    """Build AgentHandlerRegistrationService from explicit local collaborators."""

    return AgentHandlerRegistrationService(
        agent_registry=agent_registry,
        agent_handler_registry=agent_handler_registry,
    )

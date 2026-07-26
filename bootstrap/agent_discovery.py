"""Factory for safe specialized-agent manifest discovery."""

from __future__ import annotations

from core.agent_discovery import AgentDiscovery
from core.agent_manifest import AgentManifestLoader


def build_core_agent_discovery(
    manifest_loader: AgentManifestLoader,
) -> AgentDiscovery:
    """Build AgentDiscovery from an explicitly injected manifest loader."""

    return AgentDiscovery(manifest_loader)

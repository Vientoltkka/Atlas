"""Factory for safe specialized-agent manifest loading."""

from __future__ import annotations

from collections.abc import Iterable

from core.agent_manifest import AgentManifestLoader


def build_core_agent_manifest_loader(
    *,
    known_agent_ids: Iterable[str] = (),
    known_handler_ids: Iterable[str] = (),
) -> AgentManifestLoader:
    """Build an AgentManifestLoader with explicit conflict boundaries."""

    return AgentManifestLoader(
        known_agent_ids=known_agent_ids,
        known_handler_ids=known_handler_ids,
    )

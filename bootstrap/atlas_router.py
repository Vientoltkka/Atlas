"""Factory for the deterministic Atlas router."""

from __future__ import annotations

from core.atlas_router import AtlasRouter, RoutingObserver
from core.capability_execution_service import CapabilityExecutionService


def build_core_atlas_router(
    *,
    capability_execution_service: CapabilityExecutionService | None = None,
    observer: RoutingObserver | None = None,
) -> AtlasRouter:
    """Build AtlasRouter from explicitly injected services."""

    return AtlasRouter(
        capability_execution_service=capability_execution_service,
        observer=observer,
    )

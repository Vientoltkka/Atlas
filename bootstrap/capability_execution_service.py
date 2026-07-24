"""Factory for explicit capability execution service composition."""

from __future__ import annotations

from core.capability_execution_service import CapabilityExecutionService
from core.capability_orchestrator import CapabilityOrchestrator
from core.multi_capability_planner import MultiCapabilityPlanner


def build_capability_execution_service(
    capability_orchestrator: CapabilityOrchestrator,
    *,
    multi_capability_planner: MultiCapabilityPlanner | None = None,
) -> CapabilityExecutionService:
    """Build the service from the already composed capability orchestrator."""

    return CapabilityExecutionService(
        capability_orchestrator,
        multi_capability_planner=multi_capability_planner,
    )

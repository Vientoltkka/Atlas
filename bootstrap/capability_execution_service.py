"""Factory for explicit capability execution service composition."""

from __future__ import annotations

from core.capability_execution_service import CapabilityExecutionService
from core.capability_orchestrator import CapabilityOrchestrator


def build_capability_execution_service(
    capability_orchestrator: CapabilityOrchestrator,
) -> CapabilityExecutionService:
    """Build the service from the already composed capability orchestrator."""

    return CapabilityExecutionService(capability_orchestrator)

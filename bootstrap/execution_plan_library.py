"""Composition helpers for Atlas execution-plan libraries."""

from __future__ import annotations

from core.execution_plan_library import ExecutionPlanLibrary
from core.workflow_discovery import WorkflowDiscoveryService


def build_core_execution_plan_library() -> ExecutionPlanLibrary | None:
    """Return Atlas core workflows when stable production definitions exist."""
    return None


def build_workflow_discovery_service() -> WorkflowDiscoveryService:
    """Build the pure deterministic workflow discovery service."""
    return WorkflowDiscoveryService()

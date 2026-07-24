"""Composition helpers for the Atlas capability resolver."""

from __future__ import annotations

from collections.abc import Iterable

from core.capability_resolver import (
    CapabilityProvider,
    CapabilityResolver,
    ToolCapabilityProvider,
    WorkflowCapabilityProvider,
)
from core.execution_plan_library import ExecutionPlanLibrary
from tools.registry import ToolRegistry


def build_core_capability_resolver(
    *,
    tool_registry: ToolRegistry | None = None,
    execution_plan_libraries: Iterable[ExecutionPlanLibrary] = (),
    providers: Iterable[CapabilityProvider] = (),
) -> CapabilityResolver:
    """Build a pure resolver from explicitly provided capability sources."""
    capability_providers: list[CapabilityProvider] = list(providers)
    libraries = tuple(execution_plan_libraries)
    if libraries:
        capability_providers.append(WorkflowCapabilityProvider(libraries))
    elif tool_registry is not None:
        capability_providers.append(ToolCapabilityProvider(tool_registry))
    return CapabilityResolver(tuple(capability_providers))

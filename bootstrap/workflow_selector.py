"""Composition helpers for deterministic workflow selection."""

from __future__ import annotations

from core.workflow_selector import WorkflowSelectionPolicy, WorkflowSelector


def build_core_workflow_selector(
    policy: WorkflowSelectionPolicy | None = None,
) -> tuple[WorkflowSelector, WorkflowSelectionPolicy]:
    """Build the pure workflow selector and explicit default policy."""
    return WorkflowSelector(), policy or WorkflowSelectionPolicy()

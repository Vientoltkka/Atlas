"""Factory for the pure Atlas request adapter."""

from __future__ import annotations

from core.atlas_request_adapter import AtlasRequestAdapter


def build_core_atlas_request_adapter() -> AtlasRequestAdapter:
    """Build one explicit AtlasRequestAdapter instance."""

    return AtlasRequestAdapter()

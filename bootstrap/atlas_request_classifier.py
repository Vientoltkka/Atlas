"""Factory for the deterministic Atlas request classifier."""

from __future__ import annotations

from core.atlas_request_classifier import AtlasRequestClassifier


def build_core_atlas_request_classifier() -> AtlasRequestClassifier:
    """Build one explicit AtlasRequestClassifier instance."""

    return AtlasRequestClassifier()

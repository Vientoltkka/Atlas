"""Factory for the canonical Atlas request normalizer."""

from __future__ import annotations

from core.atlas_request_normalizer import AtlasRequestNormalizer


def build_core_atlas_request_normalizer() -> AtlasRequestNormalizer:
    """Build one explicit AtlasRequestNormalizer instance."""

    return AtlasRequestNormalizer()

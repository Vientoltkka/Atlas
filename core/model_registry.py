"""Safe loading of optional provider-neutral model descriptors."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from core.model_manager import ModelDescriptor


MODEL_DESCRIPTORS_ENV = "ATLAS_MODEL_DESCRIPTORS"
MODEL_REGISTRY_CONFIG_PATH_ENV = "ATLAS_MODEL_REGISTRY_CONFIG_PATH"


def load_model_descriptors_from_environment(
    *, reserved_logical_ids: Iterable[str] = (),
) -> tuple[ModelDescriptor, ...]:
    """Load descriptors from environment JSON or a JSON configuration file.

    Environment JSON takes precedence. Invalid input and entries are ignored,
    leaving ModelManager's built-in catalogue as the operational fallback.
    """
    payload = os.getenv(MODEL_DESCRIPTORS_ENV, "").strip()
    if payload:
        return load_model_descriptors(payload, reserved_logical_ids=reserved_logical_ids)
    configured_path = os.getenv(MODEL_REGISTRY_CONFIG_PATH_ENV, "").strip()
    if not configured_path:
        return ()
    try:
        payload = Path(configured_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ()
    return load_model_descriptors(payload, reserved_logical_ids=reserved_logical_ids)


def load_model_descriptors(
    payload: str | bytes | object,
    *, reserved_logical_ids: Iterable[str] = (),
) -> tuple[ModelDescriptor, ...]:
    """Deserialize only valid, unique, non-reserved descriptor entries."""
    parsed = _parse_json(payload)
    entries = parsed.get("models") if isinstance(parsed, Mapping) else parsed
    if not isinstance(entries, list):
        return ()
    logical_ids = set(reserved_logical_ids)
    descriptors: list[ModelDescriptor] = []
    for entry in entries:
        descriptor = _descriptor_from_entry(entry)
        if descriptor is None or descriptor.logical_id in logical_ids:
            continue
        logical_ids.add(descriptor.logical_id)
        descriptors.append(descriptor)
    return tuple(descriptors)


def _parse_json(payload: str | bytes | object) -> object:
    if not isinstance(payload, (str, bytes, bytearray)):
        return payload
    try:
        return json.loads(payload, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        return ()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Unsupported JSON constant: {value}")


def _descriptor_from_entry(entry: object) -> ModelDescriptor | None:
    if not isinstance(entry, Mapping):
        return None
    provider_id = entry.get("provider_id")
    model_id = entry.get("model_id")
    if not isinstance(provider_id, str) or not isinstance(model_id, str):
        return None
    capabilities = _tuple_value(entry.get("capabilities", ()))
    fallback_ids = _tuple_value(entry.get("fallback_logical_ids", ()))
    if capabilities is None or fallback_ids is None:
        return None
    try:
        return ModelDescriptor(
            logical_id=entry.get("logical_id", f"{provider_id}:{model_id}"),
            provider_id=provider_id,
            model_name=model_id,
            capabilities=capabilities,
            available=entry.get("available", True),
            relative_cost=entry.get("relative_cost"),
            relative_latency=entry.get("relative_latency"),
            context_window=entry.get("context_window"),
            local=entry.get("local", False),
            priority=entry.get("priority", 0),
            fallback_logical_ids=fallback_ids,
        )
    except (TypeError, ValueError):
        return None


def _tuple_value(value: object) -> tuple[object, ...] | None:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return None

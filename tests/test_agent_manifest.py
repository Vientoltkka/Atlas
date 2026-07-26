from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
import json
import math
import types

import pytest

from bootstrap.agent_manifest import build_core_agent_manifest_loader
from core.agent_manifest import (
    AgentManifest,
    AgentManifestConflictError,
    AgentManifestLoader,
    AgentManifestValidationStatus,
    InvalidAgentManifestError,
    agent_manifest_signature,
)
from core.agent_registry import AgentDefinition, AgentType


def _manifest(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "agent_id": "atlas.agent.project",
        "name": "Project Inspector",
        "description": "Deterministic project inspection agent.",
        "version": "1.2.3",
        "enabled": True,
        "agent_type": "project_analysis",
        "capabilities": ["project.inspect", "project.summary", "project.inspect"],
        "permissions": {
            "can_read_project": True,
            "can_write_files": False,
            "can_execute_tools": True,
            "can_modify_memory": True,
            "can_use_network": False,
            "requires_confirmation": True,
        },
        "limits": {
            "max_steps": 2,
            "max_tool_calls": 1,
            "max_context_items": 8,
            "max_memory_items": 1,
            "max_replans": 0,
        },
        "context_policy": {
            "include_project_context": True,
            "include_conversation_context": False,
            "include_runtime_context": False,
            "allow_user_input": False,
            "allow_shared_context": True,
            "allow_tool_results": True,
            "allow_workflow_results": False,
            "max_context_items": 8,
            "max_context_depth": 3,
            "max_string_length": 500,
            "max_sequence_items": 8,
            "max_mapping_items": 8,
            "max_total_items": 64,
            "allowed_context_keys": ["project.tree"],
        },
        "memory_policy": {
            "can_read_memory": True,
            "can_write_memory": False,
            "memory_scopes": ["project"],
            "max_memory_items": 1,
        },
        "security_policy": {
            "allow_network": False,
            "allow_file_write": False,
            "allowed_tools": ["read_file"],
            "blocked_tools": ["write_file"],
            "allowed_paths": ["C:\\AI\\Atlas"],
            "require_confirmation_for_writes": True,
        },
        "handler_id": "atlas.handler.project",
        "tags": ["safe", "project", "safe"],
        "metadata": {"owner": "atlas", "priority": 1, "stable": True},
    }
    payload.update(overrides)
    return payload


def test_valid_manifest_creation_is_immutable_and_normalized() -> None:
    manifest = AgentManifest(**_manifest())

    assert manifest.agent_id == "atlas.agent.project"
    assert manifest.agent_type is AgentType.PROJECT_ANALYSIS
    assert manifest.capabilities == ("project.inspect", "project.summary")
    assert manifest.tags == ("safe", "project")
    assert isinstance(manifest.metadata, MappingProxyType)

    with pytest.raises(FrozenInstanceError):
        manifest.name = "Other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.metadata["owner"] = "other"  # type: ignore[index]


def test_loader_converts_manifest_to_agent_definition_without_registering() -> None:
    definition = AgentManifestLoader().to_agent_definition(_manifest())

    assert isinstance(definition, AgentDefinition)
    assert definition.agent_id == "atlas.agent.project"
    assert definition.agent_type is AgentType.PROJECT_ANALYSIS
    assert definition.capabilities.capabilities == ("project.inspect", "project.summary")
    assert definition.capabilities.tags == ("safe", "project")
    assert definition.permissions.can_execute_tools is True
    assert definition.limits.max_tool_calls == 1
    assert definition.context_policy.allow_tool_results is True
    assert definition.memory_policy.memory_scopes == ("project",)
    assert definition.security_policy.allowed_tools == ("read_file",)
    assert definition.metadata["handler_id"] == "atlas.handler.project"


def test_validation_result_contains_agent_definition_and_signature() -> None:
    result = AgentManifestLoader().validate(_manifest())

    assert result.status is AgentManifestValidationStatus.VALID
    assert result.manifest is not None
    assert result.agent_definition is not None
    assert len(result.signature) == 64
    assert result.signature == agent_manifest_signature(result.manifest)


def test_signature_is_deterministic_and_changes_with_content() -> None:
    first = _manifest(metadata={"stable": True, "owner": "atlas"})
    same = _manifest(metadata={"owner": "atlas", "stable": True})
    different = _manifest(version="1.2.4")

    assert agent_manifest_signature(first) == agent_manifest_signature(same)
    assert agent_manifest_signature(first) != agent_manifest_signature(different)


def test_loader_rejects_known_conflicts_and_batch_duplicates() -> None:
    loader = AgentManifestLoader(
        known_agent_ids=("atlas.agent.project",),
        known_handler_ids=("atlas.handler.other",),
    )

    with pytest.raises(AgentManifestConflictError):
        loader.load(_manifest())
    with pytest.raises(AgentManifestConflictError):
        AgentManifestLoader(known_handler_ids=("atlas.handler.project",)).load(_manifest())
    with pytest.raises(AgentManifestConflictError):
        AgentManifestLoader().load_many([_manifest(), _manifest(agent_id="atlas.agent.other")])
    with pytest.raises(AgentManifestConflictError):
        AgentManifestLoader().load_many([_manifest(), _manifest(handler_id="atlas.handler.other")])


def test_validate_reports_conflict_without_exposing_secret_values() -> None:
    result = AgentManifestLoader(known_agent_ids=("atlas.agent.project",)).validate(_manifest())

    assert result.status is AgentManifestValidationStatus.CONFLICT
    assert result.error_code == "CONFLICT"
    assert "token" not in str(result.safe_message).lower()


def test_valid_json_manifest_loads_and_invalid_json_fails() -> None:
    loader = AgentManifestLoader()
    payload = json.dumps(_manifest())

    assert loader.load_json(payload).agent_id == "atlas.agent.project"

    invalid = loader.validate_json("{invalid")
    assert invalid.status is AgentManifestValidationStatus.INVALID
    with pytest.raises(InvalidAgentManifestError):
        loader.load_json('{"schema_version": NaN}')


def test_missing_unknown_and_invalid_top_level_fields_are_rejected() -> None:
    missing = _manifest()
    missing.pop("handler_id")

    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(missing)
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(extra="blocked"))
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(enabled="yes"))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("agent_id", "../agent"),
        ("schema_version", "1"),
        ("version", "v1"),
        ("agent_type", "unknown"),
        ("handler_id", "bad handler"),
    ),
)
def test_ids_versions_types_and_handler_id_are_validated(field: str, value: object) -> None:
    with pytest.raises((InvalidAgentManifestError, ValueError)):
        AgentManifestLoader().load(_manifest(**{field: value}))


def test_permissions_capabilities_and_policy_fields_are_validated() -> None:
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(capabilities=["bad capability"]))
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(permissions={"can_execute_tools": "yes"}))
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(security_policy={"allow_network": True}))
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(context_policy={"bad": True}))


def test_limits_are_validated() -> None:
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(limits={"max_steps": 0}))
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(limits={"max_tool_calls": 1001}))
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(limits={"max_steps": math.inf}))


def test_metadata_and_tags_are_safe_and_limited() -> None:
    manifest = AgentManifestLoader().load(_manifest(metadata={"value": "x", "count": 1, "stable": True}))

    assert manifest.metadata["value"] == "x"
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(tags=["bad tag"]))
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(metadata={f"k{i}": i for i in range(33)}))
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(metadata={"nested": {"value": [1, "x"]}}))


@pytest.mark.parametrize(
    "payload",
    (
        {"metadata": {"api_key": "hidden"}},
        {"metadata": {"password": "hidden"}},
        {"metadata": {"token": "hidden"}},
        {"metadata": {"authorization": "hidden"}},
        {"metadata": {"cookie": "hidden"}},
        {"metadata": {"private_key": "hidden"}},
        {"metadata": {"credential": "hidden"}},
    ),
)
def test_sensitive_manifest_keys_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(**payload))


@pytest.mark.parametrize(
    "value",
    (
        float("nan"),
        float("inf"),
        lambda: None,
        AgentManifest,
        types,
        object(),
    ),
)
def test_forbidden_objects_are_rejected(value: object) -> None:
    with pytest.raises(InvalidAgentManifestError):
        AgentManifestLoader().load(_manifest(metadata={"value": value}))


def test_agent_definition_compatibility_and_no_registry_side_effects() -> None:
    loader = build_core_agent_manifest_loader()
    definitions = loader.to_agent_definitions([_manifest()])

    assert len(definitions) == 1
    assert definitions[0].id == "atlas.agent.project"
    assert definitions[0].metadata["manifest_schema_version"] == "1.0"
    assert definitions[0].metadata["manifest_version"] == "1.2.3"
    assert definitions[0].metadata["manifest_signature"] == agent_manifest_signature(_manifest())

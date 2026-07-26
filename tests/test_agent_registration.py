from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from bootstrap.agent_registration import build_core_agent_registration_service
from core.agent_discovery import AgentDiscovery
from core.agent_manifest import AgentManifestLoader
from core.agent_registration import (
    AgentRegistrationDuplicatePolicy,
    AgentRegistrationPolicy,
    AgentRegistrationRequest,
    AgentRegistrationService,
    AgentRegistrationStatus,
    InvalidAgentRegistrationRequestError,
    agent_registration_request_signature,
)
from core.agent_registry import AgentRegistry, AgentType


def _manifest(agent_id: str = "atlas.agent.project", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "agent_id": agent_id,
        "name": f"Agent {agent_id}",
        "description": "Deterministic registration test agent.",
        "version": "1.0.0",
        "enabled": True,
        "agent_type": "project_analysis",
        "capabilities": ["project.inspect"],
        "permissions": {
            "can_read_project": True,
            "can_write_files": False,
            "can_execute_tools": True,
            "can_modify_memory": False,
            "can_use_network": False,
            "requires_confirmation": True,
        },
        "limits": {
            "max_steps": 2,
            "max_tool_calls": 1,
            "max_context_items": 8,
            "max_memory_items": 0,
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
            "can_read_memory": False,
            "can_write_memory": False,
            "memory_scopes": [],
            "max_memory_items": 0,
        },
        "security_policy": {
            "allow_network": False,
            "allow_file_write": False,
            "allowed_tools": ["read_file"],
            "blocked_tools": ["write_file"],
            "allowed_paths": ["C:\\AI\\Atlas"],
            "require_confirmation_for_writes": True,
        },
        "handler_id": f"{agent_id}.handler",
        "tags": ["safe", "test"],
        "metadata": {"owner": "atlas"},
    }
    payload.update(overrides)
    return payload


def _write_manifest(path: Path, agent_id: str = "atlas.agent.project", **overrides: object) -> dict[str, object]:
    payload = _manifest(agent_id, **overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _service(registry: AgentRegistry | None = None) -> AgentRegistrationService:
    loader = AgentManifestLoader()
    return AgentRegistrationService(AgentDiscovery(loader), loader, registry if registry is not None else AgentRegistry())


def _definition(payload: dict[str, object]):
    return AgentManifestLoader().to_agent_definition(payload)


def test_registers_one_valid_manifest(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")
    registry = AgentRegistry()
    service = _service(registry)

    result = service.register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    assert result.status is AgentRegistrationStatus.COMPLETED
    assert result.registered_agent_ids == ("atlas.agent.project",)
    assert registry.get("atlas.agent.project").agent_type is AgentType.PROJECT_ANALYSIS


def test_registers_multiple_valid_manifests(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "b.json", "atlas.agent.b")
    _write_manifest(tmp_path / "a.json", "atlas.agent.a")
    registry = AgentRegistry()

    result = _service(registry).register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    assert result.status is AgentRegistrationStatus.COMPLETED
    assert result.registered_agent_ids == ("atlas.agent.a", "atlas.agent.b")
    assert [agent.agent_id for agent in registry.list_agents()] == ["atlas.agent.a", "atlas.agent.b"]


def test_result_is_deterministic_independent_of_file_creation_order(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_manifest(left / "b.json", "atlas.agent.b")
    _write_manifest(left / "a.json", "atlas.agent.a")
    _write_manifest(right / "a.json", "atlas.agent.a")
    _write_manifest(right / "b.json", "atlas.agent.b")

    left_result = _service(AgentRegistry()).register(AgentRegistrationRequest(root_directories=(left,)))
    right_result = _service(AgentRegistry()).register(AgentRegistrationRequest(root_directories=(right,)))

    assert left_result.registered_agent_ids == right_result.registered_agent_ids == ("atlas.agent.a", "atlas.agent.b")
    assert [entry.manifest_signature for entry in left_result.entries] == [
        entry.manifest_signature for entry in right_result.entries
    ]


def test_dry_run_does_not_modify_registry(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")
    registry = AgentRegistry()

    result = _service(registry).register(
        AgentRegistrationRequest(
            root_directories=(tmp_path,),
            policy=AgentRegistrationPolicy(dry_run=True),
        )
    )

    assert result.status is AgentRegistrationStatus.DRY_RUN_COMPLETED
    assert result.entries[0].action == "would_register"
    assert len(registry) == 0


def test_atomic_batch_rejects_invalid_manifest_without_registering_anything(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "b.json", "atlas.agent.b")
    (tmp_path / "a.json").write_text("{invalid", encoding="utf-8")
    registry = AgentRegistry()

    result = _service(registry).register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    assert result.status is AgentRegistrationStatus.MANIFEST_VALIDATION_FAILED
    assert len(registry) == 0


def test_directory_without_manifests(tmp_path: Path) -> None:
    result = _service().register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    assert result.status is AgentRegistrationStatus.NO_MANIFESTS_FOUND
    assert result.manifests_processed == 0


def test_discovery_invalid_root_reports_failure(tmp_path: Path) -> None:
    result = _service().register(AgentRegistrationRequest(root_directories=(tmp_path / "missing",)))

    assert result.status is AgentRegistrationStatus.DISCOVERY_FAILED


def test_invalid_json_manifest_reports_validation_failure(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{invalid", encoding="utf-8")

    result = _service().register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    assert result.status is AgentRegistrationStatus.MANIFEST_VALIDATION_FAILED


def test_duplicate_agent_id_inside_batch_is_rejected_atomically(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "a.json", "atlas.agent.same")
    _write_manifest(tmp_path / "b.json", "atlas.agent.same")
    registry = AgentRegistry()

    result = _service(registry).register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    assert result.status is AgentRegistrationStatus.DUPLICATE_AGENT
    assert len(registry) == 0


def test_existing_agent_reject_policy_fails_batch(tmp_path: Path) -> None:
    payload = _write_manifest(tmp_path / "agent.json", "atlas.agent.same")
    registry = AgentRegistry((_definition(payload),))

    result = _service(registry).register(
        AgentRegistrationRequest(
            root_directories=(tmp_path,),
            policy=AgentRegistrationPolicy(duplicate_agent_policy=AgentRegistrationDuplicatePolicy.REJECT),
        )
    )

    assert result.status is AgentRegistrationStatus.DUPLICATE_AGENT
    assert result.rejected_agent_ids == ("atlas.agent.same",)


def test_keep_existing_policy_skips_existing_agent(tmp_path: Path) -> None:
    existing_payload = _manifest("atlas.agent.same", description="Existing definition.")
    _write_manifest(tmp_path / "agent.json", "atlas.agent.same", description="New definition.")
    registry = AgentRegistry((_definition(existing_payload),))

    result = _service(registry).register(
        AgentRegistrationRequest(
            root_directories=(tmp_path,),
            policy=AgentRegistrationPolicy(duplicate_agent_policy="KEEP_EXISTING"),
        )
    )

    assert result.status is AgentRegistrationStatus.COMPLETED
    assert result.skipped_agent_ids == ("atlas.agent.same",)
    assert registry.get("atlas.agent.same").description == "Existing definition."
    assert "conflicting" in " ".join(result.errors)


def test_replace_policy_replaces_existing_agent(tmp_path: Path) -> None:
    existing_payload = _manifest("atlas.agent.same", description="Existing definition.")
    _write_manifest(tmp_path / "agent.json", "atlas.agent.same", description="New definition.")
    registry = AgentRegistry((_definition(existing_payload),))

    result = _service(registry).register(
        AgentRegistrationRequest(
            root_directories=(tmp_path,),
            policy=AgentRegistrationPolicy(duplicate_agent_policy="REPLACE"),
        )
    )

    assert result.status is AgentRegistrationStatus.COMPLETED
    assert result.replaced_agent_ids == ("atlas.agent.same",)
    assert registry.get("atlas.agent.same").description == "New definition."


def test_conflicting_signature_with_reject_policy_is_reported(tmp_path: Path) -> None:
    existing_payload = _manifest("atlas.agent.same", description="Existing definition.")
    _write_manifest(tmp_path / "agent.json", "atlas.agent.same", description="New definition.")
    registry = AgentRegistry((_definition(existing_payload),))

    result = _service(registry).register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    assert result.status is AgentRegistrationStatus.AGENT_CONFLICT
    assert "conflicting" in " ".join(result.errors)


def test_disabled_agent_is_omitted_when_enabled_only_true(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json", enabled=False)
    registry = AgentRegistry()

    result = _service(registry).register(
        AgentRegistrationRequest(
            root_directories=(tmp_path,),
            policy=AgentRegistrationPolicy(enabled_only=True),
        )
    )

    assert result.status is AgentRegistrationStatus.NO_MANIFESTS_FOUND
    assert len(registry) == 0


def test_max_manifest_limit_is_enforced(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "a.json", "atlas.agent.a")
    _write_manifest(tmp_path / "b.json", "atlas.agent.b")

    result = _service().register(
        AgentRegistrationRequest(
            root_directories=(tmp_path,),
            policy=AgentRegistrationPolicy(max_manifests=1),
        )
    )

    assert result.status is AgentRegistrationStatus.LIMIT_EXCEEDED


def test_errors_are_sanitized(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "bad.json", metadata={"api_key": "hidden-token-value"})

    result = _service().register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    text = " ".join(result.errors).lower()
    assert result.status is AgentRegistrationStatus.MANIFEST_VALIDATION_FAILED
    assert "hidden-token-value" not in text
    assert "api_key" not in text
    assert "token" not in text


def test_request_signature_is_deterministic(tmp_path: Path) -> None:
    first = AgentRegistrationRequest(root_directories=(tmp_path,), metadata={"trace": "a"})
    same = AgentRegistrationRequest(root_directories=(tmp_path,), metadata={"trace": "a"})
    different = AgentRegistrationRequest(root_directories=(tmp_path,), recursive=True, metadata={"trace": "a"})

    assert agent_registration_request_signature(first) == agent_registration_request_signature(same)
    assert agent_registration_request_signature(first) != agent_registration_request_signature(different)


def test_request_metadata_is_immutable_and_safe(tmp_path: Path) -> None:
    request = AgentRegistrationRequest(root_directories=(tmp_path,), metadata={"trace": "safe"})

    assert isinstance(request.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        request.metadata["trace"] = "other"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        request.recursive = True  # type: ignore[misc]
    with pytest.raises(InvalidAgentRegistrationRequestError):
        AgentRegistrationRequest(root_directories=(tmp_path,), metadata={"token": "hidden"})


def test_handlers_are_not_loaded_or_executed(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json", handler_id="atlas.handler.must_not_load")

    result = _service().register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    assert result.status is AgentRegistrationStatus.COMPLETED
    assert result.registered_agent_ids == ("atlas.agent.project",)


def test_agents_are_not_executed(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")
    registry = AgentRegistry()

    result = _service(registry).register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    assert result.status is AgentRegistrationStatus.COMPLETED
    assert registry.get("atlas.agent.project").metadata["handler_id"] == "atlas.agent.project.handler"


def test_bootstrap_and_existing_components_are_compatible(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")
    loader = AgentManifestLoader()
    registry = AgentRegistry()
    service = build_core_agent_registration_service(AgentDiscovery(loader), loader, registry)

    result = service.register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    assert isinstance(service, AgentRegistrationService)
    assert result.status is AgentRegistrationStatus.COMPLETED
    assert registry.find_by_capability("project.inspect")

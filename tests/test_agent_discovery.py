from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
from types import MappingProxyType

import pytest

from bootstrap.agent_discovery import build_core_agent_discovery
from core.agent_discovery import (
    AgentDiscovery,
    AgentDiscoveryRequest,
    AgentDiscoveryStatus,
    InvalidAgentDiscoveryRequestError,
    agent_discovery_request_signature,
)
from core.agent_manifest import AgentManifestLoader
from core.agent_registry import AgentRegistry, AgentType


def _manifest(agent_id: str = "atlas.agent.project", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "agent_id": agent_id,
        "name": f"Agent {agent_id}",
        "description": "Deterministic discovered test agent.",
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


def _write_manifest(path: Path, agent_id: str = "atlas.agent.project", **overrides: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_manifest(agent_id, **overrides)), encoding="utf-8")
    return path


def _discovery() -> AgentDiscovery:
    return AgentDiscovery(AgentManifestLoader())


def test_directory_with_one_valid_manifest(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,)))

    assert result.status is AgentDiscoveryStatus.COMPLETED
    assert result.valid_manifests == 1
    assert result.agent_definitions[0].agent_id == "atlas.agent.project"
    assert result.discovered_manifests[0].relative_path == "agent.json"


def test_multiple_valid_manifests_are_loaded(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "b.json", "atlas.agent.b")
    _write_manifest(tmp_path / "a.json", "atlas.agent.a")

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,)))

    assert result.status is AgentDiscoveryStatus.COMPLETED
    assert [agent.agent_id for agent in result.agent_definitions] == ["atlas.agent.a", "atlas.agent.b"]


def test_discovery_order_is_deterministic_across_root_order(tmp_path: Path) -> None:
    first = tmp_path / "z_root"
    second = tmp_path / "a_root"
    _write_manifest(first / "z.json", "atlas.agent.z")
    _write_manifest(second / "a.json", "atlas.agent.a")

    left = _discovery().discover(AgentDiscoveryRequest(root_directories=(first, second)))
    right = _discovery().discover(AgentDiscoveryRequest(root_directories=(second, first)))

    assert [agent.agent_id for agent in left.agent_definitions] == ["atlas.agent.a", "atlas.agent.z"]
    assert [item.manifest_signature for item in left.discovered_manifests] == [
        item.manifest_signature for item in right.discovered_manifests
    ]


def test_recursive_discovery_enabled_finds_nested_manifest(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "nested" / "agent.json")

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,), recursive=True))

    assert result.status is AgentDiscoveryStatus.COMPLETED
    assert result.discovered_manifests[0].relative_path == "nested/agent.json"


def test_recursive_discovery_disabled_skips_nested_manifest(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "nested" / "agent.json")

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,), recursive=False))

    assert result.status is AgentDiscoveryStatus.NO_MANIFESTS_FOUND
    assert result.files_considered == 0


def test_nonexistent_root_returns_root_unavailable(tmp_path: Path) -> None:
    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path / "missing",)))

    assert result.status is AgentDiscoveryStatus.ROOT_UNAVAILABLE
    assert result.errors


def test_root_that_is_not_directory_returns_root_unavailable(tmp_path: Path) -> None:
    file_path = tmp_path / "not-dir.json"
    file_path.write_text("{}", encoding="utf-8")

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(file_path,)))

    assert result.status is AgentDiscoveryStatus.ROOT_UNAVAILABLE


def test_only_json_extension_is_allowed_and_other_files_are_ignored(tmp_path: Path) -> None:
    with pytest.raises(InvalidAgentDiscoveryRequestError):
        AgentDiscoveryRequest(root_directories=(tmp_path,), allowed_extensions=(".yaml",))

    (tmp_path / "agent.txt").write_text(json.dumps(_manifest()), encoding="utf-8")
    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,)))

    assert result.status is AgentDiscoveryStatus.NO_MANIFESTS_FOUND


def test_invalid_json_manifest_is_reported(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{invalid", encoding="utf-8")

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,)))

    assert result.status is AgentDiscoveryStatus.MANIFEST_INVALID
    assert result.invalid_manifests == 1
    assert "invalid" in result.errors[0].lower()


def test_semantically_invalid_manifest_is_reported(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "bad.json", agent_type="unknown")

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,)))

    assert result.status is AgentDiscoveryStatus.MANIFEST_INVALID
    assert result.rejected_files


def test_fail_fast_stops_on_first_invalid_file(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{invalid", encoding="utf-8")
    _write_manifest(tmp_path / "b.json", "atlas.agent.b")

    result = _discovery().discover(
        AgentDiscoveryRequest(root_directories=(tmp_path,), invalid_file_policy="fail_fast")
    )

    assert result.status is AgentDiscoveryStatus.MANIFEST_INVALID
    assert result.files_considered == 1
    assert result.valid_manifests == 0


def test_collect_errors_preserves_valid_manifests(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{invalid", encoding="utf-8")
    _write_manifest(tmp_path / "b.json", "atlas.agent.b")

    result = _discovery().discover(
        AgentDiscoveryRequest(root_directories=(tmp_path,), invalid_file_policy="collect_errors")
    )

    assert result.status is AgentDiscoveryStatus.COMPLETED_WITH_ERRORS
    assert result.valid_manifests == 1
    assert result.invalid_manifests == 1


def test_duplicate_reject_policy_reports_duplicate_agent(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "a.json", "atlas.agent.same")
    _write_manifest(tmp_path / "b.json", "atlas.agent.same")

    result = _discovery().discover(
        AgentDiscoveryRequest(
            root_directories=(tmp_path,),
            duplicate_agent_policy="reject",
            invalid_file_policy="collect_errors",
        )
    )

    assert result.status is AgentDiscoveryStatus.DUPLICATE_AGENT
    assert result.duplicate_agents == 1


def test_duplicate_keep_first_policy_rejects_later_file(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "a.json", "atlas.agent.same")
    _write_manifest(tmp_path / "b.json", "atlas.agent.same")

    result = _discovery().discover(
        AgentDiscoveryRequest(
            root_directories=(tmp_path,),
            duplicate_agent_policy="keep_first",
            invalid_file_policy="collect_errors",
        )
    )

    assert result.status is AgentDiscoveryStatus.COMPLETED_WITH_ERRORS
    assert [agent.agent_id for agent in result.agent_definitions] == ["atlas.agent.same"]
    assert result.rejected_files[0].endswith("b.json")


def test_duplicate_conflicting_signatures_are_reported(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "a.json", "atlas.agent.same", description="First manifest.")
    _write_manifest(tmp_path / "b.json", "atlas.agent.same", description="Different manifest.")

    result = _discovery().discover(
        AgentDiscoveryRequest(
            root_directories=(tmp_path,),
            duplicate_agent_policy="keep_first",
            invalid_file_policy="collect_errors",
        )
    )

    assert result.status is AgentDiscoveryStatus.COMPLETED_WITH_ERRORS
    assert result.duplicate_agents == 1
    assert "conflicting" in result.errors[0]


def test_file_limit_is_enforced(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "a.json", "atlas.agent.a")
    _write_manifest(tmp_path / "b.json", "atlas.agent.b")

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,), max_files=1))

    assert result.status is AgentDiscoveryStatus.LIMIT_EXCEEDED


def test_directory_limit_is_enforced(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "a" / "agent.json", "atlas.agent.a")

    result = _discovery().discover(
        AgentDiscoveryRequest(root_directories=(tmp_path,), recursive=True, max_directories=1)
    )

    assert result.status is AgentDiscoveryStatus.LIMIT_EXCEEDED


def test_manifest_size_limit_is_enforced(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")

    result = _discovery().discover(
        AgentDiscoveryRequest(root_directories=(tmp_path,), max_manifest_bytes=10)
    )

    assert result.status is AgentDiscoveryStatus.LIMIT_EXCEEDED
    assert "size limit" in result.errors[0]


def test_hidden_file_is_ignored(tmp_path: Path) -> None:
    _write_manifest(tmp_path / ".agent.json")

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,)))

    assert result.status is AgentDiscoveryStatus.NO_MANIFESTS_FOUND


def test_hidden_directory_is_ignored(tmp_path: Path) -> None:
    _write_manifest(tmp_path / ".hidden" / "agent.json")

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,), recursive=True))

    assert result.status is AgentDiscoveryStatus.NO_MANIFESTS_FOUND


def test_symlinked_manifest_is_ignored_safely(tmp_path: Path) -> None:
    target = _write_manifest(tmp_path / "target.json")
    link = tmp_path / "link.json"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlink creation is not available in this environment")

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,)))

    assert result.status is AgentDiscoveryStatus.COMPLETED
    assert [item.relative_path for item in result.discovered_manifests] == ["target.json"]


def test_errors_are_sanitized(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "bad.json", metadata={"api_key": "hidden-token-value"})

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,)))

    assert result.status is AgentDiscoveryStatus.MANIFEST_INVALID
    text = " ".join(result.errors).lower()
    assert "hidden-token-value" not in text
    assert "api_key" not in text
    assert "token" not in text


def test_request_signature_is_deterministic(tmp_path: Path) -> None:
    left = AgentDiscoveryRequest(root_directories=(tmp_path,), metadata={"trace": "a"})
    right = AgentDiscoveryRequest(root_directories=(tmp_path,), metadata={"trace": "a"})
    different = AgentDiscoveryRequest(root_directories=(tmp_path,), recursive=True, metadata={"trace": "a"})

    assert agent_discovery_request_signature(left) == agent_discovery_request_signature(right)
    assert agent_discovery_request_signature(left) != agent_discovery_request_signature(different)


def test_metadata_is_immutable_and_safe(tmp_path: Path) -> None:
    request = AgentDiscoveryRequest(root_directories=(tmp_path,), metadata={"trace": "safe"})

    assert isinstance(request.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        request.metadata["trace"] = "other"  # type: ignore[index]
    with pytest.raises(InvalidAgentDiscoveryRequestError):
        AgentDiscoveryRequest(root_directories=(tmp_path,), metadata={"token": "hidden"})
    with pytest.raises(FrozenInstanceError):
        request.recursive = True  # type: ignore[misc]


def test_no_execution_and_no_automatic_registration(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")
    registry = AgentRegistry()

    result = _discovery().discover(AgentDiscoveryRequest(root_directories=(tmp_path,)))

    assert result.status is AgentDiscoveryStatus.COMPLETED
    assert len(registry) == 0
    assert result.agent_definitions[0].agent_type is AgentType.PROJECT_ANALYSIS


def test_bootstrap_builds_discovery_with_existing_loader_and_registry_compatibility(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")
    discovery = build_core_agent_discovery(AgentManifestLoader())

    result = discovery.discover(AgentDiscoveryRequest(root_directories=(tmp_path,)))
    registry = AgentRegistry(result.agent_definitions)

    assert isinstance(discovery, AgentDiscovery)
    assert registry.get("atlas.agent.project").agent_id == "atlas.agent.project"

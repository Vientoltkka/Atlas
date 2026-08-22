from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from bootstrap.agent_system import build_core_agent_system
from core.agent_context import AgentContext
from core.agent_discovery import AgentDiscovery
from core.agent_executor import AgentExecutionRequest, AgentExecutionStatus, AgentHandlerRegistry
from core.agent_working_memory import AgentWorkingMemory
from core.agent_handler_registration import AgentHandlerRegistrationItem, AgentHandlerRegistrationRequest
from core.agent_manifest import AgentManifestLoader
from core.agent_registration import AgentRegistrationPolicy, AgentRegistrationRequest
from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolutionRequest
from core.agent_system import (
    AgentSystem,
    AgentSystemBuildRequest,
    AgentSystemBuildStatus,
    AgentSystemBuilder,
    agent_system_build_request_signature,
)


@dataclass(frozen=True)
class EchoHandler:
    agent_id: str = "atlas.agent.echo"
    calls: list[str] | None = None

    def handle(self, context: AgentContext):
        if self.calls is not None:
            self.calls.append(context.agent_id)
        return {"agent_id": context.agent_id, "ok": True}


def _manifest(agent_id: str = "atlas.agent.echo", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "agent_id": agent_id,
        "name": f"Agent {agent_id}",
        "description": "Deterministic agent-system test agent.",
        "version": "1.0.0",
        "enabled": True,
        "agent_type": "general",
        "capabilities": ["agent.echo"],
        "permissions": {
            "can_read_project": True,
            "can_write_files": False,
            "can_execute_tools": False,
            "can_modify_memory": False,
            "can_use_network": False,
            "requires_confirmation": False,
        },
        "limits": {
            "max_steps": 1,
            "max_tool_calls": 0,
            "max_context_items": 8,
            "max_memory_items": 0,
            "max_replans": 0,
        },
        "context_policy": {
            "include_project_context": True,
            "include_conversation_context": False,
            "include_runtime_context": False,
            "allow_user_input": True,
            "allow_shared_context": True,
            "allow_tool_results": False,
            "allow_workflow_results": False,
            "max_context_items": 8,
            "max_context_depth": 3,
            "max_string_length": 500,
            "max_sequence_items": 8,
            "max_mapping_items": 8,
            "max_total_items": 64,
            "allowed_context_keys": [],
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
            "allowed_tools": [],
            "blocked_tools": [],
            "allowed_paths": [],
            "require_confirmation_for_writes": True,
        },
        "handler_id": f"{agent_id}.handler",
        "tags": ["safe", "test"],
        "metadata": {"owner": "atlas"},
    }
    payload.update(overrides)
    return payload


def _write_manifest(path: Path, agent_id: str = "atlas.agent.echo", **overrides: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_manifest(agent_id, **overrides)), encoding="utf-8")


def _handler_item(agent_id: str = "atlas.agent.echo", handler=None, handler_id: str | None = None):
    return AgentHandlerRegistrationItem(
        agent_id=agent_id,
        handler_id=handler_id or f"{agent_id}.handler",
        handler=handler or EchoHandler(agent_id),
    )


def test_empty_build_composes_agent_system() -> None:
    result = AgentSystemBuilder().build()

    assert result.status is AgentSystemBuildStatus.COMPLETED
    assert isinstance(result.system, AgentSystem)
    assert len(result.system.agent_registry) == 0
    assert len(result.system.agent_handler_registry) == 0


def test_dependencies_share_the_expected_registries() -> None:
    result = build_core_agent_system()
    system = result.system
    assert system is not None

    definition = AgentManifestLoader().to_agent_definition(_manifest())
    system.agent_registry.register(definition)
    system.agent_handler_registry.register(EchoHandler())
    execution = system.agent_executor.execute(
        AgentExecutionRequest(
            resolution_request=AgentResolutionRequest(
                required_agent_ids=("atlas.agent.echo",),
                require_unique_top_score=False,
            ),
            user_input="hello",
        )
    )

    assert execution.status is AgentExecutionStatus.COMPLETED
    assert execution.output == {"agent_id": "atlas.agent.echo", "ok": True}


def test_explicit_agent_registration_through_system(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")
    result = build_core_agent_system()
    system = result.system
    assert system is not None

    registration = system.agent_registration_service.register(AgentRegistrationRequest(root_directories=(tmp_path,)))

    assert registration.registered_agent_ids == ("atlas.agent.echo",)
    assert system.agent_registry.contains("atlas.agent.echo") is True


def test_explicit_handler_registration_through_system() -> None:
    result = build_core_agent_system()
    system = result.system
    assert system is not None
    system.agent_registry.register(AgentManifestLoader().to_agent_definition(_manifest()))

    registration = system.agent_handler_registration_service.register(
        AgentHandlerRegistrationRequest(handlers=(_handler_item(),))
    )

    assert registration.registered_agent_ids == ("atlas.agent.echo",)
    assert system.agent_handler_registry.contains("atlas.agent.echo") is True


def test_build_with_discovery_opt_in_registers_agents(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")

    result = build_core_agent_system(AgentSystemBuildRequest(discovery_roots=(tmp_path,)))

    assert result.status is AgentSystemBuildStatus.COMPLETED
    assert result.system is not None
    assert result.system.agent_registry.contains("atlas.agent.echo") is True
    assert result.agent_registration_result is not None


def test_default_does_not_discover_or_register(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")

    result = build_core_agent_system()

    assert result.system is not None
    assert len(result.system.agent_registry) == 0
    assert result.agent_registration_result is None


def test_dry_run_with_discovery_does_not_mutate(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")

    result = build_core_agent_system(AgentSystemBuildRequest(discovery_roots=(tmp_path,), dry_run=True))

    assert result.status is AgentSystemBuildStatus.DRY_RUN_COMPLETED
    assert result.system is not None
    assert len(result.system.agent_registry) == 0


def test_invalid_manifest_failure_has_no_partial_mutation(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "b.json", "atlas.agent.b")
    (tmp_path / "a.json").write_text("{invalid", encoding="utf-8")

    result = build_core_agent_system(AgentSystemBuildRequest(discovery_roots=(tmp_path,)))

    assert result.status is AgentSystemBuildStatus.INITIALIZATION_FAILED
    assert result.system is not None
    assert len(result.system.agent_registry) == 0


def test_agent_registration_conflict_restores_snapshot(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json", description="new definition")
    existing = AgentManifestLoader().to_agent_definition(_manifest(description="existing definition"))
    registry = AgentRegistry((existing,))

    result = build_core_agent_system(
        AgentSystemBuildRequest(discovery_roots=(tmp_path,)),
        agent_registry=registry,
    )

    assert result.status is AgentSystemBuildStatus.INITIALIZATION_FAILED
    assert registry.get("atlas.agent.echo").description == "existing definition"


def test_handler_registration_failure_restores_agent_and_handler_registries(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")

    result = build_core_agent_system(
        AgentSystemBuildRequest(
            discovery_roots=(tmp_path,),
            handler_registration_items=(
                _handler_item(handler_id="wrong.handler"),
            ),
        )
    )

    assert result.status is AgentSystemBuildStatus.INITIALIZATION_FAILED
    assert result.system is not None
    assert len(result.system.agent_registry) == 0
    assert len(result.system.agent_handler_registry) == 0


def test_duplicate_conflict_errors_are_sanitized(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "a.json", "atlas.agent.same")
    _write_manifest(tmp_path / "b.json", "atlas.agent.same", metadata={"owner": "atlas"})

    result = build_core_agent_system(AgentSystemBuildRequest(discovery_roots=(tmp_path,)))

    assert result.status is AgentSystemBuildStatus.INITIALIZATION_FAILED
    text = " ".join(result.errors).lower()
    assert "token" not in text
    assert "secret" not in text


def test_sensitive_manifest_errors_are_sanitized(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "bad.json", metadata={"api_key": "hidden-token-value"})

    result = build_core_agent_system(AgentSystemBuildRequest(discovery_roots=(tmp_path,)))

    text = " ".join(result.errors).lower()
    assert result.status is AgentSystemBuildStatus.INITIALIZATION_FAILED
    assert "hidden-token-value" not in text
    assert "api_key" not in text
    assert "token" not in text


def test_build_request_signature_is_deterministic(tmp_path: Path) -> None:
    first = AgentSystemBuildRequest(discovery_roots=(tmp_path,), metadata={"trace": "a"})
    same = AgentSystemBuildRequest(discovery_roots=(tmp_path,), metadata={"trace": "a"})
    different = AgentSystemBuildRequest(discovery_roots=(tmp_path,), recursive=True, metadata={"trace": "a"})

    assert agent_system_build_request_signature(first) == agent_system_build_request_signature(same)
    assert agent_system_build_request_signature(first) != agent_system_build_request_signature(different)


def test_dependency_injection_for_tests() -> None:
    registry = AgentRegistry()
    handler_registry = AgentHandlerRegistry()
    loader = AgentManifestLoader()
    discovery = AgentDiscovery(loader)

    result = build_core_agent_system(
        agent_registry=registry,
        agent_handler_registry=handler_registry,
        agent_manifest_loader=loader,
        agent_discovery=discovery,
    )

    assert result.system is not None
    assert result.system.agent_registry is registry
    assert result.system.agent_handler_registry is handler_registry
    assert result.system.agent_manifest_loader is loader
    assert result.system.agent_discovery is discovery


def test_build_does_not_execute_agents_or_handlers(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")
    calls: list[str] = []

    result = build_core_agent_system(
        AgentSystemBuildRequest(
            discovery_roots=(tmp_path,),
            handler_registration_items=(_handler_item(handler=EchoHandler(calls=calls)),),
        )
    )

    assert result.status is AgentSystemBuildStatus.COMPLETED
    assert calls == []


def test_compatibility_with_existing_phase_apis(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "agent.json")
    result = build_core_agent_system(
        AgentSystemBuildRequest(
            discovery_roots=(tmp_path,),
            handler_registration_items=(_handler_item(),),
        )
    )
    system = result.system
    assert system is not None

    execution = system.agent_executor.execute(
        AgentExecutionRequest(
            resolution_request=AgentResolutionRequest(
                required_agent_ids=("atlas.agent.echo",),
                require_unique_top_score=False,
            ),
            user_input="hello",
        )
    )

    assert result.status is AgentSystemBuildStatus.COMPLETED
    assert execution.status is AgentExecutionStatus.COMPLETED
    assert system.agent_resolver.resolve(
        AgentResolutionRequest(required_agent_ids=("atlas.agent.echo",), require_unique_top_score=False)
    ).selected_agent_id == "atlas.agent.echo"


def test_working_memory_is_explicitly_composed_and_injectable() -> None:
    memory = AgentWorkingMemory()
    result = build_core_agent_system(agent_working_memory=memory)

    assert result.system is not None
    assert result.system.agent_working_memory is memory

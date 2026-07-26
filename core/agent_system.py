"""Composable safe specialized-agent system for Atlas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

from core.agent_context import AgentContextBuilder
from core.agent_delegation import AgentDelegationService
from core.agent_discovery import AgentDiscovery
from core.agent_executor import AgentExecutor, AgentHandlerRegistry
from core.agent_handler_registration import (
    AgentHandlerRegistrationItem,
    AgentHandlerRegistrationPolicy,
    AgentHandlerRegistrationRequest,
    AgentHandlerRegistrationResult,
    AgentHandlerRegistrationService,
    AgentHandlerRegistrationStatus,
)
from core.agent_manifest import AgentManifestLoader
from core.multi_agent import MultiAgentCoordinator, MultiAgentResolver
from core.skill_system import SkillSystem, build_skill_system
from core.agent_registration import (
    AgentRegistrationPolicy,
    AgentRegistrationRequest,
    AgentRegistrationResult,
    AgentRegistrationService,
    AgentRegistrationStatus,
)
from core.agent_registry import AgentRegistry
from core.agent_resolver import AgentResolver


MAX_AGENT_SYSTEM_METADATA_ITEMS = 16
MAX_AGENT_SYSTEM_LIMIT = 1_000_000
_SENSITIVE_KEY_PARTS = (
    "secret",
    "api_key",
    "apikey",
    "password",
    "token",
    "authorization",
    "cookie",
    "private_key",
    "credential",
    "prompt",
)


class AgentSystemBuildStatus(str, Enum):
    """Structured statuses for safe agent-system composition."""

    COMPLETED = "COMPLETED"
    DRY_RUN_COMPLETED = "DRY_RUN_COMPLETED"
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    INVALID_REQUEST = "INVALID_REQUEST"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentSystemBuildError(RuntimeError):
    """Base error for safe agent-system composition."""


class InvalidAgentSystemBuildRequestError(AgentSystemBuildError):
    """Raised when an agent-system build request is malformed."""


@dataclass(frozen=True, slots=True)
class AgentSystem:
    """Fully composed specialized-agent infrastructure graph."""

    agent_registry: AgentRegistry
    agent_handler_registry: AgentHandlerRegistry
    agent_manifest_loader: AgentManifestLoader
    agent_discovery: AgentDiscovery
    agent_registration_service: AgentRegistrationService
    agent_handler_registration_service: AgentHandlerRegistrationService
    agent_resolver: AgentResolver
    agent_context_builder: AgentContextBuilder
    agent_executor: AgentExecutor
    multi_agent_resolver: MultiAgentResolver
    multi_agent_coordinator: MultiAgentCoordinator
    skill_system: SkillSystem
    agent_delegation_service: AgentDelegationService

    def __post_init__(self) -> None:
        if not isinstance(self.agent_registry, AgentRegistry):
            raise InvalidAgentSystemBuildRequestError("agent_registry must be AgentRegistry.")
        if not isinstance(self.agent_handler_registry, AgentHandlerRegistry):
            raise InvalidAgentSystemBuildRequestError("agent_handler_registry must be AgentHandlerRegistry.")
        if not isinstance(self.agent_manifest_loader, AgentManifestLoader):
            raise InvalidAgentSystemBuildRequestError("agent_manifest_loader must be AgentManifestLoader.")
        if not isinstance(self.agent_discovery, AgentDiscovery):
            raise InvalidAgentSystemBuildRequestError("agent_discovery must be AgentDiscovery.")
        if not isinstance(self.agent_registration_service, AgentRegistrationService):
            raise InvalidAgentSystemBuildRequestError("agent_registration_service must be AgentRegistrationService.")
        if not isinstance(self.agent_handler_registration_service, AgentHandlerRegistrationService):
            raise InvalidAgentSystemBuildRequestError(
                "agent_handler_registration_service must be AgentHandlerRegistrationService."
            )
        if not isinstance(self.agent_resolver, AgentResolver):
            raise InvalidAgentSystemBuildRequestError("agent_resolver must be AgentResolver.")
        if not isinstance(self.agent_context_builder, AgentContextBuilder):
            raise InvalidAgentSystemBuildRequestError("agent_context_builder must be AgentContextBuilder.")
        if not isinstance(self.agent_executor, AgentExecutor):
            raise InvalidAgentSystemBuildRequestError("agent_executor must be AgentExecutor.")
        if not isinstance(self.multi_agent_resolver, MultiAgentResolver):
            raise InvalidAgentSystemBuildRequestError("multi_agent_resolver must be MultiAgentResolver.")
        if not isinstance(self.multi_agent_coordinator, MultiAgentCoordinator):
            raise InvalidAgentSystemBuildRequestError("multi_agent_coordinator must be MultiAgentCoordinator.")
        if not isinstance(self.skill_system, SkillSystem):
            raise InvalidAgentSystemBuildRequestError("skill_system must be SkillSystem.")
        if not isinstance(self.agent_delegation_service, AgentDelegationService):
            raise InvalidAgentSystemBuildRequestError("agent_delegation_service must be AgentDelegationService.")


@dataclass(frozen=True, slots=True)
class AgentSystemBuildRequest:
    """Declarative request for composing and optionally initializing AgentSystem."""

    discovery_roots: Iterable[str | Path] = ()
    recursive: bool = False
    agent_registration_policy: AgentRegistrationPolicy = field(default_factory=AgentRegistrationPolicy)
    handler_registration_items: Iterable[AgentHandlerRegistrationItem] = ()
    handler_registration_policy: AgentHandlerRegistrationPolicy = field(default_factory=AgentHandlerRegistrationPolicy)
    dry_run: bool = False
    max_directories: int = 128
    max_files: int = 256
    max_manifest_bytes: int = 64_000
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        roots = tuple(_normalize_root(root) for root in self.discovery_roots)
        object.__setattr__(self, "discovery_roots", tuple(sorted(dict.fromkeys(roots), key=_path_key)))
        if type(self.recursive) is not bool:
            raise InvalidAgentSystemBuildRequestError("recursive must be a bool.")
        if not isinstance(self.agent_registration_policy, AgentRegistrationPolicy):
            raise InvalidAgentSystemBuildRequestError("agent_registration_policy must be AgentRegistrationPolicy.")
        if not isinstance(self.handler_registration_policy, AgentHandlerRegistrationPolicy):
            raise InvalidAgentSystemBuildRequestError(
                "handler_registration_policy must be AgentHandlerRegistrationPolicy."
            )
        if type(self.dry_run) is not bool:
            raise InvalidAgentSystemBuildRequestError("dry_run must be a bool.")
        handlers = tuple(self.handler_registration_items)
        if not all(isinstance(item, AgentHandlerRegistrationItem) for item in handlers):
            raise InvalidAgentSystemBuildRequestError(
                "handler_registration_items must contain AgentHandlerRegistrationItem values."
            )
        object.__setattr__(
            self,
            "handler_registration_items",
            tuple(sorted(handlers, key=lambda item: (item.agent_id, item.handler_id))),
        )
        object.__setattr__(self, "max_directories", _positive_int(self.max_directories, "max_directories"))
        object.__setattr__(self, "max_files", _positive_int(self.max_files, "max_files"))
        object.__setattr__(self, "max_manifest_bytes", _positive_int(self.max_manifest_bytes, "max_manifest_bytes"))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class AgentSystemBuildResult:
    """Structured result for safe agent-system build."""

    status: AgentSystemBuildStatus
    system: AgentSystem | None = None
    agent_registration_result: AgentRegistrationResult | None = None
    handler_registration_result: AgentHandlerRegistrationResult | None = None
    errors: tuple[str, ...] = ()
    request_signature: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        if self.system is not None and not isinstance(self.system, AgentSystem):
            raise InvalidAgentSystemBuildRequestError("system must be AgentSystem or None.")
        object.__setattr__(self, "errors", tuple(_safe_message(error) for error in self.errors))


class AgentSystemBuilder:
    """Build one coherent specialized-agent subsystem with shared registries."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry | None = None,
        agent_handler_registry: AgentHandlerRegistry | None = None,
        agent_manifest_loader: AgentManifestLoader | None = None,
        agent_discovery: AgentDiscovery | None = None,
        agent_registration_service: AgentRegistrationService | None = None,
        agent_handler_registration_service: AgentHandlerRegistrationService | None = None,
        agent_resolver: AgentResolver | None = None,
        agent_context_builder: AgentContextBuilder | None = None,
        agent_executor: AgentExecutor | None = None,
        multi_agent_resolver: MultiAgentResolver | None = None,
        multi_agent_coordinator: MultiAgentCoordinator | None = None,
        skill_system: SkillSystem | None = None,
        agent_delegation_service: AgentDelegationService | None = None,
    ) -> None:
        self._agent_registry = agent_registry
        self._agent_handler_registry = agent_handler_registry
        self._agent_manifest_loader = agent_manifest_loader
        self._agent_discovery = agent_discovery
        self._agent_registration_service = agent_registration_service
        self._agent_handler_registration_service = agent_handler_registration_service
        self._agent_resolver = agent_resolver
        self._agent_context_builder = agent_context_builder
        self._agent_executor = agent_executor
        self._multi_agent_resolver = multi_agent_resolver
        self._multi_agent_coordinator = multi_agent_coordinator
        self._skill_system = skill_system
        self._agent_delegation_service = agent_delegation_service

    def build(
        self,
        request: AgentSystemBuildRequest | None = None,
    ) -> AgentSystemBuildResult:
        """Return a composed AgentSystem and optionally run explicit initialization."""

        try:
            request = request or AgentSystemBuildRequest()
            if not isinstance(request, AgentSystemBuildRequest):
                raise InvalidAgentSystemBuildRequestError("request must be AgentSystemBuildRequest.")
            return self._build(request)
        except InvalidAgentSystemBuildRequestError as error:
            return _result(AgentSystemBuildStatus.INVALID_REQUEST, request_signature="", errors=(str(error),))
        except (RuntimeError, ValueError, TypeError) as error:
            signature = agent_system_build_request_signature(request) if isinstance(request, AgentSystemBuildRequest) else ""
            return _result(AgentSystemBuildStatus.INTERNAL_ERROR, request_signature=signature, errors=(str(error),))

    def _build(
        self,
        request: AgentSystemBuildRequest,
    ) -> AgentSystemBuildResult:
        request_signature = agent_system_build_request_signature(request)
        registry = self._agent_registry if self._agent_registry is not None else AgentRegistry()
        handler_registry = (
            self._agent_handler_registry if self._agent_handler_registry is not None else AgentHandlerRegistry()
        )
        manifest_loader = (
            self._agent_manifest_loader if self._agent_manifest_loader is not None else AgentManifestLoader()
        )
        discovery = self._agent_discovery if self._agent_discovery is not None else AgentDiscovery(manifest_loader)
        registration_service = self._agent_registration_service if self._agent_registration_service is not None else AgentRegistrationService(
            discovery,
            manifest_loader,
            registry,
        )
        handler_registration_service = (
            self._agent_handler_registration_service
            if self._agent_handler_registration_service is not None
            else AgentHandlerRegistrationService(registry, handler_registry)
        )
        resolver = self._agent_resolver if self._agent_resolver is not None else AgentResolver(registry)
        context_builder = (
            self._agent_context_builder if self._agent_context_builder is not None else AgentContextBuilder()
        )
        executor = self._agent_executor if self._agent_executor is not None else AgentExecutor(resolver, context_builder, handler_registry)
        multi_agent_resolver = (
            self._multi_agent_resolver if self._multi_agent_resolver is not None else MultiAgentResolver(registry)
        )
        multi_agent_coordinator = (
            self._multi_agent_coordinator
            if self._multi_agent_coordinator is not None
            else MultiAgentCoordinator(multi_agent_resolver, executor)
        )
        skill_system = self._skill_system if self._skill_system is not None else build_skill_system()
        agent_delegation_service = (
            self._agent_delegation_service
            if self._agent_delegation_service is not None
            else AgentDelegationService(
                agent_registry=registry,
                agent_resolver=resolver,
                agent_context_builder=context_builder,
                agent_executor=executor,
            )
        )
        system = AgentSystem(
            agent_registry=registry,
            agent_handler_registry=handler_registry,
            agent_manifest_loader=manifest_loader,
            agent_discovery=discovery,
            agent_registration_service=registration_service,
            agent_handler_registration_service=handler_registration_service,
            agent_resolver=resolver,
            agent_context_builder=context_builder,
            agent_executor=executor,
            multi_agent_resolver=multi_agent_resolver,
            multi_agent_coordinator=multi_agent_coordinator,
            skill_system=skill_system,
            agent_delegation_service=agent_delegation_service,
        )

        agent_snapshot = registry.list_agents()
        handler_snapshot = handler_registry.list_handlers()
        agent_result: AgentRegistrationResult | None = None
        handler_result: AgentHandlerRegistrationResult | None = None
        try:
            if request.discovery_roots:
                agent_policy = AgentRegistrationPolicy(
                    duplicate_agent_policy=request.agent_registration_policy.duplicate_agent_policy,
                    enabled_only=request.agent_registration_policy.enabled_only,
                    dry_run=request.dry_run or request.agent_registration_policy.dry_run,
                    max_manifests=request.agent_registration_policy.max_manifests,
                )
                agent_result = registration_service.register(
                    AgentRegistrationRequest(
                        root_directories=request.discovery_roots,
                        recursive=request.recursive,
                        policy=agent_policy,
                        max_directories=request.max_directories,
                        max_files=request.max_files,
                        max_manifest_bytes=request.max_manifest_bytes,
                        metadata=request.metadata,
                    )
                )
                if agent_result.status not in (
                    AgentRegistrationStatus.COMPLETED,
                    AgentRegistrationStatus.DRY_RUN_COMPLETED,
                    AgentRegistrationStatus.NO_MANIFESTS_FOUND,
                ):
                    _restore_registry(registry, agent_snapshot)
                    _restore_handler_registry(handler_registry, handler_snapshot)
                    return _result(
                        AgentSystemBuildStatus.INITIALIZATION_FAILED,
                        system=system,
                        agent_registration_result=agent_result,
                        request_signature=request_signature,
                        errors=agent_result.errors,
                    )
            if request.handler_registration_items:
                handler_policy = AgentHandlerRegistrationPolicy(
                    duplicate_handler_policy=request.handler_registration_policy.duplicate_handler_policy,
                    dry_run=request.dry_run or request.handler_registration_policy.dry_run,
                    max_handlers=request.handler_registration_policy.max_handlers,
                )
                handler_result = handler_registration_service.register(
                    AgentHandlerRegistrationRequest(
                        handlers=request.handler_registration_items,
                        policy=handler_policy,
                        metadata=request.metadata,
                    )
                )
                if handler_result.status not in (
                    AgentHandlerRegistrationStatus.COMPLETED,
                    AgentHandlerRegistrationStatus.DRY_RUN_COMPLETED,
                ):
                    _restore_registry(registry, agent_snapshot)
                    _restore_handler_registry(handler_registry, handler_snapshot)
                    return _result(
                        AgentSystemBuildStatus.INITIALIZATION_FAILED,
                        system=system,
                        agent_registration_result=agent_result,
                        handler_registration_result=handler_result,
                        request_signature=request_signature,
                        errors=handler_result.errors,
                    )
        except (RuntimeError, ValueError, TypeError) as error:
            _restore_registry(registry, agent_snapshot)
            _restore_handler_registry(handler_registry, handler_snapshot)
            return _result(
                AgentSystemBuildStatus.INITIALIZATION_FAILED,
                system=system,
                agent_registration_result=agent_result,
                handler_registration_result=handler_result,
                request_signature=request_signature,
                errors=(str(error),),
            )

        status = AgentSystemBuildStatus.DRY_RUN_COMPLETED if request.dry_run else AgentSystemBuildStatus.COMPLETED
        return _result(
            status,
            system=system,
            agent_registration_result=agent_result,
            handler_registration_result=handler_result,
            request_signature=request_signature,
        )


def agent_system_build_request_signature(
    request: AgentSystemBuildRequest,
) -> str:
    """Return a deterministic SHA-256 signature for an AgentSystem build request."""

    if not isinstance(request, AgentSystemBuildRequest):
        raise InvalidAgentSystemBuildRequestError("request must be AgentSystemBuildRequest.")
    payload = {
        "discovery_roots": tuple(str(root) for root in request.discovery_roots),
        "recursive": request.recursive,
        "agent_registration_policy": {
            "duplicate_agent_policy": request.agent_registration_policy.duplicate_agent_policy.value,
            "enabled_only": request.agent_registration_policy.enabled_only,
            "dry_run": request.agent_registration_policy.dry_run,
            "max_manifests": request.agent_registration_policy.max_manifests,
        },
        "handler_registration_items": tuple(
            {
                "agent_id": item.agent_id,
                "handler_id": item.handler_id,
                "metadata": item.metadata,
            }
            for item in request.handler_registration_items
        ),
        "handler_registration_policy": {
            "duplicate_handler_policy": request.handler_registration_policy.duplicate_handler_policy.value,
            "dry_run": request.handler_registration_policy.dry_run,
            "max_handlers": request.handler_registration_policy.max_handlers,
        },
        "dry_run": request.dry_run,
        "max_directories": request.max_directories,
        "max_files": request.max_files,
        "max_manifest_bytes": request.max_manifest_bytes,
        "metadata": request.metadata,
    }
    encoded = json.dumps(_jsonable(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _restore_registry(
    registry: AgentRegistry,
    snapshot: tuple,
) -> None:
    registry.clear()
    for definition in snapshot:
        registry.register(definition)


def _restore_handler_registry(
    registry: AgentHandlerRegistry,
    snapshot: tuple,
) -> None:
    registry.clear()
    for handler in snapshot:
        registry.register(handler)


def _result(
    status: AgentSystemBuildStatus,
    *,
    system: AgentSystem | None = None,
    agent_registration_result: AgentRegistrationResult | None = None,
    handler_registration_result: AgentHandlerRegistrationResult | None = None,
    request_signature: str,
    errors: tuple[str, ...] = (),
) -> AgentSystemBuildResult:
    return AgentSystemBuildResult(
        status=status,
        system=system,
        agent_registration_result=agent_registration_result,
        handler_registration_result=handler_registration_result,
        errors=errors,
        request_signature=request_signature,
    )


def _normalize_root(
    value: str | Path,
) -> Path:
    if not isinstance(value, (str, Path)):
        raise InvalidAgentSystemBuildRequestError("discovery_roots must contain paths.")
    text = str(value)
    if not text.strip():
        raise InvalidAgentSystemBuildRequestError("discovery_roots cannot contain empty paths.")
    try:
        return Path(text).expanduser().resolve()
    except OSError as error:
        raise InvalidAgentSystemBuildRequestError("discovery root cannot be resolved.") from error


def _positive_int(
    value: int,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidAgentSystemBuildRequestError(f"{field_name} must be an integer.")
    if value <= 0 or value > MAX_AGENT_SYSTEM_LIMIT:
        raise InvalidAgentSystemBuildRequestError(f"{field_name} must be between 1 and {MAX_AGENT_SYSTEM_LIMIT}.")
    return value


def _safe_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidAgentSystemBuildRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_AGENT_SYSTEM_METADATA_ITEMS:
        raise InvalidAgentSystemBuildRequestError("metadata has too many items.")
    safe: dict[str, object] = {}
    for raw_key in sorted(metadata):
        if not isinstance(raw_key, str) or not raw_key.strip() or raw_key.strip() != raw_key:
            raise InvalidAgentSystemBuildRequestError("metadata keys must be safe strings.")
        if _is_sensitive_key(raw_key):
            raise InvalidAgentSystemBuildRequestError("metadata contains a forbidden sensitive key.")
        safe[raw_key] = _safe_metadata_value(metadata[raw_key])
    return safe


def _safe_metadata_value(
    value: object,
) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentSystemBuildRequestError("metadata floats must be finite.")
        return value
    raise InvalidAgentSystemBuildRequestError("metadata values must be primitive safe values.")


def _jsonable(
    value: object,
) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is not None and type(value) not in (bool, int, float, str):
        raise InvalidAgentSystemBuildRequestError("unsupported build signature value.")
    return value


def _path_key(
    path: Path,
) -> str:
    return str(path).replace("\\", "/").lower()


def _is_sensitive_key(
    key: str,
) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _safe_message(
    value: str,
) -> str:
    message = " ".join(value.split())
    for key in _SENSITIVE_KEY_PARTS:
        message = message.replace(key, "[redacted]")
    return message[:240]


def _status(
    value: AgentSystemBuildStatus | str,
) -> AgentSystemBuildStatus:
    if isinstance(value, AgentSystemBuildStatus):
        return value
    if isinstance(value, str):
        return AgentSystemBuildStatus(value)
    raise InvalidAgentSystemBuildRequestError("status must be AgentSystemBuildStatus.")

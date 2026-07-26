"""Controlled registration of safely discovered Atlas agent manifests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

from core.agent_discovery import (
    AgentDiscovery,
    AgentDiscoveryRequest,
    AgentDiscoveryResult,
    AgentDiscoveryStatus,
)
from core.agent_manifest import AgentManifestLoader, agent_manifest_signature
from core.agent_registry import (
    AgentAlreadyRegisteredError,
    AgentDefinition,
    AgentRegistry,
    InvalidAgentDefinitionError,
    validate_agent_id,
)


MAX_REGISTRATION_METADATA_ITEMS = 16
MAX_REGISTRATION_MANIFESTS = 1_000
MAX_REGISTRATION_LIMIT = 1_000_000
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


class AgentRegistrationStatus(str, Enum):
    """Structured statuses for controlled agent registration."""

    COMPLETED = "COMPLETED"
    DRY_RUN_COMPLETED = "DRY_RUN_COMPLETED"
    NO_MANIFESTS_FOUND = "NO_MANIFESTS_FOUND"
    DISCOVERY_FAILED = "DISCOVERY_FAILED"
    MANIFEST_VALIDATION_FAILED = "MANIFEST_VALIDATION_FAILED"
    DUPLICATE_AGENT = "DUPLICATE_AGENT"
    AGENT_CONFLICT = "AGENT_CONFLICT"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    INVALID_REQUEST = "INVALID_REQUEST"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentRegistrationDuplicatePolicy(str, Enum):
    """Deterministic policy for already registered agent ids."""

    REJECT = "REJECT"
    KEEP_EXISTING = "KEEP_EXISTING"
    REPLACE = "REPLACE"


class AgentRegistrationError(RuntimeError):
    """Base error for controlled agent registration."""


class InvalidAgentRegistrationRequestError(AgentRegistrationError):
    """Raised when an agent registration request is malformed."""


@dataclass(frozen=True, slots=True)
class AgentRegistrationPolicy:
    """Immutable policy for controlled registration into AgentRegistry."""

    duplicate_agent_policy: AgentRegistrationDuplicatePolicy | str = AgentRegistrationDuplicatePolicy.REJECT
    enabled_only: bool = False
    dry_run: bool = False
    max_manifests: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "duplicate_agent_policy", _duplicate_policy(self.duplicate_agent_policy))
        if type(self.enabled_only) is not bool:
            raise InvalidAgentRegistrationRequestError("enabled_only must be a bool.")
        if type(self.dry_run) is not bool:
            raise InvalidAgentRegistrationRequestError("dry_run must be a bool.")
        object.__setattr__(self, "max_manifests", _positive_int(self.max_manifests, "max_manifests"))


@dataclass(frozen=True, slots=True)
class AgentRegistrationRequest:
    """Explicit immutable request for discovery-backed agent registration."""

    root_directories: Iterable[str | Path]
    recursive: bool = False
    policy: AgentRegistrationPolicy = field(default_factory=AgentRegistrationPolicy)
    max_directories: int = 128
    max_files: int = 256
    max_manifest_bytes: int = 64_000
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        roots = tuple(_normalize_root(root) for root in self.root_directories)
        if not roots:
            raise InvalidAgentRegistrationRequestError("root_directories cannot be empty.")
        object.__setattr__(self, "root_directories", tuple(sorted(dict.fromkeys(roots), key=_path_key)))
        if type(self.recursive) is not bool:
            raise InvalidAgentRegistrationRequestError("recursive must be a bool.")
        if not isinstance(self.policy, AgentRegistrationPolicy):
            raise InvalidAgentRegistrationRequestError("policy must be AgentRegistrationPolicy.")
        object.__setattr__(self, "max_directories", _positive_int(self.max_directories, "max_directories"))
        object.__setattr__(self, "max_files", _positive_int(self.max_files, "max_files"))
        object.__setattr__(self, "max_manifest_bytes", _positive_int(self.max_manifest_bytes, "max_manifest_bytes"))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class AgentRegistrationEntry:
    """Immutable summary for one discovered agent registration decision."""

    agent_id: str
    manifest_signature: str
    action: str
    replaced: bool = False
    skipped: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        if not isinstance(self.manifest_signature, str) or len(self.manifest_signature) != 64:
            raise InvalidAgentRegistrationRequestError("manifest_signature must be a SHA-256 hex digest.")
        if self.action not in ("registered", "would_register", "skipped", "replaced", "would_replace"):
            raise InvalidAgentRegistrationRequestError("action is invalid.")
        if type(self.replaced) is not bool:
            raise InvalidAgentRegistrationRequestError("replaced must be a bool.")
        if type(self.skipped) is not bool:
            raise InvalidAgentRegistrationRequestError("skipped must be a bool.")
        if self.reason is not None:
            object.__setattr__(self, "reason", _safe_message(self.reason))


@dataclass(frozen=True, slots=True)
class AgentRegistrationResult:
    """Structured immutable result of controlled agent registration."""

    status: AgentRegistrationStatus
    discovered_agent_ids: tuple[str, ...] = ()
    validated_agent_ids: tuple[str, ...] = ()
    registered_agent_ids: tuple[str, ...] = ()
    skipped_agent_ids: tuple[str, ...] = ()
    replaced_agent_ids: tuple[str, ...] = ()
    rejected_agent_ids: tuple[str, ...] = ()
    entries: tuple[AgentRegistrationEntry, ...] = ()
    errors: tuple[str, ...] = ()
    request_signature: str = ""
    manifests_processed: int = 0
    discovery_result: AgentDiscoveryResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        for field_name in (
            "discovered_agent_ids",
            "validated_agent_ids",
            "registered_agent_ids",
            "skipped_agent_ids",
            "replaced_agent_ids",
            "rejected_agent_ids",
        ):
            object.__setattr__(self, field_name, _agent_id_tuple(getattr(self, field_name), field_name))
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "errors", tuple(_safe_message(error) for error in self.errors))
        if isinstance(self.manifests_processed, bool) or not isinstance(self.manifests_processed, int):
            raise InvalidAgentRegistrationRequestError("manifests_processed must be an integer.")


class AgentRegistrationService:
    """Register discovered AgentDefinitions atomically and without executing agents."""

    def __init__(
        self,
        agent_discovery: AgentDiscovery,
        manifest_loader: AgentManifestLoader,
        agent_registry: AgentRegistry,
    ) -> None:
        if not isinstance(agent_discovery, AgentDiscovery):
            raise InvalidAgentRegistrationRequestError("agent_discovery must be AgentDiscovery.")
        if not isinstance(manifest_loader, AgentManifestLoader):
            raise InvalidAgentRegistrationRequestError("manifest_loader must be AgentManifestLoader.")
        if not isinstance(agent_registry, AgentRegistry):
            raise InvalidAgentRegistrationRequestError("agent_registry must be AgentRegistry.")
        self._agent_discovery = agent_discovery
        self._manifest_loader = manifest_loader
        self._agent_registry = agent_registry

    def register(
        self,
        request: AgentRegistrationRequest,
    ) -> AgentRegistrationResult:
        """Discover, validate, and register a batch of agent definitions atomically."""

        try:
            if not isinstance(request, AgentRegistrationRequest):
                raise InvalidAgentRegistrationRequestError("request must be AgentRegistrationRequest.")
            return self._register(request)
        except InvalidAgentRegistrationRequestError as error:
            return _result(AgentRegistrationStatus.INVALID_REQUEST, request_signature="", errors=(str(error),))
        except (RuntimeError, ValueError, TypeError) as error:
            signature = agent_registration_request_signature(request) if isinstance(request, AgentRegistrationRequest) else ""
            return _result(AgentRegistrationStatus.INTERNAL_ERROR, request_signature=signature, errors=(str(error),))

    def _register(
        self,
        request: AgentRegistrationRequest,
    ) -> AgentRegistrationResult:
        request_signature = agent_registration_request_signature(request)
        discovery_request = AgentDiscoveryRequest(
            root_directories=request.root_directories,
            recursive=request.recursive,
            max_directories=request.max_directories,
            max_files=request.max_files,
            max_manifest_bytes=request.max_manifest_bytes,
            invalid_file_policy="collect_errors",
            duplicate_agent_policy="reject",
            enabled_only=request.policy.enabled_only,
            metadata=request.metadata,
        )
        discovery_result = self._agent_discovery.discover(discovery_request)
        if discovery_result.status is AgentDiscoveryStatus.NO_MANIFESTS_FOUND:
            return _result(
                AgentRegistrationStatus.NO_MANIFESTS_FOUND,
                request_signature=request_signature,
                discovery_result=discovery_result,
            )
        if discovery_result.status is AgentDiscoveryStatus.LIMIT_EXCEEDED:
            return _result(
                AgentRegistrationStatus.LIMIT_EXCEEDED,
                request_signature=request_signature,
                errors=discovery_result.errors,
                discovery_result=discovery_result,
            )
        if discovery_result.status in (AgentDiscoveryStatus.ROOT_UNAVAILABLE, AgentDiscoveryStatus.INVALID_REQUEST):
            return _result(
                AgentRegistrationStatus.DISCOVERY_FAILED,
                request_signature=request_signature,
                errors=discovery_result.errors,
                discovery_result=discovery_result,
            )
        if discovery_result.status is AgentDiscoveryStatus.DUPLICATE_AGENT:
            return _result(
                AgentRegistrationStatus.DUPLICATE_AGENT,
                request_signature=request_signature,
                discovered_agent_ids=_agent_ids(discovery_result.agent_definitions),
                rejected_agent_ids=_agent_ids(discovery_result.agent_definitions),
                errors=discovery_result.errors,
                discovery_result=discovery_result,
                manifests_processed=discovery_result.files_considered,
            )
        if any("root unavailable" in error for error in discovery_result.errors):
            return _result(
                AgentRegistrationStatus.DISCOVERY_FAILED,
                request_signature=request_signature,
                errors=discovery_result.errors,
                discovery_result=discovery_result,
                manifests_processed=discovery_result.files_considered,
            )
        if discovery_result.status not in (AgentDiscoveryStatus.COMPLETED, AgentDiscoveryStatus.COMPLETED_WITH_ERRORS):
            return _result(
                AgentRegistrationStatus.MANIFEST_VALIDATION_FAILED,
                request_signature=request_signature,
                errors=discovery_result.errors,
                discovery_result=discovery_result,
                manifests_processed=discovery_result.files_considered,
            )

        discovered = tuple(discovery_result.discovered_manifests)
        if len(discovered) > request.policy.max_manifests:
            return _result(
                AgentRegistrationStatus.LIMIT_EXCEEDED,
                request_signature=request_signature,
                discovered_agent_ids=_agent_ids(item.agent_definition for item in discovered),
                errors=("registration manifest limit exceeded.",),
                discovery_result=discovery_result,
                manifests_processed=len(discovered),
            )
        if discovery_result.errors:
            return _result(
                AgentRegistrationStatus.MANIFEST_VALIDATION_FAILED,
                request_signature=request_signature,
                discovered_agent_ids=_agent_ids(item.agent_definition for item in discovered),
                errors=discovery_result.errors,
                discovery_result=discovery_result,
                manifests_processed=discovery_result.files_considered,
            )

        definitions: list[AgentDefinition] = []
        signatures_by_agent_id: dict[str, str] = {}
        for item in discovered:
            manifest = self._manifest_loader.load(item.manifest)
            definition = self._manifest_loader.to_agent_definition(manifest)
            signature = agent_manifest_signature(manifest)
            if definition.agent_id in signatures_by_agent_id:
                return _result(
                    AgentRegistrationStatus.DUPLICATE_AGENT,
                    request_signature=request_signature,
                    discovered_agent_ids=_agent_ids(definitions),
                    rejected_agent_ids=(definition.agent_id,),
                    errors=(f"duplicate agent_id in registration batch: {definition.agent_id}",),
                    discovery_result=discovery_result,
                    manifests_processed=len(discovered),
                )
            signatures_by_agent_id[definition.agent_id] = signature
            definitions.append(definition)

        preflight = self._preflight(request, tuple(definitions), signatures_by_agent_id)
        if preflight.status is not AgentRegistrationStatus.COMPLETED:
            return _result(
                preflight.status,
                request_signature=request_signature,
                discovered_agent_ids=_agent_ids(definitions),
                validated_agent_ids=_agent_ids(definitions),
                skipped_agent_ids=preflight.skipped_agent_ids,
                replaced_agent_ids=preflight.replaced_agent_ids,
                rejected_agent_ids=preflight.rejected_agent_ids,
                entries=preflight.entries,
                errors=preflight.errors,
                discovery_result=discovery_result,
                manifests_processed=len(definitions),
            )
        if request.policy.dry_run:
            return _result(
                AgentRegistrationStatus.DRY_RUN_COMPLETED,
                request_signature=request_signature,
                discovered_agent_ids=_agent_ids(definitions),
                validated_agent_ids=_agent_ids(definitions),
                skipped_agent_ids=preflight.skipped_agent_ids,
                replaced_agent_ids=preflight.replaced_agent_ids,
                entries=tuple(
                    AgentRegistrationEntry(
                        agent_id=entry.agent_id,
                        manifest_signature=entry.manifest_signature,
                        action=(
                            "would_replace"
                            if entry.action == "replaced"
                            else "skipped"
                            if entry.action == "skipped"
                            else "would_register"
                        ),
                        replaced=entry.replaced,
                        skipped=entry.skipped,
                        reason=entry.reason,
                    )
                    for entry in preflight.entries
                ),
                discovery_result=discovery_result,
                manifests_processed=len(definitions),
            )

        registered: list[str] = []
        replaced: list[str] = []
        entries: list[AgentRegistrationEntry] = []
        try:
            for definition in definitions:
                signature = signatures_by_agent_id[definition.agent_id]
                if definition.agent_id in preflight.skipped_agent_ids:
                    entries.append(
                        AgentRegistrationEntry(
                            agent_id=definition.agent_id,
                            manifest_signature=signature,
                            action="skipped",
                            skipped=True,
                            reason="existing agent kept by policy.",
                        )
                    )
                    continue
                replace = definition.agent_id in preflight.replaced_agent_ids
                self._agent_registry.register(definition, replace=replace)
                if replace:
                    replaced.append(definition.agent_id)
                    entries.append(
                        AgentRegistrationEntry(
                            agent_id=definition.agent_id,
                            manifest_signature=signature,
                            action="replaced",
                            replaced=True,
                        )
                    )
                else:
                    registered.append(definition.agent_id)
                    entries.append(
                        AgentRegistrationEntry(
                            agent_id=definition.agent_id,
                            manifest_signature=signature,
                            action="registered",
                        )
                    )
        except (AgentAlreadyRegisteredError, InvalidAgentDefinitionError) as error:
            return _result(
                AgentRegistrationStatus.REGISTRATION_FAILED,
                request_signature=request_signature,
                discovered_agent_ids=_agent_ids(definitions),
                validated_agent_ids=_agent_ids(definitions),
                errors=(str(error),),
                discovery_result=discovery_result,
                manifests_processed=len(definitions),
            )

        return _result(
            AgentRegistrationStatus.COMPLETED,
            request_signature=request_signature,
            discovered_agent_ids=_agent_ids(definitions),
            validated_agent_ids=_agent_ids(definitions),
            registered_agent_ids=tuple(registered),
            skipped_agent_ids=preflight.skipped_agent_ids,
            replaced_agent_ids=tuple(replaced),
                entries=tuple(entries),
                errors=preflight.errors,
                discovery_result=discovery_result,
                manifests_processed=len(definitions),
            )

    def _preflight(
        self,
        request: AgentRegistrationRequest,
        definitions: tuple[AgentDefinition, ...],
        signatures_by_agent_id: Mapping[str, str],
    ) -> AgentRegistrationResult:
        skipped: list[str] = []
        replaced: list[str] = []
        rejected: list[str] = []
        entries: list[AgentRegistrationEntry] = []
        errors: list[str] = []
        policy = request.policy.duplicate_agent_policy
        for definition in definitions:
            agent_id = definition.agent_id
            signature = signatures_by_agent_id[agent_id]
            if not self._agent_registry.contains(agent_id):
                entries.append(AgentRegistrationEntry(agent_id=agent_id, manifest_signature=signature, action="registered"))
                continue
            existing = self._agent_registry.get(agent_id)
            existing_signature = str(existing.metadata.get("manifest_signature", ""))
            conflict = bool(existing_signature and existing_signature != signature)
            if policy is AgentRegistrationDuplicatePolicy.REJECT:
                rejected.append(agent_id)
                errors.append(
                    f"agent_id already registered with conflicting manifest signature: {agent_id}"
                    if conflict
                    else f"agent_id already registered: {agent_id}"
                )
                status = AgentRegistrationStatus.AGENT_CONFLICT if conflict else AgentRegistrationStatus.DUPLICATE_AGENT
                return _result(
                    status,
                    request_signature="",
                    rejected_agent_ids=tuple(rejected),
                    errors=tuple(errors),
                    entries=tuple(entries),
                )
            if policy is AgentRegistrationDuplicatePolicy.KEEP_EXISTING:
                skipped.append(agent_id)
                if conflict:
                    errors.append(f"existing agent kept despite conflicting manifest signature: {agent_id}")
                entries.append(
                    AgentRegistrationEntry(
                        agent_id=agent_id,
                        manifest_signature=signature,
                        action="skipped",
                        skipped=True,
                        reason="existing agent kept by policy.",
                    )
                )
                continue
            if policy is AgentRegistrationDuplicatePolicy.REPLACE:
                replaced.append(agent_id)
                entries.append(
                    AgentRegistrationEntry(
                        agent_id=agent_id,
                        manifest_signature=signature,
                        action="replaced",
                        replaced=True,
                        reason="existing agent replaced by explicit policy.",
                    )
                )
                continue
        return _result(
            AgentRegistrationStatus.COMPLETED,
            request_signature="",
            skipped_agent_ids=tuple(skipped),
            replaced_agent_ids=tuple(replaced),
            rejected_agent_ids=tuple(rejected),
            errors=tuple(errors),
            entries=tuple(entries),
        )


def agent_registration_request_signature(
    request: AgentRegistrationRequest,
) -> str:
    """Return a deterministic SHA-256 signature for a registration request."""

    if not isinstance(request, AgentRegistrationRequest):
        raise InvalidAgentRegistrationRequestError("request must be AgentRegistrationRequest.")
    payload = {
        "root_directories": tuple(str(path) for path in request.root_directories),
        "recursive": request.recursive,
        "policy": {
            "duplicate_agent_policy": request.policy.duplicate_agent_policy.value,
            "enabled_only": request.policy.enabled_only,
            "dry_run": request.policy.dry_run,
            "max_manifests": request.policy.max_manifests,
        },
        "max_directories": request.max_directories,
        "max_files": request.max_files,
        "max_manifest_bytes": request.max_manifest_bytes,
        "metadata": request.metadata,
    }
    encoded = json.dumps(_jsonable(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _result(
    status: AgentRegistrationStatus,
    *,
    request_signature: str,
    discovered_agent_ids: tuple[str, ...] = (),
    validated_agent_ids: tuple[str, ...] = (),
    registered_agent_ids: tuple[str, ...] = (),
    skipped_agent_ids: tuple[str, ...] = (),
    replaced_agent_ids: tuple[str, ...] = (),
    rejected_agent_ids: tuple[str, ...] = (),
    entries: tuple[AgentRegistrationEntry, ...] = (),
    errors: tuple[str, ...] = (),
    discovery_result: AgentDiscoveryResult | None = None,
    manifests_processed: int = 0,
) -> AgentRegistrationResult:
    return AgentRegistrationResult(
        status=status,
        discovered_agent_ids=discovered_agent_ids,
        validated_agent_ids=validated_agent_ids,
        registered_agent_ids=registered_agent_ids,
        skipped_agent_ids=skipped_agent_ids,
        replaced_agent_ids=replaced_agent_ids,
        rejected_agent_ids=rejected_agent_ids,
        entries=entries,
        errors=errors,
        request_signature=request_signature,
        manifests_processed=manifests_processed,
        discovery_result=discovery_result,
    )


def _duplicate_policy(
    value: AgentRegistrationDuplicatePolicy | str,
) -> AgentRegistrationDuplicatePolicy:
    if isinstance(value, AgentRegistrationDuplicatePolicy):
        return value
    if isinstance(value, str):
        try:
            return AgentRegistrationDuplicatePolicy(value.upper())
        except ValueError as error:
            raise InvalidAgentRegistrationRequestError("duplicate_agent_policy is invalid.") from error
    raise InvalidAgentRegistrationRequestError("duplicate_agent_policy must be a registration policy.")


def _normalize_root(
    value: str | Path,
) -> Path:
    if not isinstance(value, (str, Path)):
        raise InvalidAgentRegistrationRequestError("root_directories must contain paths.")
    text = str(value)
    if not text.strip():
        raise InvalidAgentRegistrationRequestError("root_directories cannot contain empty paths.")
    try:
        return Path(text).expanduser().resolve()
    except OSError as error:
        raise InvalidAgentRegistrationRequestError("root path cannot be resolved.") from error


def _positive_int(
    value: int,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidAgentRegistrationRequestError(f"{field_name} must be an integer.")
    if value <= 0 or value > MAX_REGISTRATION_LIMIT:
        raise InvalidAgentRegistrationRequestError(f"{field_name} must be between 1 and {MAX_REGISTRATION_LIMIT}.")
    return value


def _safe_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidAgentRegistrationRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_REGISTRATION_METADATA_ITEMS:
        raise InvalidAgentRegistrationRequestError("metadata has too many items.")
    safe: dict[str, object] = {}
    for raw_key in sorted(metadata):
        if not isinstance(raw_key, str) or not raw_key.strip() or raw_key.strip() != raw_key:
            raise InvalidAgentRegistrationRequestError("metadata keys must be safe strings.")
        if _is_sensitive_key(raw_key):
            raise InvalidAgentRegistrationRequestError("metadata contains a forbidden sensitive key.")
        safe[raw_key] = _safe_metadata_value(metadata[raw_key])
    return safe


def _safe_metadata_value(
    value: object,
) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentRegistrationRequestError("metadata floats must be finite.")
        return value
    raise InvalidAgentRegistrationRequestError("metadata values must be primitive safe values.")


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
        raise InvalidAgentRegistrationRequestError("unsupported registration signature value.")
    return value


def _agent_ids(
    definitions: Iterable[AgentDefinition],
) -> tuple[str, ...]:
    return tuple(definition.agent_id for definition in definitions)


def _agent_id_tuple(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentRegistrationRequestError(f"{field_name} must be an iterable of agent ids.")
    return tuple(validate_agent_id(value) for value in values)


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
    value: AgentRegistrationStatus | str,
) -> AgentRegistrationStatus:
    if isinstance(value, AgentRegistrationStatus):
        return value
    if isinstance(value, str):
        return AgentRegistrationStatus(value)
    raise InvalidAgentRegistrationRequestError("status must be AgentRegistrationStatus.")

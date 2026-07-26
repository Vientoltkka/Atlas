"""Safe deterministic filesystem discovery for declared Atlas agent manifests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

from core.agent_manifest import (
    AgentManifest,
    AgentManifestLoader,
    InvalidAgentManifestError,
    agent_manifest_signature,
)
from core.agent_registry import AgentDefinition, validate_agent_id


MAX_DISCOVERY_METADATA_ITEMS = 16
MAX_DISCOVERY_LIMIT = 1_000_000
_REPARSE_POINT_ATTRIBUTE = 0x400
_HIDDEN_ATTRIBUTE = 0x2
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


class AgentDiscoveryStatus(str, Enum):
    """Structured statuses for agent-manifest discovery."""

    COMPLETED = "COMPLETED"
    COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"
    NO_MANIFESTS_FOUND = "NO_MANIFESTS_FOUND"
    INVALID_REQUEST = "INVALID_REQUEST"
    ROOT_UNAVAILABLE = "ROOT_UNAVAILABLE"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    DUPLICATE_AGENT = "DUPLICATE_AGENT"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentDiscoveryError(RuntimeError):
    """Base error for safe manifest discovery."""


class InvalidAgentDiscoveryRequestError(AgentDiscoveryError):
    """Raised when the discovery request is malformed."""


class AgentManifestDiscoveryError(AgentDiscoveryError):
    """Raised when one manifest file cannot be discovered or loaded safely."""


class AgentManifestDuplicateError(AgentDiscoveryError):
    """Raised when duplicate manifest agent ids are discovered."""


@dataclass(frozen=True, slots=True)
class AgentDiscoveryRequest:
    """Explicit immutable request for filesystem manifest discovery."""

    root_directories: Iterable[str | Path]
    allowed_extensions: Iterable[str] = (".json",)
    recursive: bool = False
    max_directories: int = 128
    max_files: int = 256
    max_manifest_bytes: int = 64_000
    invalid_file_policy: str = "fail_fast"
    duplicate_agent_policy: str = "reject"
    enabled_only: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        roots = tuple(_normalize_root(value) for value in self.root_directories)
        if not roots:
            raise InvalidAgentDiscoveryRequestError("root_directories cannot be empty.")
        object.__setattr__(self, "root_directories", tuple(sorted(dict.fromkeys(roots), key=_path_key)))
        extensions = tuple(_normalize_extension(value) for value in self.allowed_extensions)
        if not extensions:
            raise InvalidAgentDiscoveryRequestError("allowed_extensions cannot be empty.")
        if any(extension != ".json" for extension in extensions):
            raise InvalidAgentDiscoveryRequestError("only .json manifest files are supported.")
        object.__setattr__(self, "allowed_extensions", tuple(sorted(dict.fromkeys(extensions))))
        if type(self.recursive) is not bool:
            raise InvalidAgentDiscoveryRequestError("recursive must be a bool.")
        if type(self.enabled_only) is not bool:
            raise InvalidAgentDiscoveryRequestError("enabled_only must be a bool.")
        object.__setattr__(self, "max_directories", _positive_int(self.max_directories, "max_directories"))
        object.__setattr__(self, "max_files", _positive_int(self.max_files, "max_files"))
        object.__setattr__(self, "max_manifest_bytes", _positive_int(self.max_manifest_bytes, "max_manifest_bytes"))
        if self.invalid_file_policy not in ("fail_fast", "collect_errors"):
            raise InvalidAgentDiscoveryRequestError("invalid_file_policy must be fail_fast or collect_errors.")
        if self.duplicate_agent_policy not in ("reject", "keep_first"):
            raise InvalidAgentDiscoveryRequestError("duplicate_agent_policy must be reject or keep_first.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class AgentDiscoveredManifest:
    """Immutable discovered manifest plus its converted agent definition."""

    path: str
    root_directory: str
    relative_path: str
    manifest: AgentManifest
    agent_definition: AgentDefinition
    manifest_signature: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(Path(self.path).resolve()))
        object.__setattr__(self, "root_directory", str(Path(self.root_directory).resolve()))
        object.__setattr__(self, "relative_path", _normalize_relative_path(self.relative_path))
        if not isinstance(self.manifest, AgentManifest):
            raise AgentManifestDiscoveryError("manifest must be AgentManifest.")
        if not isinstance(self.agent_definition, AgentDefinition):
            raise AgentManifestDiscoveryError("agent_definition must be AgentDefinition.")
        if self.manifest_signature != agent_manifest_signature(self.manifest):
            raise AgentManifestDiscoveryError("manifest_signature does not match manifest.")


@dataclass(frozen=True, slots=True)
class AgentDiscoveryResult:
    """Structured immutable result of safe manifest discovery."""

    status: AgentDiscoveryStatus
    discovered_manifests: tuple[AgentDiscoveredManifest, ...] = ()
    agent_definitions: tuple[AgentDefinition, ...] = ()
    rejected_files: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    roots_scanned: int = 0
    directories_scanned: int = 0
    files_considered: int = 0
    valid_manifests: int = 0
    invalid_manifests: int = 0
    duplicate_agents: int = 0
    request_signature: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(self, "discovered_manifests", tuple(self.discovered_manifests))
        object.__setattr__(self, "agent_definitions", tuple(self.agent_definitions))
        object.__setattr__(self, "rejected_files", tuple(str(path) for path in self.rejected_files))
        object.__setattr__(self, "errors", tuple(_safe_message(error) for error in self.errors))


class AgentDiscovery:
    """Discover JSON manifests and delegate loading to AgentManifestLoader only."""

    def __init__(
        self,
        manifest_loader: AgentManifestLoader,
    ) -> None:
        if not isinstance(manifest_loader, AgentManifestLoader):
            raise InvalidAgentDiscoveryRequestError("manifest_loader must be AgentManifestLoader.")
        self._manifest_loader = manifest_loader

    def discover(
        self,
        request: AgentDiscoveryRequest,
    ) -> AgentDiscoveryResult:
        """Discover manifests without importing, registering, or executing agents."""

        try:
            if not isinstance(request, AgentDiscoveryRequest):
                raise InvalidAgentDiscoveryRequestError("request must be AgentDiscoveryRequest.")
            return self._discover(request)
        except InvalidAgentDiscoveryRequestError as error:
            return _result(
                AgentDiscoveryStatus.INVALID_REQUEST,
                request_signature="",
                errors=(str(error),),
            )
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            signature = agent_discovery_request_signature(request) if isinstance(request, AgentDiscoveryRequest) else ""
            return _result(
                AgentDiscoveryStatus.INTERNAL_ERROR,
                request_signature=signature,
                errors=(str(error),),
            )

    def _discover(
        self,
        request: AgentDiscoveryRequest,
    ) -> AgentDiscoveryResult:
        request_signature = agent_discovery_request_signature(request)
        roots = tuple(Path(path) for path in request.root_directories)
        errors: list[str] = []
        rejected_files: list[str] = []
        discovered: list[AgentDiscoveredManifest] = []
        seen_signatures_by_agent_id: dict[str, str] = {}
        seen_paths_by_agent_id: dict[str, str] = {}
        roots_scanned = 0
        directories_scanned = 0
        files_considered = 0
        invalid_manifests = 0
        duplicate_agents = 0

        for root in roots:
            if not root.exists() or not root.is_dir():
                errors.append(f"root unavailable: {_display_path(root)}")
                if request.invalid_file_policy == "fail_fast":
                    return _result(
                        AgentDiscoveryStatus.ROOT_UNAVAILABLE,
                        request_signature=request_signature,
                        errors=errors,
                    )
                continue
            if _is_hidden(root) or _is_reparse_point(root):
                errors.append(f"root unavailable: {_display_path(root)}")
                if request.invalid_file_policy == "fail_fast":
                    return _result(
                        AgentDiscoveryStatus.ROOT_UNAVAILABLE,
                        request_signature=request_signature,
                        errors=errors,
                    )
                continue
            roots_scanned += 1
            root_files, directory_count, limit_error = _manifest_paths(root, request)
            directories_scanned += directory_count
            if limit_error is not None:
                errors.append(limit_error)
                return _result(
                    AgentDiscoveryStatus.LIMIT_EXCEEDED,
                    request_signature=request_signature,
                    errors=errors,
                    roots_scanned=roots_scanned,
                    directories_scanned=directories_scanned,
                    files_considered=files_considered,
                    rejected_files=tuple(rejected_files),
                )
            for path in root_files:
                files_considered += 1
                if files_considered > request.max_files:
                    errors.append("manifest file limit exceeded.")
                    return _result(
                        AgentDiscoveryStatus.LIMIT_EXCEEDED,
                        request_signature=request_signature,
                        errors=errors,
                        roots_scanned=roots_scanned,
                        directories_scanned=directories_scanned,
                        files_considered=files_considered,
                        rejected_files=tuple(rejected_files),
                    )
                loaded = self._load_manifest(path, root, request)
                if isinstance(loaded, str):
                    limit_exceeded = "limit exceeded" in loaded
                    invalid_manifests += 1
                    rejected_files.append(_display_path(path))
                    errors.append(loaded)
                    if limit_exceeded:
                        return _result(
                            AgentDiscoveryStatus.LIMIT_EXCEEDED,
                            request_signature=request_signature,
                            errors=errors,
                            roots_scanned=roots_scanned,
                            directories_scanned=directories_scanned,
                            files_considered=files_considered,
                            invalid_manifests=invalid_manifests,
                            rejected_files=tuple(rejected_files),
                            discovered_manifests=tuple(discovered),
                        )
                    if request.invalid_file_policy == "fail_fast":
                        return _result(
                            AgentDiscoveryStatus.MANIFEST_INVALID,
                            request_signature=request_signature,
                            errors=errors,
                            roots_scanned=roots_scanned,
                            directories_scanned=directories_scanned,
                            files_considered=files_considered,
                            invalid_manifests=invalid_manifests,
                            rejected_files=tuple(rejected_files),
                        )
                    continue
                agent_id = loaded.agent_definition.agent_id
                existing_signature = seen_signatures_by_agent_id.get(agent_id)
                if existing_signature is not None:
                    duplicate_agents += 1
                    rejected_files.append(loaded.path)
                    if existing_signature != loaded.manifest_signature:
                        errors.append(
                            f"duplicate agent_id with conflicting manifest signature: {agent_id}"
                        )
                    else:
                        errors.append(
                            f"duplicate agent_id already discovered: {agent_id}"
                        )
                    if request.duplicate_agent_policy == "reject" or request.invalid_file_policy == "fail_fast":
                        return _result(
                            AgentDiscoveryStatus.DUPLICATE_AGENT,
                            request_signature=request_signature,
                            errors=errors,
                            roots_scanned=roots_scanned,
                            directories_scanned=directories_scanned,
                            files_considered=files_considered,
                            invalid_manifests=invalid_manifests,
                            duplicate_agents=duplicate_agents,
                            rejected_files=tuple(rejected_files),
                            discovered_manifests=tuple(discovered),
                        )
                    continue
                seen_signatures_by_agent_id[agent_id] = loaded.manifest_signature
                seen_paths_by_agent_id[agent_id] = loaded.path
                if request.enabled_only and not loaded.agent_definition.enabled:
                    rejected_files.append(loaded.path)
                    continue
                discovered.append(loaded)

        if not discovered and not errors:
            status = AgentDiscoveryStatus.NO_MANIFESTS_FOUND
        elif errors:
            status = AgentDiscoveryStatus.COMPLETED_WITH_ERRORS if discovered else AgentDiscoveryStatus.MANIFEST_INVALID
        else:
            status = AgentDiscoveryStatus.COMPLETED
        return _result(
            status,
            request_signature=request_signature,
            discovered_manifests=tuple(discovered),
            rejected_files=tuple(rejected_files),
            errors=tuple(errors),
            roots_scanned=roots_scanned,
            directories_scanned=directories_scanned,
            files_considered=files_considered,
            valid_manifests=len(discovered),
            invalid_manifests=invalid_manifests,
            duplicate_agents=duplicate_agents,
        )

    def _load_manifest(
        self,
        path: Path,
        root: Path,
        request: AgentDiscoveryRequest,
    ) -> AgentDiscoveredManifest | str:
        try:
            if _is_reparse_point(path) or not _is_within_root(path, root):
                return f"manifest path rejected: {_display_path(path)}"
            size = path.stat().st_size
            if size > request.max_manifest_bytes:
                return f"manifest size limit exceeded: {_display_path(path)}"
            payload = path.read_text(encoding="utf-8")
            manifest = self._manifest_loader.load_json(payload)
            definition = manifest.to_agent_definition()
            signature = agent_manifest_signature(manifest)
            return AgentDiscoveredManifest(
                path=str(path),
                root_directory=str(root),
                relative_path=path.relative_to(root).as_posix(),
                manifest=manifest,
                agent_definition=definition,
                manifest_signature=signature,
            )
        except (OSError, UnicodeError, InvalidAgentManifestError, ValueError) as error:
            return f"manifest invalid: {_display_path(path)}: {_safe_message(str(error))}"


def agent_discovery_request_signature(
    request: AgentDiscoveryRequest,
) -> str:
    """Return a deterministic SHA-256 signature for a discovery request."""

    if not isinstance(request, AgentDiscoveryRequest):
        raise InvalidAgentDiscoveryRequestError("request must be AgentDiscoveryRequest.")
    payload = {
        "root_directories": tuple(str(path) for path in request.root_directories),
        "allowed_extensions": request.allowed_extensions,
        "recursive": request.recursive,
        "max_directories": request.max_directories,
        "max_files": request.max_files,
        "max_manifest_bytes": request.max_manifest_bytes,
        "invalid_file_policy": request.invalid_file_policy,
        "duplicate_agent_policy": request.duplicate_agent_policy,
        "enabled_only": request.enabled_only,
        "metadata": request.metadata,
    }
    encoded = json.dumps(_jsonable(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_paths(
    root: Path,
    request: AgentDiscoveryRequest,
) -> tuple[tuple[Path, ...], int, str | None]:
    found: list[Path] = []
    directories_scanned = 0
    stack = [root]
    while stack:
        current = stack.pop(0)
        if _is_hidden(current) or _is_reparse_point(current) or not _is_within_root(current, root):
            continue
        directories_scanned += 1
        if directories_scanned > request.max_directories:
            return tuple(found), directories_scanned, "directory traversal limit exceeded."
        try:
            entries = sorted(current.iterdir(), key=lambda path: _relative_sort_key(path, root))
        except OSError as error:
            return tuple(found), directories_scanned, f"directory unavailable: {_safe_message(str(error))}"
        for entry in entries:
            if _is_hidden(entry) or _is_reparse_point(entry) or not _is_within_root(entry, root):
                continue
            if entry.is_dir():
                if request.recursive:
                    stack.append(entry)
                continue
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in request.allowed_extensions:
                continue
            found.append(entry.resolve())
    return tuple(sorted(found, key=lambda path: _relative_sort_key(path, root))), directories_scanned, None


def _result(
    status: AgentDiscoveryStatus,
    *,
    request_signature: str,
    discovered_manifests: tuple[AgentDiscoveredManifest, ...] = (),
    rejected_files: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
    roots_scanned: int = 0,
    directories_scanned: int = 0,
    files_considered: int = 0,
    valid_manifests: int | None = None,
    invalid_manifests: int = 0,
    duplicate_agents: int = 0,
) -> AgentDiscoveryResult:
    return AgentDiscoveryResult(
        status=status,
        discovered_manifests=discovered_manifests,
        agent_definitions=tuple(item.agent_definition for item in discovered_manifests),
        rejected_files=rejected_files,
        errors=errors,
        roots_scanned=roots_scanned,
        directories_scanned=directories_scanned,
        files_considered=files_considered,
        valid_manifests=len(discovered_manifests) if valid_manifests is None else valid_manifests,
        invalid_manifests=invalid_manifests,
        duplicate_agents=duplicate_agents,
        request_signature=request_signature,
    )


def _normalize_root(
    value: str | Path,
) -> Path:
    if not isinstance(value, (str, Path)):
        raise InvalidAgentDiscoveryRequestError("root_directories must contain paths.")
    text = str(value)
    if not text.strip():
        raise InvalidAgentDiscoveryRequestError("root_directories cannot contain empty paths.")
    try:
        return Path(text).expanduser().resolve()
    except OSError as error:
        raise InvalidAgentDiscoveryRequestError("root path cannot be resolved.") from error


def _normalize_extension(
    value: str,
) -> str:
    if not isinstance(value, str):
        raise InvalidAgentDiscoveryRequestError("allowed_extensions must contain strings.")
    normalized = value.strip().lower()
    if not normalized.startswith(".") or "/" in normalized or "\\" in normalized or normalized != value:
        raise InvalidAgentDiscoveryRequestError("allowed_extensions must be normalized file extensions.")
    return normalized


def _positive_int(
    value: int,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidAgentDiscoveryRequestError(f"{field_name} must be an integer.")
    if value <= 0 or value > MAX_DISCOVERY_LIMIT:
        raise InvalidAgentDiscoveryRequestError(f"{field_name} must be between 1 and {MAX_DISCOVERY_LIMIT}.")
    return value


def _safe_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidAgentDiscoveryRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_DISCOVERY_METADATA_ITEMS:
        raise InvalidAgentDiscoveryRequestError("metadata has too many items.")
    safe: dict[str, object] = {}
    for raw_key in sorted(metadata):
        if not isinstance(raw_key, str) or not raw_key.strip() or raw_key.strip() != raw_key:
            raise InvalidAgentDiscoveryRequestError("metadata keys must be safe strings.")
        if _is_sensitive_key(raw_key):
            raise InvalidAgentDiscoveryRequestError("metadata contains a forbidden sensitive key.")
        safe[raw_key] = _safe_metadata_value(metadata[raw_key])
    return safe


def _safe_metadata_value(
    value: object,
) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentDiscoveryRequestError("metadata floats must be finite.")
        return value
    raise InvalidAgentDiscoveryRequestError("metadata values must be primitive safe values.")


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
        raise InvalidAgentDiscoveryRequestError("unsupported discovery signature value.")
    return value


def _is_within_root(
    path: Path,
    root: Path,
) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _is_hidden(
    path: Path,
) -> bool:
    if path.name.startswith("."):
        return True
    try:
        stat_result = os.lstat(path)
    except OSError:
        return True
    return bool(getattr(stat_result, "st_file_attributes", 0) & _HIDDEN_ATTRIBUTE)


def _is_reparse_point(
    path: Path,
) -> bool:
    try:
        if path.is_symlink():
            return True
        stat_result = os.lstat(path)
    except OSError:
        return True
    return bool(getattr(stat_result, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE)


def _relative_sort_key(
    path: Path,
    root: Path,
) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix().lower()
    except ValueError:
        return path.as_posix().lower()


def _path_key(
    path: Path,
) -> str:
    return str(path).replace("\\", "/").lower()


def _normalize_relative_path(
    value: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentManifestDiscoveryError("relative_path must be a string.")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or ".." in normalized.split("/"):
        raise AgentManifestDiscoveryError("relative_path is unsafe.")
    return normalized


def _display_path(
    path: Path,
) -> str:
    return str(path.resolve())


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
    value: AgentDiscoveryStatus | str,
) -> AgentDiscoveryStatus:
    if isinstance(value, AgentDiscoveryStatus):
        return value
    if isinstance(value, str):
        return AgentDiscoveryStatus(value)
    raise InvalidAgentDiscoveryRequestError("status must be AgentDiscoveryStatus.")

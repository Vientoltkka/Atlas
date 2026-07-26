"""Safe discovery of declarative skill manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType


MAX_SKILL_DISCOVERY_DIRECTORIES = 128
MAX_SKILL_DISCOVERY_FILES = 512
MAX_SKILL_DISCOVERY_BYTES = 64_000


class SkillDiscoveryStatus(str, Enum):
    """Discovery status."""

    COMPLETED = "COMPLETED"
    NO_MANIFESTS_FOUND = "NO_MANIFESTS_FOUND"
    INVALID_REQUEST = "INVALID_REQUEST"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    FAILED = "FAILED"


class SkillDiscoveryError(RuntimeError):
    """Base discovery error."""


class InvalidSkillDiscoveryRequestError(SkillDiscoveryError):
    """Raised for malformed discovery requests."""


@dataclass(frozen=True, slots=True)
class SkillDiscoveryRequest:
    """Explicit roots for skill discovery."""

    root_directories: tuple[str | Path, ...]
    recursive: bool = False
    max_directories: int = MAX_SKILL_DISCOVERY_DIRECTORIES
    max_files: int = MAX_SKILL_DISCOVERY_FILES
    max_manifest_bytes: int = MAX_SKILL_DISCOVERY_BYTES

    def __post_init__(self) -> None:
        if not self.root_directories:
            raise InvalidSkillDiscoveryRequestError("root_directories cannot be empty.")
        roots = tuple(_root(root) for root in self.root_directories)
        object.__setattr__(self, "root_directories", tuple(sorted(dict.fromkeys(roots), key=lambda path: str(path).lower())))
        if not isinstance(self.recursive, bool):
            raise InvalidSkillDiscoveryRequestError("recursive must be a bool.")
        for name in ("max_directories", "max_files", "max_manifest_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise InvalidSkillDiscoveryRequestError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class SkillDiscoveredManifest:
    """One discovered manifest file."""

    path: Path
    root: Path
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _root(self.path))
        object.__setattr__(self, "root", _root(self.root))
        if not isinstance(self.content, str):
            raise InvalidSkillDiscoveryRequestError("content must be a string.")


@dataclass(frozen=True, slots=True)
class SkillDiscoveryResult:
    """Structured discovery result."""

    status: SkillDiscoveryStatus
    manifests: tuple[SkillDiscoveredManifest, ...] = ()
    errors: tuple[str, ...] = ()
    events: tuple[MappingProxyType, ...] = ()
    metrics: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    @property
    def completed(self) -> bool:
        return self.status in (SkillDiscoveryStatus.COMPLETED, SkillDiscoveryStatus.NO_MANIFESTS_FOUND)


class SkillDiscovery:
    """Discover JSON skill manifests without importing code."""

    def discover(self, request: SkillDiscoveryRequest) -> SkillDiscoveryResult:
        try:
            if not isinstance(request, SkillDiscoveryRequest):
                raise InvalidSkillDiscoveryRequestError("request must be SkillDiscoveryRequest.")
            manifests: list[SkillDiscoveredManifest] = []
            directories_seen = 0
            for root in request.root_directories:
                if not root.exists() or not root.is_dir():
                    raise InvalidSkillDiscoveryRequestError("root directory does not exist.")
                if _unsafe_path(root):
                    raise InvalidSkillDiscoveryRequestError("root directory is unsafe.")
                directories = (path for path in root.rglob("*")) if request.recursive else (path for path in root.iterdir())
                for path in sorted(directories, key=lambda item: str(item).lower()):
                    if _hidden(path) or _unsafe_path(path):
                        continue
                    if path.is_dir():
                        directories_seen += 1
                        if directories_seen > request.max_directories:
                            return _result(SkillDiscoveryStatus.LIMIT_EXCEEDED, errors=("directory limit exceeded.",))
                        continue
                    if path.suffix.lower() != ".json":
                        continue
                    resolved = path.resolve()
                    if not _inside(root, resolved):
                        raise InvalidSkillDiscoveryRequestError("manifest escaped the discovery root.")
                    if len(manifests) >= request.max_files:
                        return _result(SkillDiscoveryStatus.LIMIT_EXCEEDED, errors=("file limit exceeded.",))
                    if path.stat().st_size > request.max_manifest_bytes:
                        raise InvalidSkillDiscoveryRequestError("manifest is too large.")
                    manifests.append(SkillDiscoveredManifest(resolved, root, path.read_text(encoding="utf-8")))
            if not manifests:
                return _result(SkillDiscoveryStatus.NO_MANIFESTS_FOUND, manifests=())
            return _result(SkillDiscoveryStatus.COMPLETED, manifests=tuple(manifests))
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            return _result(SkillDiscoveryStatus.FAILED, errors=(str(error),))


def _result(
    status: SkillDiscoveryStatus,
    *,
    manifests: tuple[SkillDiscoveredManifest, ...] = (),
    errors: tuple[str, ...] = (),
) -> SkillDiscoveryResult:
    event_name = "skill_discovery_failed" if status is SkillDiscoveryStatus.FAILED else "skill_discovery_started"
    events = [MappingProxyType({"name": event_name, "status": status.value})]
    for manifest in manifests:
        events.append(MappingProxyType({"name": "skill_discovered", "status": "FOUND", "path": manifest.path.name}))
    return SkillDiscoveryResult(
        status,
        manifests,
        tuple(str(error)[:240] for error in errors),
        tuple(events),
        MappingProxyType({"skills_discovered": len(manifests)}),
    )


def _root(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _unsafe_path(path: Path) -> bool:
    return path.is_symlink()

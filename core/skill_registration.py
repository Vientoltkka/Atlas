"""Atomic registration for declarative Atlas skills."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType

from core.skill_discovery import SkillDiscovery, SkillDiscoveryRequest, SkillDiscoveryStatus
from core.skill_manifest import SkillManifestLoader
from core.skill_registry import SkillAlreadyRegisteredError, SkillDefinition, SkillRegistry


class SkillDuplicatePolicy(str, Enum):
    """Policy for duplicate skill ids."""

    REJECT = "REJECT"
    KEEP_EXISTING = "KEEP_EXISTING"
    REPLACE = "REPLACE"


class SkillRegistrationStatus(str, Enum):
    """Skill registration status."""

    COMPLETED = "COMPLETED"
    DRY_RUN_COMPLETED = "DRY_RUN_COMPLETED"
    NO_MANIFESTS_FOUND = "NO_MANIFESTS_FOUND"
    INVALID_REQUEST = "INVALID_REQUEST"
    DUPLICATE_SKILL = "DUPLICATE_SKILL"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"


class SkillRegistrationError(RuntimeError):
    """Base registration error."""


class InvalidSkillRegistrationRequestError(SkillRegistrationError):
    """Raised for malformed registration requests."""


@dataclass(frozen=True, slots=True)
class SkillRegistrationPolicy:
    """Registration policy."""

    duplicate_policy: SkillDuplicatePolicy | str = SkillDuplicatePolicy.REJECT
    dry_run: bool = False
    max_skills: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "duplicate_policy", _duplicate_policy(self.duplicate_policy))
        if not isinstance(self.dry_run, bool):
            raise InvalidSkillRegistrationRequestError("dry_run must be a bool.")
        if isinstance(self.max_skills, bool) or not isinstance(self.max_skills, int) or self.max_skills <= 0:
            raise InvalidSkillRegistrationRequestError("max_skills must be a positive integer.")


@dataclass(frozen=True, slots=True)
class SkillRegistrationRequest:
    """Request for discovering and registering skills atomically."""

    root_directories: tuple[str, ...]
    recursive: bool = False
    policy: SkillRegistrationPolicy = field(default_factory=SkillRegistrationPolicy)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.root_directories:
            raise InvalidSkillRegistrationRequestError("root_directories cannot be empty.")
        object.__setattr__(self, "root_directories", tuple(sorted(dict.fromkeys(str(root) for root in self.root_directories))))
        if not isinstance(self.policy, SkillRegistrationPolicy):
            raise InvalidSkillRegistrationRequestError("policy must be SkillRegistrationPolicy.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SkillRegistrationResult:
    """Structured registration result."""

    status: SkillRegistrationStatus
    registered_skill_ids: tuple[str, ...] = ()
    skipped_skill_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    request_signature: str = ""
    events: tuple[Mapping[str, object], ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.status in (SkillRegistrationStatus.COMPLETED, SkillRegistrationStatus.DRY_RUN_COMPLETED)


class SkillRegistrationService:
    """Discover, load, and atomically register skills."""

    def __init__(self, discovery: SkillDiscovery, loader: SkillManifestLoader, registry: SkillRegistry) -> None:
        self._discovery = discovery
        self._loader = loader
        self._registry = registry

    def register(self, request: SkillRegistrationRequest) -> SkillRegistrationResult:
        try:
            signature = skill_registration_request_signature(request)
            discovered = self._discovery.discover(SkillDiscoveryRequest(request.root_directories, recursive=request.recursive))
            if discovered.status is SkillDiscoveryStatus.NO_MANIFESTS_FOUND:
                return _result(SkillRegistrationStatus.NO_MANIFESTS_FOUND, signature)
            if discovered.status is not SkillDiscoveryStatus.COMPLETED:
                return _result(SkillRegistrationStatus.REGISTRATION_FAILED, signature, errors=discovered.errors)
            if len(discovered.manifests) > request.policy.max_skills:
                return _result(SkillRegistrationStatus.REGISTRATION_FAILED, signature, errors=("skill limit exceeded.",))
            definitions: list[SkillDefinition] = []
            for manifest in discovered.manifests:
                loaded = self._loader.load(manifest.content)
                if not loaded.valid or loaded.definition is None:
                    return _result(SkillRegistrationStatus.REGISTRATION_FAILED, signature, errors=loaded.errors)
                definitions.append(loaded.definition)
            ids = [definition.skill_id for definition in definitions]
            if len(set(ids)) != len(ids):
                return _result(SkillRegistrationStatus.DUPLICATE_SKILL, signature, errors=("duplicate skill in manifests.",))
            snapshot = self._registry.list_skills(enabled_only=False)
            registered: list[str] = []
            skipped: list[str] = []
            if request.policy.dry_run:
                return _result(SkillRegistrationStatus.DRY_RUN_COMPLETED, signature, registered=tuple(ids))
            try:
                for definition in definitions:
                    if self._registry.contains(definition.skill_id):
                        if request.policy.duplicate_policy is SkillDuplicatePolicy.REJECT:
                            raise SkillAlreadyRegisteredError(definition.skill_id)
                        if request.policy.duplicate_policy is SkillDuplicatePolicy.KEEP_EXISTING:
                            skipped.append(definition.skill_id)
                            continue
                    self._registry.register(definition, replace=request.policy.duplicate_policy is SkillDuplicatePolicy.REPLACE)
                    registered.append(definition.skill_id)
            except Exception as error:
                self._registry.clear()
                for definition in snapshot:
                    self._registry.register(definition)
                return _result(SkillRegistrationStatus.REGISTRATION_FAILED, signature, errors=(str(error),))
            return _result(SkillRegistrationStatus.COMPLETED, signature, registered=tuple(registered), skipped=tuple(skipped))
        except Exception as error:
            return _result(SkillRegistrationStatus.INVALID_REQUEST, "", errors=(str(error),))


def skill_registration_request_signature(request: SkillRegistrationRequest) -> str:
    payload = {
        "root_directories": request.root_directories,
        "recursive": request.recursive,
        "duplicate_policy": request.policy.duplicate_policy.value,
        "dry_run": request.policy.dry_run,
        "max_skills": request.policy.max_skills,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _result(
    status: SkillRegistrationStatus,
    signature: str,
    *,
    registered: tuple[str, ...] = (),
    skipped: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
) -> SkillRegistrationResult:
    return SkillRegistrationResult(
        status,
        registered_skill_ids=registered,
        skipped_skill_ids=skipped,
        errors=tuple(str(error)[:240] for error in errors),
        request_signature=signature,
        events=(
            {"name": "skill_registration_started", "status": "STARTED"},
            {
                "name": "skill_registered" if status in (SkillRegistrationStatus.COMPLETED, SkillRegistrationStatus.DRY_RUN_COMPLETED) else "skill_registration_failed",
                "status": status.value,
                "count": len(registered),
            },
        ),
        metrics={
            "skills_registered": len(registered),
            "skill_registration_failures": 0 if status in (SkillRegistrationStatus.COMPLETED, SkillRegistrationStatus.DRY_RUN_COMPLETED) else 1,
        },
    )


def _duplicate_policy(value: SkillDuplicatePolicy | str) -> SkillDuplicatePolicy:
    if isinstance(value, SkillDuplicatePolicy):
        return value
    if isinstance(value, str):
        return SkillDuplicatePolicy(value)
    raise InvalidSkillRegistrationRequestError("duplicate_policy is invalid.")

"""Controlled registration for executable specialized-agent handlers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any

from core.agent_context import AgentContext
from core.agent_executor import (
    AgentHandler,
    AgentHandlerAlreadyRegisteredError,
    AgentHandlerRegistry,
    AgentHandlerRegistryError,
    InvalidAgentHandlerError,
)
from core.agent_registry import AgentDefinition, AgentRegistry, InvalidAgentDefinitionError, validate_agent_id


MAX_HANDLER_REGISTRATION_ITEMS = 128
MAX_HANDLER_REGISTRATION_METADATA_ITEMS = 16
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
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


class AgentHandlerRegistrationStatus(str, Enum):
    """Structured statuses for controlled handler registration."""

    COMPLETED = "COMPLETED"
    DRY_RUN_COMPLETED = "DRY_RUN_COMPLETED"
    INVALID_REQUEST = "INVALID_REQUEST"
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    HANDLER_INCOMPATIBLE = "HANDLER_INCOMPATIBLE"
    DUPLICATE_HANDLER = "DUPLICATE_HANDLER"
    HANDLER_CONFLICT = "HANDLER_CONFLICT"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentHandlerDuplicatePolicy(str, Enum):
    """Deterministic policy for already registered handlers."""

    REJECT = "REJECT"
    KEEP_EXISTING = "KEEP_EXISTING"
    REPLACE = "REPLACE"


class AgentHandlerRegistrationError(RuntimeError):
    """Base error for controlled handler registration."""


class InvalidAgentHandlerRegistrationRequestError(AgentHandlerRegistrationError):
    """Raised when a handler registration request is malformed."""


@dataclass(frozen=True, slots=True)
class _CallableAgentHandler:
    agent_id: str
    callback: Callable[[AgentContext], Mapping[str, object]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        if not callable(self.callback):
            raise InvalidAgentHandlerRegistrationRequestError("callback must be callable.")

    def handle(
        self,
        context: AgentContext,
    ) -> Mapping[str, object]:
        return self.callback(context)


@dataclass(frozen=True, slots=True)
class AgentHandlerRegistrationPolicy:
    """Immutable policy for registering handlers into AgentHandlerRegistry."""

    duplicate_handler_policy: AgentHandlerDuplicatePolicy | str = AgentHandlerDuplicatePolicy.REJECT
    dry_run: bool = False
    max_handlers: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "duplicate_handler_policy", _duplicate_policy(self.duplicate_handler_policy))
        if type(self.dry_run) is not bool:
            raise InvalidAgentHandlerRegistrationRequestError("dry_run must be a bool.")
        object.__setattr__(self, "max_handlers", _positive_int(self.max_handlers, "max_handlers"))


@dataclass(frozen=True, slots=True)
class AgentHandlerRegistrationItem:
    """Immutable declaration for one handler registration candidate."""

    handler_id: str
    agent_id: str
    handler: AgentHandler | Callable[[AgentContext], Mapping[str, object]]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "handler_id", _identifier(self.handler_id, "handler_id"))
        object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class AgentHandlerRegistrationRequest:
    """Explicit immutable request for registering executable agent handlers."""

    handlers: Iterable[AgentHandlerRegistrationItem]
    policy: AgentHandlerRegistrationPolicy = field(default_factory=AgentHandlerRegistrationPolicy)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.handlers, (str, bytes)) or not isinstance(self.handlers, Iterable):
            raise InvalidAgentHandlerRegistrationRequestError("handlers must be an iterable.")
        handlers = tuple(self.handlers)
        if not handlers:
            raise InvalidAgentHandlerRegistrationRequestError("handlers cannot be empty.")
        if len(handlers) > MAX_HANDLER_REGISTRATION_ITEMS:
            raise InvalidAgentHandlerRegistrationRequestError("handlers exceeds the item limit.")
        if not all(isinstance(item, AgentHandlerRegistrationItem) for item in handlers):
            raise InvalidAgentHandlerRegistrationRequestError("handlers must contain AgentHandlerRegistrationItem values.")
        object.__setattr__(self, "handlers", tuple(sorted(handlers, key=lambda item: (item.agent_id, item.handler_id))))
        if not isinstance(self.policy, AgentHandlerRegistrationPolicy):
            raise InvalidAgentHandlerRegistrationRequestError("policy must be AgentHandlerRegistrationPolicy.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class AgentHandlerRegistrationEntry:
    """Immutable summary for one handler registration decision."""

    agent_id: str
    handler_id: str
    action: str
    skipped: bool = False
    replaced: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        object.__setattr__(self, "handler_id", _identifier(self.handler_id, "handler_id"))
        if self.action not in ("registered", "would_register", "skipped", "replaced", "would_replace"):
            raise InvalidAgentHandlerRegistrationRequestError("action is invalid.")
        if type(self.skipped) is not bool:
            raise InvalidAgentHandlerRegistrationRequestError("skipped must be a bool.")
        if type(self.replaced) is not bool:
            raise InvalidAgentHandlerRegistrationRequestError("replaced must be a bool.")
        if self.reason is not None:
            object.__setattr__(self, "reason", _safe_message(self.reason))


@dataclass(frozen=True, slots=True)
class AgentHandlerRegistrationResult:
    """Structured immutable result for controlled handler registration."""

    status: AgentHandlerRegistrationStatus
    requested_agent_ids: tuple[str, ...] = ()
    validated_agent_ids: tuple[str, ...] = ()
    registered_agent_ids: tuple[str, ...] = ()
    skipped_agent_ids: tuple[str, ...] = ()
    replaced_agent_ids: tuple[str, ...] = ()
    rejected_agent_ids: tuple[str, ...] = ()
    entries: tuple[AgentHandlerRegistrationEntry, ...] = ()
    errors: tuple[str, ...] = ()
    request_signature: str = ""
    handlers_processed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        for field_name in (
            "requested_agent_ids",
            "validated_agent_ids",
            "registered_agent_ids",
            "skipped_agent_ids",
            "replaced_agent_ids",
            "rejected_agent_ids",
        ):
            object.__setattr__(self, field_name, _agent_id_tuple(getattr(self, field_name), field_name))
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "errors", tuple(_safe_message(error) for error in self.errors))
        if isinstance(self.handlers_processed, bool) or not isinstance(self.handlers_processed, int):
            raise InvalidAgentHandlerRegistrationRequestError("handlers_processed must be an integer.")


class AgentHandlerRegistrationService:
    """Register executable handlers against existing agent definitions atomically."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        agent_handler_registry: AgentHandlerRegistry,
    ) -> None:
        if not isinstance(agent_registry, AgentRegistry):
            raise InvalidAgentHandlerRegistrationRequestError("agent_registry must be AgentRegistry.")
        if not isinstance(agent_handler_registry, AgentHandlerRegistry):
            raise InvalidAgentHandlerRegistrationRequestError("agent_handler_registry must be AgentHandlerRegistry.")
        self._agent_registry = agent_registry
        self._agent_handler_registry = agent_handler_registry

    def register(
        self,
        request: AgentHandlerRegistrationRequest,
    ) -> AgentHandlerRegistrationResult:
        """Register handlers without importing modules or executing handlers."""

        try:
            if not isinstance(request, AgentHandlerRegistrationRequest):
                raise InvalidAgentHandlerRegistrationRequestError("request must be AgentHandlerRegistrationRequest.")
            return self._register(request)
        except InvalidAgentHandlerRegistrationRequestError as error:
            return _result(AgentHandlerRegistrationStatus.INVALID_REQUEST, request_signature="", errors=(str(error),))
        except (RuntimeError, ValueError, TypeError) as error:
            signature = (
                agent_handler_registration_request_signature(request)
                if isinstance(request, AgentHandlerRegistrationRequest)
                else ""
            )
            return _result(AgentHandlerRegistrationStatus.INTERNAL_ERROR, request_signature=signature, errors=(str(error),))

    def _register(
        self,
        request: AgentHandlerRegistrationRequest,
    ) -> AgentHandlerRegistrationResult:
        request_signature = agent_handler_registration_request_signature(request)
        handlers = tuple(request.handlers)
        requested_ids = tuple(item.agent_id for item in handlers)
        if len(handlers) > request.policy.max_handlers:
            return _result(
                AgentHandlerRegistrationStatus.LIMIT_EXCEEDED,
                request_signature=request_signature,
                requested_agent_ids=requested_ids,
                errors=("handler registration limit exceeded.",),
                handlers_processed=len(handlers),
            )

        prepared: list[tuple[AgentHandlerRegistrationItem, AgentHandler, AgentDefinition]] = []
        seen_agent_ids: set[str] = set()
        seen_handler_ids: set[str] = set()
        for item in handlers:
            if item.agent_id in seen_agent_ids:
                return _result(
                    AgentHandlerRegistrationStatus.DUPLICATE_HANDLER,
                    request_signature=request_signature,
                    requested_agent_ids=requested_ids,
                    rejected_agent_ids=(item.agent_id,),
                    errors=(f"duplicate handler agent_id in batch: {item.agent_id}",),
                    handlers_processed=len(handlers),
                )
            if item.handler_id in seen_handler_ids:
                return _result(
                    AgentHandlerRegistrationStatus.DUPLICATE_HANDLER,
                    request_signature=request_signature,
                    requested_agent_ids=requested_ids,
                    rejected_agent_ids=(item.agent_id,),
                    errors=(f"duplicate handler_id in batch: {item.handler_id}",),
                    handlers_processed=len(handlers),
                )
            seen_agent_ids.add(item.agent_id)
            seen_handler_ids.add(item.handler_id)
            if not self._agent_registry.contains(item.agent_id):
                return _result(
                    AgentHandlerRegistrationStatus.AGENT_NOT_FOUND,
                    request_signature=request_signature,
                    requested_agent_ids=requested_ids,
                    rejected_agent_ids=(item.agent_id,),
                    errors=(f"agent_id is not registered: {item.agent_id}",),
                    handlers_processed=len(handlers),
                )
            definition = self._agent_registry.get(item.agent_id)
            expected_handler_id = definition.metadata.get("handler_id")
            if expected_handler_id != item.handler_id:
                return _result(
                    AgentHandlerRegistrationStatus.HANDLER_INCOMPATIBLE,
                    request_signature=request_signature,
                    requested_agent_ids=requested_ids,
                    rejected_agent_ids=(item.agent_id,),
                    errors=(f"handler_id is incompatible with agent definition: {item.agent_id}",),
                    handlers_processed=len(handlers),
                )
            try:
                handler = _handler_for_item(item)
            except (InvalidAgentHandlerError, InvalidAgentHandlerRegistrationRequestError) as error:
                return _result(
                    AgentHandlerRegistrationStatus.HANDLER_INCOMPATIBLE,
                    request_signature=request_signature,
                    requested_agent_ids=requested_ids,
                    rejected_agent_ids=(item.agent_id,),
                    errors=(str(error),),
                    handlers_processed=len(handlers),
                )
            if handler.agent_id != item.agent_id:
                return _result(
                    AgentHandlerRegistrationStatus.HANDLER_INCOMPATIBLE,
                    request_signature=request_signature,
                    requested_agent_ids=requested_ids,
                    rejected_agent_ids=(item.agent_id,),
                    errors=(f"handler agent_id is incompatible with request: {item.agent_id}",),
                    handlers_processed=len(handlers),
                )
            prepared.append((item, handler, definition))

        preflight = self._preflight(request, prepared)
        if preflight.status is not AgentHandlerRegistrationStatus.COMPLETED:
            return _result(
                preflight.status,
                request_signature=request_signature,
                requested_agent_ids=requested_ids,
                validated_agent_ids=tuple(item.agent_id for item, _, _ in prepared),
                skipped_agent_ids=preflight.skipped_agent_ids,
                replaced_agent_ids=preflight.replaced_agent_ids,
                rejected_agent_ids=preflight.rejected_agent_ids,
                entries=preflight.entries,
                errors=preflight.errors,
                handlers_processed=len(handlers),
            )
        if request.policy.dry_run:
            return _result(
                AgentHandlerRegistrationStatus.DRY_RUN_COMPLETED,
                request_signature=request_signature,
                requested_agent_ids=requested_ids,
                validated_agent_ids=tuple(item.agent_id for item, _, _ in prepared),
                skipped_agent_ids=preflight.skipped_agent_ids,
                replaced_agent_ids=preflight.replaced_agent_ids,
                entries=tuple(
                    AgentHandlerRegistrationEntry(
                        agent_id=entry.agent_id,
                        handler_id=entry.handler_id,
                        action=(
                            "would_replace"
                            if entry.action == "replaced"
                            else "skipped"
                            if entry.action == "skipped"
                            else "would_register"
                        ),
                        skipped=entry.skipped,
                        replaced=entry.replaced,
                        reason=entry.reason,
                    )
                    for entry in preflight.entries
                ),
                errors=preflight.errors,
                handlers_processed=len(handlers),
            )

        registered: list[str] = []
        replaced: list[str] = []
        entries: list[AgentHandlerRegistrationEntry] = []
        rollback_registered: list[str] = []
        rollback_replaced: list[tuple[str, AgentHandler]] = []
        try:
            for item, handler, _ in prepared:
                if item.agent_id in preflight.skipped_agent_ids:
                    entries.append(
                        AgentHandlerRegistrationEntry(
                            agent_id=item.agent_id,
                            handler_id=item.handler_id,
                            action="skipped",
                            skipped=True,
                            reason="existing handler kept by policy.",
                        )
                    )
                    continue
                replace = item.agent_id in preflight.replaced_agent_ids
                if replace:
                    rollback_replaced.append((item.agent_id, self._agent_handler_registry.get(item.agent_id)))
                self._agent_handler_registry.register(handler, replace=replace)
                if replace:
                    replaced.append(item.agent_id)
                    entries.append(
                        AgentHandlerRegistrationEntry(
                            agent_id=item.agent_id,
                            handler_id=item.handler_id,
                            action="replaced",
                            replaced=True,
                        )
                    )
                else:
                    rollback_registered.append(item.agent_id)
                    registered.append(item.agent_id)
                    entries.append(
                        AgentHandlerRegistrationEntry(
                            agent_id=item.agent_id,
                            handler_id=item.handler_id,
                            action="registered",
                        )
                    )
        except (AgentHandlerRegistryError, InvalidAgentDefinitionError) as error:
            self._rollback(rollback_registered, rollback_replaced)
            return _result(
                AgentHandlerRegistrationStatus.REGISTRATION_FAILED,
                request_signature=request_signature,
                requested_agent_ids=requested_ids,
                validated_agent_ids=tuple(item.agent_id for item, _, _ in prepared),
                errors=(str(error),),
                handlers_processed=len(handlers),
            )

        return _result(
            AgentHandlerRegistrationStatus.COMPLETED,
            request_signature=request_signature,
            requested_agent_ids=requested_ids,
            validated_agent_ids=tuple(item.agent_id for item, _, _ in prepared),
            registered_agent_ids=tuple(registered),
            skipped_agent_ids=preflight.skipped_agent_ids,
            replaced_agent_ids=tuple(replaced),
            entries=tuple(entries),
            errors=preflight.errors,
            handlers_processed=len(handlers),
        )

    def _preflight(
        self,
        request: AgentHandlerRegistrationRequest,
        prepared: tuple[tuple[AgentHandlerRegistrationItem, AgentHandler, AgentDefinition], ...] | list[
            tuple[AgentHandlerRegistrationItem, AgentHandler, AgentDefinition]
        ],
    ) -> AgentHandlerRegistrationResult:
        skipped: list[str] = []
        replaced: list[str] = []
        rejected: list[str] = []
        entries: list[AgentHandlerRegistrationEntry] = []
        errors: list[str] = []
        policy = request.policy.duplicate_handler_policy
        for item, _, _ in prepared:
            if not self._agent_handler_registry.contains(item.agent_id):
                entries.append(
                    AgentHandlerRegistrationEntry(
                        agent_id=item.agent_id,
                        handler_id=item.handler_id,
                        action="registered",
                    )
                )
                continue
            if policy is AgentHandlerDuplicatePolicy.REJECT:
                rejected.append(item.agent_id)
                errors.append(f"handler already registered: {item.agent_id}")
                return _result(
                    AgentHandlerRegistrationStatus.DUPLICATE_HANDLER,
                    request_signature="",
                    rejected_agent_ids=tuple(rejected),
                    errors=tuple(errors),
                    entries=tuple(entries),
                )
            if policy is AgentHandlerDuplicatePolicy.KEEP_EXISTING:
                skipped.append(item.agent_id)
                entries.append(
                    AgentHandlerRegistrationEntry(
                        agent_id=item.agent_id,
                        handler_id=item.handler_id,
                        action="skipped",
                        skipped=True,
                        reason="existing handler kept by policy.",
                    )
                )
                continue
            if policy is AgentHandlerDuplicatePolicy.REPLACE:
                replaced.append(item.agent_id)
                entries.append(
                    AgentHandlerRegistrationEntry(
                        agent_id=item.agent_id,
                        handler_id=item.handler_id,
                        action="replaced",
                        replaced=True,
                        reason="existing handler replaced by explicit policy.",
                    )
                )
                continue
        return _result(
            AgentHandlerRegistrationStatus.COMPLETED,
            request_signature="",
            skipped_agent_ids=tuple(skipped),
            replaced_agent_ids=tuple(replaced),
            rejected_agent_ids=tuple(rejected),
            errors=tuple(errors),
            entries=tuple(entries),
        )

    def _rollback(
        self,
        registered: Iterable[str],
        replaced: Iterable[tuple[str, AgentHandler]],
    ) -> None:
        for agent_id in registered:
            self._agent_handler_registry.unregister(agent_id)
        for agent_id, handler in replaced:
            self._agent_handler_registry.register(handler, replace=True)


def agent_handler_registration_request_signature(
    request: AgentHandlerRegistrationRequest,
) -> str:
    """Return a deterministic SHA-256 signature for a handler registration request."""

    if not isinstance(request, AgentHandlerRegistrationRequest):
        raise InvalidAgentHandlerRegistrationRequestError("request must be AgentHandlerRegistrationRequest.")
    payload = {
        "handlers": tuple(
            {
                "agent_id": item.agent_id,
                "handler_id": item.handler_id,
                "metadata": item.metadata,
            }
            for item in request.handlers
        ),
        "policy": {
            "duplicate_handler_policy": request.policy.duplicate_handler_policy.value,
            "dry_run": request.policy.dry_run,
            "max_handlers": request.policy.max_handlers,
        },
        "metadata": request.metadata,
    }
    encoded = json.dumps(_jsonable(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _handler_for_item(
    item: AgentHandlerRegistrationItem,
) -> AgentHandler:
    handler = item.handler
    if isinstance(handler, AgentHandler):
        return handler
    if callable(handler):
        return _CallableAgentHandler(item.agent_id, handler)
    raise InvalidAgentHandlerRegistrationRequestError("handler must implement AgentHandler or be callable.")


def _result(
    status: AgentHandlerRegistrationStatus,
    *,
    request_signature: str,
    requested_agent_ids: tuple[str, ...] = (),
    validated_agent_ids: tuple[str, ...] = (),
    registered_agent_ids: tuple[str, ...] = (),
    skipped_agent_ids: tuple[str, ...] = (),
    replaced_agent_ids: tuple[str, ...] = (),
    rejected_agent_ids: tuple[str, ...] = (),
    entries: tuple[AgentHandlerRegistrationEntry, ...] = (),
    errors: tuple[str, ...] = (),
    handlers_processed: int = 0,
) -> AgentHandlerRegistrationResult:
    return AgentHandlerRegistrationResult(
        status=status,
        requested_agent_ids=requested_agent_ids,
        validated_agent_ids=validated_agent_ids,
        registered_agent_ids=registered_agent_ids,
        skipped_agent_ids=skipped_agent_ids,
        replaced_agent_ids=replaced_agent_ids,
        rejected_agent_ids=rejected_agent_ids,
        entries=entries,
        errors=errors,
        request_signature=request_signature,
        handlers_processed=handlers_processed,
    )


def _duplicate_policy(
    value: AgentHandlerDuplicatePolicy | str,
) -> AgentHandlerDuplicatePolicy:
    if isinstance(value, AgentHandlerDuplicatePolicy):
        return value
    if isinstance(value, str):
        try:
            return AgentHandlerDuplicatePolicy(value.upper())
        except ValueError as error:
            raise InvalidAgentHandlerRegistrationRequestError("duplicate_handler_policy is invalid.") from error
    raise InvalidAgentHandlerRegistrationRequestError("duplicate_handler_policy must be a handler registration policy.")


def _positive_int(
    value: int,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidAgentHandlerRegistrationRequestError(f"{field_name} must be an integer.")
    if value <= 0 or value > MAX_HANDLER_REGISTRATION_ITEMS:
        raise InvalidAgentHandlerRegistrationRequestError(
            f"{field_name} must be between 1 and {MAX_HANDLER_REGISTRATION_ITEMS}."
        )
    return value


def _safe_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidAgentHandlerRegistrationRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_HANDLER_REGISTRATION_METADATA_ITEMS:
        raise InvalidAgentHandlerRegistrationRequestError("metadata has too many items.")
    safe: dict[str, object] = {}
    for raw_key in sorted(metadata):
        if not isinstance(raw_key, str) or not raw_key.strip() or raw_key.strip() != raw_key:
            raise InvalidAgentHandlerRegistrationRequestError("metadata keys must be safe strings.")
        if _is_sensitive_key(raw_key):
            raise InvalidAgentHandlerRegistrationRequestError("metadata contains a forbidden sensitive key.")
        safe[raw_key] = _safe_metadata_value(metadata[raw_key])
    return safe


def _safe_metadata_value(
    value: object,
) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentHandlerRegistrationRequestError("metadata floats must be finite.")
        return value
    raise InvalidAgentHandlerRegistrationRequestError("metadata values must be primitive safe values.")


def _jsonable(
    value: object,
) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if value is not None and type(value) not in (bool, int, float, str):
        raise InvalidAgentHandlerRegistrationRequestError("unsupported handler registration signature value.")
    return value


def _agent_id_tuple(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentHandlerRegistrationRequestError(f"{field_name} must be an iterable of agent ids.")
    return tuple(validate_agent_id(value) for value in values)


def _identifier(
    value: object,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise InvalidAgentHandlerRegistrationRequestError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise InvalidAgentHandlerRegistrationRequestError(f"{field_name} cannot be empty.")
    if normalized != value:
        raise InvalidAgentHandlerRegistrationRequestError(f"{field_name} cannot contain surrounding whitespace.")
    if not _IDENTIFIER_PATTERN.fullmatch(normalized):
        raise InvalidAgentHandlerRegistrationRequestError(f"{field_name} contains unsupported characters.")
    return normalized


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
        message = re.sub(re.escape(key), "[redacted]", message, flags=re.IGNORECASE)
    return message[:240]


def _status(
    value: AgentHandlerRegistrationStatus | str,
) -> AgentHandlerRegistrationStatus:
    if isinstance(value, AgentHandlerRegistrationStatus):
        return value
    if isinstance(value, str):
        return AgentHandlerRegistrationStatus(value)
    raise InvalidAgentHandlerRegistrationRequestError("status must be AgentHandlerRegistrationStatus.")

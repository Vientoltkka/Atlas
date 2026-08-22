"""Temporary, bounded working memory for one agent execution."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from types import MappingProxyType

from core.agent_registry import AgentMemoryPolicy, validate_agent_id


DEFAULT_AGENT_WORKING_MEMORY_TTL = timedelta(minutes=15)


class AgentWorkingMemoryError(RuntimeError):
    """Base error for agent working memory operations."""


class InvalidAgentWorkingMemoryError(AgentWorkingMemoryError):
    """Raised when a working-memory operation is structurally invalid."""


@dataclass(frozen=True, slots=True)
class _MemoryScope:
    values: Mapping[str, object]
    expires_at: datetime


class AgentWorkingMemory:
    """In-memory state isolated by agent and execution identifier.

    The store deliberately has no persistence or dependency on conversation
    memory. Callers receive defensive immutable snapshots only.
    """

    def __init__(
        self,
        *,
        ttl: timedelta = DEFAULT_AGENT_WORKING_MEMORY_TTL,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
            raise InvalidAgentWorkingMemoryError("ttl must be a positive timedelta.")
        self._ttl = ttl
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._scopes: OrderedDict[tuple[str, str], _MemoryScope] = OrderedDict()
        self._lock = RLock()

    def read(
        self,
        agent_id: str,
        execution_id: str | None,
        policy: AgentMemoryPolicy,
    ) -> Mapping[str, object]:
        """Return the current isolated snapshot when the policy allows reads."""

        key = self._scope_key(agent_id, execution_id, policy)
        if key is None or not policy.can_read_memory:
            return MappingProxyType({})
        with self._lock:
            self._cleanup_locked()
            scope = self._scopes.get(key)
            return MappingProxyType(dict(scope.values)) if scope is not None else MappingProxyType({})

    def write(
        self,
        agent_id: str,
        execution_id: str | None,
        values: Mapping[str, object],
        policy: AgentMemoryPolicy,
    ) -> Mapping[str, object]:
        """Merge one sanitized result into a scope when writes are permitted."""

        key = self._scope_key(agent_id, execution_id, policy)
        if key is None or not policy.can_write_memory:
            return MappingProxyType({})
        if not isinstance(values, Mapping):
            raise InvalidAgentWorkingMemoryError("values must be a mapping.")
        with self._lock:
            self._cleanup_locked()
            current = self._scopes.get(key)
            merged = OrderedDict(() if current is None else current.values.items())
            for name in sorted(values):
                if not isinstance(name, str) or not name:
                    raise InvalidAgentWorkingMemoryError("memory keys must be non-empty strings.")
                merged[name] = values[name]
            while len(merged) > policy.max_memory_items:
                merged.popitem(last=False)
            if not merged:
                self._scopes.pop(key, None)
                return MappingProxyType({})
            snapshot = MappingProxyType(dict(merged))
            self._scopes[key] = _MemoryScope(
                values=snapshot,
                expires_at=self._now() + self._ttl,
            )
            return MappingProxyType(dict(snapshot))

    def cleanup(self) -> int:
        """Remove expired scopes and return their count."""

        with self._lock:
            return self._cleanup_locked()

    def _cleanup_locked(self) -> int:
        now = self._now()
        expired = tuple(
            key for key, scope in self._scopes.items() if scope.expires_at <= now
        )
        for key in expired:
            del self._scopes[key]
        return len(expired)

    def _scope_key(
        self,
        agent_id: str,
        execution_id: str | None,
        policy: AgentMemoryPolicy,
    ) -> tuple[str, str] | None:
        if not isinstance(policy, AgentMemoryPolicy):
            raise InvalidAgentWorkingMemoryError("policy must be AgentMemoryPolicy.")
        if execution_id is None:
            return None
        if not isinstance(execution_id, str) or not execution_id:
            raise InvalidAgentWorkingMemoryError("execution_id must be a non-empty string or None.")
        return validate_agent_id(agent_id), execution_id

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise InvalidAgentWorkingMemoryError("clock must return an aware datetime.")
        return value

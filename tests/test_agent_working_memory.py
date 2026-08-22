from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.agent_registry import AgentMemoryPolicy
from core.agent_working_memory import AgentWorkingMemory


NOW = datetime(2026, 8, 23, tzinfo=timezone.utc)


def _policy(*, read: bool = True, write: bool = True, limit: int = 3) -> AgentMemoryPolicy:
    return AgentMemoryPolicy(
        can_read_memory=read,
        can_write_memory=write,
        max_memory_items=limit,
    )


def test_memory_is_isolated_by_agent_and_execution() -> None:
    memory = AgentWorkingMemory(clock=lambda: NOW)
    policy = _policy()

    memory.write("agent.one", "execution.one", {"note": "one"}, policy)
    memory.write("agent.one", "execution.two", {"note": "two"}, policy)
    memory.write("agent.two", "execution.one", {"note": "other"}, policy)

    assert dict(memory.read("agent.one", "execution.one", policy)) == {"note": "one"}
    assert dict(memory.read("agent.one", "execution.two", policy)) == {"note": "two"}
    assert dict(memory.read("agent.two", "execution.one", policy)) == {"note": "other"}


def test_memory_limit_is_deterministic_and_keeps_latest_keys() -> None:
    memory = AgentWorkingMemory(clock=lambda: NOW)
    policy = _policy(limit=2)

    memory.write("agent.one", "execution.one", {"c": 3, "a": 1, "b": 2}, policy)
    memory.write("agent.one", "execution.one", {"d": 4}, policy)

    assert dict(memory.read("agent.one", "execution.one", policy)) == {"c": 3, "d": 4}


def test_expired_memory_is_cleaned_before_read() -> None:
    current = {"value": NOW}
    memory = AgentWorkingMemory(ttl=timedelta(seconds=30), clock=lambda: current["value"])
    policy = _policy()
    memory.write("agent.one", "execution.one", {"note": "temporary"}, policy)

    current["value"] = NOW + timedelta(seconds=30)

    assert dict(memory.read("agent.one", "execution.one", policy)) == {}
    assert memory.cleanup() == 0


def test_permissions_and_missing_execution_id_leave_no_state() -> None:
    memory = AgentWorkingMemory(clock=lambda: NOW)

    assert dict(memory.write("agent.one", "execution.one", {"note": "x"}, _policy(read=False, write=False))) == {}
    assert dict(memory.read("agent.one", "execution.one", _policy(read=False, write=False))) == {}
    assert dict(memory.write("agent.one", None, {"note": "x"}, _policy())) == {}
    assert dict(memory.read("agent.one", None, _policy())) == {}

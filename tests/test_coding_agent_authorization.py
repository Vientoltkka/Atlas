from __future__ import annotations

import re

import pytest

from agents.coding_agent import CodingAgent, PendingCodingChangeError


class _Client:
    def ask(self, *, model: str, messages: list[dict[str, str]]) -> str:
        return "new content\n"


class _Reader:
    def execute(self, path: str) -> str:
        return "old content\n"


class _Writer:
    def execute(self, path: str, content: str) -> str:
        return "written"


def _agent() -> CodingAgent:
    return CodingAgent(_Client(), _Reader(), _Writer())  # type: ignore[arg-type]


def test_coding_change_requires_single_use_token_and_shows_diff() -> None:
    agent = _agent()
    response = agent.run("model", [{"role": "user", "content": "corrige agents/coding_agent.py"}])
    assert "Diff:" in response
    assert "-old content" in response
    assert "+new content" in response
    token = re.search(r"APLICAR ([A-Za-z0-9_-]+)", response).group(1)  # type: ignore[union-attr]
    change = agent.authorize_pending_change(token)
    assert change.relative_path == "agents/coding_agent.py"
    with pytest.raises(PendingCodingChangeError):
        agent.authorize_pending_change(token)


@pytest.mark.parametrize("path", ["../outside.py", "C:\\outside.py"])
def test_coding_change_rejects_paths_outside_atlas(path: str) -> None:
    response = _agent().run("model", [{"role": "user", "content": f"corrige {path}"}])
    assert "dentro de C:\\AI\\Atlas" in response
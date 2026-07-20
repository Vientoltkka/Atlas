from __future__ import annotations

from agents.coding_agent import CodingAgent


class _PromptClientSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def ask(self, *, model: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((model, messages))
        return "ok"


class _ReadFileFake:
    def execute(self, path: str) -> str:
        return "content"


class _WriteFileFake:
    def execute(self, path: str, content: str) -> str:
        return "written"


def test_coding_agent_conversational_route_still_adds_system_prompt() -> None:
    prompt_client = _PromptClientSpy()
    agent = CodingAgent(prompt_client, _ReadFileFake(), _WriteFileFake())  # type: ignore[arg-type]

    result = agent.run(
        "coding-model",
        [{"role": "user", "content": "explica este codigo"}],
    )

    assert result == "ok"
    assert len(prompt_client.calls) == 1
    model, messages = prompt_client.calls[0]
    assert model == "coding-model"
    assert messages[0]["role"] == "system"
    assert "Eres Atlas Coding Agent" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "explica este codigo"}

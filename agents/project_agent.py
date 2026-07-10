"""Project analysis agent."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.prompt_client import PromptClient
from use_cases.analyze_project import AnalyzeProjectUseCase


class ProjectAgent(BaseAgent):
    """Agent specialized in project analysis."""

    SYSTEM_PROMPT = """
Eres Atlas Project Agent.

Analiza proyectos software.

Debes:

- explicar la arquitectura;
- detectar módulos;
- detectar problemas;
- proponer mejoras;
- no inventar información.
"""

    def __init__(
        self,
        prompt_client: PromptClient,
        analyze_project: AnalyzeProjectUseCase,
    ) -> None:
        self._client = prompt_client
        self._analyze_project = analyze_project

    @property
    def name(self) -> str:
        return "project"

    @property
    def description(self) -> str:
        return "Project analysis."

    def run(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:

        project = self._analyze_project.execute(".")

        summary = []

        for filename in project:
            summary.append(filename)

        conversation = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "Proyecto:\n\n" + "\n".join(summary),
            },
        ]

        conversation.extend(messages)

        return self._client.ask(
            model=model,
            messages=conversation,
        )
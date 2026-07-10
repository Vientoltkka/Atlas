"""Agent Registry."""

from agents.base_agent import BaseAgent


class AgentRegistry:

    def __init__(self):

        self._agents = {}

    def register(self, agent: BaseAgent):

        self._agents[agent.name] = agent

    def get(self, name: str):

        return self._agents.get(name)

    def list(self):

        return list(self._agents.keys())
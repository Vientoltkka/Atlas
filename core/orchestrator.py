"""Core orchestration module for Atlas."""

from __future__ import annotations

from agents.registry import AgentRegistry

from core.model_manager import ModelManager
from core.planner import Planner
from core.router import Router

from memory.conversation import ConversationMemory
from use_cases.write_file import WriteFileUseCase


class AtlasOrchestrator:
    """Main orchestrator for Atlas."""

    def __init__(
        self,
        planner: Planner,
        router: Router,
        model_manager: ModelManager,
        memory: ConversationMemory,
        registry: AgentRegistry,
        write_file: WriteFileUseCase,
    ) -> None:

        self._planner = planner
        self._router = router
        self._model_manager = model_manager
        self._memory = memory
        self._registry = registry
        self._write_file = write_file

    def start(self) -> None:

        print("Atlas iniciado correctamente.")
        print()

        while True:

            prompt = input("Tú: ")

            if prompt.lower() in ("exit", "quit", "salir"):
                print("\nHasta pronto.")
                break

            # -----------------------------
            # Memory
            # -----------------------------

            self._memory.add_user(prompt)

            # -----------------------------
            # Planner
            # -----------------------------

            plan = self._planner.create_plan(prompt)

            # -----------------------------
            # Router
            # -----------------------------

            agent_name = self._router.route(plan)

            # -----------------------------
            # Registry
            # -----------------------------

            agent = self._registry.get(agent_name)

            if agent is None:
                raise RuntimeError(
                    f"Agent '{agent_name}' is not registered."
                )

            # -----------------------------
            # Model selection
            # -----------------------------

            model = self._model_manager.choose_model(
                plan.task
            )

            # -----------------------------
            # Agent
            # -----------------------------

            response = agent.run(
                model=model,
                messages=self._memory.history(),
            )

            # -----------------------------
            # Memory
            # -----------------------------

            self._memory.add_assistant(response)

            print()
            print("Atlas:")
            print(response)
            print()
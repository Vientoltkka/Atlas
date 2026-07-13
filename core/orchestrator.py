"""Core orchestration module for Atlas."""

from __future__ import annotations

from pathlib import Path

from agents.registry import AgentRegistry

from core.model_manager import ModelManager
from core.planner import Planner
from core.router import Router

from memory.conversation import ConversationMemory
from use_cases.refactoring_interaction import RefactoringInteractionUseCase
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
        refactoring_interaction: RefactoringInteractionUseCase | None = None,
        project_root: Path | None = None,
    ) -> None:

        self._planner = planner
        self._router = router
        self._model_manager = model_manager
        self._memory = memory
        self._registry = registry
        self._write_file = write_file
        self._refactoring_interaction = refactoring_interaction
        self._project_root = project_root or Path(".")

    def start(self) -> None:

        print("Atlas iniciado correctamente.")
        print()

        while True:

            prompt = input("Tú: ")

            coding_agent = self._registry.get("coding")

            if (
                prompt.strip().lower() == "s"
                and coding_agent is not None
                and coding_agent.generated_path is not None
            ):
                result = self._write_file.execute(
                    coding_agent.generated_path,
                    coding_agent.generated_content,
                )

                coding_agent.clear_generated()

                print()
                print("Atlas:")
                print(result)
                print()

                continue

            if prompt.lower() in ("exit", "quit", "salir"):
                print("\nHasta pronto.")
                break

            if self._refactoring_interaction is not None:
                refactoring_response = self._refactoring_interaction.execute(
                    prompt=prompt,
                    project_root=self._project_root,
                    confirm=input,
                )

                if refactoring_response is not None:
                    print()
                    print("Atlas:")
                    print(refactoring_response)
                    print()

                    continue

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
                agent_name
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

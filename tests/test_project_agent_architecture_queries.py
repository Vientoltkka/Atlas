import unittest

from agents.project_agent import ProjectAgent
from core.planner import Planner
from models.architecture_graph import ArchitectureGraph, ArchitectureNode
from use_cases.query_architecture_graph import QueryArchitectureGraphUseCase


class _FailingPromptClient:
    def ask(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        raise AssertionError("Architecture queries must not use the LLM.")


class _FailingReadProject:
    def execute(
        self,
        root: str,
    ) -> list[dict[str, object]]:
        raise AssertionError("Architecture queries must not read full project.")


class ProjectAgentArchitectureQueriesTest(unittest.TestCase):
    FORBIDDEN_TERMS = (
        "Validator",
        "Executor",
        "Worker Manager",
        "Result Aggregator",
        "plan_generator.py",
        "router/dispatcher.py",
    )

    def setUp(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="models/prompt_client.py",
                    module="models.prompt_client",
                    classes=["PromptClient"],
                ),
                ArchitectureNode(
                    path="agents/chat_agent.py",
                    module="agents.chat_agent",
                    imports=["models.prompt_client.PromptClient"],
                    dependencies=["models/prompt_client.py"],
                ),
                ArchitectureNode(
                    path="agents/coding_agent.py",
                    module="agents.coding_agent",
                    imports=["models.prompt_client.PromptClient"],
                    dependencies=["models/prompt_client.py"],
                ),
                ArchitectureNode(
                    path="core/planner.py",
                    module="core.planner",
                    classes=["Plan", "Planner"],
                    functions=["Planner.create_plan"],
                ),
                ArchitectureNode(
                    path="core/router.py",
                    module="core.router",
                    classes=["Router"],
                    imports=["core.planner.Plan"],
                    dependencies=["core/planner.py"],
                ),
                ArchitectureNode(
                    path="core/model_manager.py",
                    module="core.model_manager",
                    classes=["ModelManager"],
                ),
                ArchitectureNode(
                    path="core/orchestrator.py",
                    module="core.orchestrator",
                    imports=[
                        "core.model_manager.ModelManager",
                        "core.planner.Planner",
                        "core.router.Router",
                    ],
                    dependencies=[
                        "core/model_manager.py",
                        "core/planner.py",
                        "core/router.py",
                    ],
                ),
                ArchitectureNode(
                    path="main.py",
                    module="main",
                    dependencies=["core/orchestrator.py"],
                ),
            ]
        )
        self.agent = ProjectAgent(
            _FailingPromptClient(),  # type: ignore[arg-type]
            _FailingReadProject(),  # type: ignore[arg-type]
            query_architecture_graph=QueryArchitectureGraphUseCase(graph),
        )

    def assert_forbidden_terms_absent(
        self,
        response: str,
    ) -> None:
        for term in self.FORBIDDEN_TERMS:
            self.assertNotIn(term, response)

    def test_who_uses_prompt_client(self) -> None:
        response = self.agent.run(
            model="unused",
            messages=[{"role": "user", "content": "¿Quién usa PromptClient?"}],
        )

        self.assertIn("Objetivo:\nmodels/prompt_client.py", response)
        self.assertIn("- agents/chat_agent.py", response)
        self.assertIn("- agents/coding_agent.py", response)
        self.assert_forbidden_terms_absent(response)

    def test_modules_depend_on_planner(self) -> None:
        response = self.agent.run(
            model="unused",
            messages=[
                {
                    "role": "user",
                    "content": "¿Qué módulos dependen de Planner?",
                }
            ],
        )

        self.assertIn("Objetivo:\ncore/planner.py", response)
        self.assertIn("- core/orchestrator.py", response)
        self.assertIn("- core/router.py", response)
        self.assertIn("- main.py", response)
        self.assert_forbidden_terms_absent(response)

    def test_imported_classes_of_router(self) -> None:
        response = self.agent.run(
            model="unused",
            messages=[
                {
                    "role": "user",
                    "content": "¿Qué clases importa Router?",
                }
            ],
        )

        self.assertEqual(
            response,
            "\n".join(
                [
                    "Objetivo:",
                    "core/router.py",
                    "",
                    "Clases importadas:",
                    "- Plan",
                ]
            ),
        )
        self.assert_forbidden_terms_absent(response)

    def test_impact_of_model_manager(self) -> None:
        response = self.agent.run(
            model="unused",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "¿Qué archivos se verían afectados si modifico "
                        "ModelManager?"
                    ),
                }
            ],
        )

        self.assertIn("Objetivo:\ncore/model_manager.py", response)
        self.assertIn("Archivos potencialmente afectados:", response)
        self.assertIn("- core/orchestrator.py", response)
        self.assertIn("- main.py", response)
        self.assert_forbidden_terms_absent(response)

    def test_planner_routes_architecture_queries_to_project_agent(self) -> None:
        planner = Planner()

        for query in (
            "¿Quién usa PromptClient?",
            "¿Qué módulos dependen de Planner?",
            "¿Qué clases importa Router?",
            "¿Qué archivos se verían afectados si modifico ModelManager?",
        ):
            self.assertEqual(planner.create_plan(query).task, "project")


if __name__ == "__main__":
    unittest.main()

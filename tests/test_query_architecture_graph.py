import unittest

from models.architecture_graph import ArchitectureGraph, ArchitectureNode
from use_cases.query_architecture_graph import QueryArchitectureGraphUseCase


class QueryArchitectureGraphUseCaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = ArchitectureGraph(
            nodes=[
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
                    functions=["Router.route"],
                    imports=["core.planner.Plan"],
                    dependencies=["core/planner.py"],
                ),
                ArchitectureNode(
                    path="core/orchestrator.py",
                    module="core.orchestrator",
                    classes=["AtlasOrchestrator"],
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
                    path="bootstrap/bootstrap.py",
                    module="bootstrap.bootstrap",
                    classes=["Bootstrap"],
                    imports=[
                        "core.model_manager.ModelManager",
                        "core.orchestrator.AtlasOrchestrator",
                        "core.planner.Planner",
                    ],
                    dependencies=[
                        "core/model_manager.py",
                        "core/orchestrator.py",
                        "core/planner.py",
                    ],
                ),
                ArchitectureNode(
                    path="core/model_manager.py",
                    module="core.model_manager",
                    classes=["ModelManager"],
                    functions=[
                        "ModelManager.list_models",
                        "ModelManager.choose_model",
                    ],
                ),
                ArchitectureNode(
                    path="extras/router.py",
                    module="extras.router",
                    classes=["Router"],
                ),
            ]
        )
        self.use_case = QueryArchitectureGraphUseCase(self.graph)

    def test_dependencies_of_router(self) -> None:
        result = self.use_case.dependencies_of("core.router")

        self.assertEqual(result.target, "core/router.py")
        self.assertEqual(result.dependencies, ["core/planner.py"])

    def test_dependents_of_planner(self) -> None:
        result = self.use_case.dependents_of("Planner")

        self.assertEqual(result.target, "core/planner.py")
        self.assertEqual(
            result.direct_dependents,
            [
                "bootstrap/bootstrap.py",
                "core/orchestrator.py",
                "core/router.py",
            ],
        )
        self.assertEqual(result.indirect_dependents, [])

    def test_imported_classes_of_router(self) -> None:
        result = self.use_case.imported_classes_of("core/router.py")

        self.assertEqual(result.target, "core/router.py")
        self.assertEqual(result.imported_classes, ["Plan"])

    def test_impact_of_model_manager(self) -> None:
        result = self.use_case.impact_of("ModelManager")

        self.assertEqual(result.target, "core/model_manager.py")
        self.assertEqual(
            result.affected_files,
            [
                "bootstrap/bootstrap.py",
                "core/orchestrator.py",
            ],
        )

    def test_missing_target(self) -> None:
        result = self.use_case.dependencies_of("MissingThing")

        self.assertIsNone(result.target)
        self.assertEqual(result.matches, [])
        self.assertEqual(result.dependencies, [])

    def test_multiple_matches(self) -> None:
        result = self.use_case.dependencies_of("Router")

        self.assertIsNone(result.target)
        self.assertEqual(
            result.matches,
            [
                "core/router.py",
                "extras/router.py",
            ],
        )
        self.assertEqual(result.dependencies, [])


if __name__ == "__main__":
    unittest.main()

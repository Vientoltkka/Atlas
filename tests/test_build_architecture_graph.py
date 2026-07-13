import unittest

from models.architecture_graph import ArchitectureGraph, ArchitectureNode
from use_cases.build_architecture_graph import BuildArchitectureGraphUseCase


class BuildArchitectureGraphUseCaseTest(unittest.TestCase):
    def test_resolves_internal_dependencies(self) -> None:
        index: list[dict[str, object]] = [
            {
                "path": "core/orchestrator.py",
                "imports": [
                    "use_cases.read_file.ReadFileUseCase",
                    "external_package",
                ],
                "classes": ["AtlasOrchestrator"],
                "functions": [
                    "AtlasOrchestrator.run",
                    "AtlasOrchestrator._debug",
                ],
            },
            {
                "path": "use_cases/read_file.py",
                "imports": [],
                "classes": ["ReadFileUseCase"],
                "functions": ["ReadFileUseCase.execute"],
            },
        ]

        graph = BuildArchitectureGraphUseCase().execute(index)

        self.assertEqual(
            graph.nodes,
            [
                ArchitectureNode(
                    path="core/orchestrator.py",
                    module="core.orchestrator",
                    classes=["AtlasOrchestrator"],
                    functions=["AtlasOrchestrator.run"],
                    imports=[
                        "use_cases.read_file.ReadFileUseCase",
                        "external_package",
                    ],
                    dependencies=["use_cases/read_file.py"],
                ),
                ArchitectureNode(
                    path="use_cases/read_file.py",
                    module="use_cases.read_file",
                    classes=["ReadFileUseCase"],
                    functions=["ReadFileUseCase.execute"],
                    imports=[],
                    dependencies=[],
                ),
            ],
        )


class ArchitectureGraphTest(unittest.TestCase):
    def test_summary(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="core/orchestrator.py",
                    module="core.orchestrator",
                    dependencies=[
                        "use_cases/read_file.py",
                        "use_cases/write_file.py",
                    ],
                ),
                ArchitectureNode(
                    path="use_cases/read_file.py",
                    module="use_cases.read_file",
                    dependencies=[],
                ),
            ]
        )

        self.assertEqual(
            graph.summary(),
            "\n".join(
                [
                    "Módulos analizados: 2",
                    "Relaciones: 2",
                    "Dependencias medias: 1.0",
                    "Módulo con más dependencias:",
                    "core/orchestrator.py",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()

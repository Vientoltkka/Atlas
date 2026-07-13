import unittest

from models.architecture_graph import ArchitectureGraph, ArchitectureNode
from use_cases.analyze_architecture_risks import (
    AnalyzeArchitectureRisksUseCase,
)


class AnalyzeArchitectureRisksUseCaseTest(unittest.TestCase):
    def test_detects_cycle(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="a.py",
                    module="a",
                    dependencies=["b.py"],
                ),
                ArchitectureNode(
                    path="b.py",
                    module="b",
                    dependencies=["c.py"],
                ),
                ArchitectureNode(
                    path="c.py",
                    module="c",
                    dependencies=["a.py"],
                ),
            ]
        )

        analysis = AnalyzeArchitectureRisksUseCase().execute(graph)

        self.assertEqual(analysis.cycles, [["a.py", "b.py", "c.py"]])
        self.assertEqual(self._risk_for(analysis, "a.py").risk_level, "high")

    def test_detects_isolated_module(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="isolated.py",
                    module="isolated",
                ),
                ArchitectureNode(
                    path="used.py",
                    module="used",
                ),
                ArchitectureNode(
                    path="consumer.py",
                    module="consumer",
                    dependencies=["used.py"],
                ),
            ]
        )

        analysis = AnalyzeArchitectureRisksUseCase().execute(graph)

        self.assertEqual(analysis.isolated_modules, ["isolated.py"])
        self.assertIn("módulo aislado", self._risk_for(analysis, "isolated.py").reasons)

    def test_detects_critical_module(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(path="core.py", module="core"),
                ArchitectureNode(
                    path="a.py",
                    module="a",
                    dependencies=["core.py"],
                ),
                ArchitectureNode(
                    path="b.py",
                    module="b",
                    dependencies=["core.py"],
                ),
                ArchitectureNode(
                    path="c.py",
                    module="c",
                    dependencies=["core.py"],
                ),
                ArchitectureNode(
                    path="d.py",
                    module="d",
                    dependencies=["core.py"],
                ),
            ]
        )

        analysis = AnalyzeArchitectureRisksUseCase().execute(graph)
        risk = self._risk_for(analysis, "core.py")

        self.assertEqual(risk.direct_dependents, 4)
        self.assertEqual(risk.risk_level, "critical")
        self.assertEqual(analysis.critical_modules, ["core.py"])

    def test_detects_high_coupling(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(path="a.py", module="a"),
                ArchitectureNode(path="b.py", module="b"),
                ArchitectureNode(path="c.py", module="c"),
                ArchitectureNode(
                    path="orchestrator.py",
                    module="orchestrator",
                    dependencies=["a.py", "b.py", "c.py"],
                ),
            ]
        )

        analysis = AnalyzeArchitectureRisksUseCase().execute(graph)
        risk = self._risk_for(analysis, "orchestrator.py")

        self.assertEqual(risk.outgoing_dependencies, 3)
        self.assertIn("orchestrator.py", analysis.high_coupling_modules)
        self.assertEqual(risk.risk_level, "high")

    def test_calculates_total_impact(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(path="base.py", module="base"),
                ArchitectureNode(
                    path="service.py",
                    module="service",
                    dependencies=["base.py"],
                ),
                ArchitectureNode(
                    path="api.py",
                    module="api",
                    dependencies=["service.py"],
                ),
                ArchitectureNode(
                    path="main.py",
                    module="main",
                    dependencies=["api.py"],
                ),
            ]
        )

        analysis = AnalyzeArchitectureRisksUseCase().execute(graph)
        risk = self._risk_for(analysis, "base.py")

        self.assertEqual(risk.total_impact, 3)
        self.assertEqual(
            analysis.modules_with_highest_total_impact,
            ["base.py"],
        )

    def test_project_without_cycles(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(path="a.py", module="a"),
                ArchitectureNode(
                    path="b.py",
                    module="b",
                    dependencies=["a.py"],
                ),
            ]
        )

        analysis = AnalyzeArchitectureRisksUseCase().execute(graph)

        self.assertEqual(analysis.cycles, [])
        self.assertIn("Ciclos detectados: 0", analysis.summary())

    def _risk_for(
        self,
        analysis,
        path: str,
    ):
        for result in analysis.results:
            if result.path == path:
                return result

        raise AssertionError(f"Missing risk result for {path}")


if __name__ == "__main__":
    unittest.main()

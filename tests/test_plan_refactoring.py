import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from models.architecture_graph import ArchitectureGraph, ArchitectureNode
from use_cases.plan_refactoring import (
    InvalidRefactoringNameError,
    PlanRefactoringUseCase,
)


class PlanRefactoringUseCaseTest(unittest.TestCase):
    def test_generates_refactoring_plan(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="core/service.py",
                    module="core.service",
                    classes=["Service"],
                ),
            ]
        )

        plan = PlanRefactoringUseCase(graph).execute("Service", "BetterService")

        self.assertEqual(plan.operation, "rename_symbol")
        self.assertEqual(plan.symbol_name, "Service")
        self.assertEqual(plan.new_name, "BetterService")
        self.assertEqual(plan.definition_file, "core/service.py")
        self.assertEqual(plan.affected_files, ["core/service.py"])
        self.assertTrue(plan.can_apply)
        self.assertIn("Renombrar Service a BetterService", plan.summary)

    def test_detects_definition_file(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="core/router.py",
                    module="core.router",
                    classes=["Router"],
                ),
            ]
        )

        plan = PlanRefactoringUseCase(graph).execute("Router", "RequestRouter")

        self.assertEqual(plan.definition_file, "core/router.py")

    def test_includes_dependent_files(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="core/planner.py",
                    module="core.planner",
                    classes=["Planner"],
                ),
                ArchitectureNode(
                    path="core/router.py",
                    module="core.router",
                    dependencies=["core/planner.py"],
                ),
                ArchitectureNode(
                    path="main.py",
                    module="main",
                    dependencies=["core/router.py"],
                ),
            ]
        )

        plan = PlanRefactoringUseCase(graph).execute("Planner", "SmartPlanner")

        self.assertEqual(
            plan.affected_files,
            [
                "core/planner.py",
                "core/router.py",
                "main.py",
            ],
        )

    def test_removes_duplicates_from_affected_files(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="core/self_ref.py",
                    module="core.self_ref",
                    classes=["SelfRef"],
                    dependencies=["core/self_ref.py"],
                ),
            ]
        )

        plan = PlanRefactoringUseCase(graph).execute("SelfRef", "RenamedSelfRef")

        self.assertEqual(plan.affected_files, ["core/self_ref.py"])

    def test_orders_affected_files_deterministically(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="core/base.py",
                    module="core.base",
                    classes=["Base"],
                ),
                ArchitectureNode(
                    path="z_consumer.py",
                    module="z_consumer",
                    dependencies=["core/base.py"],
                ),
                ArchitectureNode(
                    path="a_consumer.py",
                    module="a_consumer",
                    dependencies=["core/base.py"],
                ),
            ]
        )

        plan = PlanRefactoringUseCase(graph).execute("Base", "RenamedBase")

        self.assertEqual(
            plan.affected_files,
            [
                "a_consumer.py",
                "core/base.py",
                "z_consumer.py",
            ],
        )

    def test_calculates_low_risk(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="core/value.py",
                    module="core.value",
                    classes=["Value"],
                ),
                ArchitectureNode(
                    path="core/consumer.py",
                    module="core.consumer",
                    dependencies=["core/value.py"],
                ),
            ]
        )

        plan = PlanRefactoringUseCase(graph).execute("Value", "RenamedValue")

        self.assertEqual(plan.risk_level, "low")
        self.assertTrue(plan.can_apply)

    def test_calculates_medium_risk(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="core/service.py",
                    module="core.service",
                    classes=["Service"],
                ),
                ArchitectureNode(
                    path="api.py",
                    module="api",
                    dependencies=["core/service.py"],
                ),
                ArchitectureNode(
                    path="main.py",
                    module="main",
                    dependencies=["api.py"],
                ),
            ]
        )

        plan = PlanRefactoringUseCase(graph).execute("Service", "RenamedService")

        self.assertEqual(plan.risk_level, "medium")
        self.assertTrue(plan.can_apply)

    def test_calculates_high_risk(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="core/kernel.py",
                    module="core.kernel",
                    classes=["Kernel"],
                ),
                ArchitectureNode(
                    path="a.py",
                    module="a",
                    dependencies=["core/kernel.py"],
                ),
                ArchitectureNode(
                    path="b.py",
                    module="b",
                    dependencies=["core/kernel.py"],
                ),
                ArchitectureNode(
                    path="c.py",
                    module="c",
                    dependencies=["core/kernel.py"],
                ),
                ArchitectureNode(
                    path="d.py",
                    module="d",
                    dependencies=["core/kernel.py"],
                ),
                ArchitectureNode(
                    path="e.py",
                    module="e",
                    dependencies=["core/kernel.py"],
                ),
            ]
        )

        plan = PlanRefactoringUseCase(graph).execute("Kernel", "RenamedKernel")

        self.assertEqual(plan.risk_level, "high")
        self.assertFalse(plan.can_apply)
        self.assertEqual(plan.warnings, ["más de 5 archivos afectados"])

    def test_rejects_equal_names(self) -> None:
        graph = ArchitectureGraph()

        with self.assertRaises(InvalidRefactoringNameError):
            PlanRefactoringUseCase(graph).execute("Service", "Service")

    def test_missing_symbol_returns_non_applicable_plan(self) -> None:
        graph = ArchitectureGraph()

        plan = PlanRefactoringUseCase(graph).execute("Missing", "RenamedMissing")

        self.assertIsNone(plan.definition_file)
        self.assertEqual(plan.affected_files, [])
        self.assertEqual(plan.risk_level, "high")
        self.assertFalse(plan.can_apply)
        self.assertEqual(plan.warnings, ["símbolo inexistente: Missing"])

    def test_ambiguous_definition_returns_non_applicable_plan(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="core/router.py",
                    module="core.router",
                    classes=["Router"],
                ),
                ArchitectureNode(
                    path="extras/router.py",
                    module="extras.router",
                    classes=["Router"],
                ),
            ]
        )

        plan = PlanRefactoringUseCase(graph).execute("Router", "RenamedRouter")

        self.assertIsNone(plan.definition_file)
        self.assertEqual(plan.affected_files, [])
        self.assertEqual(plan.risk_level, "high")
        self.assertFalse(plan.can_apply)
        self.assertIn("definición ambigua", plan.warnings)
        self.assertIn("coincidencia: core/router.py", plan.warnings)
        self.assertIn("coincidencia: extras/router.py", plan.warnings)

    def test_does_not_modify_files_or_graph(self) -> None:
        graph = ArchitectureGraph(
            nodes=[
                ArchitectureNode(
                    path="core/service.py",
                    module="core.service",
                    classes=["Service"],
                ),
            ]
        )
        original_graph = graph

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "service.py"
            source_path.write_text("class Service:\n    pass\n", encoding="utf-8")
            original_content = source_path.read_text(encoding="utf-8")

            PlanRefactoringUseCase(graph).execute("Service", "BetterService")

            self.assertEqual(source_path.read_text(encoding="utf-8"), original_content)
            self.assertEqual(graph, original_graph)


if __name__ == "__main__":
    unittest.main()

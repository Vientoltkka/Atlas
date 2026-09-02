"""Deterministic supervised code repair for the task routing subsystem."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.self_improvement_conversation import ImprovementDiagnosis
from core.supervised_repair import RepairProposal, RepairValidation


_ROUTER_SOURCE = "core/router.py"
_ROUTER_TEST = "tests/test_router.py"
_SOURCE_OLD = '        return self._TASK_ROUTES.get(plan.task, "chat")\n'
_SOURCE_NEW = '        return self._TASK_ROUTES.get(plan.task.casefold(), "chat")\n'
_TEST_HEADER = '"""Focused tests for the task router."""\n\nfrom core.planner import Plan\nfrom core.router import Router\n'
_TEST = '''


def test_task_routing_ignores_letter_case() -> None:
    router = Router()

    assert router.route(Plan(task="Coding", objective="x")) == "coding"
    assert router.route(Plan(task="Research", objective="x")) == "chat"
'''
_METRIC = "rutas_de_tarea_sensibles_a_mayusculas_correctas"
_MEASURE_SCRIPT = (
    "from core.planner import Plan\n"
    "from core.router import Router\n"
    'print(1 if Router().route(Plan(task="Coding", objective="x")) == "coding" else 0)\n'
)


class RoutingRepairBuilder:
    """Build exactly one reviewed routing repair; it never accepts generated code."""

    def __init__(self, project_root: Path) -> None:
        self._root = project_root.resolve()
        self._before_matches: int | None = None

    def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
        return diagnosis.scope == (_ROUTER_SOURCE, _ROUTER_TEST)

    def build(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> RepairProposal | None:
        if not self.can_handle(diagnosis, _prompt):
            return None
        source_path, test_path = self._root / _ROUTER_SOURCE, self._root / _ROUTER_TEST
        try:
            source = source_path.read_text(encoding="utf-8")
            tests = test_path.read_text(encoding="utf-8") if test_path.exists() else ""
        except OSError:
            return None
        if _SOURCE_NEW in source or _SOURCE_OLD not in source or "test_task_routing_ignores_letter_case" in tests:
            return None
        self._before_matches = self._measure_case_sensitive_routes()
        return RepairProposal(
            proposal_id="repair.routing.case-insensitive-task-lookup",
            objective="Evitar que el enrutado de tareas dependa de mayúsculas accidentales.",
            files={
                _ROUTER_SOURCE: source.replace(_SOURCE_OLD, _SOURCE_NEW, 1),
                _ROUTER_TEST: (_TEST_HEADER if not tests else tests) + _TEST,
            },
            focused_tests=(_ROUTER_TEST,),
            metric_directions={_METRIC: "increase"},
        )

    def validator(self, _proposal: RepairProposal) -> RepairValidation:
        before = self._before_matches
        if before is None:
            return RepairValidation(False, detail="No existe una medición previa confiable.")
        after = self._measure_case_sensitive_routes()
        tests = self._run("-m", "pytest", "-q", _ROUTER_TEST)
        compiled = self._run("-m", "py_compile", _ROUTER_SOURCE, _ROUTER_TEST)
        diff_checked = self._git("diff", "--check")
        passed = after > before and all(result.returncode == 0 for result in (tests, compiled, diff_checked))
        detail = "tests focales, py_compile y git diff --check correctos." if passed else self._validation_error(tests, compiled, diff_checked)
        return RepairValidation(passed, {_METRIC: float(before)}, {_METRIC: float(after)}, detail)

    def _measure_case_sensitive_routes(self) -> int:
        """Count known case-sensitive routing successes in a fresh interpreter."""
        result = subprocess.run((sys.executable, "-c", _MEASURE_SCRIPT), cwd=self._root, capture_output=True, text=True, check=False)
        try:
            return int(result.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            return 0

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run((sys.executable, *arguments), cwd=self._root, capture_output=True, text=True, check=False)

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(("git", *arguments), cwd=self._root, capture_output=True, text=True, check=False)

    @staticmethod
    def _validation_error(*results: subprocess.CompletedProcess[str]) -> str:
        labels = ("tests focales", "py_compile", "git diff --check")
        failures = [label for label, result in zip(labels, results) if result.returncode != 0]
        return "Validación fallida: " + ", ".join(failures or ("métrica no mejoró",)) + "."

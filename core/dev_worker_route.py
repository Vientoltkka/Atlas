"""Bounded natural-language route for the Computer Worker dev goal (V1).

One small adapter between Atlas' prompt flow and the deterministic DevWorker:

- ``accepts`` detects exactly one natural request family (the controlled
  Computer Worker control-file goal); anything else is declined (None) so the
  normal Atlas routes keep handling it.
- ``plan`` derives the minimal DevSteps (READ -> optional EDIT -> focal TEST)
  by inspecting the controlled fixture and its focal test, never from free
  text parsing of arbitrary goals.
- ``handle`` runs DevWorker and returns its evidence report, so the terminal
  status is DONE only when the focal test passes.
"""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata

from core.dev_worker import DevStep, DevWorker, DevWorkerAction
from core.test_runner import PytestRunner

CONTROL_FILE_RELATIVE = "tests/fixtures/dev_worker_control.py"
FOCAL_TEST_RELATIVE = "tests/test_dev_worker_control.py"
BOUND_GOAL = (
    "corregir el archivo de prueba controlado del Computer Worker "
    "y verificar con su test focal"
)

_ROUTE_MARKERS = ("computer worker", "archivo de prueba controlado")

_VALUE_LINE = re.compile(r"^\s*CONTROL_VALUE\s*=.*$", re.MULTILINE)
_QUOTED_VALUE = re.compile(r"[\"'](.+?)[\"']")
_EXPECTED_IN_TEST = re.compile(
    r"^EXPECTED_CONTROL_VALUE\s*=\s*[\"'](.+?)[\"']", re.MULTILINE
)


def normalize_request(value: str) -> str:
    """Accent-insensitive, case-folded request text."""
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    ).casefold().strip()


def _current_value(path: Path) -> str | None:
    """Return the single declared CONTROL_VALUE, or None when ambiguous."""
    matches = _VALUE_LINE.findall(path.read_text(encoding="utf-8"))
    if len(matches) != 1:
        return None
    value = _QUOTED_VALUE.search(matches[0])
    return value.group(1) if value else None


def _expected_value(path: Path) -> str | None:
    """Return the expected value asserted by the focal test."""
    match = _EXPECTED_IN_TEST.search(path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


class DevWorkerRoute:
    """Select, plan and run the single bounded Computer Worker dev goal."""

    def __init__(self, root: Path, *, worker: DevWorker | None = None) -> None:
        self._root = Path(root).resolve()
        self._worker = worker

    @property
    def worker(self) -> DevWorker:
        return self._worker or DevWorker(self._root, test_runner=PytestRunner(self._root))

    @staticmethod
    def accepts(prompt: str) -> bool:
        """Accept only the bounded Computer Worker control-file request."""
        if not isinstance(prompt, str):
            return False
        text = normalize_request(prompt)
        return all(marker in text for marker in _ROUTE_MARKERS)

    def plan(self, prompt: str) -> tuple[DevStep, ...] | None:
        """Derive the minimal DevSteps, or None when the route must decline."""
        del prompt
        control = self._root / CONTROL_FILE_RELATIVE
        focal = self._root / FOCAL_TEST_RELATIVE
        if not control.is_file() or not focal.is_file():
            return None
        current = _current_value(control)
        expected = _expected_value(focal)
        if current is None or expected is None:
            return None
        steps = [
            DevStep(
                DevWorkerAction.READ,
                path=CONTROL_FILE_RELATIVE,
                description="inspeccion del archivo de control",
            ),
        ]
        if current != expected:
            source = control.read_text(encoding="utf-8")
            old_line = _VALUE_LINE.search(source).group(0).strip()
            new_line = _QUOTED_VALUE.sub(
                lambda _match: f'"{expected}"', old_line, count=1
            )
            if source.count(old_line) != 1:
                return None
            steps.append(
                DevStep(
                    DevWorkerAction.EDIT,
                    path=CONTROL_FILE_RELATIVE,
                    operation="replace_text",
                    old_text=old_line,
                    new_text=new_line,
                    description="correccion del valor incorrecto",
                ),
            )
        steps.append(
            DevStep(
                DevWorkerAction.TEST,
                test_paths=(FOCAL_TEST_RELATIVE,),
                description=f"test focal {FOCAL_TEST_RELATIVE}",
            ),
        )
        return tuple(steps)

    def handle(self, prompt: str) -> str | None:
        """Run the bounded goal and return the evidence report, or None."""
        if not self.accepts(prompt):
            return None
        steps = self.plan(prompt)
        if steps is None:
            return None
        result = self.worker.execute(BOUND_GOAL, steps)
        return result.report

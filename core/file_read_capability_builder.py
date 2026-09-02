"""Deterministic supervised capability improvement for bounded file reads."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from core.self_improvement_conversation import ImprovementClassification, ImprovementDiagnosis, normalize_prompt
from core.supervised_repair import RepairProposal, RepairValidation


_READ_TOOL_SOURCE = "tools/filesystem/read_file_tool.py"
_BOOTSTRAP_SOURCE = "bootstrap/bootstrap.py"
_READ_TOOL_TEST = "tests/test_read_file_tool.py"
_SCOPE = (_READ_TOOL_SOURCE, _BOOTSTRAP_SOURCE, _READ_TOOL_TEST)
_FILE_TERMS = ("archivo", "archivos", "fichero", "ficheros", "lectura de archivos")
_SOURCE_ANCHOR = "        return FileService.read(path)\n"
_SOURCE_NEW = (
    "        content = FileService.read(path)\n"
    "\n"
    "        limit = context.parameters.get(\"limit\")\n"
    "\n"
    "        if limit is None:\n"
    "            return content\n"
    "\n"
    "        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:\n"
    "            raise ValueError(\"Parameter 'limit' must be a positive integer.\")\n"
    "\n"
    "        return \"\\n\".join(content.splitlines()[:limit])\n"
)
_BOOTSTRAP_ANCHOR = (
    "            ReadFileTool(),\n"
    "            arguments_schema=ToolArgumentsSchema(\n"
    "                parameters=(\n"
    "                    ToolParameterSchema(\"path\", str, required=True),\n"
    "                ),\n"
)
_BOOTSTRAP_NEW = (
    "            ReadFileTool(),\n"
    "            arguments_schema=ToolArgumentsSchema(\n"
    "                parameters=(\n"
    "                    ToolParameterSchema(\"path\", str, required=True),\n"
    "                    ToolParameterSchema(\"limit\", int, minimum=1),\n"
    "                ),\n"
)
_TEST_HEADER = (
    '"""Focused tests for the bounded read variant of ReadFileTool."""\n'
    "\n"
    "import pytest\n"
    "\n"
    "from tools.filesystem.read_file_tool import ReadFileTool\n"
    "from tools.tool_context import ToolContext\n"
)
_TEST_BODY = '''

def _tool(tmp_path, content="linea1\\nlinea2\\nlinea3\\n"):
    path = tmp_path / "notas.txt"
    path.write_text(content, encoding="utf-8")
    return ReadFileTool(), str(path)


def test_read_without_limit_returns_whole_file(tmp_path) -> None:
    tool, path = _tool(tmp_path)

    assert tool.execute(ToolContext(parameters={"path": path})) == "linea1\\nlinea2\\nlinea3\\n"


def test_read_with_limit_returns_first_lines(tmp_path) -> None:
    tool, path = _tool(tmp_path)

    assert tool.execute(ToolContext(parameters={"path": path, "limit": 2})) == "linea1\\nlinea2"


@pytest.mark.parametrize("limit", [0, -1, "2", True])
def test_read_rejects_invalid_limit(tmp_path, limit) -> None:
    tool, path = _tool(tmp_path)

    with pytest.raises(ValueError):
        tool.execute(ToolContext(parameters={"path": path, "limit": limit}))
'''
_METRIC = "lecturas_de_archivo_acotadas_correctas"
_MEASURE_SCRIPT = (
    "import os\n"
    "import tempfile\n"
    "from tools.filesystem.read_file_tool import ReadFileTool\n"
    "from tools.tool_context import ToolContext\n"
    "descriptor, path = tempfile.mkstemp(suffix='.txt')\n"
    "with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:\n"
    "    handle.write('linea1\\nlinea2\\nlinea3\\n')\n"
    "try:\n"
    "    result = ReadFileTool().execute(ToolContext(parameters={'path': path, 'limit': 2}))\n"
    "    print(1 if result == 'linea1\\nlinea2' else 0)\n"
    "finally:\n"
    "    os.unlink(path)\n"
)
_PROPOSAL_ID = "improvement.read-file.bounded-limit-read"


class FileReadCapabilityImprovementBuilder:
    """Build exactly one reviewed bounded-read improvement; never generated code."""

    def __init__(self, project_root: Path) -> None:
        self._root = project_root.resolve()
        self._before_successes: int | None = None

    def diagnose(self, prompt: str) -> ImprovementDiagnosis | None:
        text = normalize_prompt(prompt)
        if not any(term in text for term in _FILE_TERMS):
            return None
        return ImprovementDiagnosis(
            ImprovementClassification.CAPABILITY_IMPROVEMENT,
            "Leer archivos de forma acotada (primeras N lineas) sin romper las funciones actuales.",
            _SCOPE,
            (_READ_TOOL_TEST,),
            ("lecturas de archivo acotadas correctas",),
            "Anade una variante opcional de lectura; no es una reparacion de bug. No toca secretos, dependencias ni red.",
            "read_file solo acepta 'path': un 'limit' se rechaza como UNKNOWN_PARAMETER y la tool lo ignora devolviendo el archivo completo.",
        )

    def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
        return diagnosis.classification is ImprovementClassification.CAPABILITY_IMPROVEMENT and diagnosis.scope == _SCOPE

    def build(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> RepairProposal | None:
        if not self.can_handle(diagnosis, _prompt):
            return None
        test_path = self._root / _READ_TOOL_TEST
        try:
            source = (self._root / _READ_TOOL_SOURCE).read_text(encoding="utf-8")
            bootstrap = (self._root / _BOOTSTRAP_SOURCE).read_text(encoding="utf-8")
            tests = test_path.read_text(encoding="utf-8") if test_path.exists() else ""
        except OSError:
            return None
        if (
            _SOURCE_ANCHOR not in source
            or "content.splitlines()[:limit]" in source
            or _BOOTSTRAP_ANCHOR not in bootstrap
            or 'ToolParameterSchema("limit"' in bootstrap
            or "test_read_with_limit_returns_first_lines" in tests
        ):
            return None
        self._before_successes = self._measure_bounded_reads()
        return RepairProposal(
            proposal_id=_PROPOSAL_ID,
            objective=diagnosis.objective,
            files={
                _READ_TOOL_SOURCE: source.replace(_SOURCE_ANCHOR, _SOURCE_NEW, 1),
                _BOOTSTRAP_SOURCE: bootstrap.replace(_BOOTSTRAP_ANCHOR, _BOOTSTRAP_NEW, 1),
                _READ_TOOL_TEST: _TEST_HEADER + _TEST_BODY,
            },
            focused_tests=(_READ_TOOL_TEST,),
            metric_directions={_METRIC: "increase"},
        )

    def validator(self, _proposal: RepairProposal) -> RepairValidation:
        before = self._before_successes
        if before is None:
            return RepairValidation(False, detail="No existe una medicion previa confiable.")
        after = self._measure_bounded_reads()
        basetemp = Path(tempfile.gettempdir()) / "atlas-supervised-pytest"
        tests = self._run("-m", "pytest", "-q", _READ_TOOL_TEST, "--basetemp", str(basetemp))
        compiled = self._run("-m", "py_compile", _READ_TOOL_SOURCE, _BOOTSTRAP_SOURCE)
        diff_checked = self._git("diff", "--check")
        passed = after > before and all(result.returncode == 0 for result in (tests, compiled, diff_checked))
        detail = "tests focales, py_compile y git diff --check correctos." if passed else self._validation_error(tests, compiled, diff_checked)
        return RepairValidation(passed, {_METRIC: float(before)}, {_METRIC: float(after)}, detail)

    def _measure_bounded_reads(self) -> int:
        """Count bounded-read successes in a fresh interpreter."""
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
        return "Validacion fallida: " + ", ".join(failures or ("metrica no mejoro",)) + "."

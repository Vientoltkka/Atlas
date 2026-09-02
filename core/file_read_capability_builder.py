"""Deterministic supervised capability improvement for bounded file reads.

The concrete patch is not stored here. It is derived from the real current
state of the read tool, its registration and its tests: the builder inspects
the code, confirms the gap is reproducible and builds one exact proposal from
what it observed. Unexpected state yields no proposal at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from core.self_improvement_conversation import ImprovementClassification, ImprovementDiagnosis, normalize_prompt
from core.supervised_repair import RepairProposal, RepairValidation


_READ_TOOL_SOURCE = "tools/filesystem/read_file_tool.py"
_BOOTSTRAP_SOURCE = "bootstrap/bootstrap.py"
_READ_TOOL_TEST = "tests/test_read_file_tool.py"
_SCOPE = (_READ_TOOL_SOURCE, _BOOTSTRAP_SOURCE, _READ_TOOL_TEST)
_FILE_TERMS = ("archivo", "archivos", "fichero", "ficheros", "lectura de archivos")
_METRIC = "lecturas_de_archivo_acotadas_correctas"
_PROPOSAL_ID = "improvement.read-file.bounded-limit-read"

_READ_RETURN = re.compile(r"(?m)^(?P<indent>[ \t]+)return (?P<call>FileService\.read\(.+\))$")
_REGISTRATION = re.compile(
    r"(?m)^(?P<call_indent>[ \t]*)ReadFileTool\(\),\n"
    r"(?P=call_indent)arguments_schema=ToolArgumentsSchema\(\n"
    r"(?P<param_indent>[ \t]*)parameters=\(\n"
    r"(?P<parameters>(?:[ \t]*ToolParameterSchema\([^\n]*\),\n)*)"
    r"(?P=param_indent)\),\n"
)
_TOOL_CLASS = re.compile(r"(?m)^class (?P<name>\w+)\(BaseTool\):")
_PARAMETER_LINE = re.compile(r"^(?P<indent>[ \t]*)ToolParameterSchema\(")
_LIMIT_MENTION = re.compile(r"\blimit\b")


@dataclass(frozen=True, slots=True)
class _ReadToolInspection:
    """Read-only snapshot of the real read tool state, used to derive one proposal."""

    tool_source: str
    bootstrap_source: str
    tests_source: str
    tool_class: str
    read_call: str
    read_indent: str
    parameter_indent: str
    parameter_lines: str
    parameter_count: int


class FileReadCapabilityImprovementBuilder:
    """Propose exactly one reviewed bounded-read improvement derived from the real code."""

    def __init__(self, project_root: Path) -> None:
        self._root = project_root.resolve()
        self._before_successes: int | None = None
        self._tool_class = "ReadFileTool"

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
            self._observed_finding(),
        )

    def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
        return diagnosis.classification is ImprovementClassification.CAPABILITY_IMPROVEMENT and diagnosis.scope == _SCOPE

    def build(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> RepairProposal | None:
        if not self.can_handle(diagnosis, _prompt):
            return None
        inspection = self._inspect()
        if inspection is None or not self._reproducible_gap(inspection):
            return None
        self._tool_class = inspection.tool_class
        self._before_successes = self._measure_bounded_reads(inspection.tool_class)
        return RepairProposal(
            proposal_id=_PROPOSAL_ID,
            objective=diagnosis.objective,
            files={
                _READ_TOOL_SOURCE: self._derived_tool_source(inspection),
                _BOOTSTRAP_SOURCE: self._derived_bootstrap(inspection),
                _READ_TOOL_TEST: self._derived_tests(inspection),
            },
            focused_tests=(_READ_TOOL_TEST,),
            metric_directions={_METRIC: "increase"},
        )

    def validator(self, _proposal: RepairProposal) -> RepairValidation:
        before = self._before_successes
        if before is None:
            return RepairValidation(False, detail="No existe una medicion previa confiable.")
        after = self._measure_bounded_reads(self._tool_class)
        basetemp = Path(tempfile.gettempdir()) / "atlas-supervised-pytest"
        tests = self._run("-m", "pytest", "-q", _READ_TOOL_TEST, "--basetemp", str(basetemp))
        compiled = self._run("-m", "py_compile", _READ_TOOL_SOURCE, _BOOTSTRAP_SOURCE)
        diff_checked = self._git("diff", "--check")
        passed = after > before and all(result.returncode == 0 for result in (tests, compiled, diff_checked))
        detail = "tests focales, py_compile y git diff --check correctos." if passed else self._validation_error(tests, compiled, diff_checked)
        return RepairValidation(passed, {_METRIC: float(before)}, {_METRIC: float(after)}, detail)

    def _observed_finding(self) -> str:
        inspection = self._inspect()
        if inspection is None:
            return "Inspeccion real: el estado actual de la tool de lectura no es reconocible; no se puede derivar un cambio seguro."
        if not self._reproducible_gap(inspection):
            return "Inspeccion real: la lectura acotada ya existe o el estado no es reconocible; no hay un gap reproducible que cubrir."
        return (
            f"Inspeccion real de {_READ_TOOL_SOURCE}: la tool devuelve '{inspection.read_call}' y su registro declara "
            f"{inspection.parameter_count} parametro(s); no existe todavia un 'limit', asi que el gap es reproducible."
        )

    def _inspect(self) -> _ReadToolInspection | None:
        try:
            tool_source = (self._root / _READ_TOOL_SOURCE).read_text(encoding="utf-8")
            bootstrap_source = (self._root / _BOOTSTRAP_SOURCE).read_text(encoding="utf-8")
            test_path = self._root / _READ_TOOL_TEST
            tests_source = test_path.read_text(encoding="utf-8") if test_path.exists() else ""
        except OSError:
            return None
        read_matches = list(_READ_RETURN.finditer(tool_source))
        class_matches = list(_TOOL_CLASS.finditer(tool_source))
        registrations = list(_REGISTRATION.finditer(bootstrap_source))
        if len(read_matches) != 1 or len(class_matches) != 1 or len(registrations) != 1:
            return None
        registration = registrations[0]
        parameter_lines = registration.group("parameters")
        if not parameter_lines:
            return None
        parameter_count = len(_PARAMETER_LINE.findall(parameter_lines))
        if parameter_count > 4:
            return None
        return _ReadToolInspection(
            tool_source=tool_source,
            bootstrap_source=bootstrap_source,
            tests_source=tests_source,
            tool_class=class_matches[0].group("name"),
            read_call=read_matches[0].group("call"),
            read_indent=read_matches[0].group("indent"),
            parameter_indent=_PARAMETER_LINE.match(parameter_lines).group("indent"),
            parameter_lines=parameter_lines,
            parameter_count=parameter_count,
        )

    @staticmethod
    def _reproducible_gap(inspection: _ReadToolInspection) -> bool:
        return (
            not _LIMIT_MENTION.search(inspection.tool_source)
            and not _LIMIT_MENTION.search(inspection.parameter_lines)
            and not _LIMIT_MENTION.search(inspection.tests_source)
            and 'ToolParameterSchema("path"' in inspection.parameter_lines
            and "context.parameters.get(\"path\")" in inspection.tool_source
        )

    @staticmethod
    def _derived_tool_source(inspection: _ReadToolInspection) -> str:
        match = _READ_RETURN.search(inspection.tool_source)
        indent, call = match.group("indent"), match.group("call")
        block = (
            f"{indent}content = {call}\n"
            f"{indent}\n"
            f"{indent}limit = context.parameters.get(\"limit\")\n"
            f"{indent}\n"
            f"{indent}if limit is None:\n"
            f"{indent}    return content\n"
            f"{indent}\n"
            f"{indent}if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:\n"
            f"{indent}    raise ValueError(\"Parameter 'limit' must be a positive integer.\")\n"
            f"{indent}\n"
            f"{indent}return \"\\n\".join(content.splitlines()[:limit])"
        )
        start, end = match.span()
        return inspection.tool_source[:start] + block + inspection.tool_source[end:]

    @staticmethod
    def _derived_bootstrap(inspection: _ReadToolInspection) -> str:
        match = _REGISTRATION.search(inspection.bootstrap_source)
        limit_line = f"{inspection.parameter_indent}ToolParameterSchema(\"limit\", int, minimum=1),\n"
        start, end = match.span("parameters")
        return inspection.bootstrap_source[:start] + inspection.parameter_lines + limit_line + inspection.bootstrap_source[end:]

    @staticmethod
    def _derived_tests(inspection: _ReadToolInspection) -> str:
        cls = inspection.tool_class
        header = (
            f'"""Focused tests for the bounded read variant of {cls}."""\n'
            "\n"
            "import pytest\n"
            "\n"
            f"from tools.filesystem.read_file_tool import {cls}\n"
            "from tools.tool_context import ToolContext\n"
        )
        body = (
            "\n"
            "\n"
            "def _tool(tmp_path, content=\"linea1\\nlinea2\\nlinea3\\n\"):\n"
            "    path = tmp_path / \"notas.txt\"\n"
            "    path.write_text(content, encoding=\"utf-8\")\n"
            f"    return {cls}(), str(path)\n"
            "\n"
            "\n"
            "def test_read_without_limit_returns_whole_file(tmp_path) -> None:\n"
            "    tool, path = _tool(tmp_path)\n"
            "\n"
            "    assert tool.execute(ToolContext(parameters={\"path\": path})) == \"linea1\\nlinea2\\nlinea3\\n\"\n"
            "\n"
            "\n"
            "def test_read_with_limit_returns_first_lines(tmp_path) -> None:\n"
            "    tool, path = _tool(tmp_path)\n"
            "\n"
            "    assert tool.execute(ToolContext(parameters={\"path\": path, \"limit\": 2})) == \"linea1\\nlinea2\"\n"
            "\n"
            "\n"
            "@pytest.mark.parametrize(\"limit\", [0, -1, \"2\", True])\n"
            "def test_read_rejects_invalid_limit(tmp_path, limit) -> None:\n"
            "    tool, path = _tool(tmp_path)\n"
            "\n"
            "    with pytest.raises(ValueError):\n"
            "        tool.execute(ToolContext(parameters={\"path\": path, \"limit\": limit}))\n"
        )
        return header + body

    def _measure_bounded_reads(self, tool_class: str) -> int:
        """Count bounded-read successes in a fresh interpreter."""
        script = (
            "import os\n"
            "import tempfile\n"
            f"from tools.filesystem.read_file_tool import {tool_class}\n"
            "from tools.tool_context import ToolContext\n"
            "descriptor, path = tempfile.mkstemp(suffix='.txt')\n"
            "with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:\n"
            "    handle.write('linea1\\nlinea2\\nlinea3\\n')\n"
            "try:\n"
            f"    result = {tool_class}().execute(ToolContext(parameters={{'path': path, 'limit': 2}}))\n"
            "    print(1 if result == 'linea1\\nlinea2' else 0)\n"
            "finally:\n"
            "    os.unlink(path)\n"
        )
        result = subprocess.run((sys.executable, "-c", script), cwd=self._root, capture_output=True, text=True, check=False)
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

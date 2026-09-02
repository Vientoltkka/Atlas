"""Deterministic supervised capability improvement for Control PC (desktop).

The concrete patch is not stored here. It is derived from the real current
state of the desktop use case, its tool, its controller and its tests: the
builder inspects the code, confirms the gap is reproducible and builds one
exact proposal from what it observed. Unexpected state yields no proposal.
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


_USE_CASE_SOURCE = "use_cases/desktop_interaction.py"
_TOOL_SOURCE = "tools/desktop/desktop_tools.py"
_CONTROLLER_SOURCE = "tools/desktop/windows_controller.py"
_TEST_SOURCE = "tests/test_desktop_interaction.py"
_SCOPE = (_USE_CASE_SOURCE, _TOOL_SOURCE, _CONTROLLER_SOURCE, _TEST_SOURCE)
_DESKTOP_TERMS = ("control pc", "desktop", "escritorio")
_UNSAFE_TERMS = (
    "powershell",
    "cmd",
    "shell",
    "ejecuta",
    "ejecutar",
    "comando",
    "comandos",
    "registro",
    "privileg",
    "administrador",
    "instala",
    "internet",
    "descarga",
    "borra",
    "elimina",
    "cierra",
)
_PROPOSAL_ID = "improvement.desktop.open-file-with-application"
_METRIC = "aperturas_de_archivo_con_aplicacion_correctas"
_OPEN_APPS_PROPOSAL_ID = "improvement.desktop.open-applications-by-name"
_OPEN_APPS_METRIC = "aperturas_de_aplicaciones_conocidas_por_nombre"
_OPEN_APPS_SCOPE = (_USE_CASE_SOURCE, _TEST_SOURCE)
_OPEN_APPS_TERMS = ("aplicaciones conocidas", "aplicacion conocida", "abrir aplicaciones", "abre aplicaciones")
_OPEN_APPS_ALIAS_ENTRY = '"la calculadora"'
_OPEN_APPS_TEST_MARKER = "abre la calculadora"
_OPEN_APPLICATION_PRIMITIVE = '"desktop.open_application"'
_CONTROLLER_CALC_ENTRY = '"calculadora": ("calc",)'
_CONTROLLER_VSCODE_ENTRY = '"vs code": ('
_TOOL_APPLICATION_ACCESS = 'context.parameters.get("application")'
_CONTROLLER_SIGNATURE = "def open_file(self, path: Path, application: str | None = None) -> None:"

_OPEN_FILE_CALL = re.compile(
    r"(?m)^(?P<i>[ \t]+)return self\._run\(\n"
    r"(?P=i)    \"desktop\.open_file\",\n"
    r"(?P=i)    \{\"path\": str\(path\)\},\n"
    r"(?P=i)    f\"Abriendo \{path\}\.\",\n"
    r"(?P=i)\)"
)
_TARGET_PREP = re.compile(
    r"(?m)^(?P<i>[ \t]+)target = self\._clean_open_target\(target\)\n"
    r"(?P=i)target = self\._resolve_application_alias\(target\)$"
)
_ALIAS_METHOD = re.compile(
    r"(?ms)^(?P<i>[ \t]+)def _resolve_application_alias\(.*?^\1[ \t]+return aliases\.get\(self\._normalize\(target\), target\)$"
)
_ALIAS_PAIR = re.compile(r'"(?P<alias>[^"\n]+)":\s*"(?P<value>[^"\n]+)"')
_OPEN_WITH_APPLICATION_MARKER = "_open_file_application"
_TESTS_COVER_GAP_MARKER = "con el bloc de notas"
_ALIASES_BLOCK = re.compile(
    r"(?m)^(?P<i>[ \t]+)aliases = \{\n"
    r"(?P=i)    \"bloc de notas\": \"notepad\",\n"
    r"(?P=i)    \"el bloc de notas\": \"notepad\",\n"
    r"(?P=i)\}\n"
    r"(?P=i)return aliases\.get\(self\._normalize\(target\), target\)$"
)


@dataclass(frozen=True, slots=True)
class _DesktopInspection:
    """Read-only snapshot of the real Control PC state, used to derive one proposal."""

    use_case_source: str
    tool_source: str
    controller_source: str
    tests_source: str


class DesktopCapabilityImprovementBuilder:
    """Propose exactly one reviewed open-with-application improvement derived from the real code."""

    def __init__(self, project_root: Path) -> None:
        self._root = project_root.resolve()
        self._before_successes: int | None = None
        self._before_open_apps_successes: int | None = None

    def diagnose(self, prompt: str) -> ImprovementDiagnosis | None:
        text = normalize_prompt(prompt)
        if not any(term in text for term in _DESKTOP_TERMS):
            return None
        if any(term in text for term in _OPEN_APPS_TERMS):
            return ImprovementDiagnosis(
                ImprovementClassification.CAPABILITY_IMPROVEMENT,
                "Abrir aplicaciones conocidas por nombre (bloc de notas, calculadora, chrome y vs code), incluidas sus formas con articulo, sin romper las funciones actuales.",
                _OPEN_APPS_SCOPE,
                (_TEST_SOURCE,),
                ("aperturas de aplicaciones conocidas por nombre",),
                "Anade entradas explicitas de nombres conocidos a la resolucion de alias de apertura; reutiliza desktop.open_application, sin shell arbitrario ni comandos libres.",
                self._observed_open_apps_finding(),
            )
        return ImprovementDiagnosis(
            ImprovementClassification.CAPABILITY_IMPROVEMENT,
            "Abrir archivos con una aplicacion conocida (variante 'con <aplicacion>') sin romper las funciones actuales.",
            _SCOPE,
            (_TEST_SOURCE,),
            ("aperturas de archivo con aplicacion correctas",),
            "Anade una variante opcional de apertura local; no toca secretos, dependencias, registro, red ni ejecucion arbitraria.",
            self._observed_finding(),
        )

    def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
        return (
            diagnosis.classification is ImprovementClassification.CAPABILITY_IMPROVEMENT
            and diagnosis.scope in (_SCOPE, _OPEN_APPS_SCOPE)
        )

    def build(self, diagnosis: ImprovementDiagnosis, prompt: str) -> RepairProposal | None:
        if not self.can_handle(diagnosis, prompt):
            return None
        if any(term in normalize_prompt(prompt) for term in _UNSAFE_TERMS):
            return None
        inspection = self._inspect()
        if inspection is None:
            return None
        if diagnosis.scope == _OPEN_APPS_SCOPE:
            return self._build_open_applications(diagnosis, inspection)
        if not self._reproducible_gap(inspection):
            return None
        use_case = self._derived_use_case(inspection)
        if use_case is None:
            return None
        self._before_successes = self._measure_open_with_application()
        return RepairProposal(
            proposal_id=_PROPOSAL_ID,
            objective=diagnosis.objective,
            files={
                _USE_CASE_SOURCE: use_case,
                _TEST_SOURCE: self._derived_tests(inspection),
            },
            focused_tests=(_TEST_SOURCE,),
            metric_directions={_METRIC: "increase"},
        )

    def validator(self, proposal: RepairProposal) -> RepairValidation:
        if proposal.proposal_id == _OPEN_APPS_PROPOSAL_ID:
            before, metric = self._before_open_apps_successes, _OPEN_APPS_METRIC
        else:
            before, metric = self._before_successes, _METRIC
        if before is None:
            return RepairValidation(False, detail="No existe una medicion previa confiable.")
        after = self._measure_open_applications_by_name() if metric == _OPEN_APPS_METRIC else self._measure_open_with_application()
        basetemp = Path(tempfile.gettempdir()) / "atlas-supervised-pytest"
        tests = self._run("-m", "pytest", "-q", _TEST_SOURCE, "--basetemp", str(basetemp))
        compiled = self._run("-m", "py_compile", _USE_CASE_SOURCE)
        diff_checked = self._git("diff", "--check")
        passed = after > before and all(result.returncode == 0 for result in (tests, compiled, diff_checked))
        detail = "tests focales, py_compile y git diff --check correctos." if passed else self._validation_error(tests, compiled, diff_checked)
        return RepairValidation(passed, {metric: float(before)}, {metric: float(after)}, detail)

    def _observed_finding(self) -> str:
        inspection = self._inspect()
        if inspection is None:
            return "Inspeccion real: el estado actual de Control PC no es reconocible; no se puede derivar un cambio seguro."
        if not self._reproducible_gap(inspection):
            return "Inspeccion real: la variante 'con aplicacion' ya existe o el estado no es reconocible; no hay un gap reproducible que cubrir."
        return (
            f"Inspeccion real de {_USE_CASE_SOURCE}: la llamada desktop.open_file solo recibe 'path' mientras la tool y el "
            "controlador ya aceptan 'application'; abrir un archivo 'con el bloc de notas' no esta soportado, asi que el gap es reproducible."
        )

    def _observed_open_apps_finding(self) -> str:
        inspection = self._inspect()
        if inspection is None:
            return "Inspeccion real: el estado actual de Control PC no es reconocible; no se puede derivar un cambio seguro."
        if not self._reproducible_open_apps_gap(inspection):
            return "Inspeccion real: las aplicaciones conocidas por nombre ya se resuelven o el estado no es reconocible; no hay un gap reproducible que cubrir."
        return (
            f"Inspeccion real de {_USE_CASE_SOURCE}: la primitiva segura desktop.open_application y el whitelist del controlador "
            "ya existen, pero 'abre la calculadora' llega con el articulo incluido ('la calculadora') y el controlador no lo reconoce; "
            "el whitelist de nombres conocidos (bloc de notas, calculadora, chrome, vs code) falta en la resolucion de alias, asi que el gap es reproducible."
        )

    def _inspect(self) -> _DesktopInspection | None:
        try:
            use_case_source = (self._root / _USE_CASE_SOURCE).read_text(encoding="utf-8")
            tool_source = (self._root / _TOOL_SOURCE).read_text(encoding="utf-8")
            controller_source = (self._root / _CONTROLLER_SOURCE).read_text(encoding="utf-8")
            test_path = self._root / _TEST_SOURCE
            tests_source = test_path.read_text(encoding="utf-8") if test_path.exists() else ""
        except OSError:
            return None
        if _OPEN_FILE_CALL.search(use_case_source) is None:
            return None
        return _DesktopInspection(
            use_case_source=use_case_source,
            tool_source=tool_source,
            controller_source=controller_source,
            tests_source=tests_source,
        )

    @staticmethod
    def _reproducible_gap(inspection: _DesktopInspection) -> bool:
        return (
            _OPEN_WITH_APPLICATION_MARKER not in inspection.use_case_source
            and _TESTS_COVER_GAP_MARKER not in inspection.tests_source
            and "class FakeToolExecutor" in inspection.tests_source
            and "DesktopInteractionUseCase" in inspection.tests_source
            and _TOOL_APPLICATION_ACCESS in inspection.tool_source
            and _CONTROLLER_SIGNATURE in inspection.controller_source
            and _TARGET_PREP.search(inspection.use_case_source) is not None
            and _ALIAS_METHOD.search(inspection.use_case_source) is not None
        )

    @staticmethod
    def _reproducible_open_apps_gap(inspection: _DesktopInspection) -> bool:
        return (
            _OPEN_APPS_ALIAS_ENTRY not in inspection.use_case_source
            and _OPEN_APPS_TEST_MARKER not in inspection.tests_source
            and _ALIASES_BLOCK.search(inspection.use_case_source) is not None
            and _OPEN_APPLICATION_PRIMITIVE in inspection.use_case_source
            and _CONTROLLER_CALC_ENTRY in inspection.controller_source
            and _CONTROLLER_VSCODE_ENTRY in inspection.controller_source
            and "class FakeToolExecutor" in inspection.tests_source
            and "DesktopInteractionUseCase" in inspection.tests_source
        )

    def _build_open_applications(self, diagnosis: ImprovementDiagnosis, inspection: _DesktopInspection) -> RepairProposal | None:
        if not self._reproducible_open_apps_gap(inspection):
            return None
        use_case = self._derived_open_apps_use_case(inspection)
        if use_case is None:
            return None
        self._before_open_apps_successes = self._measure_open_applications_by_name()
        return RepairProposal(
            proposal_id=_OPEN_APPS_PROPOSAL_ID,
            objective=diagnosis.objective,
            files={
                _USE_CASE_SOURCE: use_case,
                _TEST_SOURCE: self._derived_open_apps_tests(inspection),
            },
            focused_tests=(_TEST_SOURCE,),
            metric_directions={_OPEN_APPS_METRIC: "increase"},
        )

    @staticmethod
    def _derived_use_case(inspection: _DesktopInspection) -> str | None:
        prep = _TARGET_PREP.search(inspection.use_case_source)
        call = _OPEN_FILE_CALL.search(inspection.use_case_source)
        alias = _ALIAS_METHOD.search(inspection.use_case_source)
        if prep is None or call is None or alias is None:
            return None
        indent, call_indent = prep.group("i"), call.group("i")
        prep_replacement = (
            f"{indent}target = self._clean_open_target(target)\n"
            f"{indent}target, application = self.{_OPEN_WITH_APPLICATION_MARKER}(target)\n"
            f"{indent}target = self._resolve_application_alias(target)"
        )
        call_replacement = (
            f"{call_indent}if application is not None:\n"
            f"{call_indent}    return self._run(\n"
            f"{call_indent}        \"desktop.open_file\",\n"
            f"{call_indent}        {{\"path\": str(path), \"application\": application}},\n"
            f"{call_indent}        f\"Abriendo {{path}} con {{application}}.\",\n"
            f"{call_indent}    )\n"
            + call.group(0)
        )
        helper = (
            "\n\n"
            f"{alias.group('i')}def {_OPEN_WITH_APPLICATION_MARKER}(\n"
            f"{alias.group('i')}    self,\n"
            f"{alias.group('i')}    target: str,\n"
            f"{alias.group('i')}) -> tuple[str, str | None]:\n"
            f"{alias.group('i')}    \"\"\"Split one explicit \"con/with <aplicacion>\" suffix using known aliases only.\"\"\"\n"
            f"{alias.group('i')}    stripped = target.strip()\n"
            f"{alias.group('i')}    match = re.search(r\"\\s+(?:con|with)\\s+(.+)$\", stripped, re.IGNORECASE)\n"
            f"{alias.group('i')}    if match is None:\n"
            f"{alias.group('i')}        return target, None\n"
            f"{alias.group('i')}    requested = match.group(1).strip()\n"
            f"{alias.group('i')}    application = self._resolve_application_alias(requested)\n"
            f"{alias.group('i')}    if self._normalize(application) == self._normalize(requested):\n"
            f"{alias.group('i')}        return target, None\n"
            f"{alias.group('i')}    return stripped[: match.start()].strip(), application"
        )
        source = inspection.use_case_source
        insertion = alias.end()
        source = source[:insertion] + helper + source[insertion:]
        source = source.replace(prep.group(0), prep_replacement, 1)
        source = source.replace(call.group(0), call_replacement, 1)
        return source

    @staticmethod
    def _derived_tests(inspection: _DesktopInspection) -> str:
        block = (
            "\n\n\n"
            "def test_desktop_interaction_opens_file_with_application_alias(tmp_path: Path) -> None:\n"
            "    executor = FakeToolExecutor()\n"
            "    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)\n"
            "    file = tmp_path / \"notas.txt\"\n"
            "    file.write_text(\"demo\", encoding=\"utf-8\")\n"
            "\n"
            "    result = use_case.execute(f\"abre {file} con el bloc de notas\")\n"
            "\n"
            "    assert result == f\"\\u2713 Abriendo {file} con notepad.\"\n"
            "    assert executor.calls[0][0] == \"desktop.open_file\"\n"
            "    assert executor.calls[0][1].parameters == {\"path\": str(file), \"application\": \"notepad\"}\n"
            "\n"
            "\n"
            "def test_desktop_interaction_without_application_keeps_previous_behavior(tmp_path: Path) -> None:\n"
            "    executor = FakeToolExecutor()\n"
            "    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)\n"
            "    file = tmp_path / \"notas.txt\"\n"
            "    file.write_text(\"demo\", encoding=\"utf-8\")\n"
            "\n"
            "    use_case.execute(f\"abre {file}\")\n"
            "\n"
            "    assert executor.calls[0][0] == \"desktop.open_file\"\n"
            "    assert executor.calls[0][1].parameters == {\"path\": str(file)}\n"
        )
        return inspection.tests_source + block

    @staticmethod
    def _derived_open_apps_use_case(inspection: _DesktopInspection) -> str | None:
        block = _ALIASES_BLOCK.search(inspection.use_case_source)
        if block is None:
            return None
        indent = block.group("i")
        entry = indent + "    "
        replacement = (
            f"{indent}aliases = {{\n"
            f"{entry}\"bloc de notas\": \"notepad\",\n"
            f"{entry}\"el bloc de notas\": \"notepad\",\n"
            f"{entry}\"calculadora\": \"calculadora\",\n"
            f"{entry}\"la calculadora\": \"calculadora\",\n"
            f"{entry}\"chrome\": \"chrome\",\n"
            f"{entry}\"el chrome\": \"chrome\",\n"
            f"{entry}\"vs code\": \"vs code\",\n"
            f"{entry}\"el vs code\": \"vs code\",\n"
            f"{indent}}}\n"
            f"{indent}return aliases.get(self._normalize(target), target)"
        )
        return inspection.use_case_source.replace(block.group(0), replacement, 1)

    @staticmethod
    def _derived_open_apps_tests(inspection: _DesktopInspection) -> str:
        block = (
            "\n\n\n"
            "def test_desktop_interaction_opens_known_application_by_name_with_article(tmp_path: Path) -> None:\n"
            "    executor = FakeToolExecutor()\n"
            "    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)\n"
            "\n"
            "    use_case.execute(\"abre la calculadora\")\n"
            "\n"
            "    assert executor.calls[0][0] == \"desktop.open_application\"\n"
            "    assert executor.calls[0][1].parameters == {\"application\": \"calculadora\"}\n"
            "\n"
            "\n"
            "def test_desktop_interaction_opens_vs_code_by_name_with_article(tmp_path: Path) -> None:\n"
            "    executor = FakeToolExecutor()\n"
            "    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)\n"
            "\n"
            "    use_case.execute(\"abre el vs code\")\n"
            "\n"
            "    assert executor.calls[0][0] == \"desktop.open_application\"\n"
            "    assert executor.calls[0][1].parameters == {\"application\": \"vs code\"}\n"
        )
        return inspection.tests_source + block

    def _measure_open_applications_by_name(self) -> int:
        """Count known-application-by-name opening successes in a fresh interpreter."""
        script = (
            "from pathlib import Path\n"
            "from use_cases.desktop_interaction import DesktopInteractionUseCase\n"
            "\n"
            "class _Executor:\n"
            "    def __init__(self):\n"
            "        self.calls = []\n"
            "\n"
            "    def requires_explicit_authorization(self, tool_name):\n"
            "        return False\n"
            "\n"
            "    def execute(self, tool_name, context):\n"
            "        self.calls.append((tool_name, dict(context.parameters)))\n"
            "        return \"ok\"\n"
            "\n"
            "executor = _Executor()\n"
            f"use_case = DesktopInteractionUseCase(executor, project_root=Path({str(self._root)!r}))\n"
            "use_case.execute('abre la calculadora')\n"
            "ok = bool(executor.calls)\n"
            "ok = ok and executor.calls[0][0] == 'desktop.open_application'\n"
            "ok = ok and executor.calls[0][1].get('application') == 'calculadora'\n"
            "print(1 if ok else 0)\n"
        )
        result = subprocess.run((sys.executable, "-c", script), cwd=self._root, capture_output=True, text=True, check=False)
        try:
            return int(result.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            return 0

    def _measure_open_with_application(self) -> int:
        """Count open-with-application successes in a fresh interpreter."""
        script = (
            "import os\n"
            "import tempfile\n"
            "from pathlib import Path\n"
            "from use_cases.desktop_interaction import DesktopInteractionUseCase\n"
            "from tools.tool_context import ToolContext\n"
            "\n"
            "class _Executor:\n"
            "    def __init__(self):\n"
            "        self.calls = []\n"
            "\n"
            "    def requires_explicit_authorization(self, tool_name):\n"
            "        return False\n"
            "\n"
            "    def execute(self, tool_name, context):\n"
            "        self.calls.append((tool_name, dict(context.parameters)))\n"
            "        return \"ok\"\n"
            "\n"
            "descriptor, path = tempfile.mkstemp(suffix='.txt')\n"
            "os.close(descriptor)\n"
            "try:\n"
            "    executor = _Executor()\n"
            f"    use_case = DesktopInteractionUseCase(executor, project_root=Path({str(self._root)!r}))\n"
            "    use_case.execute('abre ' + path + ' con el bloc de notas')\n"
            "    ok = bool(executor.calls)\n"
            "    ok = ok and executor.calls[0][0] == 'desktop.open_file'\n"
            "    ok = ok and executor.calls[0][1].get('application') == 'notepad'\n"
            "    print(1 if ok else 0)\n"
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

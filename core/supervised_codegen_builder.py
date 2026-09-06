"""Single generic, approval-gated codegen producer for unowned capabilities.

Specific builders keep priority. When no specific builder recognizes a
CAPABILITY_IMPROVEMENT request, this builder reuses the existing CodingAgent
persona and PromptClient to synthesize exactly one bounded RepairProposal:
zero writes, whitelisted scope, hard file/size limits, mandatory focused
tests and a host-side validator. The model never decides whether its own
tests pass; the host executes and decides. Rollback stays native via
SupervisedRepairWorkflow.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import os
import re
import subprocess
import sys
import tempfile

from core.self_improvement_conversation import (
    ImprovementClassification,
    ImprovementDiagnosis,
    normalize_prompt,
)
from core.supervised_repair import RepairProposal, RepairValidation


_SCOPE = ("use_cases/", "skills/builtin/", "tests/", "bootstrap/skill_system.py")
_ALLOWED_PREFIXES = ("use_cases/", "tests/", "skills/builtin/")
_ALLOWED_FILES = frozenset({"bootstrap/skill_system.py"})
_FORBIDDEN_FILES = frozenset(
    {
        "core/supervised_repair.py",
        "core/self_improvement_conversation.py",
        "bootstrap/bootstrap.py",
    }
)
_FORBIDDEN_PREFIXES = ("core/", "scheduler/", "security/", ".git/")
_DEFAULT_MAX_FILES = 4
_DEFAULT_MAX_TOTAL_CHARS = 40_000
_METRIC = "pruebas_focales_superadas"
_CODEGEN_MODEL_ENV = "ATLAS_CODEGEN_MODEL"
# Empty model delegates the choice to the configured provider's own default
# model via PromptClient, so no local model name is hardcoded here.
_DEFAULT_CODEGEN_MODEL = ""

_TRIGGER = re.compile(r"\bcrea (?:la|una) capacidad(?:es)? para (?P<description>\S.*)$")
_UNSAFE = re.compile(
    r"\b(?:env|secretos?|passwords?|tokens?|credenciales?|apis?|internet|online|"
    r"descargas?|descargar|instal(?:a|ar|acion)|dependencias?|shells?|comandos?|"
    r"powershell|cmd|borrar?|elimin(?:a|ar)|commits?|push|contrasenas?|claves?|"
    r"red|nube|cloud|git|pip)\b"
)
_EXTERNAL_TERMS = ("proveedor", "infraestructura", "integracion")
_SPECIFIC_DOMAIN_TERMS = (
    "voz",
    "voice",
    "control pc",
    "desktop",
    "escritorio",
    "routing",
    "router",
    "rutas",
    "archivo",
    "archivos",
    "fichero",
    "ficheros",
    "skill",
    "temperatura",
    "celsius",
    "fahrenheit",
)

_FILE_HEADER = re.compile(r"^={2,}\s*FILE:\s*(?P<path>\S+?)(?:\s*=+)?\s*$", re.IGNORECASE)
_FILE_END = re.compile(r"^={2,}\s*END(?:\s+FILE)?(?:\s*=+)?\s*$", re.IGNORECASE)
_FENCE = re.compile(r"^`{3,}")
_PASSED = re.compile(r"(\d+) passed")

_COMPILE_SCRIPT = (
    "import sys\n"
    "for path in sys.argv[1:]:\n"
    "    with open(path, encoding='utf-8') as handle:\n"
    "        compile(handle.read(), path, 'exec')\n"
)

_USER_PROMPT_TEMPLATE = """
Objetivo del usuario:
{prompt}

Sintetiza UNA nueva capacidad local, offline y determinista para Atlas.

Reglas obligatorias:
- Responde EXCLUSIVAMENTE con bloques de archivos en este formato exacto:

=== FILE: <ruta/relativa>
<contenido completo del archivo>
=== END FILE

- Rutas permitidas unicamente: use_cases/, tests/, skills/builtin/ y bootstrap/skill_system.py.
- Maximo {max_files} archivos y {max_total_chars} caracteres en total.
- Implementa la logica en un modulo puro bajo use_cases/ (sin E/S de red, sin APIs externas, sin dependencias nuevas, sin aleatoriedad, sin reloj del sistema, sin variables de entorno).
- Si la capacidad debe quedar registrada como skill local, anade el handler en bootstrap/skill_system.py (dentro de build_builtin_skill_handler_registry) y su manifiesto declarativo skills/builtin/<skill_id>/skill.json; si no hace falta registrarla, omitelos.
- Incluye pruebas focalizadas deterministas en tests/ (sin red, sin azar) que verifiquen la salida determinista.
- No uses markdown ni bloques de codigo: solo los bloques de archivo, sin ningun texto adicional antes o despues.
"""


class SupervisedCodegenCapabilityBuilder:
    """Synthesize one bounded proposal for a capability no specific builder owns."""

    def __init__(
        self,
        project_root: Path,
        prompt_client: object | None = None,
        *,
        model: str | None = None,
        max_files: int = _DEFAULT_MAX_FILES,
        max_total_chars: int = _DEFAULT_MAX_TOTAL_CHARS,
    ) -> None:
        self._root = project_root.resolve()
        self._prompt_client = prompt_client
        self._model = model or os.getenv(_CODEGEN_MODEL_ENV, "").strip() or _DEFAULT_CODEGEN_MODEL
        self._max_files = max_files
        self._max_total_chars = max_total_chars
        self._before_passed: int | None = None

    @property
    def scope(self) -> tuple[str, ...]:
        return _SCOPE

    def diagnose(self, prompt: str) -> ImprovementDiagnosis | None:
        text = normalize_prompt(prompt)
        match = _TRIGGER.search(text)
        if match is None:
            return None
        if any(term in text for term in _SPECIFIC_DOMAIN_TERMS):
            return None
        if any(term in text for term in _EXTERNAL_TERMS) or _UNSAFE.search(text):
            return None
        description = " ".join(match.group("description").split())
        if len(re.findall(r"[a-z0-9]+", description)) < 3:
            return None
        return ImprovementDiagnosis(
            ImprovementClassification.CAPABILITY_IMPROVEMENT,
            f"Nueva capacidad local y determinista: {description.rstrip('.')}.",
            _SCOPE,
            (),
            (_METRIC,),
            "Anade una capacidad nueva acotada; el modelo solo sintetiza la propuesta, sin escrituras, y el host valida antes de conservar nada.",
            "Sin un builder especifico para esta capacidad; el Coding Agent sintetizara una propuesta acotada dentro de la whitelist y el host la validara antes de cualquier escritura.",
        )

    def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
        return diagnosis.classification is ImprovementClassification.CAPABILITY_IMPROVEMENT and diagnosis.scope == _SCOPE

    def build(self, diagnosis: ImprovementDiagnosis, prompt: str) -> RepairProposal | None:
        if not self.can_handle(diagnosis, prompt):
            return None
        files = self._synthesize(prompt)
        if files is None:
            return None
        focused_tests = tuple(sorted(path for path in files if path.startswith("tests/") and path.endswith(".py")))
        self._before_passed = self._pytest_passed(focused_tests)
        return RepairProposal(
            proposal_id=self._proposal_id(prompt),
            objective=diagnosis.objective,
            files=files,
            focused_tests=focused_tests,
            metric_directions={_METRIC: "increase"},
        )

    def validator(self, proposal: RepairProposal) -> RepairValidation:
        before = self._before_passed
        if before is None:
            return RepairValidation(False, detail="No existe una medicion previa confiable.")
        if not self._scope_respected(proposal):
            return RepairValidation(False, detail="La propuesta aplicada salio del alcance permitido; no se valida.")
        return_code, after = self._pytest_run(proposal.focused_tests)
        compiled = self._compile_ok([path for path in proposal.files if path.endswith(".py")])
        clean = self._git_clean()
        passed = return_code == 0 and after > before and compiled and clean
        failures = [
            label
            for label, ok in (
                ("tests focales", return_code == 0 and after > before),
                ("compilacion", compiled),
                ("git diff --check", clean),
            )
            if not ok
        ]
        detail = "tests focales, compilacion y git diff --check correctos." if passed else "Validacion fallida: " + ", ".join(failures) + "."
        return RepairValidation(passed, {_METRIC: float(before)}, {_METRIC: float(after)}, detail)

    def _synthesize(self, prompt: str) -> dict[str, str] | None:
        try:
            response = self._client.ask(model=self._model, messages=self._messages(prompt))
        except Exception:
            return None
        raw_files = self._parse_files(response)
        if raw_files is None:
            return None
        files: dict[str, str] = {}
        for raw_path, content in raw_files.items():
            relative = self._normalized_path(raw_path)
            if relative is None or relative in files or not self._in_scope(relative) or not content.strip():
                return None
            files[relative] = content
        if len(files) > self._max_files or sum(len(content) for content in files.values()) > self._max_total_chars:
            return None
        if not any(path.startswith("use_cases/") and path.endswith(".py") for path in files):
            return None
        if not any(path.startswith("tests/") and path.endswith(".py") for path in files):
            return None
        return files

    @property
    def _client(self) -> object:
        if self._prompt_client is None:
            from models.prompt_client import PromptClient

            self._prompt_client = PromptClient()
        return self._prompt_client

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        from agents.coding_agent import CodingAgent

        return [
            {"role": "system", "content": CodingAgent.SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(prompt=prompt, max_files=self._max_files, max_total_chars=self._max_total_chars)},
        ]

    @staticmethod
    def _parse_files(response: object) -> dict[str, str] | None:
        if not isinstance(response, str) or not response.strip():
            return None
        files: dict[str, str] = {}
        current: str | None = None
        lines: list[str] = []
        for raw_line in response.splitlines():
            stripped = raw_line.strip()
            if _FENCE.match(stripped):
                continue
            header = _FILE_HEADER.match(stripped)
            end = _FILE_END.match(stripped)
            if header is not None:
                if current is not None:
                    return None
                current, lines = header.group("path"), []
            elif end is not None:
                if current is None:
                    return None
                files[current] = "\n".join(lines).strip("\n") + "\n"
                current, lines = None, []
            elif current is not None:
                lines.append(raw_line)
        if current is not None or not files:
            return None
        return files

    @staticmethod
    def _normalized_path(raw_path: object) -> str | None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        candidate = Path(raw_path.strip().replace("\\", "/"))
        if candidate.is_absolute() or ".." in candidate.parts:
            return None
        return candidate.as_posix()

    @staticmethod
    def _in_scope(relative: str) -> bool:
        name = Path(relative).name
        if relative.endswith(".env") or ".env" in name:
            return False
        if relative in _FORBIDDEN_FILES or any(relative.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
            return False
        if relative in _ALLOWED_FILES:
            return True
        if not any(relative.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
            return False
        if Path(relative).suffix == ".py":
            return True
        return Path(relative).suffix == ".json" and relative.startswith("skills/builtin/")

    @staticmethod
    def _scope_respected(proposal: RepairProposal) -> bool:
        return all(SupervisedCodegenCapabilityBuilder._in_scope(relative) for relative in proposal.files)

    def _pytest_passed(self, focused_tests: tuple[str, ...]) -> int:
        return self._pytest_run(focused_tests)[1]

    def _pytest_run(self, focused_tests: tuple[str, ...]) -> tuple[int, int]:
        if not focused_tests:
            return 1, 0
        basetemp = Path(tempfile.gettempdir()) / "atlas-supervised-codegen"
        result = self._run(
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(basetemp),
            *focused_tests,
        )
        match = _PASSED.search(result.stdout)
        return result.returncode, int(match.group(1)) if match else 0

    def _compile_ok(self, paths: list[str]) -> bool:
        if not paths:
            return False
        return self._run(sys.executable, "-c", _COMPILE_SCRIPT, *paths).returncode == 0

    def _git_clean(self) -> bool:
        result = self._run("git", "diff", "--check")
        if result.returncode != 0 and "not a git repository" in (result.stderr + result.stdout).casefold():
            return True
        return result.returncode == 0

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(arguments, cwd=self._root, capture_output=True, text=True, check=False, env=environment)

    @staticmethod
    def _proposal_id(prompt: str) -> str:
        digest = hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()[:10]
        return f"improvement.codegen.{digest}"


def candidate_suffix(relative: str) -> str:
    return Path(relative).suffix

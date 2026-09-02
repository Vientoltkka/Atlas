from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from core.file_read_capability_builder import FileReadCapabilityImprovementBuilder
from core.self_improvement_conversation import (
    ImprovementClassification,
    ImprovementDiagnosis,
    SelfImprovementConversation,
    SupervisedRepairBuilderRegistry,
)
from core.supervised_repair import RepairProposal, RepairValidation

_ROOT = Path(__file__).resolve().parents[1]

_PROMPT = "Atlas, mejora tu capacidad para trabajar con archivos sin romper las funciones actuales."
_SCOPE = ("tools/filesystem/read_file_tool.py", "bootstrap/bootstrap.py", "tests/test_read_file_tool.py")


class _CapabilityImprovementFixtureBuilder:
    """Fixture builder exercising the supervised capability improvement path."""

    def __init__(self, *, passed: bool = True) -> None:
        self._passed = passed

    def diagnose(self, prompt: str) -> ImprovementDiagnosis | None:
        if "archivos" not in prompt:
            return None
        return ImprovementDiagnosis(
            ImprovementClassification.CAPABILITY_IMPROVEMENT,
            prompt,
            ("capability_source.txt", "tests/test_capability_fixture.py"),
            ("tests/test_capability_fixture.py",),
            ("capacidades_soportadas",),
            "fixture risk",
            "fixture finding",
        )

    def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
        return diagnosis.scope == ("capability_source.txt", "tests/test_capability_fixture.py")

    def build(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> RepairProposal | None:
        if not self.can_handle(diagnosis, _prompt):
            return None
        return RepairProposal(
            "improvement.fixture.bounded-read",
            diagnosis.objective,
            {"capability_source.txt": "improved\n"},
            ("tests/test_capability_fixture.py",),
            {"capacidades_soportadas": "increase"},
        )

    def validator(self, _proposal: RepairProposal) -> RepairValidation:
        return RepairValidation(self._passed, {"capacidades_soportadas": 1.0}, {"capacidades_soportadas": 2.0}, "fixture validated")


def _conversation(tmp_path: Path, *, passed: bool = True) -> SelfImprovementConversation:
    return SelfImprovementConversation(tmp_path, builders=(_CapabilityImprovementFixtureBuilder(passed=passed),))


def test_real_prompt_is_a_capability_improvement_request() -> None:
    assert SelfImprovementConversation.is_self_improvement_request(_PROMPT) is True


def test_real_prompt_is_classified_as_capability_improvement() -> None:
    diagnosis = SelfImprovementConversation(_ROOT).diagnose(_PROMPT)

    assert diagnosis.classification is ImprovementClassification.CAPABILITY_IMPROVEMENT
    assert diagnosis.scope == _SCOPE


def test_registry_resolves_the_file_read_builder() -> None:
    registry = SupervisedRepairBuilderRegistry((FileReadCapabilityImprovementBuilder(_ROOT),))

    diagnosis = FileReadCapabilityImprovementBuilder(_ROOT).diagnose(_PROMPT)

    assert diagnosis is not None
    assert registry.builder_for(diagnosis, _PROMPT) is not None


def test_proposal_is_concrete_with_zero_writes_before_authorization(tmp_path: Path) -> None:
    target = tmp_path / "capability_source.txt"
    target.write_text("original\n", encoding="utf-8")
    conversation = _conversation(tmp_path)

    response = conversation.handle(_PROMPT)

    assert "improvement.fixture.bounded-read" in response
    assert "mejora una capacidad" in response
    assert "No he modificado nada." in response
    assert target.read_text(encoding="utf-8") == "original\n"


def test_authorize_validate_and_accept_conserves_the_improvement(tmp_path: Path) -> None:
    target = tmp_path / "capability_source.txt"
    target.write_text("original\n", encoding="utf-8")
    conversation = _conversation(tmp_path)
    conversation.handle(_PROMPT)

    validated = conversation.handle("sí")

    assert "Antes/después: capacidades_soportadas: 1.0 -> 2.0" in validated
    assert target.read_text(encoding="utf-8") == "improved\n"
    assert conversation.handle("sí") == "Reparación aceptada. Se conserva el cambio validado."
    assert target.read_text(encoding="utf-8") == "improved\n"


def test_final_rejection_rolls_back_exact_scope(tmp_path: Path) -> None:
    target, unrelated = tmp_path / "capability_source.txt", tmp_path / "unrelated.txt"
    target.write_text("original\n", encoding="utf-8")
    unrelated.write_text("keep\n", encoding="utf-8")
    conversation = _conversation(tmp_path)
    conversation.handle(_PROMPT)
    conversation.handle("sí")

    assert "restauró exactamente el estado anterior" in conversation.handle("no")
    assert target.read_text(encoding="utf-8") == "original\n"
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


def test_failed_validation_rolls_back_exact_scope(tmp_path: Path) -> None:
    target = tmp_path / "capability_source.txt"
    target.write_text("original\n", encoding="utf-8")
    conversation = _conversation(tmp_path, passed=False)
    conversation.handle(_PROMPT)

    assert "se restauró exactamente" in conversation.handle("sí")
    assert target.read_text(encoding="utf-8") == "original\n"


def test_out_of_scope_improvement_is_blocked_before_any_write(tmp_path: Path) -> None:
    class _EscalatingBuilder(_CapabilityImprovementFixtureBuilder):
        def build(self, diagnosis: ImprovementDiagnosis, prompt: str) -> RepairProposal | None:
            return RepairProposal(
                "improvement.escalated",
                diagnosis.objective,
                {".env": "SECRET=1\n"},
                (),
                {"capacidades_soportadas": "increase"},
            )

    conversation = SelfImprovementConversation(tmp_path, builders=(_EscalatingBuilder(),))

    with pytest.raises(ValueError):
        conversation.handle(_PROMPT)
    assert not (tmp_path / ".env").exists()


def test_unknown_builder_for_improvement_stops_safely(tmp_path: Path) -> None:
    class _UnresolvableBuilder(_CapabilityImprovementFixtureBuilder):
        def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
            return False

    conversation = SelfImprovementConversation(tmp_path, builders=(_UnresolvableBuilder(),))

    response = conversation.handle(_PROMPT)

    assert response.startswith("CLARIFICATION_REQUIRED")
    assert conversation.proposal is None
    assert not conversation.active


def test_builder_prepares_the_bounded_read_proposal_without_writes(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    before = _project_snapshot(root)
    builder = FileReadCapabilityImprovementBuilder(root)
    diagnosis = builder.diagnose(_PROMPT)

    proposal = builder.build(diagnosis, _PROMPT)

    assert proposal is not None
    assert proposal.proposal_id == "improvement.read-file.bounded-limit-read"
    assert set(proposal.files) == set(_SCOPE)
    assert 'ToolParameterSchema("limit", int, minimum=1)' in proposal.files["bootstrap/bootstrap.py"]
    assert "test_read_with_limit_returns_first_lines" in proposal.files["tests/test_read_file_tool.py"]
    assert proposal.metric_directions == {"lecturas_de_archivo_acotadas_correctas": "increase"}
    assert _project_snapshot(root) == before
    assert not builder.can_handle(
        ImprovementDiagnosis(ImprovementClassification.CODE_REPAIR, "x", _SCOPE, (), (), "x", "x"),
        "x",
    )


def test_real_diagnosis_reports_the_observed_state_from_the_actual_code() -> None:
    diagnosis = FileReadCapabilityImprovementBuilder(_ROOT).diagnose(_PROMPT)

    assert "Inspeccion real" in diagnosis.finding
    assert "FileReadCapabilityImprovementBuilder" not in diagnosis.finding


def test_proposal_is_derived_from_the_inspected_source_not_from_a_constant(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    proposal = FileReadCapabilityImprovementBuilder(root).build(
        FileReadCapabilityImprovementBuilder(root).diagnose(_PROMPT), _PROMPT
    )

    assert proposal is not None
    assert "content = FileService.read(path)" in proposal.files["tools/filesystem/read_file_tool.py"]
    assert 'ToolParameterSchema("path", str, required=True),' in proposal.files["bootstrap/bootstrap.py"]
    assert "class ReadFileTool" in proposal.files["tools/filesystem/read_file_tool.py"].split("content =")[0]


_TOOL_FIXTURE = '''"""Read File Tool."""

from __future__ import annotations

from services.file_service import FileService
from tools.base_tool import BaseTool
from tools.tool_context import ToolContext


class ReadFileTool(BaseTool):
    """Read a UTF-8 text file."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read a UTF-8 text file."

    def semantic_metadata(self) -> dict[str, object]:
        """Return semantic metadata for catalog generation."""
        return {
            "capabilities": ["read_file"],
            "supported_intents": ["read a local file"],
            "input_description": "Requires a local text file path.",
            "output_description": "UTF-8 file content as text.",
            "risk_level": "low",
            "preconditions": ["path must exist", "path must point to a file"],
            "limitations": ["does not interpret file contents", "does not read remote paths"],
            "negative_examples": ["explain what a file is", "write new file content"],
            "compatible_tools": ["write_file"],
            "tags": ["filesystem", "read"],
            "positive_examples": ["lee el archivo notas.txt"],
            "category": "filesystem",
        }

    def execute(
        self,
        context: ToolContext,
    ) -> str:

        path = context.parameters.get("path")

        if not path:
            raise ValueError("Missing parameter 'path'.")

        return FileService.read(path)
'''

_BOOTSTRAP_FIXTURE = (
    "from tools.filesystem.read_file_tool import ReadFileTool\n"
    "from tools.tool_schema import ToolArgumentsSchema, ToolParameterSchema\n"
    "\n"
    "tool_registry.register(\n"
    "    ReadFileTool(),\n"
    "    arguments_schema=ToolArgumentsSchema(\n"
    "        parameters=(\n"
    '            ToolParameterSchema("path", str, required=True),\n'
    "        ),\n"
    "    ),\n"
    ")\n"
)


def _fixture_root(tmp_path: Path, *, tool_source: str | None = None, bootstrap_source: str = _BOOTSTRAP_FIXTURE) -> Path:
    root = tmp_path / "project"
    (root / "tools" / "filesystem").mkdir(parents=True)
    (root / "bootstrap").mkdir()
    (root / "tools" / "filesystem" / "read_file_tool.py").write_text(tool_source if tool_source is not None else _TOOL_FIXTURE, encoding="utf-8")
    (root / "bootstrap" / "bootstrap.py").write_text(bootstrap_source, encoding="utf-8")
    return root


def test_proposal_adapts_when_the_inspected_state_changes(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    tool_path = root / "tools" / "filesystem" / "read_file_tool.py"
    tool_path.write_text(tool_path.read_text(encoding="utf-8").replace("FileService.read(path)", "FileService.read(str(path))"), encoding="utf-8")
    snapshots = {path: path.read_text(encoding="utf-8") for path in (tool_path, root / "bootstrap" / "bootstrap.py")}

    builder = FileReadCapabilityImprovementBuilder(root)
    proposal = builder.build(builder.diagnose(_PROMPT), _PROMPT)

    assert proposal is not None
    tool_content = proposal.files["tools/filesystem/read_file_tool.py"]
    assert "content = FileService.read(str(path))" in tool_content
    assert "content = FileService.read(path)" not in tool_content
    for path, snapshot in snapshots.items():
        assert path.read_text(encoding="utf-8") == snapshot


def test_unexpected_states_yield_no_safe_proposal(tmp_path: Path) -> None:
    implemented = _TOOL_FIXTURE
    with_limit = implemented.replace(
        "        return FileService.read(path)",
        "        content = FileService.read(path)\n"
        "        limit = context.parameters.get(\"limit\")\n"
        "        if limit is not None:\n"
        "            return \"\\n\".join(content.splitlines()[:limit])\n"
        "        return content",
    )
    without_return = implemented.replace("        return FileService.read(path)", "        content = FileService.read(path)\n        return content")
    without_path_parameter = _BOOTSTRAP_FIXTURE.replace('ToolParameterSchema("path", str, required=True),', 'ToolParameterSchema("other", str, required=True),')

    for kwargs in ({"tool_source": with_limit}, {"tool_source": without_return}, {"bootstrap_source": without_path_parameter}):
        root = _fixture_root(tmp_path / f"case{len(list(tmp_path.iterdir()))}", **kwargs)
        builder = FileReadCapabilityImprovementBuilder(root)

        diagnosis = builder.diagnose(_PROMPT)
        assert diagnosis is not None
        assert "no hay un gap reproducible" in diagnosis.finding or "no es reconocible" in diagnosis.finding
        assert builder.build(diagnosis, _PROMPT) is None


def _project_snapshot(root: Path) -> list[tuple[str, bytes]]:
    return sorted(
        (str(path.relative_to(root)), path.read_bytes())
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def test_diagnosis_and_build_never_write_any_file(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    before = _project_snapshot(root)

    builder = FileReadCapabilityImprovementBuilder(root)
    diagnosis = builder.diagnose(_PROMPT)
    proposal = builder.build(diagnosis, _PROMPT)

    assert proposal is not None
    assert _project_snapshot(root) == before


def test_proposed_scope_never_exceeds_the_declared_maximum(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    builder = FileReadCapabilityImprovementBuilder(root)
    proposal = builder.build(builder.diagnose(_PROMPT), _PROMPT)

    assert proposal is not None
    assert set(proposal.files) <= set(_SCOPE)
    assert all(path.endswith(".py") for path in proposal.files)

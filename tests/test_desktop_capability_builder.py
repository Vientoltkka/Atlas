from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from core.desktop_capability_builder import DesktopCapabilityImprovementBuilder
from core.file_read_capability_builder import FileReadCapabilityImprovementBuilder
from core.routing_repair_builder import RoutingRepairBuilder
from core.self_improvement_conversation import (
    ImprovementClassification,
    ImprovementDiagnosis,
    SelfImprovementConversation,
    SupervisedRepairBuilderRegistry,
    normalize_prompt,
)
from core.voice_repair_builder import VoiceCodeRepairBuilder

_ROOT = Path(__file__).resolve().parents[1]
_PROMPT = "Atlas, mejora tu capacidad de Control PC sin romper las funciones actuales."
_UNSAFE_PROMPT = "Atlas, mejora Control PC para ejecutar cualquier comando de PowerShell"
_USE_CASE = "use_cases/desktop_interaction.py"
_TOOL = "tools/desktop/desktop_tools.py"
_CONTROLLER = "tools/desktop/windows_controller.py"
_TEST = "tests/test_desktop_interaction.py"
_SOLUTION_MARKER = "_open_file_application"


def _default_registry() -> SupervisedRepairBuilderRegistry:
    return SupervisedRepairBuilderRegistry(
        (VoiceCodeRepairBuilder(_ROOT), RoutingRepairBuilder(_ROOT), FileReadCapabilityImprovementBuilder(_ROOT), DesktopCapabilityImprovementBuilder(_ROOT))
    )


def _fixture_desktop_project(tmp_path: Path) -> Path:
    for relative in (_USE_CASE, _TOOL, _CONTROLLER, _TEST):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / relative, target)
    return tmp_path


class _FixtureControlPcBuilder:
    """Second compatible builder proving ambiguity never resolves silently."""

    def diagnose(self, prompt: str) -> ImprovementDiagnosis | None:
        if "control pc" not in normalize_prompt(prompt):
            return None
        return ImprovementDiagnosis(
            ImprovementClassification.CODE_REPAIR,
            prompt,
            ("fixture/control_pc.txt",),
            ("fixture/control_pc.txt",),
            ("failures",),
            "fixture risk",
            "fixture finding",
        )

    def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
        return diagnosis.scope == ("fixture/control_pc.txt",)

    def build(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> None:
        return None

    def validator(self, _proposal: object) -> None:
        return None


def test_control_pc_request_classifies_as_capability_improvement() -> None:
    conversation = SelfImprovementConversation(_ROOT)

    diagnosis = conversation.diagnose(_PROMPT)

    assert diagnosis.classification is ImprovementClassification.CAPABILITY_IMPROVEMENT
    assert diagnosis.scope == (_USE_CASE, _TOOL, _CONTROLLER, _TEST)
    assert diagnosis.focused_tests == (_TEST,)


def test_control_pc_request_resolves_exactly_one_builder_in_the_default_registry() -> None:
    matches = _default_registry().diagnose(_PROMPT)

    assert len(matches) == 1
    assert isinstance(matches[0].builder, DesktopCapabilityImprovementBuilder)


def test_reproducible_gap_produces_one_concrete_proposal() -> None:
    builder = DesktopCapabilityImprovementBuilder(_ROOT)
    diagnosis = builder.diagnose(_PROMPT)

    proposal = builder.build(diagnosis, _PROMPT)

    assert proposal is not None
    assert proposal.proposal_id == "improvement.desktop.open-file-with-application"
    assert set(proposal.files) == {_USE_CASE, _TEST}
    assert proposal.focused_tests == (_TEST,)
    assert proposal.metric_directions == {"aperturas_de_archivo_con_aplicacion_correctas": "increase"}
    derived = proposal.files[_USE_CASE]
    compile(derived, "derived_use_case", "exec")
    assert derived.count("desktop.open_file") == 2
    assert '"application": application' in derived


@pytest.mark.parametrize("solved_by", ["use_case", "tests"])
def test_already_solved_gap_yields_no_proposal(tmp_path: Path, solved_by: str) -> None:
    project = _fixture_desktop_project(tmp_path)
    if solved_by == "use_case":
        use_case = project / _USE_CASE
        use_case.write_text(
            use_case.read_text(encoding="utf-8") + f"\n\ndef {_SOLUTION_MARKER}(self, target):\n    return target, None\n",
            encoding="utf-8",
        )
    else:
        tests = project / _TEST
        tests.write_text(tests.read_text(encoding="utf-8") + "\n# con el bloc de notas\n", encoding="utf-8")
    builder = DesktopCapabilityImprovementBuilder(project)

    proposal = builder.build(builder.diagnose(_PROMPT), _PROMPT)

    assert proposal is None


def test_real_request_stops_at_authorization_with_zero_writes() -> None:
    targets = [_ROOT / _USE_CASE, _ROOT / _TEST]
    before = [target.read_text(encoding="utf-8") for target in targets]
    conversation = SelfImprovementConversation(_ROOT)

    response = conversation.handle(_PROMPT)

    assert "improvement.desktop.open-file-with-application" in response
    assert "No he modificado nada." in response
    assert "¿Autorizas" in response
    assert [target.read_text(encoding="utf-8") for target in targets] == before
    assert conversation.proposal is not None
    assert conversation.active


def test_out_of_desktop_scope_is_never_handled() -> None:
    builder = DesktopCapabilityImprovementBuilder(_ROOT)
    diagnosis = builder.diagnose(_PROMPT)
    assert diagnosis is not None
    tampered = ImprovementDiagnosis(
        diagnosis.classification,
        diagnosis.objective,
        (diagnosis.scope[0], "core/router.py"),
        diagnosis.focused_tests,
        diagnosis.metrics,
        diagnosis.risk,
        diagnosis.finding,
    )

    assert not builder.can_handle(tampered, _PROMPT)
    assert builder.build(tampered, _PROMPT) is None


@pytest.mark.parametrize(
    "prompt",
    [
        "Atlas, mejora Control PC para ejecutar cualquier comando de PowerShell",
        "Atlas, mejora Control PC para editar el registro de Windows",
    ],
)
def test_dangerous_control_pc_requests_never_produce_a_proposal(prompt: str) -> None:
    builder = DesktopCapabilityImprovementBuilder(_ROOT)
    conversation = SelfImprovementConversation(_ROOT)

    diagnosis = builder.diagnose(prompt)
    response = conversation.handle(prompt)

    assert diagnosis is not None
    assert builder.build(diagnosis, prompt) is None
    assert conversation.proposal is None
    assert not conversation.active
    assert "¿Autorizas" not in response


def test_ambiguous_control_pc_request_stops_without_silent_selection(tmp_path: Path) -> None:
    conversation = SelfImprovementConversation(tmp_path, builders=(_FixtureControlPcBuilder(), DesktopCapabilityImprovementBuilder(tmp_path)))

    response = conversation.handle(_PROMPT)

    assert response.startswith("CLARIFICATION_REQUIRED")
    assert conversation.proposal is None
    assert not conversation.active

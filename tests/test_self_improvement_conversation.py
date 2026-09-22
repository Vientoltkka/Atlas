from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from core.orchestrator import AtlasOrchestrator
from core.router import Router
from core.self_improvement_conversation import ImprovementDiagnosis, SelfImprovementConversation
from core.supervised_repair import ImprovementClassification, RepairProposal, RepairValidation
from core.voice_repair_builder import VoiceCodeRepairBuilder
from memory.conversation import ConversationMemory


def _ensure_git_repo(root: Path) -> None:
    """Create a real temporary Git repository for isolated-repair tests."""
    if (root / ".git").exists():
        return

    subprocess.run(
        ("git", "init"),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.email", "atlas-tests@example.invalid"),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Atlas Tests"),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ("git", "add", "-A"),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ("git", "commit", "--allow-empty", "-m", "test fixture"),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )


class _Chat:
    name = "chat"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, model, messages, provider_id=None):
        self.calls += 1
        return "ruta normal"


class _Models:
    def choose_model(self, _task):
        return "test"


_ROOT = Path(__file__).resolve().parents[1]


def _conversation(root: Path, *, passed: bool = True) -> SelfImprovementConversation:
    _ensure_git_repo(root)
    def build(diagnosis: ImprovementDiagnosis, _prompt: str) -> RepairProposal | None:
        if diagnosis.classification is not ImprovementClassification.CODE_REPAIR:
            return None
        return RepairProposal(
            "repair.dialogue-fixture",
            diagnosis.objective,
            {"fixture.txt": "fixed\n"},
            ("tests/test_self_improvement_conversation.py",),
            {"failures": "decrease"},
        )

    return SelfImprovementConversation(
        root,
        proposal_builder=build,
        validator_factory=lambda _proposal: lambda _: RepairValidation(passed, {"failures": 1}, {"failures": 0}, "fixture validated"),
    )


def _orchestrator(root: Path, *, passed: bool = True):
    chat = _Chat()
    app = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=_Models(),
        memory=ConversationMemory(),
        registry=SimpleNamespace(get=lambda name: chat if name == "chat" else None),
        write_file=SimpleNamespace(execute=lambda *_: None),
        project_root=root,
        self_improvement_conversation=_conversation(root, passed=passed),
    )
    return app, chat


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Atlas, corrige los fallos de la voz", True),
        ("Atlas, optimiza tu voz", True),
        ("Haz que puedas organizar mis tareas", True),
        ("Crea la capacidad para que Atlas lea calendarios", True),
        ("En que puedes automejorarte?", True),
        ("Qué mejora te falta tener", True),
        ("Qué herramientas necesitas para ser mejor", True),
        ("Qué te falta para mejorar de cara al futuro", True),
        ("Hola Atlas, ¿Qué mejorarías de tu sistema operativo?", True),
        ("Atlas, que mejor irías en tu sistema operativo", True),
        ("Atlas, qué mejorarías en tu sistema operativo", True),
        ("Cómo mejoro mi sistema operativo", False),
        ("¿En qué puedes mejorar esta carta?", False),
        ("¿En qué puedes mejorar este texto?", False),
        ("corrige este texto", False),
        ("mejora esta carta", False),
    ],
)
def test_detects_only_atlas_self_improvement_intent(prompt: str, expected: bool) -> None:
    assert SelfImprovementConversation.is_self_improvement_request(prompt) is expected


def test_self_improvement_question_is_honest_and_requires_authorization() -> None:
    response = SelfImprovementConversation(_ROOT).handle("En que puedes automejorarte?")

    assert response is not None
    assert "diagnosticar oportunidades con evidencia operativa" in response
    assert "propuestas acotadas con pruebas focalizadas" in response
    assert "autorizacion humana" in response
    for limit in ("codigo", "permisos", "memoria", "modelos", "datos"):
        assert limit in response
    assert "por mi cuenta" in response
    assert "aprendo constantemente" not in response


def test_operating_system_question_uses_bounded_verifiable_response() -> None:
    response = SelfImprovementConversation(_ROOT).handle(
        "Hola, Atlas: ¿Qué mejorarías de tu sistema operativo?"
    )

    assert response is not None
    assert "capacidades acotadas" in response
    assert "evidencia" in response
    assert "autorizar y validar" in response
    assert "kernel" in response
    assert "latencia o errores cero" in response
    assert "conciencia" not in response


@pytest.mark.parametrize(
    "prompt",
    [
        "Qué mejora te falta tener",
        "Qué herramientas necesitas para ser mejor",
        "Qué te falta para mejorar de cara al futuro",
    ],
)
def test_self_diagnosis_questions_have_the_bounded_response(prompt: str) -> None:
    response = SelfImprovementConversation(_ROOT).handle(prompt)

    assert response is not None
    assert "capacidades actuales" in response
    assert "autorizacion humana" in response
    assert "No me automejoro por mi cuenta" in response


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Atlas usa una capacidad existente", ImprovementClassification.REUSE),
        ("Atlas crea una skill para algo", ImprovementClassification.SKILL_GAP),
        ("Atlas corrige los fallos de la voz", ImprovementClassification.CODE_REPAIR),
        ("Atlas crea la capacidad para un proveedor externo", ImprovementClassification.CAPABILITY_GAP),
        ("Atlas mejora", ImprovementClassification.CLARIFICATION_REQUIRED),
    ],
)
def test_classifies_self_improvement_requests(prompt: str, expected: ImprovementClassification) -> None:
    assert SelfImprovementConversation(_ROOT).diagnose(prompt).classification is expected


def test_voice_diagnosis_is_limited_to_voice_scope() -> None:
    diagnosis = SelfImprovementConversation(_ROOT).diagnose("Atlas, corrige los fallos de la voz")
    assert diagnosis.scope == ("use_cases/voice_conversation.py", "tests/test_voice_conversation.py")


def test_real_voice_builder_prepares_only_the_reviewed_voice_repair() -> None:
    root = _ROOT
    diagnosis = SelfImprovementConversation(root).diagnose("Atlas, corrige los fallos de la voz sin romper las funciones actuales")
    proposal = VoiceCodeRepairBuilder(root).build(diagnosis, "Atlas, corrige los fallos de la voz sin romper las funciones actuales")

    assert proposal is not None
    assert proposal.proposal_id == "repair.voice.expired-model-worker-wait"
    assert set(proposal.files) == {"use_cases/voice_conversation.py", "tests/test_voice_conversation.py"}
    assert "_EXPIRED_MODEL_WORKER_WAIT_TIMEOUT_SECONDS" in proposal.files["use_cases/voice_conversation.py"]
    assert "test_expired_model_worker_wait_is_bounded" in proposal.files["tests/test_voice_conversation.py"]
    assert "expired_model_worker_wait_ms" in proposal.metric_directions


def test_real_voice_builder_rejects_an_expanded_scope() -> None:
    diagnosis = ImprovementDiagnosis(
        ImprovementClassification.CODE_REPAIR,
        "x",
        ("use_cases/voice_conversation.py", "outside.py"),
        (), (), "x", "x",
    )
    assert VoiceCodeRepairBuilder(Path(__file__).resolve().parents[1]).build(diagnosis, "x") is None


def test_real_voice_order_creates_a_concrete_proposal_without_writing() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "use_cases/voice_conversation.py"
    tests = root / "tests/test_voice_conversation.py"
    before = (source.read_text(encoding="utf-8"), tests.read_text(encoding="utf-8"))
    chat = _Chat()
    app = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=_Models(),
        memory=ConversationMemory(),
        registry=SimpleNamespace(get=lambda name: chat if name == "chat" else None),
        write_file=SimpleNamespace(execute=lambda *_: None),
        project_root=root,
    )

    response = app.process_prompt("Atlas, corrige los fallos de la voz sin romper las funciones actuales.", confirm=lambda _: "")

    assert "repair.voice.expired-model-worker-wait" in response
    assert "No he modificado nada." in response
    assert before == (source.read_text(encoding="utf-8"), tests.read_text(encoding="utf-8"))
    assert chat.calls == 0


def test_e2e_propose_authorize_validate_and_accept(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("original\n", encoding="utf-8")
    app, _ = _orchestrator(tmp_path)

    proposal = app.process_prompt(
        "Atlas, corrige los fallos de la voz sin romper las funciones actuales.",
        confirm=lambda _: "",
    )
    assert "proposal_id: repair.dialogue-fixture" in proposal
    assert target.read_text(encoding="utf-8") == "original\n"

    validated = app.process_prompt("s\u00ed", confirm=lambda _: "")
    assert "Antes/despu\u00e9s: failures: 1 -> 0" in validated

    # First approval applies and validates only inside the isolated worktree.
    assert target.read_text(encoding="utf-8") == "original\n"

    accepted = app.process_prompt("s\u00ed", confirm=lambda _: "")
    assert target.read_text(encoding="utf-8") == "fixed\n"

def test_rejection_rolls_back_exact_fixture_scope(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    unrelated = tmp_path / "unrelated.txt"
    target.write_text("original\n", encoding="utf-8")
    unrelated.write_text("keep\n", encoding="utf-8")
    app, _ = _orchestrator(tmp_path)
    app.process_prompt("Atlas, corrige los fallos de la voz sin romper las funciones actuales.", confirm=lambda _: "")
    app.process_prompt("sí", confirm=lambda _: "")

    assert "restauró exactamente" in app.process_prompt("no", confirm=lambda _: "")
    assert target.read_text(encoding="utf-8") == "original\n"
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


def test_validation_failure_rolls_back(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("original\n", encoding="utf-8")
    app, _ = _orchestrator(tmp_path, passed=False)
    app.process_prompt("Atlas, corrige los fallos de la voz sin romper las funciones actuales.", confirm=lambda _: "")

    assert "se restauró exactamente" in app.process_prompt("sí", confirm=lambda _: "")
    assert target.read_text(encoding="utf-8") == "original\n"


def test_capability_gap_stops_without_changes(tmp_path: Path) -> None:
    app, _ = _orchestrator(tmp_path)
    response = app.process_prompt("Atlas, crea la capacidad para un proveedor externo", confirm=lambda _: "")
    assert response.startswith("CAPABILITY_GAP")
    assert not (tmp_path / "fixture.txt").exists()


def test_normal_conversation_is_not_captured(tmp_path: Path) -> None:
    app, chat = _orchestrator(tmp_path)
    assert app.process_prompt("corrige este texto", confirm=lambda _: "") == "ruta normal"
    assert chat.calls == 1

def test_acceptance_conflict_is_not_reported_as_success(tmp_path: Path) -> None:
    conversation = _conversation(tmp_path)
    proposed = conversation.handle("Atlas, corrige los fallos de la voz")

    assert proposed is not None
    validated = conversation.handle("s\u00ed")
    assert validated is not None
    assert conversation.active

    workflow = conversation._workflow
    assert workflow is not None

    target = tmp_path / "fixture.txt"
    target.write_text("human edit\n", encoding="utf-8")

    response = conversation.handle("s\u00ed")

    assert response is not None
    assert "No se pudo conservar" in response
    assert "Reparaci\u00f3n aceptada" not in response
    assert conversation.active
    assert target.read_text(encoding="utf-8") == "human edit\n"

    rejected = conversation.handle("no")
    assert rejected is not None
    assert not conversation.active
    assert target.read_text(encoding="utf-8") == "human edit\n"

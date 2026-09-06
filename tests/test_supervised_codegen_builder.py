from __future__ import annotations

from pathlib import Path

import pytest

from core.file_read_capability_builder import FileReadCapabilityImprovementBuilder
from core.self_improvement_conversation import (
    ImprovementClassification,
    SelfImprovementConversation,
    SupervisedRepairBuilderRegistry,
)
from core.supervised_codegen_builder import SupervisedCodegenCapabilityBuilder
from core.supervised_repair import RepairProposal, RepairValidation, SupervisedRepairWorkflow

_ROOT = Path(__file__).resolve().parents[1]

_PROMPT = "Atlas, crea una capacidad para generar un informe estructurado de un texto local de forma determinista."
_FILE_READ_PROMPT = "Atlas, mejora tu capacidad para trabajar con archivos sin romper las funciones actuales."
_USE_CASE_PATH = "use_cases/text_report.py"
_TESTS_PATH = "tests/test_text_report.py"
_BOOTSTRAP_PATH = "bootstrap/skill_system.py"
_MANIFEST_PATH = "skills/builtin/skill.text-report/skill.json"

_BOOTSTRAP = """class SkillHandlerRegistry:
    def __init__(self) -> None:
        self._handlers = {}

    def register(self, handler_id, handler) -> None:
        self._handlers[handler_id] = handler

    def get(self, handler_id):
        return self._handlers[handler_id]


def build_builtin_skill_handler_registry() -> SkillHandlerRegistry:
    registry = SkillHandlerRegistry()
    registry.register("handler.text-uppercase", lambda inputs, **_: {"result": str(inputs.get("text", "")).upper()})
    return registry
"""

_BOOTSTRAP_PROPOSED = _BOOTSTRAP.replace(
    "    return registry",
    '    registry.register("handler.text-report", _text_report_handler)\n    return registry',
) + '''

def _text_report_handler(inputs, *, execution_context=None):
    from use_cases.text_report import text_report
    text = inputs.get("text")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    top = inputs.get("top", 5)
    if not isinstance(top, int) or isinstance(top, bool) or top < 1:
        raise ValueError("top must be a positive integer")
    return {"report": text_report(text, top)}
'''

_USE_CASE = '''def text_report(text: str, top: int = 5) -> dict[str, object]:
    """Return a deterministic structural report for a local text."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not isinstance(top, int) or isinstance(top, bool) or top < 1:
        raise ValueError("top must be a positive integer")
    words = text.split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    top_words = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top]
    return {
        "lines": len(text.splitlines()),
        "words": len(words),
        "characters": len(text),
        "unique_words": len(counts),
        "top_words": top_words,
    }
'''

_TESTS = '''from bootstrap.skill_system import build_builtin_skill_handler_registry
from use_cases.text_report import text_report


def test_report_counts_are_deterministic():
    text = "hola hola mundo\\nsegunda linea\\n"
    report = text_report(text)
    assert report["lines"] == 2
    assert report["words"] == 5
    assert report["characters"] == len(text)
    assert report["unique_words"] == 4
    assert report["top_words"] == [("hola", 2), ("linea", 1), ("mundo", 1), ("segunda", 1)]
    assert text_report(text) == report


def test_skill_handler_exposes_the_capability():
    registry = build_builtin_skill_handler_registry()
    handler = registry.get("handler.text-report")
    report = handler({"text": "a b b", "top": 1}, execution_context=None)["report"]
    assert report == {"lines": 1, "words": 3, "characters": 5, "unique_words": 2, "top_words": [("b", 2)]}
'''

_TESTS_BROKEN = _TESTS.replace('report["words"] == 5', 'report["words"] == 4')

_MANIFEST = """{
  "schema_version": "1.0",
  "skill_id": "skill.text-report",
  "name": "Informe de texto",
  "version": "1.0",
  "description": "Genera un informe estructurado y determinista de un texto local.",
  "enabled": true,
  "input_fields": [{"name": "text", "type": "str", "required": true}],
  "output_fields": [{"name": "report", "type": "dict", "required": true}],
  "execution_target": {"type": "handler", "handler_id": "handler.text-report"},
  "execution_target_type": "handler",
  "handler_id": "handler.text-report"
}
"""

_PROPOSAL_FILES = {_USE_CASE_PATH: _USE_CASE, _TESTS_PATH: _TESTS, _BOOTSTRAP_PATH: _BOOTSTRAP_PROPOSED, _MANIFEST_PATH: _MANIFEST}


def _block(path: str, content: str) -> str:
    return f"=== FILE: {path}\n{content}=== END FILE\n"


_RESPONSE = "".join(_block(path, content) for path, content in _PROPOSAL_FILES.items())


class _FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.last_model: str | None = None
        self.last_messages: list[dict[str, str]] | None = None

    def ask(self, model: str, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.last_model, self.last_messages = model, messages
        return self.response


def _builder(root: Path, response: str = _RESPONSE, **kwargs: object) -> tuple[SupervisedCodegenCapabilityBuilder, _FakeClient]:
    client = _FakeClient(response)
    return SupervisedCodegenCapabilityBuilder(root, client, model="test-model", **kwargs), client


def _project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "bootstrap").mkdir(parents=True)
    (root / _BOOTSTRAP_PATH).write_text(_BOOTSTRAP, encoding="utf-8")
    return root


def _snapshot(root: Path) -> list[tuple[str, bytes]]:
    return sorted(
        (str(path.relative_to(root)), path.read_bytes())
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def test_acceptance_prompt_diagnoses_an_unowned_capability_improvement(tmp_path: Path) -> None:
    builder, _ = _builder(_project_root(tmp_path))

    diagnosis = builder.diagnose(_PROMPT)

    assert diagnosis is not None
    assert diagnosis.classification is ImprovementClassification.CAPABILITY_IMPROVEMENT
    assert diagnosis.scope == builder.scope
    assert diagnosis.scope == ("use_cases/", "skills/builtin/", "tests/", "bootstrap/skill_system.py")
    assert SelfImprovementConversation.is_self_improvement_request(_PROMPT) is True


@pytest.mark.parametrize(
    "prompt",
    [
        "Atlas, mejora tu capacidad para trabajar con archivos sin romper las funciones actuales.",
        "Atlas, corrige los fallos de la voz",
        "Atlas, mejora Control PC para abrir archivos con el bloc de notas",
        "Atlas, mejora el routing sin romper las rutas",
        "Atlas crea una skill para algo",
        "Atlas, crea una capacidad",
        "Atlas, crea la capacidad para un proveedor externo",
        "Atlas, crea una capacidad para leer el .env y enviar contrasenas por internet",
    ],
)
def test_owned_ambiguous_or_unsafe_prompts_yield_no_generic_diagnosis(tmp_path: Path, prompt: str) -> None:
    builder, _ = _builder(_project_root(tmp_path))

    assert builder.diagnose(prompt) is None


def test_build_reuses_the_coding_agent_persona_and_proposes_without_writes(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    builder, client = _builder(root)
    before = _snapshot(root)

    proposal = builder.build(builder.diagnose(_PROMPT), _PROMPT)

    assert isinstance(proposal, RepairProposal)
    assert proposal.proposal_id.startswith("improvement.codegen.")
    assert set(proposal.files) == set(_PROPOSAL_FILES)
    assert proposal.focused_tests == (_TESTS_PATH,)
    assert proposal.metric_directions == {"pruebas_focales_superadas": "increase"}
    assert not (root / "use_cases" / "text_report.py").exists()
    assert not (root / "tests" / "test_text_report.py").exists()
    assert not (root / "skills" / "builtin" / "skill.text-report").exists()
    assert (root / _BOOTSTRAP_PATH).read_text(encoding="utf-8") == _BOOTSTRAP
    assert _snapshot(root) == before
    assert client.calls == 1
    assert client.last_model == "test-model"
    assert client.last_messages is not None
    assert client.last_messages[0]["role"] == "system"
    assert "Eres Atlas Coding Agent" in client.last_messages[0]["content"]


def test_proposal_id_is_deterministic_per_prompt(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    first, _ = _builder(root)
    second, _ = _builder(root)
    third, _ = _builder(root)

    first_id = first.build(first.diagnose(_PROMPT), _PROMPT).proposal_id
    second_id = second.build(second.diagnose(_PROMPT), _PROMPT).proposal_id
    third_id = third.build(third.diagnose("Atlas, crea una capacidad para generar indices de un documento local de forma determinista."), "otro").proposal_id

    assert first_id == second_id
    assert third_id != first_id


@pytest.mark.parametrize(
    "path",
    [
        "core/atlas.py",
        "core/supervised_repair.py",
        "core/self_improvement_conversation.py",
        "scheduler/supervisor.py",
        "security/policy.py",
        "bootstrap/bootstrap.py",
        ".env",
        "../outside.py",
        "C:/AI/Atlas/use_cases/evil.py",
    ],
)
def test_responses_outside_the_whitelist_are_rejected_without_writes(tmp_path: Path, path: str) -> None:
    root = _project_root(tmp_path)
    builder, _ = _builder(root, _block(path, "SECRET = 'x'\n"))
    before = _snapshot(root)

    assert builder.build(builder.diagnose(_PROMPT), _PROMPT) is None
    assert _snapshot(root) == before


def test_limits_of_files_and_total_size_are_enforced(tmp_path: Path) -> None:
    root = _project_root(tmp_path)

    too_many, _ = _builder(root, max_files=3)
    assert too_many.build(too_many.diagnose(_PROMPT), _PROMPT) is None

    too_large, _ = _builder(root, max_total_chars=200)
    assert too_large.build(too_large.diagnose(_PROMPT), _PROMPT) is None


def test_proposals_without_focused_tests_or_logic_are_rejected(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    without_tests, _ = _builder(root, "".join(_block(path, content) for path, content in _PROPOSAL_FILES.items() if path != _TESTS_PATH))
    without_logic, _ = _builder(root, _block(_TESTS_PATH, _TESTS) + _block(_MANIFEST_PATH, _MANIFEST))

    assert without_tests.build(without_tests.diagnose(_PROMPT), _PROMPT) is None
    assert without_logic.build(without_logic.diagnose(_PROMPT), _PROMPT) is None


def test_unparsable_model_response_yields_no_proposal(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    builder, _ = _builder(root, "Lo siento, aqui tienes una idea general del cambio.")

    assert builder.build(builder.diagnose(_PROMPT), _PROMPT) is None


def test_synthesis_prompt_carries_the_real_repo_contracts(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    manifest_dir = root / "skills" / "builtin" / "text_uppercase"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "skill.json").write_text(_MANIFEST, encoding="utf-8")
    builder, client = _builder(root)

    builder.build(builder.diagnose(_PROMPT), _PROMPT)

    user_prompt = client.last_messages[1]["content"]
    assert "Contratos reales del repositorio" in user_prompt
    assert "def build_builtin_skill_handler_registry" in user_prompt
    assert '"skill_id": "skill.text-report"' in user_prompt
    assert "SIN reescribirlo" in user_prompt
    assert "execution_context: SkillExecutionContext" in user_prompt


def test_synthesis_prompt_without_real_repo_files_still_proposes(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    builder, client = _builder(root)

    proposal = builder.build(builder.diagnose(_PROMPT), _PROMPT)

    assert proposal is not None
    user_prompt = client.last_messages[1]["content"]
    assert "Contratos reales del repositorio" in user_prompt
    assert "def build_builtin_skill_handler_registry" not in user_prompt


def test_bootstrap_rewrites_dropping_the_real_api_are_rejected(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    invented_bootstrap = "def build_builtin_skill_handler_registry():\n    return {}\n"
    response = (
        _block(_USE_CASE_PATH, _USE_CASE)
        + _block(_TESTS_PATH, _TESTS)
        + _block(_BOOTSTRAP_PATH, invented_bootstrap)
    )
    builder, _ = _builder(root, response)
    before = _snapshot(root)

    assert builder.build(builder.diagnose(_PROMPT), _PROMPT) is None
    assert _snapshot(root) == before


@pytest.mark.parametrize("module", ["random", "uuid", "time", "datetime", "locale"])
@pytest.mark.parametrize("placement", ["top_level", "nested"])
def test_nondeterministic_imports_in_the_capability_module_are_rejected_without_writes(
    tmp_path: Path, module: str, placement: str
) -> None:
    root = _project_root(tmp_path)
    if placement == "top_level":
        use_case = f"import {module}\n\n" + _USE_CASE
    else:
        use_case = _USE_CASE.replace("    return {", f"    import {module}\n    return {{")
    response = (
        _block(_USE_CASE_PATH, use_case)
        + _block(_TESTS_PATH, _TESTS)
        + _block(_BOOTSTRAP_PATH, _BOOTSTRAP_PROPOSED)
        + _block(_MANIFEST_PATH, _MANIFEST)
    )
    builder, _ = _builder(root, response)
    before = _snapshot(root)

    assert builder.build(builder.diagnose(_PROMPT), _PROMPT) is None
    assert _snapshot(root) == before


def test_deterministic_capability_module_without_banned_imports_is_still_proposed(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    builder, _ = _builder(root)

    proposal = builder.build(builder.diagnose(_PROMPT), _PROMPT)

    assert proposal is not None
    assert _USE_CASE_PATH in proposal.files


def test_synthesis_prompt_carries_the_determinism_contract(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    _, client = _builder(root)

    builder = SupervisedCodegenCapabilityBuilder(root, client, model="test-model")
    builder.build(builder.diagnose(_PROMPT), _PROMPT)

    user_prompt = client.last_messages[1]["content"]
    assert "prohibido importar o usar random, uuid, time, datetime o locale" in user_prompt
    assert "ordena siempre con sorted" in user_prompt
    assert "ejemplo concreto entrada -> salida" in user_prompt


def test_failed_focal_tests_report_the_pytest_summary(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    builder, _ = _builder(root, _RESPONSE.replace(_TESTS, _TESTS_BROKEN))
    proposal = builder.build(builder.diagnose(_PROMPT), _PROMPT)
    workflow = SupervisedRepairWorkflow(root, validator=builder.validator)
    workflow.propose(proposal)
    workflow.authorize_and_apply(proposal.authorization)

    validation = workflow.validate()

    assert not validation.passed
    assert "tests focales" in validation.detail
    assert "FAILED tests/test_text_report.py" in validation.detail


def test_failing_model_call_yields_no_proposal_and_no_writes(tmp_path: Path) -> None:
    root = _project_root(tmp_path)

    class _BrokenClient:
        def ask(self, model: str, messages: list[dict[str, str]]) -> str:
            raise RuntimeError("404 model not found")

    builder = SupervisedCodegenCapabilityBuilder(root, _BrokenClient(), model="test-model")
    before = _snapshot(root)

    assert builder.build(builder.diagnose(_PROMPT), _PROMPT) is None
    assert _snapshot(root) == before


def test_default_model_delegates_to_the_configured_provider_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project_root(tmp_path)
    monkeypatch.delenv("ATLAS_CODEGEN_MODEL", raising=False)
    default_client = _FakeClient(_RESPONSE)
    default_builder = SupervisedCodegenCapabilityBuilder(root, default_client)
    pinned_client = _FakeClient(_RESPONSE)
    pinned_builder = SupervisedCodegenCapabilityBuilder(root, pinned_client, model="pinned-model")
    monkeypatch.setenv("ATLAS_CODEGEN_MODEL", "env-model")
    env_builder = SupervisedCodegenCapabilityBuilder(root, _FakeClient(_RESPONSE))

    default_builder.build(default_builder.diagnose(_PROMPT), _PROMPT)
    pinned_builder.build(pinned_builder.diagnose(_PROMPT), _PROMPT)
    env_builder.build(env_builder.diagnose(_PROMPT), _PROMPT)

    assert default_client.last_model == ""
    assert pinned_client.last_model == "pinned-model"
    assert env_builder._model == "env-model"


def test_file_headers_with_trailing_equal_signs_are_parsed(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    response = "".join(f"=== FILE: {path} ===\n{content}=== END FILE ===\n" for path, content in _PROPOSAL_FILES.items())
    builder, _ = _builder(root, response)

    proposal = builder.build(builder.diagnose(_PROMPT), _PROMPT)

    assert proposal is not None
    assert set(proposal.files) == set(_PROPOSAL_FILES)


def test_acceptance_prompt_with_a_valid_proposal_is_not_a_clarification(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    _, client = _builder(root)
    conversation = SelfImprovementConversation(root, builders=(SupervisedCodegenCapabilityBuilder(root, client, model="test-model"),))
    before = _snapshot(root)

    response = conversation.handle(_PROMPT)

    assert response is not None
    assert not response.startswith("CLARIFICATION_REQUIRED")
    assert "proposal_id: improvement.codegen." in response
    assert client.calls == 1
    assert client.last_model == "test-model"
    assert _snapshot(root) == before


def test_host_validator_runs_focal_tests_and_the_model_is_not_consulted(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    builder, client = _builder(root)
    proposal = builder.build(builder.diagnose(_PROMPT), _PROMPT)
    workflow = SupervisedRepairWorkflow(root, validator=builder.validator)
    workflow.propose(proposal)
    assert workflow.authorize_and_apply(proposal.authorization)
    calls_after_build = client.calls

    validation = workflow.validate()

    assert isinstance(validation, RepairValidation)
    assert validation.passed
    assert validation.before_metrics["pruebas_focales_superadas"] == 0.0
    assert validation.after_metrics["pruebas_focales_superadas"] >= 2.0
    assert "tests focales" in validation.detail
    assert client.calls == calls_after_build
    assert workflow.finalize(accepted=True)
    assert workflow.state is not None and workflow.state.value == "ACCEPTED"


def test_full_flow_conserves_the_new_capability(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    builder, _ = _builder(root)
    proposal = builder.build(builder.diagnose(_PROMPT), _PROMPT)
    workflow = SupervisedRepairWorkflow(root, validator=builder.validator)
    workflow.propose(proposal)
    assert not (root / _USE_CASE_PATH).exists()
    assert workflow.authorize_and_apply(proposal.authorization)

    validation = workflow.validate()

    assert validation.passed
    assert (root / _USE_CASE_PATH).read_text(encoding="utf-8") == _USE_CASE
    assert 'registry.register("handler.text-report", _text_report_handler)' in (root / _BOOTSTRAP_PATH).read_text(encoding="utf-8")
    assert (root / _MANIFEST_PATH).exists()
    assert workflow.finalize(accepted=True)


def test_failed_focal_tests_roll_back_the_exact_scope(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    builder, _ = _builder(root, _RESPONSE.replace(_TESTS, _TESTS_BROKEN))
    proposal = builder.build(builder.diagnose(_PROMPT), _PROMPT)
    workflow = SupervisedRepairWorkflow(root, validator=builder.validator)
    workflow.propose(proposal)
    assert workflow.authorize_and_apply(proposal.authorization)

    validation = workflow.validate()

    assert not validation.passed
    assert workflow.state is not None and workflow.state.value == "ROLLED_BACK"
    assert not (root / _USE_CASE_PATH).exists()
    assert not (root / _TESTS_PATH).exists()
    assert not (root / "skills" / "builtin" / "skill.text-report").exists()
    assert (root / _BOOTSTRAP_PATH).read_text(encoding="utf-8") == _BOOTSTRAP


def test_specific_builders_keep_priority_over_the_generic_builder() -> None:
    specific = FileReadCapabilityImprovementBuilder(_ROOT)
    generic = SupervisedCodegenCapabilityBuilder(_ROOT)

    assert generic.diagnose(_FILE_READ_PROMPT) is None
    diagnosis = specific.diagnose(_FILE_READ_PROMPT)
    registry = SupervisedRepairBuilderRegistry((specific, generic))
    assert registry.builder_for(diagnosis, _FILE_READ_PROMPT) is specific

    conversation = SelfImprovementConversation(_ROOT)
    resolved = conversation.diagnose(_FILE_READ_PROMPT)
    assert resolved.classification is ImprovementClassification.CAPABILITY_IMPROVEMENT
    assert resolved.scope == specific.diagnose(_FILE_READ_PROMPT).scope


def test_default_conversation_routes_the_acceptance_prompt_to_the_generic_builder() -> None:
    conversation = SelfImprovementConversation(_ROOT)
    diagnosis = conversation.diagnose(_PROMPT)

    assert diagnosis.classification is ImprovementClassification.CAPABILITY_IMPROVEMENT
    assert diagnosis.scope == SupervisedCodegenCapabilityBuilder(_ROOT).scope

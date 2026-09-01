from core.atlas import Atlas
from bootstrap.bootstrap import Bootstrap
from dataclasses import replace
from types import SimpleNamespace
from core.agent_registry import AgentRegistry
from core.orchestrator import AtlasOrchestrator
from core.router import Router
from core.supervised_capability_gap import SupervisedCapabilityGapDetector
from core.skill_registry import SkillDefinition, SkillRegistry, SkillNotFoundError
from core.skill_executor import SkillExecutionRequest, SkillExecutionStatus
from memory.conversation import ConversationMemory
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext

class TemperatureConversionTool(BaseTool):
    @property
    def name(self): return "convert_temperature"
    @property
    def description(self): return "Convert temperature from Celsius to Fahrenheit."
    def execute(self, context: ToolContext): raise AssertionError("must not execute")

class ChatAgent:
    name, generated_path = "chat", None
    def __init__(self): self.calls = 0
    def run(self, *, model, messages): self.calls += 1; return "ruta normal"

class ModelManager:
    def __init__(self): self.calls = 0
    def choose_model(self, _): self.calls += 1; return "test-model"

class WriteFile:
    def __init__(self): self.calls = 0
    def execute(self, *args): self.calls += 1; return "written"

def _detector(*tools, skills=()):
    registry = ToolRegistry()
    for tool in tools: registry.register(tool)
    return SupervisedCapabilityGapDetector.from_registries(tool_registry=registry, skill_system=SimpleNamespace(skill_registry=SkillRegistry(skills)), agent_registry=AgentRegistry())

def _orchestrator(detector):
    chat, writer, model_manager = ChatAgent(), WriteFile(), ModelManager()
    app = AtlasOrchestrator(planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)), router=Router(), model_manager=model_manager, memory=ConversationMemory(), registry=SimpleNamespace(get=lambda name: chat if name == "chat" else None), write_file=writer, capability_gap_detector=detector)
    return app, chat, writer, model_manager

def test_existing_temperature_capability_keeps_the_normal_route():
    app, chat, writer, model_manager = _orchestrator(_detector(TemperatureConversionTool()))
    assert app.process_prompt("Convierte 37 grados Celsius a Fahrenheit.", confirm=lambda _: "") == "ruta normal"
    assert chat.calls == 1 and writer.calls == 0 and model_manager.calls == 1

def test_missing_temperature_capability_proposes_improvement_and_blocks_execution():
    app, chat, writer, model_manager = _orchestrator(_detector())
    response = app.process_prompt("Convierte 37 grados Celsius a Fahrenheit.", confirm=lambda _: "sí")
    assert "No dispongo de una capacidad registrada" in response
    assert "tools, skills y agentes" in response
    assert "Celsius–Fahrenheit" in response
    assert "¿Quieres que prepare esta mejora para tu aprobación?" in response
    assert chat.calls == 0 and writer.calls == 0 and model_manager.calls == 0

def test_detector_is_not_triggered_for_unrelated_requests():
    assert _detector().proposal_for("Hola Atlas") is None

def test_bootstrap_and_atlas_process_prompt_return_the_supervised_proposal():
    prompt = "Convierte 37 grados Celsius a Fahrenheit."
    atlas = Atlas()
    try:
        orchestrator = atlas._orchestrator
        request = orchestrator._request_gateway.from_text(prompt)
        decision = orchestrator.classify_request(request)
        assert decision.route.value == "direct_response"
        assert decision.matched_rules == ("direct.simple_request",)
        assert decision.target_tool_name is None
        assert decision.target_agent_name is None
        assert orchestrator._capability_gap_detector is not None
        response = atlas.process_prompt(prompt)
    finally:
        atlas.close()

    assert response == "37 grados Celsius equivalen a 98.6 grados Fahrenheit."
    assert response == "37 grados Celsius equivalen a 98.6 grados Fahrenheit."

class PlanningCodingAgent:
    def __init__(self): self.preparations = 0; self.run_calls = 0; self.applies = 0
    def run(self, *, model, messages): self.run_calls += 1; raise AssertionError("must not run")
    def apply_prepared_capability_plan(self, capability_id): self.applies += 1; return "applied"
    def prepare_capability_plan(self, **details): self.preparations += 1; return "\n".join(("Preparación supervisada de mejora:", f"- Capacidad ausente: {details['capability_id']}.", f"- Implementación mínima propuesta: {details['implementation']}.", "- Archivos previsiblemente afectados:", *(f"  - {path}" for path in details['planned_files']), "- Tests focalizados necesarios:", *(f"  - {test}" for test in details['focused_tests']), f"- Riesgo/impacto: {details['risk']}", "- Estado: todavía NO se han realizado cambios."))
def _pending_orchestrator():
    chat, coding, writer, manager = ChatAgent(), PlanningCodingAgent(), WriteFile(), ModelManager(); app = AtlasOrchestrator(planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)), router=Router(), model_manager=manager, memory=ConversationMemory(), registry=SimpleNamespace(get=lambda name: {"chat": chat, "coding": coding}.get(name)), write_file=writer, capability_gap_detector=_detector()); return app, chat, coding, writer, manager
def test_pending_proposal_and_yes_prepares_without_running_or_writing():
    app, chat, coding, writer, manager = _pending_orchestrator(); app.process_prompt("Convierte 37 grados Celsius a Fahrenheit.", confirm=lambda _: ""); response = app.process_prompt("si", confirm=lambda _: "")
    assert "Capacidad ausente: unit.temperature-conversion" in response and "Archivos previsiblemente afectados" in response and "Tests focalizados necesarios" in response and "Riesgo/impacto" in response and "todavía NO se han realizado cambios" in response
    assert coding.preparations == 1 and coding.run_calls == 0 and chat.calls == 0 and writer.calls == 0 and manager.calls == 0
def test_pending_proposal_and_supported_affirmatives_prepare_plan():
    for answer in ("Sí", "vale", "ok", "de acuerdo", "adelante"):
        app, _, coding, writer, manager = _pending_orchestrator(); app.process_prompt("Convierte 37 grados Celsius a Fahrenheit.", confirm=lambda _: ""); response = app.process_prompt(answer, confirm=lambda _: "")
        assert "Preparación supervisada de mejora" in response and coding.preparations == 1 and coding.run_calls == 0 and writer.calls == 0 and manager.calls == 0
def test_second_authorization_applies_only_the_prepared_plan():
    app, chat, coding, writer, manager = _pending_orchestrator(); app.process_prompt("Convierte 37 grados Celsius a Fahrenheit.", confirm=lambda _: ""); app.process_prompt("si", confirm=lambda _: ""); response = app.process_prompt("si", confirm=lambda _: "")
    assert response == "applied" and coding.applies == 1 and writer.calls == 0 and chat.calls == 0 and manager.calls == 0

def test_second_authorization_bypasses_capability_status_and_clears_the_plan():
    app, _, coding, writer, _ = _pending_orchestrator(); app.process_prompt("Convierte 37 grados Celsius a Fahrenheit.", confirm=lambda _: ""); app.process_prompt("si", confirm=lambda _: "")
    app._capability_status_response = lambda _prompt: (_ for _ in ()).throw(AssertionError("must not reach capability status"))
    assert app.process_prompt("si", confirm=lambda _: "") == "applied"
    assert coding.applies == 1 and writer.calls == 0 and app._prepared_capability_proposal is None

def test_rejection_after_preparation_does_not_apply_plan():
    app, _, coding, writer, _ = _pending_orchestrator(); app.process_prompt("Convierte 37 grados Celsius a Fahrenheit.", confirm=lambda _: ""); app.process_prompt("si", confirm=lambda _: ""); response = app.process_prompt("no", confirm=lambda _: "")
    assert "cancelada" in response and coding.applies == 0 and writer.calls == 0
def test_yes_without_pending_proposal_does_not_prepare_capability_improvement():
    app, _, coding, _, _ = _pending_orchestrator(); app.process_prompt("si", confirm=lambda _: ""); app.process_prompt("ok", confirm=lambda _: ""); assert coding.preparations == 0 and coding.run_calls == 0
def test_rejected_pending_proposal_does_not_prepare_or_write():
    app, chat, coding, writer, manager = _pending_orchestrator(); app.process_prompt("Convierte 37 grados Celsius a Fahrenheit.", confirm=lambda _: ""); response = app.process_prompt("No", confirm=lambda _: "")
    assert response == "Preparación de mejora cancelada. No se han realizado cambios."
    assert coding.preparations == 0 and coding.run_calls == 0 and chat.calls == 0 and writer.calls == 0 and manager.calls == 0
def test_bootstrap_prepares_authorized_capability_plan_without_applying_changes():
    atlas = Atlas()
    try:
        response = atlas.process_prompt("Convierte 37 grados Celsius a Fahrenheit.")
    finally:
        atlas.close()
    assert response == "37 grados Celsius equivalen a 98.6 grados Fahrenheit."

class FinalizationCodingAgent(PlanningCodingAgent):
    def __init__(self):
        super().__init__()
        self.capability_validation_status = None
        self.closures: list[bool] = []

    def apply_prepared_capability_plan(self, capability_id):
        self.applies += 1
        self.capability_validation_status = "VALIDATED"
        return "Validación completada correctamente. ¿Apruebas cerrar y versionar esta mejora?"

    def close_validated_capability_plan(self, capability_id, *, approved):
        self.closures.append(approved)
        if not approved:
            self.capability_validation_status = "CLOSURE_DECLINED"
            return "Cierre/versionado no aprobado."
        self.capability_validation_status = None
        return "Mejora cerrada y versionada correctamente. Commit: abc123."


def _finalization_orchestrator():
    chat, coding, writer, manager = ChatAgent(), FinalizationCodingAgent(), WriteFile(), ModelManager()
    app = AtlasOrchestrator(planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)), router=Router(), model_manager=manager, memory=ConversationMemory(), registry=SimpleNamespace(get=lambda name: {"chat": chat, "coding": coding}.get(name)), write_file=writer, capability_gap_detector=_detector())
    return app, coding, writer


def test_final_approval_closes_only_a_validated_capability():
    app, coding, writer = _finalization_orchestrator()
    app.process_prompt("Convierte 37 grados Celsius a Fahrenheit.", confirm=lambda _: "")
    app.process_prompt("si", confirm=lambda _: "")
    validation = app.process_prompt("si", confirm=lambda _: "")
    response = app.process_prompt("si", confirm=lambda _: "")

    assert "Validación completada correctamente" in validation
    assert "Commit: abc123" in response
    assert coding.closures == [True]
    assert writer.calls == 0
    assert app._validated_capability_proposal is None


def test_final_rejection_never_closes_the_validated_capability():
    app, coding, writer = _finalization_orchestrator()
    app.process_prompt("Convierte 37 grados Celsius a Fahrenheit.", confirm=lambda _: "")
    app.process_prompt("si", confirm=lambda _: "")
    app.process_prompt("si", confirm=lambda _: "")
    response = app.process_prompt("no", confirm=lambda _: "")

    assert "no aprobado" in response
    assert coding.closures == [False]
    assert writer.calls == 0
    assert app._validated_capability_proposal is None
def test_skill_creation_reuses_registered_skill_without_effects():
    skill = SkillDefinition(skill_id="skill.text-uppercase", name="Text Uppercase", version="1.0", description="Convert dynamic input text to uppercase.", execution_target="handler.text-uppercase")
    app, chat, writer, model_manager = _orchestrator(_detector(skills=(skill,)))
    response = app.process_prompt("Atlas, crea una skill que convierta texto a mayúsculas", confirm=lambda _: "")
    assert "REUSE" in response and "skill.text-uppercase" in response and "Tipo: skill" in response
    assert chat.calls == writer.calls == model_manager.calls == 0

def test_skill_creation_for_missing_capability_returns_read_only_proposal():
    app, chat, writer, model_manager = _orchestrator(_detector())
    response = app.process_prompt("Atlas, crea una skill que catalogue constelaciones por brillo", confirm=lambda _: "")
    assert "CREATE_PROPOSAL" in response and "skill." in response and "AUTORIZAR" in response
    assert chat.calls == writer.calls == model_manager.calls == 0

def test_skill_creation_without_capability_requests_clarification_without_effects():
    app, chat, writer, model_manager = _orchestrator(_detector())
    response = app.process_prompt("Atlas, crea una skill", confirm=lambda _: "")
    assert "CLARIFICATION_REQUIRED" in response
    assert chat.calls == writer.calls == model_manager.calls == 0
def _bootstrap_skill_creation(tmp_path):
    app = Bootstrap.build()
    app._project_root = tmp_path
    detector = app._capability_gap_detector
    assert detector is not None and app._skill_system is not None
    return app, detector, app._skill_system


def _creation_proposal(detector):
    proposal = detector.skill_creation_response_for(
        "Crea una skill que normalice mensajes usando handler.text-uppercase"
    )
    assert proposal is not None and proposal.status == "CREATE_PROPOSAL"
    return proposal


def _authorization(proposal):
    return f"AUTORIZAR {proposal.skill_id} {proposal.authorization_token}"


def _handler_skill(skill_id, handler_id):
    return SkillDefinition(
        skill_id=skill_id,
        name=skill_id,
        version="1.0",
        description=skill_id,
        execution_target=handler_id,
        execution_target_type="handler",
        handler_id=handler_id,
    )


def test_bootstrap_creates_authorized_skill_manifest_registers_and_executes(tmp_path):
    app, detector, system = _bootstrap_skill_creation(tmp_path)
    proposal = _creation_proposal(detector)

    response = app.process_prompt(
        "Crea una skill que normalice mensajes usando handler.text-uppercase",
        confirm=lambda _: "",
    )
    created = app.process_prompt(_authorization(proposal), confirm=lambda _: "")
    manifest = tmp_path / "skills" / "builtin" / proposal.skill_id.removeprefix("skill.") / "skill.json"
    execution = system.skill_executor.execute(
        SkillExecutionRequest(system.skill_registry.get(proposal.skill_id), inputs={"text": "Atlas"})
    )

    assert "CREATE_PROPOSAL" in response
    assert "SKILL_CREATED" in created
    assert manifest.exists()
    assert execution.status is SkillExecutionStatus.COMPLETED
    assert execution.output == {"result": "ATLAS"}


def test_authorization_requires_the_active_exact_proposal(tmp_path):
    app, detector, system = _bootstrap_skill_creation(tmp_path)
    proposal = _creation_proposal(detector)
    before = system.skill_registry.list_skills(enabled_only=False)

    assert app._handle_pending_skill_creation(_authorization(proposal)) is None
    app.process_prompt("Crea una skill que normalice mensajes usando handler.text-uppercase", confirm=lambda _: "")
    wrong = f"AUTORIZAR skill.other {proposal.authorization_token}"
    assert "UNSUPPORTED_FOR_SAFE_CREATION" in app.process_prompt(wrong, confirm=lambda _: "")
    assert system.skill_registry.list_skills(enabled_only=False) == before
    assert not (tmp_path / "skills").exists()


def test_unknown_handler_is_unsupported_without_changes(tmp_path):
    app, detector, system = _bootstrap_skill_creation(tmp_path)
    proposal = detector.skill_creation_response_for(
        "Crea una skill que catalogue estrellas usando handler.unknown"
    )
    assert proposal is not None and proposal.status == "CREATE_PROPOSAL"
    before = system.skill_registry.list_skills(enabled_only=False)

    result = detector.apply_declarative_skill(proposal, _authorization(proposal), tmp_path)

    assert "UNSUPPORTED_FOR_SAFE_CREATION" in result
    assert system.skill_registry.list_skills(enabled_only=False) == before
    assert not (tmp_path / "skills").exists()


def test_bootstrap_creation_rollback_preserves_existing_skills_and_files(tmp_path, monkeypatch):
    _, detector, system = _bootstrap_skill_creation(tmp_path)
    registry = system.skill_registry
    source = registry.get("skill.text-uppercase")
    a = replace(source, skill_id="skill.a", name="A", description="Existing A")
    b = replace(source, skill_id="skill.b", name="B", description="Existing B")
    registry.register(a)
    registry.register(b)
    before = registry.list_skills(enabled_only=False)
    proposal = _creation_proposal(detector)
    original_register = registry.register

    def fail_only_new(definition, *, replace=False):
        if definition.skill_id == proposal.skill_id:
            raise RuntimeError("controlled registration failure")
        return original_register(definition, replace=replace)

    monkeypatch.setattr(registry, "register", fail_only_new)
    result = detector.apply_declarative_skill(proposal, _authorization(proposal), tmp_path)

    assert "SKILL_CREATION_ROLLED_BACK" in result
    assert registry.get("skill.a") == a
    assert registry.get("skill.b") == b
    assert not registry.contains(proposal.skill_id)
    assert registry.list_skills(enabled_only=False) == before
    assert not (tmp_path / "skills").exists()

def test_skill_registry_unregister_removes_only_requested_skill():
    a, b, c = _handler_skill("skill.a", "handler.a"), _handler_skill("skill.b", "handler.b"), _handler_skill("skill.c", "handler.c")
    registry = SkillRegistry((a, b, c)); assert registry.unregister("skill.c") == c
    assert tuple(registry.list_skills()) == (a, b)
    try: registry.unregister("skill.c")
    except SkillNotFoundError: pass
    else: raise AssertionError("missing skill must raise SkillNotFoundError")

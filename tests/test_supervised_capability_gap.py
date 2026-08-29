from core.atlas import Atlas
from types import SimpleNamespace
from core.agent_registry import AgentRegistry
from core.orchestrator import AtlasOrchestrator
from core.router import Router
from core.supervised_capability_gap import SupervisedCapabilityGapDetector
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

def _detector(*tools):
    registry = ToolRegistry()
    for tool in tools: registry.register(tool)
    return SupervisedCapabilityGapDetector.from_registries(tool_registry=registry, agent_registry=AgentRegistry())

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

    assert "No dispongo de una capacidad registrada" in response
    assert "¿Quieres que prepare esta mejora para tu aprobación?" in response

class PlanningCodingAgent:
    def __init__(self): self.preparations = 0; self.run_calls = 0
    def run(self, *, model, messages): self.run_calls += 1; raise AssertionError("must not run")
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
def test_yes_without_pending_proposal_does_not_prepare_capability_improvement():
    app, _, coding, _, _ = _pending_orchestrator(); app.process_prompt("si", confirm=lambda _: ""); app.process_prompt("ok", confirm=lambda _: ""); assert coding.preparations == 0 and coding.run_calls == 0
def test_rejected_pending_proposal_does_not_prepare_or_write():
    app, chat, coding, writer, manager = _pending_orchestrator(); app.process_prompt("Convierte 37 grados Celsius a Fahrenheit.", confirm=lambda _: ""); response = app.process_prompt("No", confirm=lambda _: "")
    assert response == "Preparación de mejora cancelada. No se han realizado cambios."
    assert coding.preparations == 0 and coding.run_calls == 0 and chat.calls == 0 and writer.calls == 0 and manager.calls == 0
def test_bootstrap_prepares_authorized_capability_plan_without_applying_changes():
    atlas = Atlas()
    try:
        atlas.process_prompt("Convierte 37 grados Celsius a Fahrenheit."); response = atlas.process_prompt("si")
    finally:
        atlas.close()
    assert "Preparación supervisada de mejora" in response and "todavía NO se han realizado cambios" in response

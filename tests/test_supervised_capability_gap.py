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
"""E2E #3 regression: an explicit use of a registered Skill must execute it.

Guards the real conversational path: an explicit "usa la skill ..." order that
names an already registered builtin Skill must resolve and execute the real
handler instead of falling into the self-improvement SKILL_GAP diagnosis.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from bootstrap.skill_system import (
    build_builtin_skill_handler_registry,
    build_core_skill_system,
    register_builtin_skills,
)
from core.orchestrator import AtlasOrchestrator
from core.router import Router
from memory.conversation import ConversationMemory


_PROMPT = (
    "usa la skill skill.informe-texto-estructurado "
    "con el texto: Atlas aprende. Atlas mejora. Gemini ayuda."
)

_ROOT = Path(__file__).resolve().parents[1]


class _Chat:
    name = "chat"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, model, messages):
        self.calls += 1
        return "ruta normal"


class _Models:
    def choose_model(self, _task):
        return "test"


def _real_skill_system():
    """Runtime wiring exactly as bootstrap does: real registry + handlers."""
    skill_system = build_core_skill_system(
        skill_handler_registry=build_builtin_skill_handler_registry(),
    )
    register_builtin_skills(skill_system)
    return skill_system


def _orchestrator():
    chat = _Chat()
    app = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=_Models(),
        memory=ConversationMemory(),
        registry=SimpleNamespace(get=lambda name: chat if name == "chat" else None),
        write_file=SimpleNamespace(execute=lambda *_: None),
        project_root=_ROOT,
        skill_system=_real_skill_system(),
    )
    return app, chat


def test_registered_builtin_skill_is_in_the_real_runtime_registry() -> None:
    skill_system = _real_skill_system()
    skill = skill_system.skill_registry.get("skill.informe-texto-estructurado")
    assert skill is not None
    assert skill.enabled is True
    assert skill.handler_id == "handler.informe-texto-estructurado"


def test_explicit_use_of_registered_skill_executes_the_real_handler() -> None:
    app, chat = _orchestrator()

    response = app.process_prompt(_PROMPT, confirm=lambda _x: None)

    assert "SKILL_GAP" not in response
    assert chat.calls == 0
    assert "Informe Estructurado de Texto" in response
    assert "Palabras: 6" in response
    assert "Caracteres: 42" in response
    assert "1. atlas: 2" in response


def test_unknown_skill_use_keeps_the_existing_gate() -> None:
    app, _chat = _orchestrator()

    response = app.process_prompt(
        "usa la skill skill.desconocida con el texto: Atlas mejora. Atlas aprende.",
        confirm=lambda _x: None,
    )

    assert "SKILL_GAP" in response

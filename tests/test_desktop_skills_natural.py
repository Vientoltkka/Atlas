from __future__ import annotations

import pytest

from bootstrap.bootstrap import Bootstrap
from core.skill_intent import requested_skill_id
from core.skill_resolver import SkillResolutionRequest, SkillResolutionStatus


NEW_DESKTOP_SKILL_IDS = (
    "skill.modo-trabajo",
    "skill.modo-escritura",
    "skill.modo-investigacion",
    "skill.preparar-ventana",
)


def _built_system():
    orchestrator = Bootstrap.build()
    agent_system = orchestrator._atlas_router._agent_system
    return orchestrator, agent_system.skill_system


def test_real_bootstrap_discovers_and_lists_all_new_desktop_skills() -> None:
    _, skill_system = _built_system()

    listed = {skill.skill_id for skill in skill_system.skill_registry.list_skills()}

    for skill_id in NEW_DESKTOP_SKILL_IDS:
        assert skill_id in listed
        resolution = skill_system.skill_resolver.resolve(
            SkillResolutionRequest(required_skill_ids=(skill_id,))
        )
        assert resolution.status is SkillResolutionStatus.RESOLVED


@pytest.mark.parametrize(
    ("prompt", "expected_skill_id"),
    (
        ("Usa la skill Modo Trabajo", "skill.modo-trabajo"),
        ("Usa la skill Modo Escritura con la ventana \"Notas\"", "skill.modo-escritura"),
        (
            "ejecuta la skill Modo Investigacion con \"Notas\" y \"Navegador\"",
            "skill.modo-investigacion",
        ),
        (
            'Usa la skill Preparar Ventana con "Bloc de notas" a la derecha en 800x700',
            "skill.preparar-ventana",
        ),
    ),
)
def test_new_skills_are_discovered_from_natural_text(
    prompt: str,
    expected_skill_id: str,
) -> None:
    _, skill_system = _built_system()

    assert requested_skill_id(prompt, skill_system) == expected_skill_id


def test_natural_text_execution_reports_missing_window_safely(
    monkeypatch,
    capsys,
) -> None:
    orchestrator, _ = _built_system()
    inputs = iter(
        (
            "Usa la skill Modo Trabajo con la ventana \"VentanaQueNoExiste12345\"",
            "salir",
        )
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    orchestrator.start()

    visible = capsys.readouterr().out
    assert "No se encontraron ventanas para 'VentanaQueNoExiste12345'" in visible
    assert "Hasta pronto." in visible


def test_natural_text_execution_reports_contract_violation_safely(
    monkeypatch,
    capsys,
) -> None:
    orchestrator, _ = _built_system()
    inputs = iter(
        (
            "Usa la skill Preparar Ventana con \"Bloc de notas\"",
            "salir",
        )
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    orchestrator.start()

    visible = capsys.readouterr().out
    assert "skill input contract violation" in visible
    assert "Hasta pronto." in visible

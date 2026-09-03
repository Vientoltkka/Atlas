"""Tests for skill intent detection, input extraction and output presentation (V3.2.1-1)."""

from __future__ import annotations

import pytest

from core.skill_intent import (
    present_skill_output,
    requested_skill_id,
    skill_inputs_from_text,
)
from core.skill_registry import SkillDefinition, SkillFieldDefinition, SkillRegistry
from core.skill_system import build_skill_system


def make_skill(
    skill_id: str = "skill.text-uppercase",
    name: str = "Text Uppercase",
    enabled: bool = True,
    input_fields=(),
    input_names: tuple[str, ...] = (),
) -> SkillDefinition:
    return SkillDefinition(
        skill_id=skill_id,
        name=name,
        version="1.0",
        description="Convert text to uppercase.",
        enabled=enabled,
        input_fields=input_fields,
        input_names=input_names,
        execution_target="tool.text-uppercase",
    )


def make_skill_system(*skills: SkillDefinition):
    registry = SkillRegistry(skills)
    return build_skill_system(skill_registry=registry)


def test_detects_explicit_skill_request() -> None:
    system = make_skill_system(make_skill())
    assert requested_skill_id("usa la skill Text Uppercase", system) == "skill.text-uppercase"


def test_detects_by_skill_id() -> None:
    system = make_skill_system(make_skill())
    assert requested_skill_id("ejecuta skill.text-uppercase ahora", system) == "skill.text-uppercase"


def test_no_keyword_means_no_intent() -> None:
    system = make_skill_system(make_skill())
    assert requested_skill_id("hola que tal", system) is None


def test_keyword_without_registered_match_is_none() -> None:
    system = make_skill_system(make_skill())
    assert requested_skill_id("que es una skill", system) is None


def test_disabled_skills_are_ignored() -> None:
    system = make_skill_system(
        make_skill(enabled=False),
        make_skill(skill_id="skill.other", name="Other"),
    )
    assert requested_skill_id("usa la skill Text Uppercase", system) is None


def test_first_registered_match_wins_when_multiple_match() -> None:
    system = make_skill_system(
        make_skill(),
        make_skill(skill_id="skill.text-uppercase-2", name="Text Uppercase Pro"),
    )
    result = requested_skill_id("skill text uppercase", system)
    assert result == "skill.text-uppercase"


@pytest.mark.parametrize(
    "prompt",
    ["di el texto HOLA en mayusculas", "skill sin texto"],
)
def test_no_intent_without_keyword(prompt: str) -> None:
    system = make_skill_system(make_skill())
    assert requested_skill_id(prompt, system) is None


def test_inputs_from_quoted_text() -> None:
    skill = make_skill(input_fields=(SkillFieldDefinition("text", "string", True),))
    inputs = skill_inputs_from_text('convierte "hola mundo" con la skill', skill)
    assert inputs == {"text": "hola mundo"}


def test_inputs_from_text_label() -> None:
    skill = make_skill(input_fields=(SkillFieldDefinition("text", "string", True),))
    inputs = skill_inputs_from_text("skill texto: hola atlas", skill)
    assert inputs == {"text": "hola atlas"}


def test_inputs_empty_for_non_text_skills() -> None:
    skill = make_skill(input_names=("source", "target"))
    assert skill_inputs_from_text('skill "algo"', skill) == {}


def test_inputs_empty_without_extractable_value() -> None:
    skill = make_skill(input_fields=(SkillFieldDefinition("text", "string", True),))
    assert skill_inputs_from_text("ejecuta la skill", skill) == {}


def _layout_skill(*fields: SkillFieldDefinition) -> SkillDefinition:
    return make_skill(
        skill_id="skill.preparar-ventana",
        name="Preparar Ventana",
        input_fields=fields,
    )


_WINDOW_FIELDS = (
    SkillFieldDefinition("window_title", "string", True),
    SkillFieldDefinition("width", "integer", True),
    SkillFieldDefinition("height", "integer", True),
    SkillFieldDefinition("position", "string", False),
)


def test_layout_inputs_extract_title_size_and_position() -> None:
    skill = _layout_skill(*_WINDOW_FIELDS)
    inputs = skill_inputs_from_text(
        'Usa la skill Preparar Ventana con "Bloc de notas" a la derecha en 800x700',
        skill,
    )
    assert inputs == {
        "window_title": "Bloc de notas",
        "width": 800,
        "height": 700,
        "position": "derecha",
    }


def test_layout_inputs_support_spaced_size_and_center() -> None:
    skill = _layout_skill(*_WINDOW_FIELDS)
    inputs = skill_inputs_from_text(
        'skill Preparar Ventana con "Notas" en 800 x 700 al centro',
        skill,
    )
    assert inputs == {
        "window_title": "Notas",
        "width": 800,
        "height": 700,
        "position": "centro",
    }


def test_layout_inputs_default_to_empty_without_parameters() -> None:
    skill = _layout_skill(*_WINDOW_FIELDS)
    assert skill_inputs_from_text("usa la skill Preparar Ventana", skill) == {}


def test_layout_inputs_map_second_quote_to_secondary_title() -> None:
    skill = _layout_skill(
        SkillFieldDefinition("window_title", "string", False),
        SkillFieldDefinition("secondary_title", "string", False),
    )
    inputs = skill_inputs_from_text(
        'skill Modo Investigacion con "Notas" y "Navegador"',
        skill,
    )
    assert inputs == {
        "window_title": "Notas",
        "secondary_title": "Navegador",
    }


def test_layout_inputs_ignored_for_non_layout_skills() -> None:
    skill = make_skill(input_names=("source", "target"))
    assert skill_inputs_from_text(
        'skill rara "titulo" 800x700 a la derecha',
        skill,
    ) == {}


def test_presentation_prefers_result_key() -> None:
    assert present_skill_output({"result": "OK", "other": "x"}) == "OK"


def test_presentation_single_value() -> None:
    assert present_skill_output({"text": "valor"}) == "valor"


def test_presentation_multiple_values_without_result() -> None:
    output = present_skill_output({"a": 1, "b": 2})
    assert output == "a: 1\nb: 2"


def test_presentation_handles_empty_and_none() -> None:
    assert present_skill_output({}) == ""
    assert present_skill_output(None) == ""

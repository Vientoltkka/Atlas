"""Tests for deterministic structured text report generation capability."""

from __future__ import annotations

import time

import pytest

from bootstrap.skill_system import (
    INFORME_TEXTO_ESTRUCTURADO_HANDLER_ID,
    _informe_texto_estructurado_handler,
    build_builtin_skill_handler_registry,
)
from core.skill_execution_context import SkillExecutionContext
from use_cases.structured_text_report import generate_structured_text_report


def test_generate_structured_text_report_exact_example() -> None:
    text = "Hola mundo.\nHola Atlas."
    result = generate_structured_text_report(text)

    assert result["char_count"] == 23
    assert result["word_count"] == 4
    assert result["line_count"] == 2
    assert result["paragraph_count"] == 1
    assert result["unique_word_count"] == 3

    expected_report = (
        "# Informe Estructurado de Texto\n\n"
        "## Metricas Generales\n"
        "- Caracteres: 23\n"
        "- Palabras: 4\n"
        "- Lineas: 2\n"
        "- Parrafos: 1\n"
        "- Palabras Unicas: 3\n\n"
        "## Frecuencia de Palabras\n"
        "1. hola: 2\n"
        "2. atlas: 1\n"
        "3. mundo: 1"
    )
    assert result["report"] == expected_report


def test_generate_structured_text_report_empty_string() -> None:
    result = generate_structured_text_report("")

    assert result["char_count"] == 0
    assert result["word_count"] == 0
    assert result["line_count"] == 0
    assert result["paragraph_count"] == 0
    assert result["unique_word_count"] == 0
    assert "(Sin palabras)" in result["report"]


def test_generate_structured_text_report_invalid_input() -> None:
    with pytest.raises(ValueError, match="text must be a string"):
        generate_structured_text_report(123)  # type: ignore[arg-type]


def test_informe_texto_estructurado_handler_success() -> None:
    context = SkillExecutionContext(deadline=time.monotonic() + 1.0)
    inputs = {"text": "Hola mundo.\nHola Atlas."}

    outcome = _informe_texto_estructurado_handler(inputs, execution_context=context)

    assert outcome["char_count"] == 23
    assert outcome["word_count"] == 4
    assert "Informe Estructurado de Texto" in str(outcome["report"])


def test_informe_texto_estructurado_handler_cancelled_context() -> None:
    context = SkillExecutionContext(deadline=time.monotonic() + 1.0)
    context.cancel()
    inputs = {"text": "Hola mundo."}

    with pytest.raises(
        RuntimeError, match="informe texto estructurado execution context is unavailable"
    ):
        _informe_texto_estructurado_handler(inputs, execution_context=context)


def test_informe_texto_estructurado_handler_invalid_input() -> None:
    context = SkillExecutionContext(deadline=time.monotonic() + 1.0)
    inputs = {"text": 123}

    with pytest.raises(ValueError, match="text must be a string"):
        _informe_texto_estructurado_handler(inputs, execution_context=context)


def test_handler_registered_in_builtin_registry() -> None:
    registry = build_builtin_skill_handler_registry()
    handler = registry.get(INFORME_TEXTO_ESTRUCTURADO_HANDLER_ID)
    assert handler is not None

    context = SkillExecutionContext(deadline=time.monotonic() + 1.0)
    outcome = handler({"text": "Test atlas skill."}, execution_context=context)
    assert outcome["word_count"] == 3

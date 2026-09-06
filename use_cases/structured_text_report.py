"""UseCase for deterministic structured text report generation."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any


def generate_structured_text_report(text: str) -> dict[str, Any]:
    """Generate a deterministic structured analysis report from a text string.

    Example input:
        text = "Hola mundo.\nHola Atlas."

    Example output dict:
        {
            "report": "# Informe Estructurado de Texto\n\n## Metricas Generales\n- Caracteres: 23\n- Palabras: 4\n- Lineas: 2\n- Parrafos: 1\n- Palabras Unicas: 3\n\n## Frecuencia de Palabras\n1. hola: 2\n2. atlas: 1\n3. mundo: 1",
            "char_count": 23,
            "word_count": 4,
            "line_count": 2,
            "paragraph_count": 1,
            "unique_word_count": 3,
        }
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    char_count = len(text)
    lines = text.splitlines() if text else []
    line_count = len(lines)

    paragraphs = [p for p in text.split("\n\n") if p.strip()] if text else []
    paragraph_count = len(paragraphs)

    words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    word_count = len(words)
    unique_word_count = len(set(words))

    counts = Counter(words)
    sorted_words = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    report_lines = [
        "# Informe Estructurado de Texto",
        "",
        "## Metricas Generales",
        f"- Caracteres: {char_count}",
        f"- Palabras: {word_count}",
        f"- Lineas: {line_count}",
        f"- Parrafos: {paragraph_count}",
        f"- Palabras Unicas: {unique_word_count}",
        "",
        "## Frecuencia de Palabras",
    ]

    if sorted_words:
        for idx, (word, count) in enumerate(sorted_words[:10], start=1):
            report_lines.append(f"{idx}. {word}: {count}")
    else:
        report_lines.append("(Sin palabras)")

    report = "\n".join(report_lines)

    return {
        "report": report,
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "paragraph_count": paragraph_count,
        "unique_word_count": unique_word_count,
    }

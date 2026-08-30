"""Small PDF rendering service for Atlas documents."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import WrapMode


class PdfService:
    """Render plain Spanish text into a paginated PDF file."""

    def create(self, content: str, target: Path) -> None:
        if not isinstance(content, str):
            raise TypeError("content debe ser str.")
        if not content.strip():
            raise ValueError("content no puede estar vacio.")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("helvetica", size=11)

        usable_width = pdf.w - pdf.l_margin - pdf.r_margin
        for line in self._export_lines(content):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                usable_width,
                6,
                text=line or " ",
                markdown=True,
                new_x="LMARGIN",
                new_y="NEXT",
                wrapmode=WrapMode.CHAR,
            )

        pdf.output(str(target))

    @staticmethod
    def _export_lines(content: str) -> list[str]:
        """Keep simple training text while dropping the PDF-refusal artifact."""
        replacements = str.maketrans(
            {
                "\u2013": "-",
                "\u2014": "-",
                "\u2026": "...",
                "\u00a0": " ",
            }
        )
        return [
            line.translate(replacements)
            for line in (content.splitlines() or [content])
            if "no puedo crear archivos pdf" not in line.casefold()
        ]
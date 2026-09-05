"""Generated directories must never break the bootstrap project index scan."""

from __future__ import annotations

from pathlib import Path

from use_cases.read_project_index import ReadProjectIndexUseCase


def test_generated_directories_are_excluded_from_the_index(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    broken = "def (bad):\n"
    (tmp_path / "pytest-tmp").mkdir()
    (tmp_path / "pytest-tmp" / "new.py").write_text(broken, encoding="utf-8")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "broken.py").write_text(broken, encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "broken.py").write_text(broken, encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "broken.py").write_text(broken, encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "broken.py").write_text(broken, encoding="utf-8")

    index = ReadProjectIndexUseCase().execute(str(tmp_path))

    indexed = {entry["path"] for entry in index}
    assert indexed == {str(tmp_path / "app.py")}

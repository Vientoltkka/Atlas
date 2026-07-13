"""List Python files use case."""

from __future__ import annotations

from pathlib import Path


class ListPythonFilesUseCase:
    """Return every Python file inside a project."""

    def execute(
        self,
        root: str,
    ) -> list[str]:

        project = Path(root)

        return sorted(
            str(path)
            for path in project.rglob("*.py")
            if ".venv" not in path.parts
            and "__pycache__" not in path.parts
        )
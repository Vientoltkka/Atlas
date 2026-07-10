"""Project service."""

from __future__ import annotations

from pathlib import Path


class ProjectService:
    """High level project operations."""

    @staticmethod
    def list_files(
        root: str,
        extensions: tuple[str, ...] = (".py",),
    ) -> list[str]:
        """Return every project file."""

        project = Path(root)

        if not project.exists():
            raise FileNotFoundError(root)

        return sorted(
            str(path)
            for path in project.rglob("*")
            if path.is_file()
            and path.suffix in extensions
        )
"""Read an entire Python project."""

from __future__ import annotations

from use_cases.list_python_files import ListPythonFilesUseCase
from use_cases.read_file import ReadFileUseCase


class ReadProjectUseCase:
    """Read every Python file in a project."""

    def __init__(
        self,
        list_python_files: ListPythonFilesUseCase,
        read_file: ReadFileUseCase,
    ) -> None:
        self._list_python_files = list_python_files
        self._read_file = read_file

    def execute(
        self,
        root: str,
    ) -> list[dict[str, str]]:
        project = []

        files = self._list_python_files.execute(root)

        for path in files:
            content = self._read_file.execute(path)

            project.append(
                {
                    "path": path,
                    "content": content,
                }
            )

        return project

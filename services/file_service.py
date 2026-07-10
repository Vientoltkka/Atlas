"""High level file operations."""

from pathlib import Path


class FileService:
    """High level file operations."""

    @staticmethod
    def read(path: str) -> str:

        return Path(path).read_text(
            encoding="utf-8"
        )

    @staticmethod
    def write(
        path: str,
        content: str,
    ) -> None:

        Path(path).write_text(
            content,
            encoding="utf-8",
        )
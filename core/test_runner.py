"""Generalized, bounded pytest execution primitive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

DEFAULT_TEST_TIMEOUT_SECONDS = 120.0
DEFAULT_OUTPUT_TAIL_LINES = 40


@dataclass(frozen=True, slots=True)
class TestRunResult:
    """    Structured outcome with the minimum output needed for diagnosis."""

    __test__ = False

    passed: bool
    exit_code: int | None
    timed_out: bool
    detail: str
    output_tail: str
    command: tuple[str, ...]
    basetemp: Path | None


class PytestRunner:
    """Run parametrizable pytest paths in isolation with a hard timeout."""

    def __init__(
        self,
        project_root: Path,
        *,
        default_timeout: float = DEFAULT_TEST_TIMEOUT_SECONDS,
        output_tail_lines: int = DEFAULT_OUTPUT_TAIL_LINES,
    ) -> None:
        if default_timeout <= 0:
            raise ValueError("default_timeout must be positive.")
        if output_tail_lines <= 0:
            raise ValueError("output_tail_lines must be positive.")
        self._root = project_root.resolve()
        self._default_timeout = float(default_timeout)
        self._tail_lines = int(output_tail_lines)

    def run(
        self,
        test_paths: Sequence[str],
        *,
        timeout: float | None = None,
        basetemp: Path | str | None = None,
        cleanup_basetemp: bool = True,
    ) -> TestRunResult:
        if not test_paths:
            raise ValueError("at least one test path is required.")
        effective_timeout = self._default_timeout if timeout is None else float(timeout)
        if effective_timeout <= 0:
            raise ValueError("timeout must be positive.")
        arguments = tuple(self._validated_argument(item) for item in test_paths)
        if basetemp is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="atlas-pytest-"))
        else:
            temp_dir = Path(basetemp)
            temp_dir.mkdir(parents=True, exist_ok=True)
        command = (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *arguments,
            "--basetemp",
            str(temp_dir),
        )
        try:
            completed = subprocess.run(
                command,
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            tail = self._tail(_decoded(expired.stdout) + _decoded(expired.stderr))
            if cleanup_basetemp:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return TestRunResult(
                passed=False,
                exit_code=None,
                timed_out=True,
                detail=f"pytest timed out after {effective_timeout} seconds.",
                output_tail=tail,
                command=command,
                basetemp=temp_dir,
            )
        if cleanup_basetemp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        tail = self._tail(completed.stdout + completed.stderr)
        return TestRunResult(
            passed=completed.returncode == 0,
            exit_code=completed.returncode,
            timed_out=False,
            detail=f"pytest exited with code {completed.returncode}.",
            output_tail=tail,
            command=command,
            basetemp=temp_dir,
        )

    def _validated_argument(self, item: str) -> str:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("test paths must be non-empty strings.")
        path_part = item.split("::", 1)[0]
        candidate = Path(path_part)
        resolved = candidate.resolve() if candidate.is_absolute() else (self._root / candidate).resolve()
        if resolved == self._root or self._root not in resolved.parents:
            raise ValueError("test paths must remain inside the project root.")
        return item

    def _tail(self, text: str) -> str:
        lines = [line for line in text.splitlines() if line.strip()]
        return "\n".join(lines[-self._tail_lines :])


def _decoded(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

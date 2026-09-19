from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile


class RepairWorkspaceError(RuntimeError):
    """Raised when an isolated repair workspace cannot be safely managed."""


@dataclass(frozen=True, slots=True)
class RepairWorkspace:
    source_root: Path
    workspace_root: Path
    base_revision: str


class RepairWorkspaceManager:
    """Create and remove detached Git worktrees for supervised repairs."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = Path(project_root).resolve()

    def create(self) -> RepairWorkspace:
        if not (self._project_root / ".git").exists():
            raise RepairWorkspaceError(
                "project root is not a Git working tree."
            )

        tracked_status = self._git(
            "status",
            "--porcelain",
            "--untracked-files=no",
            cwd=self._project_root,
        ).strip()
        if tracked_status:
            raise RepairWorkspaceError(
                "project has tracked changes not represented by HEAD."
            )

        tracked_status = self._git(
            "status",
            "--porcelain",
            "--untracked-files=no",
            cwd=self._project_root,
        ).strip()
        if tracked_status:
            raise RepairWorkspaceError(
                "project has tracked changes not represented by HEAD."
            )

        revision = self._git(
            "rev-parse",
            "--verify",
            "HEAD",
            cwd=self._project_root,
        ).strip()
        if not revision:
            raise RepairWorkspaceError("could not resolve HEAD.")

        parent = Path(
            tempfile.mkdtemp(prefix="atlas-repair-")
        ).resolve()
        workspace_root = parent / "workspace"

        try:
            self._git(
                "worktree",
                "add",
                "--detach",
                str(workspace_root),
                revision,
                cwd=self._project_root,
            )
        except Exception:
            shutil.rmtree(parent, ignore_errors=True)
            raise

        return RepairWorkspace(
            source_root=self._project_root,
            workspace_root=workspace_root,
            base_revision=revision,
        )

    def remove(self, workspace: RepairWorkspace) -> None:
        source = workspace.source_root.resolve()
        target = workspace.workspace_root.resolve()

        if source != self._project_root:
            raise RepairWorkspaceError(
                "workspace does not belong to this project."
            )
        if target == self._project_root:
            raise RepairWorkspaceError(
                "refusing to remove the source working tree."
            )

        parent = target.parent

        try:
            self._git(
                "worktree",
                "remove",
                "--force",
                str(target),
                cwd=self._project_root,
            )
        finally:
            shutil.rmtree(parent, ignore_errors=True)

    def _git(
        self,
        *args: str,
        cwd: Path,
    ) -> str:
        try:
            completed = subprocess.run(
                ("git", *args),
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RepairWorkspaceError(
                f"git command failed: {error}"
            ) from error

        if completed.returncode != 0:
            detail = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"exit code {completed.returncode}"
            )
            raise RepairWorkspaceError(
                f"git {' '.join(args)} failed: {detail}"
            )

        return completed.stdout

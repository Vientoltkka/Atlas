from pathlib import Path
import subprocess

import pytest

from core.repair_workspace import (
    RepairWorkspaceError,
    RepairWorkspaceManager,
)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()

    _git(root, "init")
    _git(root, "config", "user.email", "atlas-tests@example.invalid")
    _git(root, "config", "user.name", "Atlas Tests")

    (root / "sample.txt").write_text("original\n", encoding="utf-8")
    _git(root, "add", "sample.txt")
    _git(root, "commit", "-m", "initial")

    return root


def test_create_uses_detached_isolated_worktree(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    manager = RepairWorkspaceManager(root)

    workspace = manager.create()

    try:
        assert workspace.source_root == root.resolve()
        assert workspace.workspace_root != root.resolve()
        assert workspace.workspace_root.is_dir()
        assert workspace.base_revision == _git(root, "rev-parse", "HEAD")
        assert (
            _git(workspace.workspace_root, "rev-parse", "--is-inside-work-tree")
            == "true"
        )
        assert (
            _git(workspace.workspace_root, "rev-parse", "--abbrev-ref", "HEAD")
            == "HEAD"
        )
    finally:
        manager.remove(workspace)


def test_edit_in_workspace_does_not_touch_source(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    manager = RepairWorkspaceManager(root)
    workspace = manager.create()

    try:
        isolated_file = workspace.workspace_root / "sample.txt"
        isolated_file.write_text("changed\n", encoding="utf-8")

        assert isolated_file.read_text(encoding="utf-8") == "changed\n"
        assert (root / "sample.txt").read_text(encoding="utf-8") == "original\n"
    finally:
        manager.remove(workspace)


def test_remove_deletes_isolated_workspace_only(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    manager = RepairWorkspaceManager(root)
    workspace = manager.create()
    target = workspace.workspace_root

    manager.remove(workspace)

    assert not target.exists()
    assert root.exists()
    assert (root / "sample.txt").read_text(encoding="utf-8") == "original\n"


def test_non_git_root_is_rejected(tmp_path: Path) -> None:
    manager = RepairWorkspaceManager(tmp_path)

    with pytest.raises(RepairWorkspaceError):
        manager.create()

def test_create_rejects_tracked_dirty_source(tmp_path: Path) -> None:
    import subprocess

    root = tmp_path

    subprocess.run(
        ("git", "init"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "config", "user.email", "atlas-tests@example.invalid"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Atlas Tests"),
        cwd=root,
        check=True,
        capture_output=True,
    )

    tracked = root / "tracked.txt"
    tracked.write_text("baseline\\n", encoding="utf-8")

    subprocess.run(
        ("git", "add", "-A"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "commit", "-m", "tracked baseline"),
        cwd=root,
        check=True,
        capture_output=True,
    )

    tracked.write_text("dirty\\n", encoding="utf-8")

    manager = RepairWorkspaceManager(root)

    with pytest.raises(RepairWorkspaceError):
        manager.create()


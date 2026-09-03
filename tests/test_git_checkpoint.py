from __future__ import annotations

from pathlib import Path

import pytest

from core.git_checkpoint import GitCheckpointManager


def test_write_and_restore_roundtrip_returns_exact_original(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("original\n", encoding="utf-8")
    manager = GitCheckpointManager(tmp_path)

    manager.checkpoint(("fixture.txt",))
    manager.write("fixture.txt", "changed\n")
    assert target.read_text(encoding="utf-8") == "changed\n"

    manager.restore()

    assert target.read_text(encoding="utf-8") == "original\n"
    assert manager.active is None
    assert [entry["event"] for entry in manager.audit_log] == [
        "checkpoint_created",
        "write",
        "restored",
    ]


def test_restore_refuses_unexpected_external_modification(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("original\n", encoding="utf-8")
    manager = GitCheckpointManager(tmp_path)
    manager.checkpoint(("fixture.txt",))
    target.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="externally"):
        manager.restore()

    assert target.read_text(encoding="utf-8") == "tampered\n"
    assert manager.active is not None


def test_detect_external_changes_reports_modified_deleted_and_created(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_text("keep\n", encoding="utf-8")
    absent = tmp_path / "absent.txt"
    manager = GitCheckpointManager(tmp_path)
    manager.checkpoint(("existing.txt", "absent.txt"))
    manager.write("absent.txt", "created\n")

    existing.write_text("tampered\n", encoding="utf-8")
    (tmp_path / "absent.txt").unlink()

    changes = {change.relative: change.kind for change in manager.detect_external_changes()}

    assert changes == {"existing.txt": "modified", "absent.txt": "deleted"}


def test_restore_of_file_created_after_checkpoint_deletes_it(tmp_path: Path) -> None:
    manager = GitCheckpointManager(tmp_path)
    manager.checkpoint(("sub/created.txt",))
    manager.write("sub/created.txt", "created\n")
    target = tmp_path / "sub" / "created.txt"
    assert target.exists()

    manager.restore()

    assert not target.exists()
    assert not (tmp_path / "sub").exists()


def test_targets_outside_allowed_scope_are_rejected(tmp_path: Path) -> None:
    inside = tmp_path / "tests"
    inside.mkdir()
    (inside / "scoped.txt").write_text("scoped\n", encoding="utf-8")
    outside = tmp_path / "fixture.txt"
    outside.write_text("outside\n", encoding="utf-8")
    manager = GitCheckpointManager(tmp_path, allowed_scope=("tests",))

    with pytest.raises(ValueError, match="scope"):
        manager.checkpoint(("fixture.txt",))

    scoped = manager.checkpoint(("tests/scoped.txt",))
    assert scoped.snapshots == {"tests/scoped.txt": "scoped\n"}


def test_paths_escaping_project_root_are_rejected(tmp_path: Path) -> None:
    manager = GitCheckpointManager(tmp_path)

    with pytest.raises(ValueError, match="project root"):
        manager.checkpoint(("../outside.txt",))


def test_secret_files_are_rejected(tmp_path: Path) -> None:
    manager = GitCheckpointManager(tmp_path)

    with pytest.raises(ValueError, match="secret"):
        manager.checkpoint((".env",))


def test_write_only_accepts_tracked_paths_and_single_active_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "tracked.txt").write_text("a\n", encoding="utf-8")
    manager = GitCheckpointManager(tmp_path)
    manager.checkpoint(("tracked.txt",))

    with pytest.raises(ValueError, match="tracked by the active checkpoint"):
        manager.write("other.txt", "b\n")
    with pytest.raises(RuntimeError, match="active checkpoint already exists"):
        manager.checkpoint(("tracked.txt",))

    manager.restore()


def test_accept_keeps_current_state_and_deactivates_checkpoint(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("original\n", encoding="utf-8")
    manager = GitCheckpointManager(tmp_path)
    manager.checkpoint(("fixture.txt",))
    manager.write("fixture.txt", "changed\n")

    manager.accept()

    assert target.read_text(encoding="utf-8") == "changed\n"
    assert manager.active is None


def test_operations_without_active_checkpoint_are_rejected(tmp_path: Path) -> None:
    manager = GitCheckpointManager(tmp_path)

    with pytest.raises(RuntimeError, match="no active checkpoint"):
        manager.restore()
    with pytest.raises(RuntimeError, match="no active checkpoint"):
        manager.detect_external_changes()
    with pytest.raises(RuntimeError, match="no active checkpoint"):
        manager.accept()

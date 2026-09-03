"""Local snapshot/restore checkpoint primitive for supervised autonomous work.

This module never spawns git processes: push, reset and stash are structurally
impossible because no git command is executed. Snapshots are plain in-memory
copies of scoped text files, and restore refuses to run when a tracked file
changed outside the sanctioned write path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

MAX_CHECKPOINT_FILES = 64


@dataclass(frozen=True, slots=True)
class GitCheckpoint:
    """Immutable snapshot of the scoped files taken before any change."""

    checkpoint_id: str
    snapshots: Mapping[str, str | None]
    digests: Mapping[str, str]
    created_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", MappingProxyType(dict(self.snapshots)))
        object.__setattr__(self, "digests", MappingProxyType(dict(self.digests)))


@dataclass(frozen=True, slots=True)
class ExternalChange:
    """One tracked file whose on-disk state diverged from the expected state."""

    relative: str
    kind: str


class GitCheckpointManager:
    """Create and restore local checkpoints for files inside an allowed scope."""

    def __init__(self, project_root: Path, *, allowed_scope: Sequence[str] = (".",)) -> None:
        self._root = project_root.resolve()
        scope_entries = tuple(dict.fromkeys(allowed_scope)) or (".",)
        self._scope_dirs = tuple(
            self._safe_scope_dir(entry) for entry in scope_entries
        )
        self._active: GitCheckpoint | None = None
        self._expected: dict[str, str | None] = {}
        self.audit_log: list[dict[str, object]] = []

    @property
    def active(self) -> GitCheckpoint | None:
        return self._active

    def checkpoint(self, relative_paths: Sequence[str]) -> GitCheckpoint:
        """Snapshot the scoped files without writing anything."""
        if self._active is not None:
            raise RuntimeError("an active checkpoint already exists; restore or accept it first.")
        relatives = tuple(dict.fromkeys(relative_paths))
        if not relatives:
            raise ValueError("at least one relative path is required.")
        if len(relatives) > MAX_CHECKPOINT_FILES:
            raise ValueError("checkpoint exceeds the safe file limit.")
        targets = {relative: self._target(relative) for relative in relatives}
        snapshots: dict[str, str | None] = {}
        digests: dict[str, str] = {}
        for relative, target in targets.items():
            content = self._read_current(target)
            snapshots[relative] = content
            if content is not None:
                digests[relative] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        created_at = datetime.now(timezone.utc).isoformat()
        checkpoint = GitCheckpoint(
            checkpoint_id=self._checkpoint_id(digests, created_at),
            snapshots=snapshots,
            digests=digests,
            created_at=created_at,
        )
        self._active = checkpoint
        self._expected = dict(snapshots)
        self._record("checkpoint_created", checkpoint)
        return checkpoint

    def write(self, relative: str, content: str) -> None:
        """Apply a sanctioned write through the only path restore trusts."""
        checkpoint = self._require_active()
        if relative not in self._expected:
            raise ValueError("write target is not tracked by the active checkpoint.")
        if not isinstance(content, str):
            raise TypeError("content must be text.")
        target = self._target(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._expected[relative] = content
        self._record("write", checkpoint, detail=relative)

    def detect_external_changes(self) -> tuple[ExternalChange, ...]:
        """Return every tracked file whose on-disk state is not the expected one."""
        self._require_active()
        changes: list[ExternalChange] = []
        for relative, expected in self._expected.items():
            current = self._read_current(self._target(relative))
            if current == expected:
                continue
            if expected is None:
                kind = "created"
            elif current is None:
                kind = "deleted"
            else:
                kind = "modified"
            changes.append(ExternalChange(relative, kind))
        return tuple(changes)

    def restore(self, *, reason: str = "restore") -> None:
        """Restore the snapshot, refusing unexpected external changes."""
        checkpoint = self._require_active()
        changes = self.detect_external_changes()
        if changes:
            described = ", ".join(f"{item.relative}:{item.kind}" for item in changes)
            self._record("restore_refused", checkpoint, detail=reason + ": " + described)
            raise RuntimeError(
                "checkpoint restore refused because tracked files changed externally: " + described
            )
        for relative, original in checkpoint.snapshots.items():
            target = self._target(relative)
            if original is None:
                if target.exists():
                    target.unlink()
                    self._remove_empty_parents(target.parent)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(original, encoding="utf-8")
            self._expected[relative] = original
        self._active = None
        self._record("restored", checkpoint, detail=reason)

    def accept(self) -> None:
        """Deactivate the checkpoint while keeping the current file state."""
        checkpoint = self._require_active()
        self._active = None
        self._record("accepted", checkpoint)

    def _safe_scope_dir(self, entry: str) -> Path:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("allowed scope entries must be non-empty relative paths.")
        scope_dir = (self._root / entry).resolve()
        if scope_dir != self._root and self._root not in scope_dir.parents:
            raise ValueError("allowed scope must remain inside the project root.")
        return scope_dir

    def _target(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("relative paths must be non-empty strings.")
        candidate = (self._root / relative).resolve()
        if candidate == self._root or self._root not in candidate.parents:
            raise ValueError("checkpoint scope must remain inside the project root.")
        if not any(
            scope_dir == candidate.parent
            or scope_dir in candidate.parents
            or scope_dir == self._root
            for scope_dir in self._scope_dirs
        ):
            raise ValueError("checkpoint target is outside the allowed scope.")
        if candidate.suffix == ".env" or ".env" in candidate.name:
            raise ValueError("secret files are outside the checkpoint scope.")
        return candidate

    @staticmethod
    def _read_current(target: Path) -> str | None:
        if not target.exists():
            return None
        if not target.is_file():
            raise ValueError("checkpoint targets must be regular files.")
        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError("checkpoint requires readable utf-8 text files.") from error

    @staticmethod
    def _checkpoint_id(digests: Mapping[str, str], created_at: str) -> str:
        payload = json.dumps(
            {"digests": dict(digests), "created_at": created_at},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "ckpt-" + hashlib.sha256(payload).hexdigest()[:12]

    def _require_active(self) -> GitCheckpoint:
        if self._active is None:
            raise RuntimeError("no active checkpoint exists.")
        return self._active

    def _record(self, event: str, checkpoint: GitCheckpoint, detail: str = "") -> None:
        self.audit_log.append(
            {
                "event": event,
                "checkpoint_id": checkpoint.checkpoint_id,
                "files": tuple(checkpoint.snapshots),
                "detail": detail,
            }
        )

    def _remove_empty_parents(self, path: Path) -> None:
        while path != self._root:
            try:
                path.rmdir()
            except OSError:
                return
            path = path.parent

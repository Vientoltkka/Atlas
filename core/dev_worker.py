"""Bounded computer worker for in-repo development goals (V1).

Composes the existing primitives (PytestRunner, path-sandbox patterns from
GitCheckpointManager / AutonomousTaskRunner) into a deterministic
GOAL -> INSPECT -> MODIFY -> TEST -> REPORT loop that always terminates in
DONE, BLOCKED or WAITING_APPROVAL.

Safety invariants, by construction:

- Every path is resolved against the workspace root; anything outside it is
  blocked before a single byte is touched.
- Secret and protected files (.env, secrets, supervisor/security/authorization
  modules) can never be edited by this worker.
- Sensitive commands (push, reset, stash, deletes, dependency installs, any
  contact with secrets or the network) are never executed: they stop the run
  in WAITING_APPROVAL with the offending action recorded as evidence.
- Focal tests get at most ``MAX_TEST_RETRIES`` re-runs; no infinite loops.
- The worker never spawns git write operations and never deletes files.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import shlex
import subprocess

from core.test_runner import PytestRunner

MAX_TEST_RETRIES = 2
MAX_READ_CHARS = 4000
MAX_SEARCH_FILES = 2000
MAX_SEARCH_MATCHES = 50
MAX_EVIDENCE_CHARS = 300
COMMAND_TIMEOUT_SECONDS = 60.0
DEFAULT_COMMAND_TAIL_LINES = 20

SEARCH_SKIP_DIRS = frozenset(
    {".git", ".venv", "__pycache__", ".tmp", "node_modules", "artifacts", "dist", "logs", "models", "memory", "data"}
)
SEARCH_TEXT_EXTENSIONS = frozenset(
    {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".js", ".ts", ".ps1", ".sh", ".bat", ".csv", ".sql", ".html", ".css"}
)

SENSITIVE_COMMAND_TOKENS = (
    "push",
    "reset",
    "stash",
    "clean",
    "rebase",
    "filter-branch",
    "install",
    "uninstall",
    "curl",
    "wget",
    "invoke-webrequest",
    "invoke-restmethod",
)
DELETE_COMMAND_TOKENS = ("rm", "del", "rmdir", "remove-item", "rd")
SECRET_PATH_TOKENS = (".env", "secret")
PROTECTED_EDIT_TOKENS = (
    "supervisor",
    "security",
    "authorization",
    "effect_permissions",
    "secrets",
    ".env",
    ".git",
)


class SandboxViolationError(RuntimeError):
    """Raised when a step targets a path outside the workspace or protected."""


class ActionRequiresApprovalError(RuntimeError):
    """Raised when a step maps to a sensitive action that needs approval."""

    def __init__(self, description: str, reason: str) -> None:
        super().__init__(reason)
        self.description = description
        self.reason = reason


class DevWorkerAction(str, Enum):
    """The four bounded actions a dev-worker step may perform."""

    READ = "read"
    SEARCH = "search"
    EDIT = "edit"
    TEST = "test"
    COMMAND = "command"


@dataclass(frozen=True, slots=True)
class DevStep:
    """One declarative, bounded step of a dev goal."""

    action: DevWorkerAction
    path: str | None = None
    operation: str = "replace_text"
    old_text: str = ""
    new_text: str = ""
    content: str = ""
    pattern: str | None = None
    test_paths: tuple[str, ...] = ()
    command: str | tuple[str, ...] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.action, DevWorkerAction):
            raise ValueError("action must be a DevWorkerAction value.")
        if self.action is DevWorkerAction.EDIT:
            if not self.path:
                raise ValueError("edit steps require a path.")
            if self.operation not in ("replace_text", "append_text", "full_file"):
                raise ValueError("edit operation must be replace_text, append_text or full_file.")
            if self.operation == "replace_text" and (not self.old_text or self.new_text == ""):
                raise ValueError("replace_text requires non-empty old_text and new_text.")
            if self.operation == "append_text" and not self.content:
                raise ValueError("append_text requires non-empty content.")
        if self.action is DevWorkerAction.TEST and not self.test_paths:
            raise ValueError("test steps require at least one test path.")
        if self.action is DevWorkerAction.COMMAND and not self.command:
            raise ValueError("command steps require a command.")


@dataclass(frozen=True, slots=True)
class DevEvidence:
    """One auditable record produced while executing the goal."""

    step_index: int
    action: DevWorkerAction
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "detail",
            _bound(self.detail, MAX_EVIDENCE_CHARS),
        )


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Bounded outcome of one sandboxed command execution."""

    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class DevWorkerResult:
    """Terminal outcome of one bounded dev-worker run."""

    goal: str
    status: str
    evidence: tuple[DevEvidence, ...] = ()
    error: str | None = None
    pending_action: str | None = None
    pending_reason: str | None = None

    @property
    def report(self) -> str:
        """Single human-readable report with evidence."""
        lines = [f"GOAL: {self.goal}", f"STATUS: {self.status}"]
        if self.error:
            lines.append(f"ERROR: {self.error}")
        if self.pending_action:
            lines.append(f"PENDING_ACTION: {self.pending_action}")
            lines.append(f"PENDING_REASON: {self.pending_reason}")
        lines.append("EVIDENCE:")
        lines.extend(f"- {item.detail}" for item in self.evidence)
        return "\n".join(lines)


def default_command_runner(
    root: Path,
    command: tuple[str, ...],
    *,
    timeout: float = COMMAND_TIMEOUT_SECONDS,
    tail_lines: int = DEFAULT_COMMAND_TAIL_LINES,
) -> CommandResult:
    """Run one non-sensitive command inside the workspace root."""
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        return CommandResult(
            exit_code=None,
            stdout_tail=_tail(_decoded(expired.stdout), tail_lines),
            stderr_tail=_tail(_decoded(expired.stderr), tail_lines),
            timed_out=True,
        )
    return CommandResult(
        exit_code=completed.returncode,
        stdout_tail=_tail(completed.stdout, tail_lines),
        stderr_tail=_tail(completed.stderr, tail_lines),
    )


def classify_command(command: str | tuple[str, ...]) -> str | None:
    """Return an approval reason for sensitive commands, else None."""
    if isinstance(command, str):
        tokens = [token.lower() for token in shlex.split(command, posix=False) if token.strip('"')]
    else:
        tokens = [token.lower() for token in command]
    if not tokens:
        return None
    joined = " ".join(tokens)
    if tokens[0] == "git" and any(token in SENSITIVE_COMMAND_TOKENS[:6] for token in tokens[1:]):
        return "git write operation (push/reset/stash/clean/rebase) requires approval."
    if tokens[0] in ("pip", "pip3", "npm", "poetry", "conda", "uv") and any(
        token in ("install", "uninstall", "add", "remove") for token in tokens[1:]
    ):
        return "installing or removing dependencies requires approval."
    if tokens[0] in DELETE_COMMAND_TOKENS:
        return "deleting files requires approval."
    if tokens[0] in ("curl", "wget", "invoke-webrequest", "invoke-restmethod"):
        return "external network actions require approval."
    if any(token in SECRET_PATH_TOKENS for token in tokens):
        return "commands touching secrets or .env require approval."
    if "python" in tokens[0] and any(token in ("push", "install") for token in tokens[1:]):
        return "sensitive action inside python invocation requires approval."
    return None


def resolve_in_workspace(root: Path, target: str | Path) -> Path:
    """Resolve a path against the workspace, refusing anything outside it."""
    candidate = Path(target)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise SandboxViolationError(f"path is outside the workspace root: {target}")
    return resolved


def ensure_editable(root: Path, target: str | Path) -> Path:
    """Resolve a path and refuse secret/protected edit targets."""
    resolved = resolve_in_workspace(root, target)
    relative = resolved.relative_to(root.resolve()).as_posix().lower()
    if any(token in relative for token in PROTECTED_EDIT_TOKENS):
        raise SandboxViolationError(f"path is protected and cannot be edited: {relative}")
    return resolved


def _decoded(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _tail(text: str, lines: int) -> str:
    kept = [line for line in text.splitlines() if line.strip()]
    return "\n".join(kept[-lines:])


def _bound(text: str, limit: int) -> str:
    compact = text.replace("\r", " ").replace("\n", " ").strip()
    if len(compact) > limit:
        return compact[: limit - 3] + "..."
    return compact


class DevWorker:
    """Execute a bounded, declarative dev plan inside the workspace root."""

    def __init__(
        self,
        root: Path,
        *,
        test_runner: PytestRunner | None = None,
        command_runner: Callable[[tuple[str, ...]], CommandResult] | None = None,
        max_test_retries: int = MAX_TEST_RETRIES,
    ) -> None:
        self._root = Path(root).resolve()
        self._test_runner = test_runner or PytestRunner(self._root)
        self._command_runner = command_runner
        if max_test_retries < 0:
            raise ValueError("max_test_retries must not be negative.")
        self._max_test_retries = max_test_retries

    @property
    def root(self) -> Path:
        return self._root

    def execute(self, goal: str, steps: Sequence[DevStep]) -> DevWorkerResult:
        """Run the steps in order; stop at the first blocking outcome."""
        if not goal.strip():
            raise ValueError("goal must be a non-empty string.")
        evidence: list[DevEvidence] = []
        for index, step in enumerate(steps):
            try:
                evidence.extend(self._run_step(index, step))
            except SandboxViolationError as violation:
                return self._blocked(
                    goal, evidence, index, step, f"BLOCKED: {violation}"
                )
            except ActionRequiresApprovalError as approval:
                return DevWorkerResult(
                    goal=goal,
                    status="WAITING_APPROVAL",
                    evidence=tuple(evidence),
                    pending_action=approval.description,
                    pending_reason=approval.reason,
                )
            except (OSError, ValueError) as error:
                return self._blocked(goal, evidence, index, step, f"BLOCKED: {error}")
            except RuntimeError as error:
                return self._blocked(goal, evidence, index, step, f"BLOCKED: {error}")
        return DevWorkerResult(goal=goal, status="DONE", evidence=tuple(evidence))

    def _blocked(
        self,
        goal: str,
        evidence: list[DevEvidence],
        index: int,
        step: DevStep,
        error: str,
    ) -> DevWorkerResult:
        return DevWorkerResult(
            goal=goal,
            status="BLOCKED",
            evidence=tuple(evidence),
            error=f"{error} (step {index}: {step.action.value})",
        )

    def _run_step(self, index: int, step: DevStep) -> list[DevEvidence]:
        handlers = {
            DevWorkerAction.READ: self._read,
            DevWorkerAction.SEARCH: self._search,
            DevWorkerAction.EDIT: self._edit,
            DevWorkerAction.TEST: self._test,
            DevWorkerAction.COMMAND: self._command,
        }
        detail = handlers[step.action](step)
        label = step.description or step.action.value
        return [DevEvidence(index, step.action, f"{label}: {detail}")]

    def _read(self, step: DevStep) -> str:
        if not step.path:
            raise ValueError("read steps require a path.")
        resolved = resolve_in_workspace(self._root, step.path)
        if not resolved.is_file():
            raise OSError(f"file not found: {step.path}")
        text = resolved.read_text(encoding="utf-8", errors="replace")
        preview = text[:MAX_READ_CHARS]
        return f"read {len(text)} chars; head: {_bound(preview, 160)}"

    def _search(self, step: DevStep) -> str:
        if not step.pattern:
            raise ValueError("search steps require a pattern.")
        matches: list[str] = []
        scanned = 0
        for file_path in self._iter_search_files():
            scanned += 1
            if scanned > MAX_SEARCH_FILES or len(matches) >= MAX_SEARCH_MATCHES:
                break
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(content.splitlines(), start=1):
                if step.pattern in line:
                    relative = file_path.relative_to(self._root).as_posix()
                    matches.append(f"{relative}:{number}: {_bound(line, 120)}")
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        break
        detail = "; ".join(matches[:MAX_SEARCH_MATCHES])
        if not detail:
            detail = f"no matches for {step.pattern!r}"
        return f"{len(matches)} matches (scanned {scanned} files): {detail}" if matches else detail

    def _edit(self, step: DevStep) -> str:
        if not step.path:
            raise ValueError("edit steps require a path.")
        resolved = ensure_editable(self._root, step.path)
        current = (
            resolved.read_text(encoding="utf-8", errors="replace")
            if resolved.is_file()
            else ""
        )
        if step.operation == "replace_text":
            if current.count(step.old_text) != 1:
                raise ValueError(
                    f"old_text must match exactly once in {step.path} (found {current.count(step.old_text)})."
                )
            updated = current.replace(step.old_text, step.new_text, 1)
        elif step.operation == "append_text":
            updated = current + step.content
        else:
            updated = step.content
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(updated, encoding="utf-8")
        return f"{step.operation} applied to {step.path}"

    def _test(self, step: DevStep) -> str:
        attempts: list[str] = []
        for attempt in range(self._max_test_retries + 1):
            result = self._test_runner.run(step.test_paths)
            attempts.append(
                f"attempt {attempt + 1}: {'PASS' if result.passed else 'FAIL'} ({result.detail})"
            )
            if result.passed:
                return (
                    "PASS after " + "; ".join(attempts)
                    if len(attempts) > 1
                    else f"PASS ({result.detail}; tests: {', '.join(step.test_paths)})"
                )
            tail = _bound(result.output_tail, MAX_EVIDENCE_CHARS)
            if tail:
                attempts[-1] += f" tail: {tail}"
        raise RuntimeError(
            f"focal tests failed after {len(attempts)} attempts (retry limit {self._max_test_retries}). "
            + "; ".join(attempts)
        )

    def _command(self, step: DevStep) -> str:
        if step.command is None:
            raise ValueError("command steps require a command.")
        reason = classify_command(step.command)
        if reason is not None:
            raise ActionRequiresApprovalError(
                _bound(str(step.command), MAX_EVIDENCE_CHARS),
                reason,
            )
        tokens = (
            tuple(part for part in shlex.split(step.command, posix=False) if part.strip('"'))
            if isinstance(step.command, str)
            else tuple(step.command)
        )
        if not tokens:
            raise ValueError("command is empty after tokenization.")
        runner = self._command_runner or (
            lambda command: default_command_runner(self._root, command)
        )
        result = runner(tokens)
        if result.timed_out:
            raise RuntimeError(f"command timed out after {COMMAND_TIMEOUT_SECONDS} seconds.")
        status = "OK" if result.exit_code == 0 else f"EXIT {result.exit_code}"
        output = _bound(result.stdout_tail or result.stderr_tail, MAX_EVIDENCE_CHARS)
        return f"{status} stdout/stderr: {output}"

    def _iter_search_files(self) -> Iterable[Path]:
        for current, dir_names, file_names in os.walk(self._root):
            dir_names[:] = [name for name in dir_names if name not in SEARCH_SKIP_DIRS]
            for name in file_names:
                file_path = Path(current) / name
                if file_path.suffix.lower() in SEARCH_TEXT_EXTENSIONS:
                    yield file_path

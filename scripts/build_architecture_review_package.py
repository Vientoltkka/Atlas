"""Build the safe Atlas architecture-review package from a closed allowlist."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Iterable
import zipfile


PACKAGE_FILENAME = "atlas-architecture-review-v1.0.zip"
MANIFEST_FILENAME = "MANIFEST.json"
MANIFEST_SCHEMA_VERSION = 1
MAX_FILES = 1_000
MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 25_000_000
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

REQUIRED_REVIEW_DOCUMENTS = (
    "00_EXECUTIVE_SUMMARY.md",
    "01_SYSTEM_ARCHITECTURE.md",
    "02_MODULE_INVENTORY.md",
    "03_END_TO_END_FLOWS.md",
    "04_PUBLIC_CONTRACTS.md",
    "05_SECURITY_AND_SAFETY.md",
    "06_PERSISTENCE_AND_RECOVERY.md",
    "07_TESTING_EVIDENCE.md",
    "08_KNOWN_LIMITATIONS.md",
    "09_REVIEW_QUESTIONS.md",
    "10_FILES_TO_SHARE.md",
    "REVIEW_PROMPT_MASTER.md",
)

PUBLIC_ROOT_FILES = (
    ".env.example",
    "CHECKLIST_FINAL.md",
    "README.md",
    "RELEASE.md",
    "VERSION",
    "main.py",
    "requirements.txt",
)

PUBLIC_DOCUMENT_FILES = (
    "docs/ARCHITECTURE.md",
    "docs/execution_decision.md",
)

PYTHON_SOURCE_TREES = (
    "agents",
    "api",
    "bootstrap",
    "core",
    "domain",
    "memory",
    "models",
    "scripts",
    "services",
    "tools",
    "use_cases",
    "voice",
)

PUBLIC_DOCUMENT_TREES = ("docs/manual",)

AUTHORIZED_TEST_FILES = (
    "tests/test_agent_system.py",
    "tests/test_agent_executor.py",
    "tests/test_architecture_review_package.py",
    "tests/test_atlas_router.py",
    "tests/test_execution_arguments.py",
    "tests/test_execution_authorization.py",
    "tests/test_execution_context.py",
    "tests/test_execution_plan_executor.py",
    "tests/test_execution_plan_validator.py",
    "tests/test_execution_retry.py",
    "tests/test_execution_session_persistence.py",
    "tests/test_execution_strategy.py",
    "tests/test_objective_correction.py",
    "tests/test_objective_outcome_verification.py",
    "tests/test_operational_end_to_end.py",
    "tests/test_operational_multi_step_end_to_end.py",
    "tests/test_operational_request_router.py",
    "tests/test_operational_route_executor.py",
    "tests/test_request_gateway.py",
    "tests/test_resumable_execution_store.py",
    "tests/test_skill_system.py",
    "tests/test_structured_execution_orchestrator.py",
    "tests/test_windows_startup.py",
)

FORBIDDEN_PARTS = frozenset(
    {
        ".atlas",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "artifacts",
        "dist",
        "execution_sessions",
        "logs",
        "notebooks",
        "screenshots",
        "traces",
        "venv",
    }
)
FORBIDDEN_FILENAMES = frozenset({".env", "execution_state.json"})
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".flac",
        ".jpeg",
        ".jpg",
        ".log",
        ".mp3",
        ".onnx",
        ".png",
        ".pyc",
        ".tmp",
        ".trace",
        ".wav",
        ".zip",
    }
)

_SECRET_PATTERNS = (
    ("private_key", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    (
        "provider_token",
        re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "jwt",
        re.compile(rb"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\."),
    ),
)

class ArchitectureReviewPackageError(RuntimeError):
    """Raised when the review package cannot be built safely."""


@dataclass(frozen=True, slots=True)
class PackageEntry:
    """One immutable file entry in the package manifest."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PackageBuildResult:
    """Paths and immutable metadata produced by one package build."""

    zip_path: Path
    manifest_path: Path
    entries: tuple[PackageEntry, ...]
    total_size: int
    generated_at_utc: str
    zip_sha256: str


def build_architecture_review_package(
    *,
    project_root: Path | str | None = None,
    output_path: Path | str | None = None,
    extra_paths: Iterable[str] = (),
    generated_at: datetime | None = None,
) -> PackageBuildResult:
    """Build a validated ZIP without reading any file outside the allowlist."""

    root = Path(project_root or Path.cwd()).resolve()
    _validate_project_root(root)
    authorized = _authorized_paths(root)
    extras = tuple(extra_paths)
    _validate_extra_paths(extras, authorized)
    entries = _load_entries(root, authorized)
    generated_at_utc = _utc_text(generated_at or datetime.now(timezone.utc))
    manifest = _manifest_payload(entries, generated_at_utc)
    manifest_bytes = _json_bytes(manifest)

    destination = (
        Path(output_path)
        if output_path is not None
        else root / "dist" / "review" / PACKAGE_FILENAME
    )
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    _require_within_root(root, destination)
    if destination.suffix.lower() != ".zip":
        raise ArchitectureReviewPackageError("output_path must end in .zip.")
    manifest_path = destination.with_suffix(".manifest.json")

    _write_outputs_atomically(
        destination,
        manifest_path,
        root,
        entries,
        manifest_bytes,
    )
    return PackageBuildResult(
        zip_path=destination,
        manifest_path=manifest_path,
        entries=entries,
        total_size=sum(item.size for item in entries),
        generated_at_utc=generated_at_utc,
        zip_sha256=_sha256_bytes(destination.read_bytes()),
    )


def validate_source_path(
    project_root: Path,
    relative_path: str,
    *,
    authorized_paths: frozenset[str],
) -> Path:
    """Validate one authorized source path without following unsafe links."""

    normalized = _normalize_relative_path(relative_path)
    if normalized not in authorized_paths:
        raise ArchitectureReviewPackageError(
            f"Path is not authorized for the review package: {normalized}"
        )
    _reject_forbidden_path(normalized)
    root = project_root.resolve()
    candidate = root / PurePosixPath(normalized)
    if not candidate.exists() or not candidate.is_file():
        raise ArchitectureReviewPackageError(
            f"Authorized package file does not exist: {normalized}"
        )
    if _path_has_symlink(root, candidate):
        raise ArchitectureReviewPackageError(
            f"Symbolic links are not allowed in the review package: {normalized}"
        )
    resolved = candidate.resolve(strict=True)
    _require_within_root(root, resolved)
    if resolved.stat().st_size > MAX_FILE_BYTES:
        raise ArchitectureReviewPackageError(
            f"Package file exceeds the size limit: {normalized}"
        )
    return resolved


def _authorized_paths(root: Path) -> frozenset[str]:
    paths: list[str] = list(PUBLIC_ROOT_FILES)
    paths.extend(PUBLIC_DOCUMENT_FILES)
    paths.extend(AUTHORIZED_TEST_FILES)
    paths.extend(
        f"docs/architecture_review/{name}"
        for name in REQUIRED_REVIEW_DOCUMENTS
    )
    for tree in PYTHON_SOURCE_TREES:
        base = root / tree
        if not base.is_dir():
            raise ArchitectureReviewPackageError(
                f"Authorized source tree does not exist: {tree}"
            )
        paths.extend(
            path.relative_to(root).as_posix()
            for path in base.rglob("*.py")
            if path.is_file()
        )
    for tree in PUBLIC_DOCUMENT_TREES:
        base = root / tree
        if not base.is_dir():
            raise ArchitectureReviewPackageError(
                f"Authorized document tree does not exist: {tree}"
            )
        paths.extend(
            path.relative_to(root).as_posix()
            for path in base.rglob("*.md")
            if path.is_file()
        )
    normalized = tuple(_normalize_relative_path(path) for path in paths)
    if len(normalized) != len(set(normalized)):
        raise ArchitectureReviewPackageError(
            "The package allowlist contains duplicate paths."
        )
    return frozenset(normalized)


def _load_entries(
    root: Path,
    authorized: frozenset[str],
) -> tuple[PackageEntry, ...]:
    if len(authorized) > MAX_FILES:
        raise ArchitectureReviewPackageError("Package exceeds the file limit.")
    loaded: list[PackageEntry] = []
    total_size = 0
    for relative_path in sorted(authorized):
        source = validate_source_path(
            root,
            relative_path,
            authorized_paths=authorized,
        )
        content = source.read_bytes()
        _scan_content(relative_path, content)
        total_size += len(content)
        if total_size > MAX_TOTAL_BYTES:
            raise ArchitectureReviewPackageError(
                "Package exceeds the total size limit."
            )
        loaded.append(
            PackageEntry(
                path=relative_path,
                size=len(content),
                sha256=_sha256_bytes(content),
            )
        )
    return tuple(loaded)


def _validate_extra_paths(
    paths: tuple[str, ...],
    authorized: frozenset[str],
) -> None:
    seen: set[str] = set()
    for raw_path in paths:
        normalized = _normalize_relative_path(raw_path)
        _reject_forbidden_path(normalized)
        if normalized in seen:
            raise ArchitectureReviewPackageError(
                f"Duplicate requested package path: {normalized}"
            )
        seen.add(normalized)
        if normalized not in authorized:
            raise ArchitectureReviewPackageError(
                f"Requested path is not authorized: {normalized}"
            )
        raise ArchitectureReviewPackageError(
            f"Requested path duplicates the fixed allowlist: {normalized}"
        )


def _validate_project_root(root: Path) -> None:
    required = ("main.py", "VERSION", "docs/architecture_review")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise ArchitectureReviewPackageError(
            "Run the package builder from the Atlas project root."
        )


def _normalize_relative_path(raw_path: str) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ArchitectureReviewPackageError(
            "Package paths must be non-empty relative strings."
        )
    if "\\" in raw_path or "\x00" in raw_path:
        raise ArchitectureReviewPackageError(
            "Package paths must use safe POSIX separators."
        )
    path = PurePosixPath(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArchitectureReviewPackageError(
            f"Unsafe package path: {raw_path}"
        )
    return path.as_posix()


def _reject_forbidden_path(relative_path: str) -> None:
    path = PurePosixPath(relative_path)
    lowered_parts = {part.casefold() for part in path.parts}
    if lowered_parts.intersection(FORBIDDEN_PARTS):
        raise ArchitectureReviewPackageError(
            f"Forbidden package path category: {relative_path}"
        )
    if path.name.casefold() in FORBIDDEN_FILENAMES:
        raise ArchitectureReviewPackageError(
            f"Forbidden package filename: {relative_path}"
        )
    if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
        raise ArchitectureReviewPackageError(
            f"Forbidden package file type: {relative_path}"
        )


def _path_has_symlink(root: Path, candidate: Path) -> bool:
    current = candidate
    while current != root:
        if current.is_symlink():
            return True
        current = current.parent
    return root.is_symlink()


def _require_within_root(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ArchitectureReviewPackageError(
            "Package path escapes the project root."
        ) from error


def _scan_content(relative_path: str, content: bytes) -> None:
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(content):
            raise ArchitectureReviewPackageError(
                f"Potential {label} detected in authorized file: {relative_path}"
            )
    if relative_path == ".env.example":
        _validate_env_example(content)


def _validate_env_example(content: bytes) -> None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArchitectureReviewPackageError(
            ".env.example must be UTF-8 text."
        ) from error
    keys: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ArchitectureReviewPackageError(
                ".env.example contains an invalid line."
            )
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ArchitectureReviewPackageError(
                ".env.example contains an invalid variable name."
            )
        if key in keys:
            raise ArchitectureReviewPackageError(
                ".env.example contains a duplicate variable name."
            )
        if value.strip():
            raise ArchitectureReviewPackageError(
                ".env.example values must be empty."
            )
        keys.add(key)


def _manifest_payload(
    entries: tuple[PackageEntry, ...],
    generated_at_utc: str,
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "package": "atlas-architecture-review-v1.0",
        "generated_at_utc": generated_at_utc,
        "file_count": len(entries),
        "total_size": sum(item.size for item in entries),
        "files": [
            {
                "path": item.path,
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in entries
        ],
    }


def _write_outputs_atomically(
    destination: Path,
    manifest_path: Path,
    root: Path,
    entries: tuple[PackageEntry, ...],
    manifest_bytes: bytes,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    zip_temp = _temporary_path(destination.parent, ".zip.tmp")
    manifest_temp = _temporary_path(destination.parent, ".manifest.tmp")
    try:
        _write_zip(zip_temp, root, entries, manifest_bytes)
        manifest_temp.write_bytes(manifest_bytes)
        os.replace(manifest_temp, manifest_path)
        os.replace(zip_temp, destination)
    except Exception:
        zip_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
        raise


def _write_zip(
    destination: Path,
    root: Path,
    entries: tuple[PackageEntry, ...],
    manifest_bytes: bytes,
) -> None:
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for entry in entries:
            _write_zip_entry(
                archive,
                entry.path,
                (root / PurePosixPath(entry.path)).read_bytes(),
            )
        _write_zip_entry(archive, MANIFEST_FILENAME, manifest_bytes)


def _write_zip_entry(
    archive: zipfile.ZipFile,
    relative_path: str,
    content: bytes,
) -> None:
    info = zipfile.ZipInfo(relative_path, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100444 << 16
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _temporary_path(parent: Path, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        dir=parent,
        prefix=".atlas-review-",
        suffix=suffix,
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the safe Atlas architecture-review ZIP.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional ZIP path inside the project root.",
    )
    return parser.parse_args()


def main() -> int:
    """Build the package and print only safe relative metadata."""

    args = _parse_args()
    try:
        result = build_architecture_review_package(output_path=args.output)
    except ArchitectureReviewPackageError as error:
        print(f"Package build rejected: {error}")
        return 1
    root = Path.cwd().resolve()
    print(f"ZIP: {result.zip_path.relative_to(root).as_posix()}")
    print(f"Manifest: {result.manifest_path.relative_to(root).as_posix()}")
    print(f"Files: {len(result.entries)}")
    print(f"Bytes: {result.total_size}")
    print(f"ZIP SHA-256: {result.zip_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

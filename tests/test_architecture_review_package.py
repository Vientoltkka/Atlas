from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.build_architecture_review_package import (
    ArchitectureReviewPackageError,
    FORBIDDEN_FILENAMES,
    FORBIDDEN_PARTS,
    MANIFEST_FILENAME,
    REQUIRED_REVIEW_DOCUMENTS,
    _scan_content,
    build_architecture_review_package,
    validate_source_path,
)


FIXED_GENERATION_TIME = datetime(2026, 7, 30, 22, 0, tzinfo=timezone.utc)
TEST_OUTPUT_DIR = Path("dist/test-temp/architecture-review-package-tests")


@pytest.fixture(scope="module")
def built_package():
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = TEST_OUTPUT_DIR / "review.zip"
    return build_architecture_review_package(
        output_path=output,
        generated_at=FIXED_GENERATION_TIME,
    )


def test_zip_contains_all_required_documents(built_package) -> None:
    with zipfile.ZipFile(built_package.zip_path) as archive:
        names = archive.namelist()

    assert archive_safe_names(names)
    assert MANIFEST_FILENAME in names
    for name in REQUIRED_REVIEW_DOCUMENTS:
        assert f"docs/architecture_review/{name}" in names


def test_zip_excludes_private_runtime_and_cache_paths(built_package) -> None:
    with zipfile.ZipFile(built_package.zip_path) as archive:
        names = archive.namelist()

    for name in names:
        parts = {part.casefold() for part in Path(name).parts}
        assert not parts.intersection(FORBIDDEN_PARTS)
        assert Path(name).name.casefold() not in FORBIDDEN_FILENAMES
        assert ".env" not in parts


def test_manifest_matches_zip_content_and_hashes(built_package) -> None:
    external = json.loads(built_package.manifest_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(built_package.zip_path) as archive:
        internal = json.loads(archive.read(MANIFEST_FILENAME))
        names = [
            name
            for name in archive.namelist()
            if name != MANIFEST_FILENAME
        ]
        assert names == sorted(names)
        assert internal == external
        assert internal["file_count"] == len(names)
        assert internal["total_size"] == sum(
            item["size"] for item in internal["files"]
        )
        assert [item["path"] for item in internal["files"]] == names
        for item in internal["files"]:
            content = archive.read(item["path"])
            assert item["size"] == len(content)
            assert item["sha256"] == hashlib.sha256(content).hexdigest()


def test_two_generations_have_identical_logical_content(
    built_package,
    tmp_path: Path,
) -> None:
    second = build_architecture_review_package(
        output_path=TEST_OUTPUT_DIR / "second.zip",
        generated_at=FIXED_GENERATION_TIME,
    )

    assert second.entries == built_package.entries
    assert second.zip_path.read_bytes() == built_package.zip_path.read_bytes()
    assert second.zip_sha256 == built_package.zip_sha256


@pytest.mark.parametrize(
    "unsafe_path",
    (
        ".env",
        "../outside.txt",
        "/absolute.txt",
        "C:/absolute.txt",
        ".git/config",
        ".atlas/execution_state.json",
        "logs/atlas.log",
    ),
)
def test_forbidden_entry_rejects_build_without_partial_zip(
    unsafe_path: str,
) -> None:
    output = TEST_OUTPUT_DIR / "rejected.zip"

    with pytest.raises(ArchitectureReviewPackageError):
        build_architecture_review_package(
            output_path=output,
            extra_paths=(unsafe_path,),
            generated_at=FIXED_GENERATION_TIME,
        )

    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()


def test_duplicate_requested_entry_is_rejected() -> None:
    output = TEST_OUTPUT_DIR / "duplicate.zip"

    with pytest.raises(ArchitectureReviewPackageError):
        build_architecture_review_package(
            output_path=output,
            extra_paths=("VERSION", "VERSION"),
            generated_at=FIXED_GENERATION_TIME,
        )

    assert not output.exists()


def test_unsafe_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = root / "allowed.py"
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.build_architecture_review_package._path_has_symlink",
        lambda _root, _candidate: True,
    )

    with pytest.raises(ArchitectureReviewPackageError, match="Symbolic links"):
        validate_source_path(
            root,
            "allowed.py",
            authorized_paths=frozenset({"allowed.py"}),
        )


def test_env_example_has_only_empty_values() -> None:
    lines = Path(".env.example").read_text(encoding="utf-8").splitlines()
    assignments = [
        line
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert assignments
    assert all("=" in line and not line.split("=", 1)[1].strip() for line in assignments)


def test_high_confidence_secret_pattern_is_rejected() -> None:
    with pytest.raises(
        ArchitectureReviewPackageError,
        match="Potential provider_token",
    ):
        _scan_content(
            "tests/fixture.py",
            b'test_token = "' + b"sk-" + b"proj-" + b"a" * 24 + b'"\n',
        )


def test_builder_does_not_modify_authorized_sources() -> None:
    observed = (
        Path("VERSION"),
        Path("RELEASE.md"),
        Path("docs/architecture_review/10_FILES_TO_SHARE.md"),
        Path("core/orchestrator.py"),
    )
    before = {
        path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in observed
    }

    build_architecture_review_package(
        output_path=TEST_OUTPUT_DIR / "source-safe.zip",
        generated_at=FIXED_GENERATION_TIME,
    )

    after = {
        path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in observed
    }
    assert after == before


def test_zip_can_be_opened_and_fully_verified(built_package) -> None:
    with zipfile.ZipFile(built_package.zip_path) as archive:
        assert archive.testzip() is None
        assert archive_safe_names(archive.namelist())


def archive_safe_names(names: list[str]) -> bool:
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            return False
    return len(names) == len(set(names))

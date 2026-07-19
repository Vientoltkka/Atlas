"""CLI for consulting the internal Atlas manual."""

from __future__ import annotations

import sys

from use_cases.atlas_manual import (
    AtlasManualLoader,
    AtlasManualValidator,
    ManualLoadStatus,
)


def main(argv: list[str] | None = None) -> int:
    """Run the manual CLI."""
    args = list(sys.argv[1:] if argv is None else argv)
    loader = AtlasManualLoader()

    if not args or args[0] in {"-h", "--help"}:
        _print_usage()
        return 0 if args else 2

    command = args[0]
    if command == "list" and len(args) == 1:
        for section in loader.list_sections():
            print(f"{section.id}\t{section.title}\t{section.summary}")
        return 0

    if command == "show" and len(args) == 2:
        result = loader.get_section(args[1])
        if result.status is not ManualLoadStatus.FOUND or result.content is None:
            print(result.message)
            return 1
        print(_console_text(result.content.rstrip()))
        return 0

    if command == "search" and len(args) == 2:
        matches = loader.search(args[1])
        if not matches:
            print("No hay resultados.")
            return 1
        for section in matches:
            print(f"{section.id}\t{section.title}\t{section.summary}")
        return 0

    if command == "validate" and len(args) == 1:
        return _validate()

    _print_usage()
    return 2


def _validate() -> int:
    result = AtlasManualValidator().validate()
    if result.valid:
        print("Manual valido.")
        return 0

    print("Manual invalido:")
    for issue in result.issues:
        section = f" [{issue.section_id}]" if issue.section_id else ""
        print(f"- {issue.code}{section}: {issue.message}")
    return 1


def _print_usage() -> None:
    print("Usage:")
    print("  python -B -m tools.atlas_manual list")
    print("  python -B -m tools.atlas_manual show <section_id>")
    print("  python -B -m tools.atlas_manual search <text>")
    print("  python -B -m tools.atlas_manual validate")


def _console_text(value: object) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate the internal Atlas manual."""

from __future__ import annotations

from tools.atlas_manual import main as manual_main


def main() -> int:
    """Run manual validation."""
    return manual_main(["validate"])


if __name__ == "__main__":
    raise SystemExit(main())

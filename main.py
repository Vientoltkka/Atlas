"""Atlas entry point."""

from __future__ import annotations

from core.atlas import Atlas


def main() -> None:
    """Start Atlas."""
    atlas = Atlas()
    atlas.start()


if __name__ == "__main__":
    main()

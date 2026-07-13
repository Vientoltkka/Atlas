"""Resolve direct internal dependencies for a Python project file."""

from __future__ import annotations

from pathlib import Path


class ResolveProjectDependenciesUseCase:
    """Resolve direct project imports using a previously generated index."""

    def execute(
        self,
        main_path: str,
        index: list[dict[str, object]],
    ) -> list[str]:
        """Return the main file and its direct internal dependency paths."""
        normalized_main_path = self._normalize_path(main_path)
        module_paths = self._build_module_paths(index)
        main_item = self._find_index_item(normalized_main_path, index)

        if main_item is None:
            return [normalized_main_path]

        resolved_paths = [normalized_main_path]

        for imported_name in self._imported_names(main_item):
            module_name = self._resolve_relative_import(
                imported_name,
                normalized_main_path,
            )
            dependency_path = self._resolve_module_path(
                module_name,
                module_paths,
            )

            if dependency_path and dependency_path not in resolved_paths:
                resolved_paths.append(dependency_path)

        return resolved_paths

    def _build_module_paths(
        self,
        index: list[dict[str, object]],
    ) -> dict[str, str]:
        """Map internal module names to project file paths."""
        module_paths: dict[str, str] = {}

        for item in index:
            path = self._normalize_path(str(item["path"]))
            module_name = Path(path).with_suffix("").as_posix().replace("/", ".")
            module_paths[module_name] = path

        return module_paths

    def _find_index_item(
        self,
        path: str,
        index: list[dict[str, object]],
    ) -> dict[str, object] | None:
        """Find one indexed item by normalized path."""
        for item in index:
            if self._normalize_path(str(item["path"])) == path:
                return item

        return None

    def _imported_names(
        self,
        item: dict[str, object],
    ) -> list[str]:
        """Return import names stored in an indexed item."""
        imports = item.get("imports", [])

        if not isinstance(imports, list):
            return []

        return [str(imported_name) for imported_name in imports]

    def _resolve_module_path(
        self,
        imported_name: str,
        module_paths: dict[str, str],
    ) -> str | None:
        """Resolve the longest internal module prefix for one import."""
        candidates = imported_name.split(".")

        while candidates:
            module_name = ".".join(candidates)

            if module_name in module_paths:
                return module_paths[module_name]

            candidates.pop()

        return None

    def _resolve_relative_import(
        self,
        imported_name: str,
        main_path: str,
    ) -> str:
        """Convert a relative import into an absolute project module name."""
        if not imported_name.startswith("."):
            return imported_name

        level = len(imported_name) - len(imported_name.lstrip("."))
        suffix = imported_name[level:]
        package_parts = Path(main_path).with_suffix("").parts[:-1]
        base_parts = list(package_parts[: max(len(package_parts) - level + 1, 0)])

        if suffix:
            base_parts.extend(suffix.split("."))

        return ".".join(base_parts)

    def _normalize_path(
        self,
        path: str,
    ) -> str:
        """Normalize paths to project-relative POSIX format."""
        return Path(path).as_posix()

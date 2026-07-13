"""Build an internal architecture graph from a project index."""

from __future__ import annotations

from pathlib import Path

from models.architecture_graph import ArchitectureGraph, ArchitectureNode


class BuildArchitectureGraphUseCase:
    """Build an in-memory architecture graph using AST-derived index data."""

    def execute(
        self,
        index: list[dict[str, object]],
    ) -> ArchitectureGraph:
        """Return architecture nodes with resolved internal dependencies."""
        normalized_items = self._normalize_index(index)
        module_paths = {
            item["module"]: item["path"]
            for item in normalized_items
        }
        nodes: list[ArchitectureNode] = []

        for item in normalized_items:
            imports = self._string_list(item["imports"])
            dependencies = self._resolve_dependencies(
                imports=imports,
                current_module=item["module"],
                module_paths=module_paths,
            )
            nodes.append(
                ArchitectureNode(
                    path=item["path"],
                    module=item["module"],
                    classes=self._string_list(item["classes"]),
                    functions=self._public_functions(item["functions"]),
                    imports=imports,
                    dependencies=dependencies,
                )
            )

        return ArchitectureGraph(nodes=nodes)

    def _normalize_index(
        self,
        index: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Normalize indexed file metadata into graph-ready strings."""
        paths = [self._normalize_path(str(item["path"])) for item in index]
        project_root = self._common_root(paths)
        normalized_items: list[dict[str, object]] = []

        for item, path in zip(index, paths, strict=True):
            normalized_items.append(
                {
                    "path": path,
                    "module": self._module_name(path, project_root),
                    "classes": item.get("classes", []),
                    "functions": item.get("functions", []),
                    "imports": item.get("imports", []),
                }
            )

        return normalized_items

    def _resolve_dependencies(
        self,
        imports: list[str],
        current_module: str,
        module_paths: dict[str, str],
    ) -> list[str]:
        """Resolve imported module names to internal project paths."""
        dependencies: list[str] = []

        for imported_name in imports:
            module_name = self._resolve_relative_import(
                imported_name=imported_name,
                current_module=current_module,
            )
            dependency = self._resolve_module_path(
                imported_name=module_name,
                module_paths=module_paths,
            )

            if dependency and dependency not in dependencies:
                dependencies.append(dependency)

        return dependencies

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
        current_module: str,
    ) -> str:
        """Convert a relative import into an absolute project module name."""
        if not imported_name.startswith("."):
            return imported_name

        level = len(imported_name) - len(imported_name.lstrip("."))
        suffix = imported_name[level:]
        package_parts = current_module.split(".")[:-1]
        base_parts = package_parts[: max(len(package_parts) - level + 1, 0)]

        if suffix:
            base_parts.extend(suffix.split("."))

        return ".".join(base_parts)

    def _module_name(
        self,
        path: str,
        project_root: str,
    ) -> str:
        """Convert a Python file path into a dotted module name."""
        relative_path = path

        if project_root and path.startswith(f"{project_root}/"):
            relative_path = path[len(project_root) + 1 :]

        module_path = str(Path(relative_path).with_suffix("")).replace("\\", "/")

        return module_path.replace("/", ".")

    def _common_root(
        self,
        paths: list[str],
    ) -> str:
        """Infer the project root from indexed Python file paths."""
        if not paths:
            return ""

        parent_parts = [
            path.rsplit("/", 1)[0].split("/")
            for path in paths
            if "/" in path
        ]

        if not parent_parts:
            return ""

        common_parts = parent_parts[0]

        for parts in parent_parts[1:]:
            common_parts = [
                left
                for left, right in zip(common_parts, parts, strict=False)
                if left == right
            ]

        return "/".join(common_parts)

    def _public_functions(
        self,
        values: object,
    ) -> list[str]:
        """Return functions whose final qualified name is public."""
        return [
            name
            for name in self._string_list(values)
            if not name.rsplit(".", 1)[-1].startswith("_")
        ]

    def _string_list(
        self,
        values: object,
    ) -> list[str]:
        """Return a list of strings for index values."""
        if not isinstance(values, list):
            return []

        return [str(value) for value in values]

    def _normalize_path(
        self,
        path: str,
    ) -> str:
        """Normalize paths to POSIX format."""
        return Path(path).as_posix()

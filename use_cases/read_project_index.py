"""Read a structural index for a Python project."""

from __future__ import annotations

import ast
from pathlib import Path


class _ProjectIndexVisitor(ast.NodeVisitor):
    """Collect imports, classes, and functions from a Python syntax tree."""

    def __init__(self) -> None:
        self.imports: list[str] = []
        self.classes: list[str] = []
        self.functions: list[str] = []
        self._class_stack: list[str] = []

    def visit_Import(
        self,
        node: ast.Import,
    ) -> None:
        """Collect direct import statements."""
        for alias in node.names:
            self.imports.append(alias.name)

        self.generic_visit(node)

    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ) -> None:
        """Collect from-import statements as module-level dependencies."""
        module = f"{'.' * node.level}{node.module or ''}"

        for alias in node.names:
            if module:
                self.imports.append(f"{module}.{alias.name}")
            else:
                self.imports.append(alias.name)

        self.generic_visit(node)

    def visit_ClassDef(
        self,
        node: ast.ClassDef,
    ) -> None:
        """Collect class names and keep context for methods."""
        self.classes.append(node.name)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:
        """Collect function and method names."""
        self._collect_function(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        """Collect async function and method names."""
        self._collect_function(node.name)
        self.generic_visit(node)

    def _collect_function(
        self,
        name: str,
    ) -> None:
        """Store a qualified method name when inside a class."""
        if self._class_stack:
            self.functions.append(f"{self._class_stack[-1]}.{name}")
            return

        self.functions.append(name)


class ReadProjectIndexUseCase:
    """Build an index of Python files without reading project behavior."""

    def execute(
        self,
        root: str,
    ) -> list[dict[str, object]]:
        """Return path, filename, imports, classes, and functions per file."""
        project = Path(root)
        index: list[dict[str, object]] = []

        for path in self._iter_python_files(project):
            visitor = self._read_file_index(path)

            index.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "imports": visitor.imports,
                    "classes": visitor.classes,
                    "functions": visitor.functions,
                }
            )

        return index

    def _iter_python_files(
        self,
        root: Path,
    ) -> list[Path]:
        """Return Python files while ignoring generated and virtualenv files."""
        return sorted(
            path
            for path in root.rglob("*.py")
            if not self._is_generated_directory(path.parts)
        )

    @staticmethod
    def _is_generated_directory(parts: tuple[str, ...]) -> bool:
        """Generated/temporary trees never contain project source modules."""
        return bool(
            {"pytest-tmp", ".pytest_cache", ".git", ".venv", "__pycache__"}
            & set(parts)
        )

    def _read_file_index(
        self,
        path: Path,
    ) -> _ProjectIndexVisitor:
        """Parse one file and collect its structural metadata."""
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        visitor = _ProjectIndexVisitor()
        visitor.visit(tree)

        return visitor

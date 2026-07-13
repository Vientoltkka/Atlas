"""In-memory architecture graph models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArchitectureNode:
    """Architecture metadata for one Python module."""

    path: str
    module: str
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArchitectureGraph:
    """In-memory graph of project modules and internal dependencies."""

    nodes: list[ArchitectureNode] = field(default_factory=list)

    def summary(self) -> str:
        """Return a compact summary of the architecture graph."""
        module_count = len(self.nodes)
        relation_count = sum(len(node.dependencies) for node in self.nodes)
        average_dependencies = (
            relation_count / module_count
            if module_count
            else 0.0
        )
        most_connected = self._most_connected_module()

        return "\n".join(
            [
                f"Módulos analizados: {module_count}",
                f"Relaciones: {relation_count}",
                f"Dependencias medias: {average_dependencies:.1f}",
                "Módulo con más dependencias:",
                most_connected,
            ]
        )

    def _most_connected_module(self) -> str:
        """Return the path with the highest number of dependencies."""
        if not self.nodes:
            return "ninguno"

        return max(
            self.nodes,
            key=lambda node: len(node.dependencies),
        ).path

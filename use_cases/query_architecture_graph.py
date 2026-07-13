"""Query an in-memory architecture graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from models.architecture_graph import ArchitectureGraph, ArchitectureNode


@dataclass(frozen=True)
class ArchitectureQueryResult:
    """Deterministic result for architecture graph queries."""

    target: str | None = None
    matches: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    direct_dependents: list[str] = field(default_factory=list)
    indirect_dependents: list[str] = field(default_factory=list)
    imported_classes: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)


class QueryArchitectureGraphUseCase:
    """Query dependencies, dependents, imports, and impact in a graph."""

    def __init__(
        self,
        graph: ArchitectureGraph,
    ) -> None:
        self._graph = graph
        self._nodes_by_path = {
            node.path: node
            for node in graph.nodes
        }

    def dependencies_of(
        self,
        target: str,
    ) -> ArchitectureQueryResult:
        """Return direct internal modules used by the target."""
        node_result = self._resolve_single_target(target)

        if node_result.target is None:
            return node_result

        node = self._nodes_by_path[node_result.target]

        return ArchitectureQueryResult(
            target=node.path,
            dependencies=sorted(node.dependencies),
        )

    def dependents_of(
        self,
        target: str,
    ) -> ArchitectureQueryResult:
        """Return modules depending directly or indirectly on the target."""
        node_result = self._resolve_single_target(target)

        if node_result.target is None:
            return node_result

        direct_dependents = self._direct_dependents(node_result.target)
        indirect_dependents = self._indirect_dependents(
            target_path=node_result.target,
            direct_dependents=direct_dependents,
        )

        return ArchitectureQueryResult(
            target=node_result.target,
            direct_dependents=direct_dependents,
            indirect_dependents=indirect_dependents,
        )

    def imported_classes_of(
        self,
        target: str,
    ) -> ArchitectureQueryResult:
        """Return internal classes imported by the target."""
        node_result = self._resolve_single_target(target)

        if node_result.target is None:
            return node_result

        node = self._nodes_by_path[node_result.target]

        return ArchitectureQueryResult(
            target=node.path,
            imported_classes=self._imported_classes(node),
        )

    def impact_of(
        self,
        target: str,
    ) -> ArchitectureQueryResult:
        """Return files potentially affected by a change to the target."""
        dependents = self.dependents_of(target)

        if dependents.target is None:
            return dependents

        affected_files = [
            *dependents.direct_dependents,
            *dependents.indirect_dependents,
        ]

        return ArchitectureQueryResult(
            target=dependents.target,
            direct_dependents=dependents.direct_dependents,
            indirect_dependents=dependents.indirect_dependents,
            affected_files=affected_files,
        )

    def _resolve_single_target(
        self,
        target: str,
    ) -> ArchitectureQueryResult:
        """Resolve a target to one node or return deterministic matches."""
        matches = self._resolve_matches(target)

        if len(matches) == 1:
            return ArchitectureQueryResult(target=matches[0].path)

        return ArchitectureQueryResult(
            matches=[node.path for node in matches],
        )

    def _resolve_matches(
        self,
        target: str,
    ) -> list[ArchitectureNode]:
        """Resolve target by path, module, file, class, or function."""
        normalized_target = self._normalize_target(target)
        scored_matches: list[tuple[int, str, ArchitectureNode]] = []

        for node in self._graph.nodes:
            score = self._match_score(normalized_target, node)

            if score is not None:
                scored_matches.append((score, node.path, node))

        scored_matches.sort(key=lambda item: (item[0], item[1]))

        return [node for _, _, node in scored_matches]

    def _match_score(
        self,
        target: str,
        node: ArchitectureNode,
    ) -> int | None:
        """Return a relevance score for one target-node match."""
        node_path = self._normalize_target(node.path)
        node_module = node.module.lower()
        filename = Path(node.path).name.lower()
        target_lower = target.lower()

        if target_lower == node_path:
            return 0

        if target_lower == node_module:
            return 1

        if target_lower == filename:
            return 2

        if target in node.classes:
            return 3

        if any(self._matches_function(target, function) for function in node.functions):
            return 4

        return None

    def _matches_function(
        self,
        target: str,
        function: str,
    ) -> bool:
        """Return whether target names a function or qualified method."""
        return target == function or target == function.rsplit(".", 1)[-1]

    def _direct_dependents(
        self,
        target_path: str,
    ) -> list[str]:
        """Return direct dependents of a target path."""
        return sorted(
            node.path
            for node in self._graph.nodes
            if target_path in node.dependencies
        )

    def _indirect_dependents(
        self,
        target_path: str,
        direct_dependents: list[str],
    ) -> list[str]:
        """Return transitive dependents excluding direct dependents."""
        visited = {target_path, *direct_dependents}
        pending = list(direct_dependents)
        indirect: list[str] = []

        while pending:
            current_path = pending.pop(0)

            for dependent_path in self._direct_dependents(current_path):
                if dependent_path in visited:
                    continue

                visited.add(dependent_path)
                pending.append(dependent_path)
                indirect.append(dependent_path)

        return sorted(indirect)

    def _imported_classes(
        self,
        node: ArchitectureNode,
    ) -> list[str]:
        """Resolve imported internal classes from stored AST import names."""
        classes: list[str] = []

        for imported_name in node.imports:
            class_name = self._resolve_imported_class(imported_name)

            if class_name and class_name not in classes:
                classes.append(class_name)

        return sorted(classes)

    def _resolve_imported_class(
        self,
        imported_name: str,
    ) -> str | None:
        """Resolve one import to an internal class when possible."""
        parts = imported_name.split(".")

        while len(parts) > 1:
            module_name = ".".join(parts[:-1])
            class_name = parts[-1]
            node = self._node_by_module(module_name)

            if node and class_name in node.classes:
                return class_name

            parts.pop()

        return None

    def _node_by_module(
        self,
        module: str,
    ) -> ArchitectureNode | None:
        """Find one node by exact module name."""
        for node in self._graph.nodes:
            if node.module == module:
                return node

        return None

    def _normalize_target(
        self,
        target: str,
    ) -> str:
        """Normalize user-provided target text."""
        return target.strip().replace("\\", "/")

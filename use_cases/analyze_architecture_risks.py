"""Analyze coupling and architectural risks in an architecture graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from models.architecture_graph import ArchitectureGraph, ArchitectureNode


RiskLevel = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class ArchitectureRiskResult:
    """Risk metrics for one architecture module."""

    path: str
    outgoing_dependencies: int
    direct_dependents: int
    total_impact: int
    risk_level: RiskLevel
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArchitectureRiskAnalysis:
    """Deterministic architecture risk analysis."""

    results: list[ArchitectureRiskResult]
    modules_with_most_outgoing_dependencies: list[str]
    modules_with_most_direct_dependents: list[str]
    modules_with_highest_total_impact: list[str]
    cycles: list[list[str]]
    isolated_modules: list[str]
    high_coupling_modules: list[str]
    critical_modules: list[str]

    def summary(self) -> str:
        """Return a compact risk summary."""
        return "\n".join(
            [
                f"Módulos analizados: {len(self.results)}",
                f"Módulos críticos: {len(self.critical_modules)}",
                (
                    "Módulos con alto acoplamiento: "
                    f"{len(self.high_coupling_modules)}"
                ),
                f"Ciclos detectados: {len(self.cycles)}",
                "Módulo de mayor impacto:",
                self._highest_impact_module(),
            ]
        )

    def _highest_impact_module(self) -> str:
        """Return the first highest-impact module or an explicit empty value."""
        if not self.modules_with_highest_total_impact:
            return "ninguno"

        return self.modules_with_highest_total_impact[0]


class AnalyzeArchitectureRisksUseCase:
    """Detect coupling, impact, cycles, and critical modules."""

    HIGH_OUTGOING_DEPENDENCIES = 3
    HIGH_DIRECT_DEPENDENTS = 3
    HIGH_TOTAL_IMPACT = 3
    CRITICAL_DIRECT_DEPENDENTS = 4
    CRITICAL_TOTAL_IMPACT = 4

    def execute(
        self,
        graph: ArchitectureGraph,
    ) -> ArchitectureRiskAnalysis:
        """Return deterministic architecture risk metrics for a graph."""
        nodes_by_path = {
            node.path: node
            for node in graph.nodes
        }
        dependents_by_path = self._dependents_by_path(graph.nodes)
        cycles = self._detect_cycles(graph.nodes, nodes_by_path)
        cycle_nodes = {
            path
            for cycle in cycles
            for path in cycle
        }

        results = [
            self._analyze_node(
                node=node,
                dependents_by_path=dependents_by_path,
                cycle_nodes=cycle_nodes,
            )
            for node in graph.nodes
        ]
        results.sort(key=lambda result: result.path)

        return ArchitectureRiskAnalysis(
            results=results,
            modules_with_most_outgoing_dependencies=self._max_paths(
                results,
                "outgoing_dependencies",
            ),
            modules_with_most_direct_dependents=self._max_paths(
                results,
                "direct_dependents",
            ),
            modules_with_highest_total_impact=self._max_paths(
                results,
                "total_impact",
            ),
            cycles=cycles,
            isolated_modules=[
                result.path
                for result in results
                if "módulo aislado" in result.reasons
            ],
            high_coupling_modules=[
                result.path
                for result in results
                if "acoplamiento alto" in result.reasons
            ],
            critical_modules=[
                result.path
                for result in results
                if result.risk_level == "critical"
            ],
        )

    def _analyze_node(
        self,
        node: ArchitectureNode,
        dependents_by_path: dict[str, list[str]],
        cycle_nodes: set[str],
    ) -> ArchitectureRiskResult:
        """Return risk metrics for one module."""
        outgoing_dependencies = len(node.dependencies)
        direct_dependents = len(dependents_by_path[node.path])
        total_impact = len(
            self._transitive_dependents(
                path=node.path,
                dependents_by_path=dependents_by_path,
            )
        )
        reasons = self._reasons(
            path=node.path,
            outgoing_dependencies=outgoing_dependencies,
            direct_dependents=direct_dependents,
            total_impact=total_impact,
            cycle_nodes=cycle_nodes,
        )

        return ArchitectureRiskResult(
            path=node.path,
            outgoing_dependencies=outgoing_dependencies,
            direct_dependents=direct_dependents,
            total_impact=total_impact,
            risk_level=self._risk_level(
                outgoing_dependencies=outgoing_dependencies,
                direct_dependents=direct_dependents,
                total_impact=total_impact,
                in_cycle=node.path in cycle_nodes,
            ),
            reasons=reasons,
        )

    def _reasons(
        self,
        path: str,
        outgoing_dependencies: int,
        direct_dependents: int,
        total_impact: int,
        cycle_nodes: set[str],
    ) -> list[str]:
        """Return deterministic risk reasons."""
        reasons: list[str] = []

        if outgoing_dependencies == 0 and direct_dependents == 0:
            reasons.append("módulo aislado")

        if (
            outgoing_dependencies >= self.HIGH_OUTGOING_DEPENDENCIES
            or direct_dependents >= self.HIGH_DIRECT_DEPENDENTS
        ):
            reasons.append("acoplamiento alto")

        if direct_dependents >= self.HIGH_DIRECT_DEPENDENTS:
            reasons.append("muchos dependientes directos")

        if total_impact >= self.HIGH_TOTAL_IMPACT:
            reasons.append("impacto total alto")

        if path in cycle_nodes:
            reasons.append("participa en ciclo")

        if (
            direct_dependents >= self.CRITICAL_DIRECT_DEPENDENTS
            or total_impact >= self.CRITICAL_TOTAL_IMPACT
        ):
            reasons.append("módulo crítico")

        if not reasons:
            reasons.append("sin señales relevantes de riesgo")

        return reasons

    def _risk_level(
        self,
        outgoing_dependencies: int,
        direct_dependents: int,
        total_impact: int,
        in_cycle: bool,
    ) -> RiskLevel:
        """Classify risk using deterministic thresholds."""
        if (
            direct_dependents >= self.CRITICAL_DIRECT_DEPENDENTS
            or total_impact >= self.CRITICAL_TOTAL_IMPACT
        ):
            return "critical"

        if (
            in_cycle
            or outgoing_dependencies >= self.HIGH_OUTGOING_DEPENDENCIES
            or direct_dependents >= self.HIGH_DIRECT_DEPENDENTS
            or total_impact >= self.HIGH_TOTAL_IMPACT
        ):
            return "high"

        if outgoing_dependencies >= 2 or direct_dependents >= 2 or total_impact >= 2:
            return "medium"

        return "low"

    def _dependents_by_path(
        self,
        nodes: list[ArchitectureNode],
    ) -> dict[str, list[str]]:
        """Return direct dependents keyed by target path."""
        paths = {node.path for node in nodes}
        dependents = {
            path: []
            for path in paths
        }

        for node in nodes:
            for dependency in node.dependencies:
                if dependency in dependents:
                    dependents[dependency].append(node.path)

        return {
            path: sorted(values)
            for path, values in dependents.items()
        }

    def _transitive_dependents(
        self,
        path: str,
        dependents_by_path: dict[str, list[str]],
    ) -> set[str]:
        """Return every direct or indirect dependent of a module."""
        visited = {path}
        affected: set[str] = set()
        pending = list(dependents_by_path[path])

        while pending:
            current_path = pending.pop(0)

            if current_path in visited:
                continue

            visited.add(current_path)
            affected.add(current_path)
            pending.extend(dependents_by_path[current_path])

        return affected

    def _detect_cycles(
        self,
        nodes: list[ArchitectureNode],
        nodes_by_path: dict[str, ArchitectureNode],
    ) -> list[list[str]]:
        """Return canonical dependency cycles."""
        cycles: set[tuple[str, ...]] = set()

        for node in nodes:
            self._visit_cycles(
                start=node.path,
                current=node.path,
                nodes_by_path=nodes_by_path,
                path=[],
                cycles=cycles,
            )

        return [
            list(cycle)
            for cycle in sorted(cycles)
        ]

    def _visit_cycles(
        self,
        start: str,
        current: str,
        nodes_by_path: dict[str, ArchitectureNode],
        path: list[str],
        cycles: set[tuple[str, ...]],
    ) -> None:
        """Depth-first cycle detection from one start node."""
        if current in path:
            if current == start:
                cycles.add(self._canonical_cycle(path))
            return

        node = nodes_by_path[current]
        next_path = [*path, current]

        for dependency in node.dependencies:
            if dependency not in nodes_by_path:
                continue

            if dependency != start and dependency in path:
                continue

            self._visit_cycles(
                start=start,
                current=dependency,
                nodes_by_path=nodes_by_path,
                path=next_path,
                cycles=cycles,
            )

    def _canonical_cycle(
        self,
        cycle: list[str],
    ) -> tuple[str, ...]:
        """Rotate a cycle so the lexical first path is first."""
        rotations = [
            tuple(cycle[index:] + cycle[:index])
            for index in range(len(cycle))
        ]

        return min(rotations)

    def _max_paths(
        self,
        results: list[ArchitectureRiskResult],
        attribute: str,
    ) -> list[str]:
        """Return paths tied for the highest non-negative metric."""
        if not results:
            return []

        highest = max(
            getattr(result, attribute)
            for result in results
        )

        return [
            result.path
            for result in results
            if getattr(result, attribute) == highest
        ]

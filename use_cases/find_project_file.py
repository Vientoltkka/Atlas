"""Find Python project files using the structural project index."""

from __future__ import annotations

from pathlib import Path

from use_cases.read_project_index import ReadProjectIndexUseCase


class FindProjectFileUseCase:
    """Locate project files by path, filename, class, or function names."""

    def __init__(
        self,
        read_project_index: ReadProjectIndexUseCase,
    ) -> None:
        self._read_project_index = read_project_index

    def execute(
        self,
        query: str,
        root: str = ".",
        index: list[dict[str, object]] | None = None,
    ) -> str | list[str] | None:
        """Return one matching path or ranked matching paths."""
        normalized_queries = self._query_candidates(query)

        if not normalized_queries:
            return None

        matches: list[tuple[int, str]] = []

        project_index = index or self._read_project_index.execute(root)

        for item in project_index:
            path = Path(str(item["path"])).as_posix()
            score = max(
                self._score_item(normalized_query, item)
                for normalized_query in normalized_queries
            )

            if score > 0:
                matches.append((score, path))

        ranked_paths = self._rank_paths(matches)

        if not ranked_paths:
            return None

        if len(ranked_paths) == 1:
            return ranked_paths[0]

        return ranked_paths

    def _score_item(
        self,
        query: str,
        item: dict[str, object],
    ) -> int:
        """Calculate relevance for one indexed file."""
        filename = self._normalize(str(item["filename"]))
        path = self._normalize(str(item["path"])).replace("\\", "/")
        path_without_suffix = self._normalize(str(Path(str(item["path"])).with_suffix("")))
        query_path = query.replace("\\", "/")

        scores = [
            self._score_text(query, filename, exact=100, contains=65),
            self._score_text(query_path, path, exact=95, contains=55),
            self._score_text(query, path_without_suffix, exact=90, contains=50),
            self._score_names(query, item.get("classes", []), exact=85, contains=45),
            self._score_names(query, item.get("functions", []), exact=80, contains=40),
        ]

        return max(scores)

    def _score_text(
        self,
        query: str,
        value: str,
        exact: int,
        contains: int,
    ) -> int:
        """Score a normalized query against a normalized text value."""
        if query == value:
            return exact

        if value.endswith(f"/{query}") or value.endswith(f"\\{query}"):
            return exact - 5

        if query in value:
            return contains

        return 0

    def _score_names(
        self,
        query: str,
        values: object,
        exact: int,
        contains: int,
    ) -> int:
        """Score a query against class or function names."""
        if not isinstance(values, list):
            return 0

        score = 0

        for value in values:
            normalized_value = self._normalize(str(value))
            short_name = normalized_value.rsplit(".", maxsplit=1)[-1]

            if query == normalized_value or query == short_name:
                score = max(score, exact)
            elif query in normalized_value:
                score = max(score, contains)

        return score

    def _rank_paths(
        self,
        matches: list[tuple[int, str]],
    ) -> list[str]:
        """Return unique paths ordered by relevance."""
        best_scores: dict[str, int] = {}

        for score, path in matches:
            best_scores[path] = max(score, best_scores.get(path, 0))

        return [
            path
            for path, _ in sorted(
                best_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

    def _normalize(
        self,
        value: str,
    ) -> str:
        """Normalize user queries and indexed values for matching."""
        return value.strip().strip("\"'`").lower()

    def _query_candidates(
        self,
        query: str,
    ) -> list[str]:
        """Extract searchable candidates from direct queries or sentences."""
        normalized_query = self._normalize(query)

        if not normalized_query:
            return []

        candidates = [normalized_query]
        ignored_tokens = {
            "analiza",
            "analizar",
            "archivo",
            "clase",
            "donde",
            "dónde",
            "el",
            "encuentra",
            "esta",
            "está",
            "funcion",
            "función",
            "la",
            "localiza",
            "path",
            "ruta",
        }

        for token in query.split():
            clean_token = self._normalize(token.strip(".,:;()[]{}¿?¡!"))

            if clean_token and clean_token not in ignored_tokens:
                candidates.append(clean_token)

        return list(dict.fromkeys(candidates))

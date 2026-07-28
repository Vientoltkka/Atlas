"""Deterministic bounded operational context built from Atlas memory."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from core.operational_request_router import RequestRoute, RouteDecision
from core.request_gateway import AtlasRequest
from memory.operational import (
    MemoryCategory,
    MemoryEntry,
    MemoryPolicy,
    freeze_safe_mapping,
    normalize_content,
    normalize_tag,
)


_STOP_WORDS = frozenset(
    {
        "a",
        "al",
        "con",
        "de",
        "del",
        "el",
        "en",
        "es",
        "esta",
        "este",
        "la",
        "las",
        "lo",
        "los",
        "para",
        "por",
        "que",
        "un",
        "una",
        "y",
    }
)


class OperationalContextError(RuntimeError):
    code = "OPERATIONAL_CONTEXT_ERROR"


@dataclass(frozen=True, slots=True)
class MemoryRelevanceScore:
    memory_id: str
    exact_tag_score: float
    keyword_overlap_score: float
    category_score: float
    recency_score: float
    importance_score: float
    conversation_score: float
    final_score: float


@dataclass(frozen=True, slots=True)
class OperationalContext:
    request_id: str
    conversation_id: str | None
    recent_messages: tuple[Mapping[str, str], ...]
    relevant_memories: tuple[MemoryEntry, ...]
    user_preferences: tuple[MemoryEntry, ...]
    project_context: tuple[MemoryEntry, ...]
    execution_context: Mapping[str, Any]
    selected_memory_ids: tuple[str, ...]
    total_characters: int
    truncated: bool
    generated_at: datetime

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware.")
        if self.total_characters < 0:
            raise ValueError("total_characters cannot be negative.")
        object.__setattr__(
            self,
            "recent_messages",
            tuple(
                MappingProxyType(
                    {
                        "role": str(message.get("role", "")),
                        "content": str(message.get("content", "")),
                    }
                )
                for message in self.recent_messages
            ),
        )
        object.__setattr__(self, "relevant_memories", tuple(self.relevant_memories))
        object.__setattr__(self, "user_preferences", tuple(self.user_preferences))
        object.__setattr__(self, "project_context", tuple(self.project_context))
        object.__setattr__(
            self,
            "execution_context",
            freeze_safe_mapping(self.execution_context),
        )
        object.__setattr__(
            self,
            "selected_memory_ids",
            tuple(self.selected_memory_ids),
        )

    def safe_summary(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "selected_memory_ids": self.selected_memory_ids,
                "selected_count": len(self.selected_memory_ids),
                "total_characters": self.total_characters,
                "truncated": self.truncated,
            }
        )

    def prompt_context(self) -> str:
        """Return bounded plain context without metadata or relevance scores."""
        parts: list[str] = []
        if self.user_preferences:
            parts.append(
                "Preferencias: "
                + " | ".join(entry.content for entry in self.user_preferences)
            )
        non_preferences = tuple(
            entry
            for entry in self.relevant_memories
            if entry.category is not MemoryCategory.USER_PREFERENCE
        )
        if non_preferences:
            parts.append(
                "Contexto relevante: "
                + " | ".join(entry.content for entry in non_preferences)
            )
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class OperationalContextEvent:
    event_type: str
    request_id: str
    memory_id: str | None
    category: MemoryCategory | None
    operation: str
    selected_count: int
    content_length: int
    truncated: bool
    reason_code: str
    timestamp: datetime


class OperationalContextBuilder:
    """Select a small memory snapshot using reproducible local rules."""

    def __init__(
        self,
        memory: object | None,
        *,
        policy: MemoryPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._memory = memory
        self._policy = policy or getattr(memory, "policy", None) or MemoryPolicy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._events: list[OperationalContextEvent] = []

    @property
    def events(self) -> tuple[OperationalContextEvent, ...]:
        return tuple(self._events)

    def build(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
    ) -> OperationalContext:
        if request.request_id != decision.request_id:
            raise OperationalContextError("Request and RouteDecision do not correspond.")
        now = self._clock()
        self._record(
            "operational_context_build_started",
            request,
            operation="build",
            reason_code="started",
            timestamp=now,
        )
        recent = self._recent_messages(request, decision)
        scored = self._score_entries(request, decision, now)
        selected, memories_truncated = self._select_memories(scored)
        recent, selected, size_truncated = self._apply_character_limit(recent, selected)
        truncated = memories_truncated or size_truncated
        user_preferences = tuple(
            entry
            for entry in selected
            if entry.category is MemoryCategory.USER_PREFERENCE
        )
        project_context = tuple(
            entry
            for entry in selected
            if entry.category
            in {
                MemoryCategory.PROJECT_FACT,
                MemoryCategory.DECISION,
                MemoryCategory.TASK_CONTEXT,
                MemoryCategory.EXECUTION_RESULT,
            }
        )
        execution_context = (
            asdict(request.execution_context)
            if request.execution_context is not None
            else {}
        )
        total_characters = sum(
            len(message["content"]) for message in recent
        ) + sum(len(entry.content) for entry in selected)
        context = OperationalContext(
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            recent_messages=recent,
            relevant_memories=selected,
            user_preferences=user_preferences,
            project_context=project_context,
            execution_context=execution_context,
            selected_memory_ids=tuple(entry.memory_id for entry in selected),
            total_characters=total_characters,
            truncated=truncated,
            generated_at=now,
        )
        if truncated:
            self._record(
                "context_truncated",
                request,
                operation="build",
                selected_count=len(selected),
                content_length=total_characters,
                truncated=True,
                reason_code="configured_limit",
                timestamp=now,
            )
        self._record(
            "operational_context_built",
            request,
            operation="build",
            selected_count=len(selected),
            content_length=total_characters,
            truncated=truncated,
            reason_code="built",
            timestamp=now,
        )
        return context

    def _recent_messages(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
    ) -> tuple[Mapping[str, str], ...]:
        if decision.route not in {
            RequestRoute.DIRECT_RESPONSE,
            RequestRoute.AGENT_DELEGATION,
        }:
            return ()
        history = getattr(self._memory, "history", None)
        if not callable(history):
            return ()
        messages = []
        for message in history():
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role", "")).strip()
            content = " ".join(str(message.get("content", "")).split())
            if not role or not content or content == request.content:
                continue
            messages.append({"role": role, "content": content})
        return tuple(messages[-self._policy.recent_entry_limit :])

    def _score_entries(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        now: datetime,
    ) -> tuple[tuple[MemoryEntry, MemoryRelevanceScore], ...]:
        list_entries = getattr(self._memory, "list_entries", None)
        if not callable(list_entries):
            return ()
        allowed_categories = _categories_for_route(decision)
        if not allowed_categories:
            return ()
        request_tokens = _significant_tokens(request.content)
        request_tags = _request_tags(request)
        scored: list[tuple[MemoryEntry, MemoryRelevanceScore]] = []
        for entry in list_entries(
            active_only=True,
            include_expired=False,
            include_sensitive=False,
            limit=self._policy.max_entries_per_query,
        ):
            if not isinstance(entry, MemoryEntry):
                continue
            reason = _rejection_reason(entry, request, decision, allowed_categories, now)
            if reason is not None:
                self._record(
                    "memory_candidate_rejected",
                    request,
                    memory_id=entry.memory_id,
                    category=entry.category,
                    operation="select",
                    content_length=len(entry.content),
                    reason_code=reason,
                    timestamp=now,
                )
                continue
            score = _score_entry(
                entry,
                request,
                request_tokens=request_tokens,
                request_tags=request_tags,
                now=now,
            )
            if not _entry_is_relevant(entry, score, decision.route):
                self._record(
                    "memory_candidate_rejected",
                    request,
                    memory_id=entry.memory_id,
                    category=entry.category,
                    operation="select",
                    content_length=len(entry.content),
                    reason_code="insufficient_relevance",
                    timestamp=now,
                )
                continue
            scored.append((entry, score))
            self._record(
                "memory_candidate_selected",
                request,
                memory_id=entry.memory_id,
                category=entry.category,
                operation="select",
                content_length=len(entry.content),
                reason_code="relevant",
                timestamp=now,
            )
        return tuple(
            sorted(
                scored,
                key=lambda item: (
                    -item[1].final_score,
                    -item[0].importance,
                    -item[0].updated_at.timestamp(),
                    item[0].memory_id,
                ),
            )
        )

    def _select_memories(
        self,
        scored: tuple[tuple[MemoryEntry, MemoryRelevanceScore], ...],
    ) -> tuple[tuple[MemoryEntry, ...], bool]:
        selected = tuple(
            entry for entry, _score in scored[: self._policy.max_context_entries]
        )
        return selected, len(scored) > len(selected)

    def _apply_character_limit(
        self,
        recent: tuple[Mapping[str, str], ...],
        selected: tuple[MemoryEntry, ...],
    ) -> tuple[
        tuple[Mapping[str, str], ...],
        tuple[MemoryEntry, ...],
        bool,
    ]:
        limit = self._policy.max_context_characters
        kept_messages: list[Mapping[str, str]] = []
        kept_entries: list[MemoryEntry] = []
        used = 0
        truncated = False
        for message in reversed(recent):
            size = len(message["content"])
            if used + size > limit:
                truncated = True
                continue
            kept_messages.append(message)
            used += size
        kept_messages.reverse()
        for entry in selected:
            size = len(entry.content)
            if used + size > limit:
                truncated = True
                continue
            kept_entries.append(entry)
            used += size
        return tuple(kept_messages), tuple(kept_entries), truncated

    def _record(
        self,
        event_type: str,
        request: AtlasRequest,
        *,
        memory_id: str | None = None,
        category: MemoryCategory | None = None,
        operation: str,
        selected_count: int = 0,
        content_length: int = 0,
        truncated: bool = False,
        reason_code: str,
        timestamp: datetime,
    ) -> None:
        self._events.append(
            OperationalContextEvent(
                event_type=event_type,
                request_id=request.request_id,
                memory_id=memory_id,
                category=category,
                operation=operation,
                selected_count=selected_count,
                content_length=content_length,
                truncated=truncated,
                reason_code=reason_code,
                timestamp=timestamp,
            )
        )


def _categories_for_route(decision: RouteDecision) -> frozenset[MemoryCategory]:
    if decision.route is RequestRoute.DIRECT_RESPONSE:
        return frozenset(
            {
                MemoryCategory.USER_PREFERENCE,
                MemoryCategory.USER_FACT,
                MemoryCategory.CONVERSATION_NOTE,
            }
        )
    if decision.route is RequestRoute.AGENT_DELEGATION:
        return frozenset(
            {
                MemoryCategory.USER_PREFERENCE,
                MemoryCategory.PROJECT_FACT,
                MemoryCategory.DECISION,
                MemoryCategory.TASK_CONTEXT,
            }
        )
    if decision.route is RequestRoute.AUTONOMOUS_EXECUTION:
        return frozenset(
            {
                MemoryCategory.PROJECT_FACT,
                MemoryCategory.DECISION,
                MemoryCategory.TASK_CONTEXT,
                MemoryCategory.EXECUTION_RESULT,
            }
        )
    if decision.route is RequestRoute.MEMORY_QUERY:
        return frozenset(MemoryCategory)
    return frozenset()


def _rejection_reason(
    entry: MemoryEntry,
    request: AtlasRequest,
    decision: RouteDecision,
    categories: frozenset[MemoryCategory],
    now: datetime,
) -> str | None:
    if not entry.active:
        return "inactive"
    if entry.is_expired(now):
        return "expired"
    if entry.sensitive:
        return "sensitive"
    if entry.category not in categories:
        return "incompatible_category"
    if entry.user_id is not None and request.user_id is not None:
        if entry.user_id != request.user_id:
            return "different_user"
    if decision.route is RequestRoute.RESUME_EXECUTION:
        return "resume_uses_session_state"
    return None


def _score_entry(
    entry: MemoryEntry,
    request: AtlasRequest,
    *,
    request_tokens: frozenset[str],
    request_tags: frozenset[str],
    now: datetime,
) -> MemoryRelevanceScore:
    exact_tags = request_tags.intersection(entry.tags)
    exact_tag_score = float(len(exact_tags) * 2)
    entry_tokens = _significant_tokens(entry.content)
    overlap = request_tokens.intersection(entry_tokens)
    keyword_overlap_score = (
        float(len(overlap)) / max(1, len(request_tokens)) * 4.0
    )
    category_score = 1.0
    age_seconds = max(0.0, (now - entry.updated_at).total_seconds())
    recency_score = 1.0 / (1.0 + age_seconds / 86_400.0)
    importance_score = entry.importance * 2.0
    conversation_score = (
        2.0
        if request.conversation_id is not None
        and entry.conversation_id == request.conversation_id
        else 0.0
    )
    final = (
        exact_tag_score
        + keyword_overlap_score
        + category_score
        + recency_score
        + importance_score
        + conversation_score
    )
    return MemoryRelevanceScore(
        memory_id=entry.memory_id,
        exact_tag_score=round(exact_tag_score, 6),
        keyword_overlap_score=round(keyword_overlap_score, 6),
        category_score=round(category_score, 6),
        recency_score=round(recency_score, 6),
        importance_score=round(importance_score, 6),
        conversation_score=round(conversation_score, 6),
        final_score=round(final, 6),
    )


def _entry_is_relevant(
    entry: MemoryEntry,
    score: MemoryRelevanceScore,
    route: RequestRoute,
) -> bool:
    if entry.category is MemoryCategory.USER_PREFERENCE:
        general_markers = {
            "breve",
            "breves",
            "conciso",
            "concisos",
            "idioma",
            "espanol",
            "español",
            "respuesta",
            "respuestas",
            "tono",
        }
        return bool(
            general_markers.intersection(_significant_tokens(entry.content))
            or score.keyword_overlap_score > 0
            or score.exact_tag_score > 0
            or score.conversation_score > 0
        )
    if route is RequestRoute.MEMORY_QUERY:
        return True
    return (
        score.keyword_overlap_score > 0
        or score.exact_tag_score > 0
        or score.conversation_score > 0
    )


def _significant_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in normalize_content(value).split()
        if len(token) > 2 and token not in _STOP_WORDS
    )


def _request_tags(request: AtlasRequest) -> frozenset[str]:
    raw = request.metadata.get("tags", ())
    if isinstance(raw, str):
        values: Sequence[str] = (raw,)
    elif isinstance(raw, Sequence):
        values = tuple(str(item) for item in raw)
    else:
        values = ()
    return frozenset(normalize_tag(item) for item in values if normalize_tag(item))

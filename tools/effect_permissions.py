"""Central explicit authorization for effectful Atlas tools."""

from __future__ import annotations

from dataclasses import dataclass
import secrets


class ToolPermissionDeniedError(PermissionError):
    """Raised before an effectful tool reaches its real adapter."""


@dataclass(frozen=True, slots=True)
class ToolEffectAuthorization:
    """One-use authorization bound to one tool and its declared effects."""

    token: str
    tool_name: str
    permissions: tuple[str, ...]


class ToolEffectPermissionPolicy:
    """Deny declared effects unless a caller explicitly authorizes them."""

    def __init__(self) -> None:
        self._pending: dict[str, tuple[str, tuple[str, ...]]] = {}

    def requires_authorization(self, permissions: tuple[str, ...]) -> bool:
        return bool(permissions)

    def authorize(self, tool_name: str, permissions: tuple[str, ...]) -> ToolEffectAuthorization:
        normalized = _normalized_permissions(permissions)
        if not normalized:
            raise ValueError("Cannot authorize a tool without declared effects.")
        token = secrets.token_urlsafe(24)
        self._pending[token] = (tool_name, normalized)
        return ToolEffectAuthorization(token, tool_name, normalized)

    def require(
        self,
        tool_name: str,
        permissions: tuple[str, ...],
        authorization: ToolEffectAuthorization | None,
    ) -> None:
        normalized = _normalized_permissions(permissions)
        if not normalized:
            return
        if not isinstance(authorization, ToolEffectAuthorization):
            raise ToolPermissionDeniedError(
                f"Explicit authorization is required before executing '{tool_name}'."
            )
        expected = self._pending.pop(authorization.token, None)
        if expected != (tool_name, normalized) or authorization != ToolEffectAuthorization(
            authorization.token, tool_name, normalized
        ):
            raise ToolPermissionDeniedError(
                f"Authorization does not permit executing '{tool_name}'."
            )


def _normalized_permissions(permissions: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(permissions, tuple) or any(
        not isinstance(permission, str) or not permission.strip()
        for permission in permissions
    ):
        raise TypeError("Tool effect permissions must be a tuple of non-empty strings.")
    return tuple(sorted(set(permissions)))

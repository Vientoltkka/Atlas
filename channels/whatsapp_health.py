"""Local-only health diagnostics for the WhatsApp channel (F5.2).

Mirrors the stable-result pattern of ``core/model_health.py`` but is
specific to the channel. Every check is local and side-effect free:
no calls to Meta, no ``send_text``, no network probes. Results contain
only booleans, generic component names and stable codes — never token
values, phone numbers, media ids, URLs or message content.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable, Mapping

from channels.whatsapp_sender import WhatsAppGraphSender


class WhatsAppHealthStatus(str, Enum):
    """Overall diagnostic outcome for the channel."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class WhatsAppHealthCode(str, Enum):
    """Stable classifications for one component check."""

    OK = "OK"
    MISSING_CREDENTIAL = "MISSING_CREDENTIAL"
    STORE_UNAVAILABLE = "STORE_UNAVAILABLE"
    COMPONENT_UNAVAILABLE = "COMPONENT_UNAVAILABLE"


@dataclass(frozen=True)
class WhatsAppChannelHealthResult:
    """Controlled diagnostic result; safe to log or repr by construction."""

    status: WhatsAppHealthStatus
    checks: Mapping[str, str]

    @property
    def healthy(self) -> bool:
        return self.status is WhatsAppHealthStatus.HEALTHY


def _default_store_probe(store: Any) -> bool:
    """Local reachability probe for the configured idempotency store.

    In-memory stores are always reachable. SQLite stores are probed by
    opening a local connection only (no writes to business tables).
    """
    from channels.webhook_idempotency import SqliteIdempotencyStore

    if isinstance(store, SqliteIdempotencyStore):
        try:
            connection = store._connect()
        except Exception:
            return False
        try:
            connection.close()
        except Exception:
            return False
        return True
    return True


def _default_component_check(builder: Callable[[], Any]) -> bool:
    try:
        builder()
    except Exception:
        return False
    return True


class WhatsAppChannelHealthChecker:
    """Evaluate channel readiness using only local configuration checks."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        store: Any,
        transcriber: Any,
        voice_renderer: Any,
        store_probe: Callable[[Any], bool] | None = None,
        sender_builder: Callable[[], Any] | None = None,
    ) -> None:
        self._config = dict(config)
        self._store = store
        self._transcriber = transcriber
        self._voice_renderer = voice_renderer
        self._store_probe = store_probe or _default_store_probe
        self._sender_builder = sender_builder or self._build_default_sender
        self._lock = Lock()

    def check(self) -> WhatsAppChannelHealthResult:
        with self._lock:
            return self._run_checks()

    def _run_checks(self) -> WhatsAppChannelHealthResult:
        checks: dict[str, str] = {}

        for name in ("verify_token", "access_token", "phone_number_id"):
            value = self._config.get(name)
            if isinstance(value, str) and value.strip():
                checks[name] = WhatsAppHealthCode.OK.value
            else:
                checks[name] = WhatsAppHealthCode.MISSING_CREDENTIAL.value

        try:
            store_ok = bool(self._store_probe(self._store))
        except Exception:
            store_ok = False
        checks["idempotency_store"] = (
            WhatsAppHealthCode.OK.value
            if store_ok
            else WhatsAppHealthCode.STORE_UNAVAILABLE.value
        )

        try:
            sender_ok = bool(_default_component_check(self._sender_builder))
        except Exception:
            sender_ok = False
        checks["sender"] = (
            WhatsAppHealthCode.OK.value
            if sender_ok
            else WhatsAppHealthCode.COMPONENT_UNAVAILABLE.value
        )

        required_ok = (
            all(code == WhatsAppHealthCode.OK.value for code in checks.values())
        )
        optional_checks = {
            "transcriber": self._transcriber is not None,
            "voice_renderer": self._voice_renderer is not None,
        }
        for name, available in optional_checks.items():
            checks[name] = (
                WhatsAppHealthCode.OK.value
                if available
                else WhatsAppHealthCode.COMPONENT_UNAVAILABLE.value
            )

        if not required_ok:
            status = WhatsAppHealthStatus.UNHEALTHY
        elif not all(optional_checks.values()):
            status = WhatsAppHealthStatus.DEGRADED
        else:
            status = WhatsAppHealthStatus.HEALTHY
        return WhatsAppChannelHealthResult(status=status, checks=checks)

    def _build_default_sender(self) -> Any:
        return WhatsAppGraphSender(
            access_token=str(self._config.get("access_token", "")),
            phone_number_id=str(self._config.get("phone_number_id", "")),
        )

    def __repr__(self) -> str:
        return "WhatsAppChannelHealthChecker(configured=True)"

"""Safely diagnose one externally configured chat inference provider."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from models.chat_inference import (
    ChatInferenceError,
    ChatInferenceProvider,
    configured_provider_id,
    default_provider_registry,
)


def diagnose_provider(
    provider: ChatInferenceProvider,
    *,
    provider_id: str,
    model: str,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Run one minimal health request and return a safe, provider-neutral result."""
    started = clock()
    try:
        response = provider.health(model=model)
    except ChatInferenceError as error:
        return _failure(
            provider_id=error.provider_id,
            model=error.model,
            latency_ms=_elapsed_ms(started, clock),
            code="CHAT_INFERENCE_ERROR",
            message="Provider health check failed. Verify provider configuration and connectivity.",
        )
    except (ValueError, RuntimeError) as error:
        return _failure(
            provider_id=provider_id,
            model=model,
            latency_ms=_elapsed_ms(started, clock),
            code="PROVIDER_CONFIGURATION_ERROR",
            message="Provider health check failed. Verify provider configuration and connectivity.",
        )
    return {
        "provider": provider_id,
        "model": model,
        "health": "healthy",
        "latency_ms": _elapsed_ms(started, clock),
        "response": _brief_response(response),
        "error": None,
    }


def diagnose_from_environment(
    *,
    registry_builder=default_provider_registry,
    environment: Callable[[str, str], str] = os.getenv,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Resolve the configured provider and model without exposing credentials."""
    provider_id = configured_provider_id()
    model = _configured_model(provider_id, environment)
    started = clock()
    try:
        registry = registry_builder(
            timeout=_timeout(environment),
            keep_alive=environment("ATLAS_OLLAMA_KEEP_ALIVE", "10m").strip() or "10m",
            provider_id=provider_id,
        )
        provider = registry.get(provider_id)
    except (ValueError, RuntimeError) as error:
        return _failure(
            provider_id=provider_id,
            model=model,
            latency_ms=_elapsed_ms(started, clock),
            code="PROVIDER_CONFIGURATION_ERROR",
            message="Provider health check failed. Verify provider configuration and connectivity.",
        )
    return diagnose_provider(provider, provider_id=provider_id, model=model, clock=clock)


def _configured_model(provider_id: str, environment: Callable[[str, str], str]) -> str:
    explicit_model = environment("ATLAS_PROVIDER_DIAGNOSTIC_MODEL", "").strip()
    if explicit_model:
        return explicit_model
    if provider_id == "gemini":
        return environment("ATLAS_GEMINI_MODEL", "").strip()
    return environment("ATLAS_OPENAI_MODEL", "").strip()


def _timeout(environment: Callable[[str, str], str]) -> float:
    try:
        return min(max(float(environment("ATLAS_OLLAMA_TIMEOUT", "120")), 1.0), 600.0)
    except ValueError:
        return 120.0


def _brief_response(response: Any, limit: int = 240) -> str:
    del response, limit
    return "health check completed"


def _failure(*, provider_id: str, model: str, latency_ms: int, code: str, message: str) -> dict[str, Any]:
    return {
        "provider": provider_id,
        "model": model,
        "health": "unhealthy",
        "latency_ms": latency_ms,
        "response": None,
        "error": {"code": code, "message": _redact(message)},
    }


def _elapsed_ms(started: float, clock: Callable[[], float]) -> int:
    return max(0, round((clock() - started) * 1000))


def _redact(text: str) -> str:
    return re.sub(
        r"(?i)(api[_-]?key|token|secret|password|credential)(\s*[:=]\s*)([^\s,}]+)",
        r"\1\2[REDACTED]",
        text,
    )


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> int:
    _load_dotenv()
    result = diagnose_from_environment()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["health"] == "healthy" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Prompt client for Atlas."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Any

import ollama


class PromptClient:
    """Client used to communicate with Ollama."""

    def __init__(self) -> None:
        self._client = ollama.Client(
            timeout=_read_float("ATLAS_OLLAMA_TIMEOUT", 120.0, 1.0, 600.0)
        )
        self._keep_alive = os.getenv("ATLAS_OLLAMA_KEEP_ALIVE", "10m").strip() or "10m"
        self._seen_models: set[str] = set()
        self.last_metrics: dict[str, str | float | bool] = {}

    def ask(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Send a conversation to the selected model."""
        return self.ask_messages(model=model, messages=messages)

    def ask_messages(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Send exactly the provided messages to the selected model."""
        response = self._client.chat(
            model=model,
            messages=messages,
            keep_alive=self._keep_alive,
        )
        self._record_metrics(model, response)
        return self._extract_content(response)

    def stream_messages(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        """Stream exactly the provided messages and yield content fragments."""
        stream = self._client.chat(
            model=model,
            messages=messages,
            stream=True,
            keep_alive=self._keep_alive,
        )

        for chunk in stream:
            content = self._extract_stream_content(chunk)
            if content:
                yield content

    def _record_metrics(self, requested_model: str, response: Any) -> None:
        model = str(self._response_value(response, "model") or requested_model)
        load_seconds = self._duration_seconds(response, "load_duration")
        generation_seconds = self._duration_seconds(response, "eval_duration")
        total_seconds = self._duration_seconds(response, "total_duration")
        previously_seen = model in self._seen_models
        reused_loaded_model = previously_seen and load_seconds < 1.0
        self._seen_models.add(model)
        self.last_metrics = {
            "model": model,
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "total_seconds": total_seconds,
            "reused_loaded_model": reused_loaded_model,
            "keep_alive": self._keep_alive,
        }
        if _read_bool("ATLAS_VOICE_METRICS", False) or _read_bool(
            "ATLAS_VOICE_DEBUG", False
        ):
            print(
                "[ollama-metrics] "
                f"modelo={model} "
                f"carga={load_seconds:.3f}s "
                f"generacion={generation_seconds:.3f}s "
                f"total={total_seconds:.3f}s "
                f"modelo_reutilizado={'si' if reused_loaded_model else 'no'} "
                f"keep_alive={self._keep_alive}"
            )

    def _duration_seconds(self, response: Any, name: str) -> float:
        value = self._response_value(response, name)
        try:
            return max(0.0, float(value) / 1_000_000_000.0)
        except (TypeError, ValueError):
            return 0.0

    def _response_value(self, response: Any, name: str) -> Any:
        if isinstance(response, dict):
            return response.get(name)
        return getattr(response, name, None)
    def _extract_content(self, response: Any) -> str:
        """Extract text from the Ollama response."""

        if isinstance(response, dict):
            return response["message"]["content"]

        return response.message.content

    def _extract_stream_content(self, chunk: Any) -> str | None:
        """Extract one streamed content fragment from an Ollama chunk."""
        if isinstance(chunk, dict):
            message = chunk.get("message")
            if message is None:
                return None
            if not isinstance(message, dict):
                raise ValueError("Malformed Ollama stream chunk.")
            content = message.get("content")
        else:
            message = getattr(chunk, "message", None)
            if message is None:
                return None
            content = getattr(message, "content", None)

        if content is None:
            return None
        if not isinstance(content, str):
            raise ValueError("Malformed Ollama stream chunk.")
        return content


def _read_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)

def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "s", "si", "sí"}
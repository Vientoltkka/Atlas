"""Prompt client for Atlas."""

from __future__ import annotations

from collections.abc import Iterator
import os
from time import perf_counter
from typing import Any

from models.chat_inference import (
    ChatInferenceError,
    ChatInferenceProvider,
    configured_provider_id,
    default_provider_registry,
    ollama,
)


class InferenceBackendError(RuntimeError, ValueError):
    """Failure attributable to the configured inference backend or its response."""

    def __init__(self, model: str, reason: str) -> None:
        self.model = model
        self.reason = reason.strip() or "Inference backend failed."
        super().__init__(f"Inference failed for model '{model}': {self.reason}")


class PromptClient:
    """Client used to communicate with Ollama."""

    def __init__(self, provider: ChatInferenceProvider | None = None) -> None:
        timeout = _read_float("ATLAS_OLLAMA_TIMEOUT", 120.0, 1.0, 600.0)
        keep_alive = os.getenv("ATLAS_OLLAMA_KEEP_ALIVE", "10m").strip() or "10m"
        provider_id = configured_provider_id()
        self._providers = default_provider_registry(
            timeout=timeout,
            keep_alive=keep_alive,
            provider_id=provider_id,
        )
        self._provider = provider or self._providers.get(provider_id)
        self._keep_alive = os.getenv("ATLAS_OLLAMA_KEEP_ALIVE", "10m").strip() or "10m"
        self._seen_models: set[str] = set()
        self.last_metrics: dict[str, str | float | bool] = {}

    def ask(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        provider_id: str | None = None,
    ) -> str:
        """Send a conversation to the selected model."""
        return self.ask_messages(
            model=model,
            messages=messages,
            provider_id=provider_id,
        )

    def check_model_health(self, model: str, *, provider_id: str | None = None) -> None:
        """Verify one model with a minimal non-streaming provider request."""
        try:
            response = self._provider_for(provider_id).health(model=model)
        except ChatInferenceError as error:
            raise InferenceBackendError(model, error.reason) from error
        try:
            content = self._extract_content(response)
        except (AttributeError, KeyError, TypeError) as error:
            raise InferenceBackendError(model, "Provider returned a malformed health-check response.") from error
        if not isinstance(content, str):
            raise InferenceBackendError(model, "Provider returned a non-text health-check response.")
    def ask_messages(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        provider_id: str | None = None,
    ) -> str:
        """Send exactly the provided messages to the selected model."""
        started = perf_counter()
        first_fragment_seconds: float | None = None
        fragments: list[str] = []
        final_chunk: Any = None
        try:
            stream = self._provider_for(provider_id).chat(
                model=model,
                messages=messages,
                stream=True,
                            )
            for chunk in stream:
                final_chunk = chunk
                content = self._extract_stream_content(chunk)
                if content:
                    if first_fragment_seconds is None:
                        first_fragment_seconds = perf_counter() - started
                    fragments.append(content)
        except InferenceBackendError:
            raise
        except ChatInferenceError as error:
            raise InferenceBackendError(model, str(error)) from error
        if final_chunk is None:
            raise InferenceBackendError(
                model,
                "Ollama returned an empty response stream.",
            )
        self._record_metrics(
            model,
            final_chunk,
            first_fragment_seconds=first_fragment_seconds,
            wall_total_seconds=perf_counter() - started,
        )
        return "".join(fragments)

    def stream_messages(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        provider_id: str | None = None,
    ) -> Iterator[str]:
        """Stream exactly the provided messages and yield content fragments."""
        started = perf_counter()
        first_fragment_seconds: float | None = None
        final_chunk: Any = None
        yielded_content = False
        try:
            try:
                stream = self._provider_for(provider_id).chat(
                    model=model,
                    messages=messages,
                    stream=True,
                                    )
                for chunk in stream:
                    final_chunk = chunk
                    content = self._extract_stream_content(chunk)
                    if content:
                        if first_fragment_seconds is None:
                            first_fragment_seconds = perf_counter() - started
                        yield content
                        yielded_content = True
            except ChatInferenceError as error:
                raise InferenceBackendError(model, error.reason) from error
        finally:
            if final_chunk is not None:
                self._record_metrics(
                    model,
                    final_chunk,
                    first_fragment_seconds=first_fragment_seconds,
                    wall_total_seconds=perf_counter() - started,
                )
        if not yielded_content:
            raise InferenceBackendError(
                model,
                "Ollama returned no observable stream content.",
            )

    def _provider_for(self, provider_id: str | None) -> ChatInferenceProvider:
        if provider_id is None or provider_id == self._provider.provider_id:
            return self._provider
        return self._providers.get(provider_id)

    def _record_metrics(
        self,
        requested_model: str,
        response: Any,
        *,
        first_fragment_seconds: float | None = None,
        wall_total_seconds: float | None = None,
    ) -> None:
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
            "first_fragment_seconds": max(0.0, first_fragment_seconds or 0.0),
            "wall_total_seconds": max(0.0, wall_total_seconds or 0.0),
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
                f"primer_fragmento={max(0.0, first_fragment_seconds or 0.0):.3f}s "
                f"pared_total={max(0.0, wall_total_seconds or 0.0):.3f}s "
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

"""Benchmark installed Ollama models through Atlas PromptClient streaming."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any
import unicodedata

import ollama

from models.prompt_client import PromptClient


DEFAULT_MODELS = (
    "llama3.1:8b",
    "glm4:9b",
    "gemma4:e4b",
    "glm-5.2-local:latest",
)
PROMPTS = (
    "¿Cuál es la capital de Francia?",
    "¿Qué es CrossFit? Respóndeme en dos frases.",
    (
        "Créame un entrenamiento HYROX de 40 minutos para parejas con carrera "
        "sincronizada, sled push, burpees y sandbag lunges. Sé conciso."
    ),
)
WARMUP_PROMPT = "Responde únicamente con: OK"


class _CapturingOllamaClient:
    """Delegate Ollama calls while retaining the latest streamed chunk."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.final_chunk: Any = None

    def chat(self, **kwargs: Any) -> Iterator[Any]:
        self.final_chunk = None
        stream = self._delegate.chat(**kwargs)

        def capture() -> Iterator[Any]:
            for chunk in stream:
                self.final_chunk = chunk
                yield chunk

        return capture()


def main() -> int:
    """Run a controlled warm benchmark and optionally persist exact JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    models = tuple(args.models) if args.models else DEFAULT_MODELS

    installed = _installed_models()
    missing = tuple(model for model in models if model not in installed)
    if missing:
        parser.error("models are not installed locally: " + ", ".join(missing))

    results: list[dict[str, Any]] = []
    for model in models:
        client = PromptClient()
        capturing = _CapturingOllamaClient(client._client)  # type: ignore[attr-defined]
        client._client = capturing  # type: ignore[attr-defined]

        warmup = _measure(
            client,
            capturing,
            model,
            WARMUP_PROMPT,
            query_index=0,
            expected_reused=False,
        )
        print(
            json.dumps(
                {
                    "event": "warmup_completed",
                    "model": model,
                    "total_seconds": warmup["wall_total_seconds"],
                    "load_seconds": warmup["load_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        queries = [
            _measure(
                client,
                capturing,
                model,
                prompt,
                query_index=index,
                expected_reused=True,
            )
            for index, prompt in enumerate(PROMPTS, start=1)
        ]
        results.append(
            {
                "model": model,
                "model_metadata": _model_metadata(model),
                "loaded_resource": _loaded_resource(model),
                "warmup": warmup,
                "queries": queries,
            }
        )
        for query in queries:
            print(
                json.dumps(
                    {
                        "event": "query_completed",
                        "model": model,
                        "query_index": query["query_index"],
                        "first_fragment_seconds": query["first_fragment_seconds"],
                        "wall_total_seconds": query["wall_total_seconds"],
                        "eval_count": query["eval_count"],
                        "tokens_per_second": query["tokens_per_second"],
                        "automatic_correct": query["automatic_correct"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    payload = {
        "library": "ollama",
        "method": "PromptClient.stream_messages",
        "streaming": True,
        "keep_alive": "10m",
        "historical_context": False,
        "warmup_prompt": WARMUP_PROMPT,
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output)}, ensure_ascii=False), flush=True)
    else:
        print(rendered)
    return 0


def _measure(
    client: PromptClient,
    capturing: _CapturingOllamaClient,
    model: str,
    prompt: str,
    *,
    query_index: int,
    expected_reused: bool,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    started = perf_counter()
    first_fragment_seconds: float | None = None
    fragments: list[str] = []
    for fragment in client.stream_messages(model=model, messages=messages):
        if first_fragment_seconds is None:
            first_fragment_seconds = perf_counter() - started
        fragments.append(fragment)
    wall_total_seconds = perf_counter() - started
    response = "".join(fragments).strip()
    final_chunk = capturing.final_chunk
    if final_chunk is None:
        raise RuntimeError(f"Ollama returned no chunks for {model} query {query_index}.")

    load_seconds = _duration_seconds(final_chunk, "load_duration")
    eval_seconds = _duration_seconds(final_chunk, "eval_duration")
    native_total_seconds = _duration_seconds(final_chunk, "total_duration")
    eval_count = _integer_value(final_chunk, "eval_count")
    prompt_eval_count = _integer_value(final_chunk, "prompt_eval_count")
    tokens_per_second = (
        eval_count / eval_seconds if eval_count is not None and eval_seconds > 0 else None
    )
    return {
        "query_index": query_index,
        "prompt": prompt,
        "message_count": len(messages),
        "context_characters": len(prompt),
        "first_fragment_seconds": _rounded(first_fragment_seconds),
        "wall_total_seconds": round(wall_total_seconds, 6),
        "native_total_seconds": round(native_total_seconds, 6),
        "load_seconds": round(load_seconds, 6),
        "generation_seconds": round(eval_seconds, 6),
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "tokens_per_second": _rounded(tokens_per_second),
        "reused_loaded_model": expected_reused,
        "load_below_one_second": load_seconds < 1.0,
        "response_characters": len(response),
        "response": response,
        "automatic_correct": _automatic_correct(query_index, response),
    }


def _automatic_correct(query_index: int, response: str) -> bool | None:
    if query_index == 0:
        return None
    normalized = _normalize(response)
    if query_index == 1:
        return "paris" in normalized
    if query_index == 2:
        sentence_count = len(re.findall(r"[.!?](?=\s|$)", response))
        describes_training = any(
            term in normalized
            for term in ("entrenamiento", "acondicionamiento", "ejercicio", "fitness")
        )
        return "crossfit" in normalized and describes_training and sentence_count <= 2
    required_groups = (
        ("40",),
        ("pareja", "parejas"),
        ("sincronizada", "sincronizado", "synchronized"),
        ("sled push", "empuje de trineo", "trineo"),
        ("burpee", "burpees"),
        ("sandbag lunge", "sandbag lunges", "zancada", "zancadas"),
    )
    return all(any(term in normalized for term in group) for group in required_groups)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(character for character in decomposed if unicodedata.category(character) != "Mn")


def _duration_seconds(response: Any, name: str) -> float:
    value = _value(response, name)
    try:
        return max(0.0, float(value) / 1_000_000_000.0)
    except (TypeError, ValueError):
        return 0.0


def _integer_value(response: Any, name: str) -> int | None:
    value = _value(response, name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _value(response: Any, name: str) -> Any:
    if isinstance(response, dict):
        return response.get(name)
    return getattr(response, name, None)


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _installed_models() -> frozenset[str]:
    response = ollama.Client().list()
    models = response.get("models", []) if isinstance(response, dict) else response.models
    return frozenset(
        str(model.get("model") if isinstance(model, dict) else model.model)
        for model in models
    )


def _model_metadata(model: str) -> dict[str, Any]:
    response = ollama.Client().show(model)
    details = response.get("details", {}) if isinstance(response, dict) else response.details
    return {
        "parameter_size": _value(details, "parameter_size"),
        "quantization_level": _value(details, "quantization_level"),
        "family": _value(details, "family"),
    }


def _loaded_resource(model: str) -> dict[str, int] | None:
    response = ollama.Client().ps()
    models = response.get("models", []) if isinstance(response, dict) else response.models
    for item in models:
        name = _value(item, "model") or _value(item, "name")
        if name == model:
            return {
                "size_bytes": int(_value(item, "size") or 0),
                "size_vram_bytes": int(_value(item, "size_vram") or 0),
            }
    return None


if __name__ == "__main__":
    raise SystemExit(main())

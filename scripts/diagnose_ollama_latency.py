"""Measure Atlas PromptClient latency against one local Ollama model."""

from __future__ import annotations

import argparse
import json
import os
from time import perf_counter

from models.prompt_client import PromptClient


DEFAULT_MODEL = "glm-5.2-local:latest"
DEFAULT_PROMPT = "¿Cuál es la capital de Francia?"


def main() -> int:
    """Run consecutive streamed calls through Atlas PromptClient."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    initialization_started = perf_counter()
    client = PromptClient()
    initialization_seconds = perf_counter() - initialization_started
    messages = [{"role": "user", "content": args.prompt}]
    configuration = {
        "model": args.model,
        "keep_alive": os.getenv("ATLAS_OLLAMA_KEEP_ALIVE", "10m").strip() or "10m",
        "timeout_seconds": _effective_timeout(),
        "streaming": True,
        "message_count": len(messages),
        "context_characters": sum(len(message["content"]) for message in messages),
        "client_initialization_seconds": round(initialization_seconds, 6),
    }
    print(json.dumps({"configuration": configuration}, ensure_ascii=False))

    for sequence in range(1, args.repetitions + 1):
        started = perf_counter()
        response = client.ask_messages(model=args.model, messages=messages)
        wall_seconds = perf_counter() - started
        result = {
            "sequence": sequence,
            "wall_response_seconds": round(wall_seconds, 6),
            "response_characters": len(response),
            "response": response,
            "ollama_metrics": client.last_metrics,
        }
        print(json.dumps(result, ensure_ascii=False))

    return 0


def _effective_timeout() -> float:
    raw = os.getenv("ATLAS_OLLAMA_TIMEOUT", "").strip()
    if not raw:
        return 120.0
    try:
        value = float(raw)
    except ValueError:
        return 120.0
    return min(max(value, 1.0), 600.0)


if __name__ == "__main__":
    raise SystemExit(main())

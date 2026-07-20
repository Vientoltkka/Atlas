"""Safe manual probe for structured hybrid planning.

This script plans only. It does not execute tools or start the Atlas orchestrator.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from bootstrap.bootstrap import Bootstrap
from core.deterministic_multi_tool_planner import DeterministicMultiToolPlanner
from core.hybrid_execution_planner import HybridExecutionPlanner, StructuredPlanningProgress


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a structured execution plan without executing tools.")
    parser.add_argument("objective", help="Objective to plan.")
    parser.add_argument(
        "--diagnose-raw-response",
        action="store_true",
        help="Include a limited, redacted diagnostic view of the provider raw response.",
    )
    parser.add_argument(
        "--diagnostic-preview-chars",
        type=int,
        default=300,
        help="Maximum characters to show at the start and end of the raw response diagnostic.",
    )
    parser.add_argument(
        "--diagnose-provider-performance",
        action="store_true",
        help="Print temporary provider prompt-size and timing diagnostics to stderr.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use opt-in provider streaming while still parsing only the complete response.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print safe provider progress metadata to stderr during streaming.",
    )
    args = parser.parse_args()

    registry = Bootstrap.build_tool_registry()
    selector = Bootstrap.build_tool_selector(registry)
    schemas = Bootstrap.build_argument_schema_registry()
    catalog = Bootstrap.build_semantic_tool_catalog(registry, selector, schemas)
    hybrid_planner: HybridExecutionPlanner = Bootstrap.build_hybrid_execution_planner(
        registry,
        schemas,
    )
    provider = Bootstrap.build_structured_plan_provider(
        diagnostic_sink=(
            _print_provider_performance_diagnostic
            if args.diagnose_provider_performance
            else None
        ),
        structured_plan_streaming_enabled=False,
    )
    if args.stream and provider is not None:
        provider = _StreamingPlanProvider(
            provider,
            on_progress=(
                _progress_printer()
                if args.show_progress
                else None
            ),
        )

    result = hybrid_planner.plan(
        args.objective,
        deterministic_planner=DeterministicMultiToolPlanner(),
        catalog=catalog,
        selector=selector,
        plan_provider=provider,
    )

    payload = {
        "success": result.success,
        "handled": result.handled,
        "source": result.source,
        "error_code": result.error_code,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "requires_clarification": result.requires_clarification,
        "missing_information": list(result.missing_information),
        "validation": (
            {
                "is_valid": result.validation_result.is_valid,
                "status": result.validation_result.status,
                "errors": result.validation_result.errors,
                "warnings": result.validation_result.warnings,
                "requires_confirmation": result.validation_result.requires_confirmation,
            }
            if result.validation_result is not None
            else None
        ),
        "plan_diagnostic": _plan_diagnostic(result),
        "provider_performance_diagnostic": _provider_performance_diagnostic(result),
        "plan": _plan_to_dict(result.plan),
    }
    if args.diagnose_raw_response:
        payload["raw_response_diagnostic"] = _raw_response_diagnostic(
            result,
            preview_chars=args.diagnostic_preview_chars,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _plan_to_dict(plan) -> dict | None:
    if plan is None:
        return None
    return {
        "goal": plan.goal,
        "estimated_steps": plan.estimated_steps,
        "required_tools": list(plan.required_tools),
        "detected_risks": list(plan.detected_risks),
        "requires_confirmation": plan.requires_confirmation,
        "status": plan.status,
        "steps": [
            {
                "id": step.id,
                "description": step.description,
                "tool": step.tool,
                "dependencies": list(step.dependencies),
                "status": step.status,
                "arguments": dict(step.arguments),
            }
            for step in plan.ordered_steps
        ],
    }


def _raw_response_diagnostic(result, *, preview_chars: int = 300) -> dict:
    """Return a bounded diagnostic view of a provider response for manual debugging."""
    model_result = getattr(result, "model_result", None)
    raw_response = getattr(result, "raw_response", None)
    if raw_response is None and model_result is not None:
        raw_response = getattr(model_result, "raw_response", None)

    provider_name = getattr(model_result, "provider_name", None) if model_result is not None else None
    model_name = getattr(model_result, "model_name", None) if model_result is not None else None
    message_count = getattr(model_result, "message_count", None) if model_result is not None else None
    message_roles = getattr(model_result, "message_roles", ()) if model_result is not None else ()
    planning_prompt_version = (
        getattr(model_result, "planning_prompt_version", None)
        if model_result is not None
        else None
    )
    preview_limit = max(0, min(preview_chars, 1000))

    if raw_response is None:
        return {
            "has_response": False,
            "is_string": False,
            "length": 0,
            "provider_name": provider_name,
            "model_name": model_name,
            "message_count": message_count,
            "message_roles": list(message_roles),
            "planning_prompt_version": planning_prompt_version,
            "starts_with_json_fence": False,
            "contains_json_fence": False,
            "starts_with_json_object": False,
            "has_text_before_first_json_object": False,
            "contains_nested_content_property": False,
            "looks_like_serialized_error": False,
            "first_chars": "",
            "last_chars": "",
            "safe_repr": "None",
        }

    is_string = isinstance(raw_response, str)
    raw_text = raw_response if is_string else str(raw_response)
    redacted = _redact_sensitive_text(raw_text)
    stripped = raw_text.lstrip()
    lowered = stripped.lower()
    first_json_index = raw_text.find("{")
    first_non_space_index = len(raw_text) - len(stripped)

    return {
        "has_response": True,
        "is_string": is_string,
        "length": len(raw_text),
        "provider_name": provider_name,
        "model_name": model_name,
        "message_count": message_count,
        "message_roles": list(message_roles),
        "planning_prompt_version": planning_prompt_version,
        "starts_with_json_fence": stripped.startswith("```json") or stripped.startswith("```"),
        "contains_json_fence": "```json" in lowered or "```" in raw_text,
        "starts_with_json_object": stripped.startswith("{"),
        "has_text_before_first_json_object": first_json_index > first_non_space_index,
        "contains_nested_content_property": '"content"' in lowered or "'content'" in lowered,
        "looks_like_serialized_error": lowered.startswith("error") or '"error"' in lowered[:200],
        "first_chars": _clip(redacted, preview_limit, from_end=False),
        "last_chars": _clip(redacted, preview_limit, from_end=True),
        "safe_repr": _clip(repr(redacted), preview_limit * 2, from_end=False),
    }


def _plan_diagnostic(result) -> dict:
    """Return parse and validation metadata without raw prompt or execution."""
    model_result = getattr(result, "model_result", None)
    parsed_plan = getattr(model_result, "plan", None) if model_result is not None else None
    visible_plan = getattr(result, "plan", None) or parsed_plan
    validation_result = getattr(result, "validation_result", None)

    proposed_tools: list[str] = []
    if visible_plan is not None:
        for step in visible_plan.ordered_steps:
            if step.tool is not None and step.tool not in proposed_tools:
                proposed_tools.append(step.tool)

    return {
        "parse_success": bool(getattr(model_result, "success", False)) if model_result is not None else False,
        "parsed_status": getattr(model_result, "status", None) if model_result is not None else None,
        "parsed_step_count": len(parsed_plan.ordered_steps) if parsed_plan is not None else 0,
        "proposed_tools": proposed_tools,
        "parser_error_code": getattr(model_result, "error_code", None) if model_result is not None else None,
        "validation_is_valid": (
            validation_result.is_valid
            if validation_result is not None
            else None
        ),
        "validation_errors": (
            validation_result.errors
            if validation_result is not None
            else []
        ),
        "locally_recalculated_confirmation": (
            visible_plan.requires_confirmation
            if visible_plan is not None
            else None
        ),
        "source": getattr(result, "source", None),
        "executed_tools": 0,
    }


def _provider_performance_diagnostic(result) -> dict:
    """Return provider prompt-size and timing metadata when available."""
    model_result = getattr(result, "model_result", None)
    return {
        "prompt_system_chars": getattr(model_result, "prompt_system_chars", None) if model_result is not None else None,
        "prompt_user_chars": getattr(model_result, "prompt_user_chars", None) if model_result is not None else None,
        "prompt_total_chars": getattr(model_result, "prompt_total_chars", None) if model_result is not None else None,
        "prompt_approx_tokens": getattr(model_result, "prompt_approx_tokens", None) if model_result is not None else None,
        "prompt_build_ms": getattr(model_result, "prompt_build_ms", None) if model_result is not None else None,
        "ollama_response_ms": getattr(model_result, "ollama_response_ms", None) if model_result is not None else None,
        "catalog_total_tools": getattr(model_result, "catalog_total_tools", None) if model_result is not None else None,
        "catalog_sent_tools": getattr(model_result, "catalog_sent_tools", None) if model_result is not None else None,
        "catalog_token_reduction": getattr(model_result, "catalog_token_reduction", None) if model_result is not None else None,
    }


def _print_provider_performance_diagnostic(payload: dict) -> None:
    print(
        "PROVIDER_PERFORMANCE "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


class _StreamingPlanProvider:
    """Script-only adapter that opts into streaming without changing the planner."""

    def __init__(
        self,
        provider: Any,
        *,
        on_progress=None,
    ) -> None:
        self._provider = provider
        self._on_progress = on_progress

    def generate_plan(
        self,
        objective: str,
        catalog_json: str,
    ):
        return self._provider.generate_plan_streaming(
            objective,
            catalog_json,
            on_progress=self._on_progress,
        )


def _progress_printer():
    state = {
        "first_token_printed": False,
        "receiving_printed": False,
    }

    def print_progress(progress: StructuredPlanningProgress) -> None:
        if progress.phase == "preparing":
            line = "[planning] preparando contexto"
        elif progress.phase == "waiting_model":
            line = "[planning] esperando modelo"
        elif progress.phase == "receiving" and not state["first_token_printed"]:
            state["first_token_printed"] = True
            line = f"[planning] primer token recibido en {progress.elapsed_ms / 1000:.1f} s"
        elif progress.phase == "receiving" and not state["receiving_printed"]:
            state["receiving_printed"] = True
            line = "[planning] recibiendo respuesta..."
        elif progress.phase == "completed":
            line = f"[planning] respuesta completa en {progress.elapsed_ms / 1000:.1f} s"
        elif progress.phase == "failed":
            line = f"[planning] fallo: {progress.message}"
        else:
            return
        print(line, file=sys.stderr, flush=True)

    return print_progress


def _clip(text: str, limit: int, *, from_end: bool) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return ("..." + text[-limit:]) if from_end else (text[:limit] + "...")


def _redact_sensitive_text(text: str) -> str:
    redacted = re.sub(
        r"(?i)(api[_-]?key|token|secret|password|credential)(\s*[:=]\s*)(['\"]?)[^'\"\s,}]+",
        r"\1\2\3[REDACTED]",
        text,
    )
    return re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+",
        r"\1[REDACTED]",
        redacted,
    )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from types import SimpleNamespace

from scripts.test_structured_planning import (
    _StreamingPlanProvider,
    _plan_diagnostic,
    _provider_performance_diagnostic,
    _progress_printer,
    _raw_response_diagnostic,
)


def _result(raw_response, *, provider_name: str = "fake", model_name: str = "fake-model"):
    model_result = SimpleNamespace(
        raw_response=raw_response,
        provider_name=provider_name,
        model_name=model_name,
        message_count=2,
        message_roles=("system", "user"),
        planning_prompt_version="structured-planning-v1",
        prompt_system_chars=100,
        prompt_user_chars=900,
        prompt_total_chars=1000,
        prompt_approx_tokens=250,
        prompt_build_ms=3,
        ollama_response_ms=456,
        catalog_total_tools=40,
        catalog_sent_tools=5,
        catalog_token_reduction=7000,
    )
    return SimpleNamespace(raw_response=raw_response, model_result=model_result)


def _planning_result(*, model_result=None, validation_result=None, plan=None):
    return SimpleNamespace(
        source="model",
        model_result=model_result,
        validation_result=validation_result,
        plan=plan,
    )


def test_raw_response_diagnostic_reports_absent_response() -> None:
    diagnostic = _raw_response_diagnostic(_result(None))

    assert diagnostic["has_response"] is False
    assert diagnostic["length"] == 0
    assert diagnostic["provider_name"] == "fake"
    assert diagnostic["model_name"] == "fake-model"
    assert diagnostic["message_count"] == 2
    assert diagnostic["message_roles"] == ["system", "user"]
    assert diagnostic["planning_prompt_version"] == "structured-planning-v1"


def test_raw_response_diagnostic_detects_empty_string() -> None:
    diagnostic = _raw_response_diagnostic(_result(""))

    assert diagnostic["has_response"] is True
    assert diagnostic["is_string"] is True
    assert diagnostic["length"] == 0
    assert diagnostic["first_chars"] == ""


def test_raw_response_diagnostic_handles_non_string_response() -> None:
    diagnostic = _raw_response_diagnostic(_result({"message": {"content": "{}"}}))

    assert diagnostic["has_response"] is True
    assert diagnostic["is_string"] is False
    assert diagnostic["contains_nested_content_property"] is True


def test_raw_response_diagnostic_detects_markdown_json_fence() -> None:
    diagnostic = _raw_response_diagnostic(_result("```json\n{\"status\":\"plan\"}\n```"))

    assert diagnostic["starts_with_json_fence"] is True
    assert diagnostic["contains_json_fence"] is True
    assert diagnostic["starts_with_json_object"] is False


def test_raw_response_diagnostic_detects_reasoning_before_json() -> None:
    diagnostic = _raw_response_diagnostic(_result("Voy a crear el plan:\n{\"status\":\"plan\"}"))

    assert diagnostic["has_text_before_first_json_object"] is True
    assert diagnostic["starts_with_json_object"] is False


def test_raw_response_diagnostic_detects_nested_content_property() -> None:
    diagnostic = _raw_response_diagnostic(_result('{"message":{"content":"{\\"status\\":\\"plan\\"}"}}'))

    assert diagnostic["contains_nested_content_property"] is True
    assert diagnostic["starts_with_json_object"] is True


def test_raw_response_diagnostic_detects_serialized_error_text() -> None:
    diagnostic = _raw_response_diagnostic(_result('{"error":"model failed"}'))

    assert diagnostic["looks_like_serialized_error"] is True


def test_raw_response_diagnostic_limits_and_redacts_output() -> None:
    raw = "token=abc123 " + ("x" * 50) + " password:supersecret"

    diagnostic = _raw_response_diagnostic(_result(raw), preview_chars=10)

    assert len(diagnostic["first_chars"]) <= 13
    assert len(diagnostic["last_chars"]) <= 13
    assert "abc123" not in diagnostic["safe_repr"]
    assert "supersecret" not in diagnostic["safe_repr"]


def test_plan_diagnostic_reports_parse_and_validation_metadata() -> None:
    step = SimpleNamespace(tool="write_file")
    plan = SimpleNamespace(
        ordered_steps=(step,),
        requires_confirmation=True,
    )
    model_result = SimpleNamespace(
        success=True,
        status="plan",
        plan=plan,
        error_code=None,
    )
    validation_result = SimpleNamespace(
        is_valid=True,
        errors=[],
    )

    diagnostic = _plan_diagnostic(
        _planning_result(
            model_result=model_result,
            validation_result=validation_result,
            plan=plan,
        )
    )

    assert diagnostic == {
        "parse_success": True,
        "parsed_status": "plan",
        "parsed_step_count": 1,
        "proposed_tools": ["write_file"],
        "parser_error_code": None,
        "validation_is_valid": True,
        "validation_errors": [],
        "locally_recalculated_confirmation": True,
        "source": "model",
        "executed_tools": 0,
    }


def test_plan_diagnostic_reports_parser_failure_without_plan() -> None:
    model_result = SimpleNamespace(
        success=False,
        status=None,
        plan=None,
        error_code="MODEL_PLAN_PARSE_ERROR",
    )

    diagnostic = _plan_diagnostic(_planning_result(model_result=model_result))

    assert diagnostic["parse_success"] is False
    assert diagnostic["parsed_status"] is None
    assert diagnostic["parsed_step_count"] == 0
    assert diagnostic["proposed_tools"] == []
    assert diagnostic["parser_error_code"] == "MODEL_PLAN_PARSE_ERROR"
    assert diagnostic["validation_is_valid"] is None
    assert diagnostic["locally_recalculated_confirmation"] is None
    assert diagnostic["executed_tools"] == 0


def test_provider_performance_diagnostic_reports_prompt_sizes_and_timings() -> None:
    diagnostic = _provider_performance_diagnostic(_result("{}"))

    assert diagnostic == {
        "prompt_system_chars": 100,
        "prompt_user_chars": 900,
        "prompt_total_chars": 1000,
        "prompt_approx_tokens": 250,
        "prompt_build_ms": 3,
        "ollama_response_ms": 456,
        "catalog_total_tools": 40,
        "catalog_sent_tools": 5,
        "catalog_token_reduction": 7000,
    }


def test_streaming_script_adapter_uses_streaming_provider_path() -> None:
    calls = []

    class Provider:
        def generate_plan_streaming(self, objective, catalog_json, *, on_progress=None):
            calls.append((objective, catalog_json, on_progress))
            return "ok"

    def on_progress(_progress):
        return None

    adapter = _StreamingPlanProvider(Provider(), on_progress=on_progress)

    assert adapter.generate_plan("lee", "{}") == "ok"
    assert calls == [("lee", "{}", on_progress)]


def test_progress_printer_emits_safe_metadata_without_partial_json(capsys) -> None:
    printer = _progress_printer()
    progress = SimpleNamespace(
        phase="receiving",
        elapsed_ms=800,
        received_chars=20,
        chunk_count=2,
        first_token_received=True,
        message='{"status":"partial"}',
    )

    printer(progress)

    captured = capsys.readouterr()
    assert "primer token recibido" in captured.err
    assert "status" not in captured.err

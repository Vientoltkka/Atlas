from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agents.registry import AgentRegistry
from core.orchestrator import AtlasOrchestrator
from core.planner import Plan
from core.request_gateway import (
    AtlasRequest,
    EmptyRequestContentError,
    InvalidRequestAttachmentError,
    InvalidRequestExecutionContextError,
    InvalidRequestMetadataError,
    RequestAttachment,
    RequestExecutionContext,
    RequestGateway,
    RequestGatewayLimits,
    RequestSafetyContext,
    RequestSource,
)
from core.router import Router


NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _gateway(*, request_id: str = "request-1", limits: RequestGatewayLimits | None = None):
    return RequestGateway(clock=lambda: NOW, id_generator=lambda: request_id, limits=limits)


def test_text_request_is_created_and_normalized() -> None:
    request = _gateway().from_text("  Hola\r\nmundo  ")

    assert request.request_id == "request-1"
    assert request.content == "Hola\nmundo"
    assert request.raw_content == "  Hola\r\nmundo  "
    assert request.source is RequestSource.TEXT
    assert request.created_at == NOW
    assert request.locale == "es-ES"
    assert request.attachments == ()


def test_empty_text_is_rejected() -> None:
    with pytest.raises(EmptyRequestContentError):
        _gateway().from_text(" \n\t ")


def test_voice_request_preserves_voice_metadata() -> None:
    request = _gateway().from_voice(
        " Qué hora es ",
        confidence=0.91,
        language="es-ES",
        audio_device="mic-1",
        wake_word_detected=True,
    )

    assert request.source is RequestSource.VOICE
    assert request.content == "Qué hora es"
    assert request.metadata["confidence"] == 0.91
    assert request.metadata["audio_device"] == "mic-1"
    assert request.metadata["wake_word_detected"] is True


def test_system_request_does_not_elevate_privileges() -> None:
    request = _gateway().from_system("internal check")

    assert request.source is RequestSource.SYSTEM
    assert request.safety_context.trusted_source is False
    assert request.safety_context.allow_side_effects is False
    assert request.safety_context.allow_external_calls is False
    assert request.safety_context.user_present is False


def test_resume_request_transports_session_context() -> None:
    request = _gateway().from_resume(
        "session-1",
        confirmation_response=True,
        recovery_authorization=True,
        content="reanuda",
    )

    assert request.source is RequestSource.RESUME
    assert request.execution_context.session_id == "session-1"
    assert request.execution_context.dry_run is None
    assert request.execution_context.confirmation_response is True


def test_explicit_request_id_is_preserved_and_timestamp_is_aware() -> None:
    request = _gateway(request_id="generated").from_text(
        "hola",
        request_id="explicit-1",
        conversation_id="conversation-1",
        correlation_id="correlation-1",
    )

    assert request.request_id == "explicit-1"
    assert request.created_at.tzinfo is not None
    assert request.conversation_id == "conversation-1"
    assert request.correlation_id == "correlation-1"


def test_attachment_validation_and_immutability() -> None:
    attachment = RequestAttachment(
        attachment_id="attachment-1",
        name="note.txt",
        media_type="text/plain",
        size_bytes=10,
        local_reference="local-ref",
    )
    request = _gateway().from_text("hola", attachments=(attachment,))

    assert request.attachments == (attachment,)
    with pytest.raises(AttributeError):
        request.attachments[0].name = "other"  # type: ignore[misc]
    with pytest.raises(InvalidRequestAttachmentError):
        RequestAttachment("bad", "x", "text/plain", size_bytes=-1)
    with pytest.raises(InvalidRequestAttachmentError):
        RequestAttachment("bad2", "x", "text/plain", local_reference="a", external_reference="b")
    with pytest.raises(InvalidRequestAttachmentError):
        _gateway().from_text("hola", attachments=(attachment, attachment))


def test_metadata_is_json_safe_limited_and_immutable() -> None:
    request = _gateway().from_text("hola", metadata={"b": [1, True], "a": {"x": "y"}})

    assert request.metadata["a"]["x"] == "y"
    with pytest.raises(TypeError):
        request.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(InvalidRequestMetadataError):
        _gateway().from_text("hola", metadata={"callback": object()})
    with pytest.raises(InvalidRequestMetadataError):
        _gateway(limits=RequestGatewayLimits(max_metadata_size=4)).from_text(
            "hola",
            metadata={"safe": "too-large"},
        )
    with pytest.raises(InvalidRequestMetadataError):
        _gateway().from_text("hola", metadata={"api_token": "secret"})


def test_limits_reject_large_content_and_too_many_attachments() -> None:
    gateway = _gateway(limits=RequestGatewayLimits(max_content_length=4, max_attachments=1))
    attachment = RequestAttachment("a1", "a.txt", "text/plain")

    with pytest.raises(ValueError):
        gateway.from_text("12345")
    with pytest.raises(InvalidRequestAttachmentError):
        gateway.from_text(
            "hola",
            attachments=(attachment, RequestAttachment("a2", "b.txt", "text/plain")),
        )


def test_execution_context_validation_and_dry_run_transport() -> None:
    context = RequestExecutionContext(session_id="session-1", dry_run=True, requested_budget=1.5)
    request = _gateway().from_text("hola", execution_context=context)

    assert request.execution_context.dry_run is True
    assert request.execution_context.requested_budget == 1.5
    with pytest.raises(InvalidRequestExecutionContextError):
        RequestExecutionContext(requested_timeout=-1)
    with pytest.raises(InvalidRequestExecutionContextError):
        RequestExecutionContext(requested_budget=-1)
    with pytest.raises(InvalidRequestExecutionContextError):
        RequestExecutionContext(confirmation_response="yes")  # type: ignore[arg-type]


def test_safety_defaults_are_conservative() -> None:
    request = _gateway().from_text("hola")

    assert request.safety_context == RequestSafetyContext()
    assert request.safety_context.trusted_source is False
    assert request.safety_context.allow_side_effects is False


def test_determinism_with_injected_clock_and_id_generator() -> None:
    first = _gateway().from_text("hola", metadata={"z": 1, "a": 2})
    second = _gateway().from_text("hola", metadata={"a": 2, "z": 1})

    assert first == second


def test_router_route_request_accepts_atlas_request_and_route_still_works() -> None:
    request = _gateway().from_text("analiza router.py")
    router = Router()

    assert router.route_request(request, plan=Plan("project", request.content)) == "project"
    assert router.route(Plan("chat", "hola")) == "chat"


def test_gateway_dispatch_delegates_explicitly_to_router() -> None:
    class FakeRouter:
        def __init__(self) -> None:
            self.requests: list[AtlasRequest] = []

        def route_request(self, request: AtlasRequest):
            self.requests.append(request)
            return "chat"

    router = FakeRouter()
    gateway = RequestGateway(clock=lambda: NOW, id_generator=lambda: "request-1", router=router)
    request = gateway.from_text("hola")

    assert gateway.dispatch(request) == "chat"
    assert router.requests == [request]
    assert gateway.events[-1].event_type == "request_dispatched"


def test_events_do_not_include_full_content() -> None:
    gateway = _gateway()
    gateway.from_text("texto secreto completo")

    assert [event.event_type for event in gateway.events] == [
        "request_received",
        "request_normalized",
        "request_created",
    ]
    assert all(not hasattr(event, "content") for event in gateway.events)
    assert gateway.events[0].content_length == len("texto secreto completo")


def test_orchestrator_text_uses_gateway_before_router_without_visible_change() -> None:
    calls: list[AtlasRequest] = []

    class RequestAwareRouter(Router):
        def route_request(self, request: AtlasRequest, *, plan=None):
            calls.append(request)
            return super().route_request(request, plan=plan)

    agent = _ChatAgent()
    registry = AgentRegistry()
    registry.register(agent)
    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda text: Plan("chat", text)),
        router=RequestAwareRouter(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=SimpleNamespace(
            events=[],
            add_user=lambda prompt: None,
            add_assistant=lambda response: None,
            history=lambda: [{"role": "user", "content": "hola"}],
        ),
        registry=registry,
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        request_gateway=_gateway(),
    )

    assert orchestrator.process_prompt("  hola  ", confirm=lambda _prompt: "") == "ok"
    assert calls[0].content == "hola"


def test_orchestrator_voice_uses_gateway_and_empty_transcription_stops_before_router() -> None:
    class FailingRouter(Router):
        def route_voice_command(self, _prompt):  # pragma: no cover
            raise AssertionError("router must not receive empty transcription")

    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda text: Plan("chat", text)),
        router=FailingRouter(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=SimpleNamespace(add_user=lambda _prompt: None, add_assistant=lambda _response: None, history=list),
        registry=AgentRegistry(),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        request_gateway=_gateway(),
    )

    with pytest.raises(EmptyRequestContentError):
        orchestrator.process_voice_prompt("   ", confirm=lambda _prompt: "")


def test_gateway_does_not_plan_execute_or_call_autonomous_components() -> None:
    gateway = _gateway()
    request = gateway.from_text("haz algo")

    assert request.content == "haz algo"
    assert not hasattr(gateway, "planner")
    assert not hasattr(gateway, "executor")
    assert not hasattr(gateway, "autonomous_execution_orchestrator")


class _ChatAgent:
    name = "chat"
    generated_path = None
    generated_content = None

    def run(self, *, model, messages):
        del model, messages
        return "ok"

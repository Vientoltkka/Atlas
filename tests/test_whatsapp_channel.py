"""Tests for the WhatsApp channel adapter (Atlas V4.0 Phase 1)."""

from __future__ import annotations

import pytest

from channels.base_channel import InvalidChannelMessageError
from channels.whatsapp_channel import WhatsAppChannel
from core.agent_executor import (
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentExecutionStatus,
)
from core.agent_resolver import AgentResolutionRequest


@pytest.fixture()
def channel() -> WhatsAppChannel:
    return WhatsAppChannel()


def _valid_message() -> dict:
    return {"id": "wamid.abc123", "from": "34600111222", "text": "Hola Atlas"}


def test_channel_name(channel: WhatsAppChannel) -> None:
    assert channel.name == "whatsapp"


def test_valid_message_becomes_agent_execution_request(channel: WhatsAppChannel) -> None:
    request = channel.parse_inbound(_valid_message())
    assert isinstance(request, AgentExecutionRequest)
    assert isinstance(request.resolution_request, AgentResolutionRequest)
    assert request.user_input == "Hola Atlas"


def test_sender_id_extracted(channel: WhatsAppChannel) -> None:
    request = channel.parse_inbound(_valid_message())
    assert request.session_id == "whatsapp-34600111222"
    assert request.metadata["sender_id"] == "34600111222"


def test_text_extracted_and_stripped(channel: WhatsAppChannel) -> None:
    message = _valid_message()
    message["text"] = "  Hola Atlas  "
    request = channel.parse_inbound(message)
    assert request.user_input == "Hola Atlas"


def test_correlation_id_from_message_id(channel: WhatsAppChannel) -> None:
    request = channel.parse_inbound(_valid_message())
    assert request.correlation_id == "wamid.abc123"


@pytest.mark.parametrize(
    "message",
    [
        "not-a-mapping",
        {},
        {"text": "Hola"},
        {"from": "34600111222"},
        {"from": "", "text": "Hola"},
        {"from": "34600111222", "text": ""},
        {"from": "34600111222", "text": 123},
        {"from": "34600111222", "text": "x" * 5000},
    ],
)
def test_invalid_messages_raise_controlled_error(
    channel: WhatsAppChannel, message: object
) -> None:
    with pytest.raises(InvalidChannelMessageError):
        channel.parse_inbound(message)  # type: ignore[arg-type]


def test_completed_result_formats_outbound(channel: WhatsAppChannel) -> None:
    result = AgentExecutionResult(
        status=AgentExecutionStatus.COMPLETED,
        request_signature="sig",
        correlation_id="wamid.abc123",
        output={"text": "Respuesta de Atlas"},
    )
    payload = channel.format_outbound(result)
    assert payload == {
        "channel": "whatsapp",
        "status": "COMPLETED",
        "correlation_id": "wamid.abc123",
        "body": "Respuesta de Atlas",
    }


def test_failed_result_uses_safe_message(channel: WhatsAppChannel) -> None:
    result = AgentExecutionResult(
        status=AgentExecutionStatus.NO_AGENT_CANDIDATES,
        request_signature="sig",
        safe_message="No hay agente disponible.",
    )
    payload = channel.format_outbound(result)
    assert payload["status"] == "NO_AGENT_CANDIDATES"
    assert payload["body"] == "No hay agente disponible."


def test_format_outbound_rejects_foreign_contract(channel: WhatsAppChannel) -> None:
    with pytest.raises(InvalidChannelMessageError):
        channel.format_outbound({"status": "COMPLETED"})  # type: ignore[arg-type]


def test_channel_does_not_import_legacy_agent_registry() -> None:
    import sys

    from agents import registry as legacy_registry

    legacy_module_name = legacy_registry.__name__
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("channels"):
            continue
        source = getattr(module, "__dict__", {})
        assert all(
            value is not legacy_registry.AgentRegistry
            for value in source.values()
            if isinstance(value, type)
        ), f"{module_name} importa agents.registry"
    assert legacy_module_name == "agents.registry"

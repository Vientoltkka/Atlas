"""WhatsApp channel adapter for Atlas (Phase 1: simulated inbound messages)."""

from __future__ import annotations

from typing import Any, Mapping

from channels.base_channel import BaseChannel, InvalidChannelMessageError
from core.agent_executor import AgentExecutionRequest, AgentExecutionResult, AgentExecutionStatus
from core.agent_resolver import AgentResolutionRequest


MAX_WHATSAPP_TEXT_LENGTH = 4_096


class WhatsAppChannel(BaseChannel):
    """Translates WhatsApp messages to AgentExecutionRequest contracts.

    Phase 1 accepts simulated WhatsApp message payloads shaped like the
    WhatsApp Cloud API webhook value structure. No HTTP transport, no
    credentials and no direct agent execution happen here.
    """

    @property
    def name(self) -> str:
        return "whatsapp"

    def parse_inbound(self, message: Mapping[str, Any]) -> AgentExecutionRequest:
        if not isinstance(message, Mapping):
            raise InvalidChannelMessageError("whatsapp message must be a mapping.")
        sender_id = self.extract_sender_id(message)
        text = self.extract_text(message)
        metadata = {
            "channel": self.name,
            "sender_id": sender_id,
            "message_id": _optional_str(message.get("id")),
        }
        return AgentExecutionRequest(
            resolution_request=AgentResolutionRequest(),
            user_input=text,
            session_id=f"whatsapp-{sender_id}",
            correlation_id=_optional_str(message.get("id")),
            metadata=metadata,
        )

    def format_outbound(self, result: AgentExecutionResult) -> Mapping[str, Any]:
        if not isinstance(result, AgentExecutionResult):
            raise InvalidChannelMessageError("outbound payload must be AgentExecutionResult.")
        if result.status is AgentExecutionStatus.COMPLETED:
            body = _extract_output_text(result.output)
        else:
            body = result.safe_message or f"Atlas no pudo procesar el mensaje ({result.status.value})."
        return {
            "channel": self.name,
            "status": result.status.value,
            "correlation_id": result.correlation_id,
            "body": body,
        }

    def extract_sender_id(self, message: Mapping[str, Any]) -> str:
        sender = message.get("from")
        if not isinstance(sender, str) or not sender.strip():
            raise InvalidChannelMessageError("whatsapp message is missing a valid 'from' sender.")
        return sender.strip()

    def extract_text(self, message: Mapping[str, Any]) -> str:
        text = message.get("text")
        if not isinstance(text, str):
            raise InvalidChannelMessageError("whatsapp message is missing string 'text'.")
        text = text.strip()
        if not text:
            raise InvalidChannelMessageError("whatsapp message text must not be empty.")
        if len(text) > MAX_WHATSAPP_TEXT_LENGTH:
            raise InvalidChannelMessageError("whatsapp message text exceeds maximum length.")
        return text


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _extract_output_text(output: Mapping[str, object] | None) -> str:
    if not output:
        return ""
    for key in ("text", "message", "response", "output"):
        value = output.get(key)
        if isinstance(value, str):
            return value
    return ""

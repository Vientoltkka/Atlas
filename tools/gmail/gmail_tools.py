"""Atlas Gmail tools: bounded list, read and confirmed send operations."""

from __future__ import annotations

from tools.base_tool import BaseTool
from tools.gmail.gmail_service import (
    GmailConfigurationError,
    GmailOperationError,
    GmailService,
)
from tools.tool_context import ToolContext

MAX_RESULTS_LIMIT = 20


class _GmailToolBase(BaseTool):
    def __init__(self, service: GmailService | None = None) -> None:
        self._service = service or GmailService()


class GmailListTool(_GmailToolBase):
    """List the most recent Gmail inbox messages."""

    @property
    def name(self) -> str:
        return "gmail_list"

    @property
    def description(self) -> str:
        return "List recent Gmail inbox messages with sender, subject, date and snippet (read-only)."

    def semantic_metadata(self) -> dict[str, object]:
        return {
            "capabilities": ["gmail_list"],
            "supported_intents": ["list recent email messages"],
            "input_description": "Optional bounded max_results between 1 and 20.",
            "output_description": "Message id, sender, subject, date and snippet.",
            "risk_level": "low",
            "limitations": ["read-only", "maximum 20 messages", "inbox only"],
            "tags": ["gmail", "email", "read"],
            "category": "gmail",
        }

    def execute(self, context: ToolContext) -> dict[str, object]:
        max_results = context.parameters.get("max_results", 5)
        if not isinstance(max_results, int):
            raise ValueError("max_results debe ser un numero entero.")
        messages: list[dict[str, str]] = self._service.list_messages(
            max_results=max_results,
        )
        return {"messages": messages}


class GmailReadTool(_GmailToolBase):
    """Read one concrete Gmail message by id or unambiguous sender."""

    @property
    def name(self) -> str:
        return "gmail_read"

    @property
    def description(self) -> str:
        return "Read one Gmail message by message id or by sender when unambiguous (read-only)."

    def semantic_metadata(self) -> dict[str, object]:
        return {
            "capabilities": ["gmail_read"],
            "supported_intents": ["read one email message"],
            "input_description": "Requires message_id or a sender address/name.",
            "output_description": "Message id, sender, subject, date and body preview.",
            "risk_level": "low",
            "limitations": ["read-only", "single message", "body preview capped"],
            "tags": ["gmail", "email", "read"],
            "category": "gmail",
        }

    def execute(self, context: ToolContext) -> dict[str, str]:
        message_id = context.parameters.get("message_id")
        sender = context.parameters.get("sender")
        if not (isinstance(message_id, str) and message_id.strip()) and not (
            isinstance(sender, str) and sender.strip()
        ):
            raise ValueError(
                "Indica el id del mensaje o el remitente del correo que quieres leer."
            )
        return dict(
            self._service.read_message(
                message_id=message_id if isinstance(message_id, str) else None,
                sender=sender if isinstance(sender, str) else None,
            )
        )


class GmailSendTool(_GmailToolBase):
    """Send one plain-text email; always requires explicit confirmation."""

    @property
    def name(self) -> str:
        return "gmail_send"

    @property
    def description(self) -> str:
        return "Send one plain-text email to a recipient with subject and body. Requires confirmation (sends email)."

    @property
    def requires_confirmation(self) -> bool:
        return True

    @property
    def required_permissions(self) -> tuple[str, ...]:
        return ("email.send",)

    def semantic_metadata(self) -> dict[str, object]:
        return {
            "capabilities": ["gmail_send"],
            "supported_intents": ["send one email"],
            "input_description": "Requires to, subject and body.",
            "output_description": "Sent recipient and subject.",
            "risk_level": "high",
            "limitations": ["requires confirmation", "plain text only", "one recipient"],
            "tags": ["gmail", "email", "send"],
            "category": "gmail",
        }

    def execute(self, context: ToolContext) -> dict[str, str]:
        to = context.parameters.get("to")
        subject = context.parameters.get("subject")
        body = context.parameters.get("body")
        if not isinstance(to, str) or not to.strip():
            raise ValueError("Missing parameter 'to'.")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("Missing parameter 'subject'.")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("Missing parameter 'body'.")
        return dict(
            self._service.send_message(
                to=to,
                subject=subject,
                body=body,
            )
        )

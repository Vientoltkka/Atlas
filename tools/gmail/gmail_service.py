"""Minimal Gmail adapter over IMAP/SMTP using app-password credentials.

Credentials are read exclusively from the environment:

- ``ATLAS_GMAIL_ADDRESS``: full Gmail address.
- ``ATLAS_GMAIL_APP_PASSWORD``: Google app password (16 characters).

Atlas never writes these values and never performs OAuth flows. When the
credentials are absent, every operation fails with a controlled
``GmailConfigurationError`` carrying the exact setup instructions.
"""

from __future__ import annotations

from email import policy
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.parser import BytesParser
import email.utils
import imaplib
import os
import re
import smtplib

IMAP_HOST = "imap.gmail.com"
SMTP_HOST = "smtp.gmail.com"
IMAP_PORT = 993
SMTP_PORT = 465
SNIPPET_CHARACTER_LIMIT = 200
BODY_CHARACTER_LIMIT = 4000
MAX_RESULTS_LIMIT = 20

GMAIL_SETUP_INSTRUCTIONS = (
    "Gmail no esta configurado. Define en el entorno ATLAS_GMAIL_ADDRESS y "
    "ATLAS_GMAIL_APP_PASSWORD (contrasena de aplicacion de Google, generada en "
    "https://myaccount.google.com/apppasswords). Atlas no guarda ni modifica "
    "estas credenciales."
)


class GmailConfigurationError(RuntimeError):
    """Gmail credentials are missing or incomplete."""


class GmailOperationError(RuntimeError):
    """A Gmail IMAP/SMTP operation failed in a controlled way."""


def require_gmail_max_results(value: object) -> object:
    """Argument validator bounding the Gmail listing size."""
    if type(value) is not int or not 1 <= value <= MAX_RESULTS_LIMIT:
        raise ValueError(
            f"max_results must be an integer between 1 and {MAX_RESULTS_LIMIT}."
        )
    return value


def require_gmail_message_id(value: object) -> object:
    """Argument validator for a Gmail message id value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("message_id must be a non-empty string.")
    return value


def require_gmail_sender(value: object) -> object:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("sender must be a non-empty string.")
    return value


def require_email_address(value: object) -> object:
    if not isinstance(value, str) or not re.fullmatch(
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        value.strip(),
    ):
        raise ValueError("to must be a valid email address.")
    return value


def require_gmail_subject(value: object) -> object:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("subject must be a non-empty string.")
    return value


def require_gmail_body(value: object) -> object:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("body must be a non-empty string.")
    return value


def gmail_credentials_missing_message() -> str:
    return GMAIL_SETUP_INSTRUCTIONS


class GmailService:
    """Bounded Gmail read/list/send operations through IMAP and SMTP."""

    def list_messages(self, max_results: int = 5) -> list[dict[str, str]]:
        """Return the most recent inbox messages without marking them read."""
        address, password = _credentials()
        connection = _connect_imap(address, password)
        try:
            status, _data = connection.select("INBOX", readonly=True)
            if status != "OK":
                raise GmailOperationError("No se pudo abrir la bandeja de entrada.")
            status, data = connection.uid("search", None, "ALL")
            if status != "OK":
                raise GmailOperationError("No se pudo listar los correos.")
            identifiers = (data[0] or b"").split()
            messages: list[dict[str, str]] = []
            for uid in reversed(identifiers[-max(1, max_results) :]):
                messages.append(_fetch_summary(connection, uid.decode("ascii")))
            return messages
        finally:
            _close_imap(connection)

    def read_message(self, message_id: str | None = None, sender: str | None = None) -> dict[str, str]:
        """Read one concrete message by UID, or by sender when unambiguous."""
        address, password = _credentials()
        connection = _connect_imap(address, password)
        try:
            status, _data = connection.select("INBOX", readonly=True)
            if status != "OK":
                raise GmailOperationError("No se pudo abrir la bandeja de entrada.")
            uid: str | None = None
            if message_id and message_id.strip():
                uid = message_id.strip()
            elif sender and sender.strip():
                status, data = connection.uid(
                    "search",
                    None,
                    f'(FROM "{_escape_imap_query(sender.strip())}")',
                )
                if status != "OK":
                    raise GmailOperationError("No se pudo buscar por remitente.")
                identifiers = (data[0] or b"").split()
                if not identifiers:
                    raise GmailOperationError(
                        f"No hay correos del remitente '{sender.strip()}'."
                    )
                if len(identifiers) > 1:
                    raise GmailOperationError(
                        "Hay varios correos de ese remitente; indica el id del "
                        "mensaje concreto (gmail_list muestra los ids)."
                    )
                uid = identifiers[-1].decode("ascii")
            if not uid:
                raise GmailOperationError(
                    "Indica el id del mensaje o el remitente del correo."
                )
            return _fetch_full_message(connection, uid)
        finally:
            _close_imap(connection)

    def send_message(self, to: str, subject: str, body: str) -> dict[str, str]:
        """Send one plain-text email through SMTP over TLS."""
        address, password = _credentials()
        message = EmailMessage()
        message["From"] = address
        message["To"] = to.strip()
        message["Subject"] = subject.strip()
        message.set_content(body.strip())
        try:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server:
                server.login(address, password)
                server.send_message(message)
        except smtplib.SMTPAuthenticationError as error:
            raise GmailOperationError(
                "Gmail rechazo la autenticacion. Revisa ATLAS_GMAIL_APP_PASSWORD."
            ) from error
        except (smtplib.SMTPException, OSError) as error:
            raise GmailOperationError(f"No se pudo enviar el email: {error}") from error
        return {"sent_to": to.strip(), "subject": subject.strip()}


def _credentials() -> tuple[str, str]:
    address = os.environ.get("ATLAS_GMAIL_ADDRESS", "").strip()
    password = os.environ.get("ATLAS_GMAIL_APP_PASSWORD", "").strip()
    if not address or not password:
        raise GmailConfigurationError(GMAIL_SETUP_INSTRUCTIONS)
    return address, password


def _connect_imap(address: str, password: str) -> imaplib.IMAP4_SSL:
    try:
        connection = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        connection.login(address, password)
        return connection
    except imaplib.IMAP4.error as error:
        raise GmailOperationError(
            "Gmail rechazo la autenticacion IMAP. Revisa ATLAS_GMAIL_APP_PASSWORD."
        ) from error
    except OSError as error:
        raise GmailOperationError(
            "No se pudo conectar con Gmail (imap.gmail.com)."
        ) from error


def _close_imap(connection: imaplib.IMAP4_SSL) -> None:
    try:
        connection.logout()
    except Exception:
        pass


def _escape_imap_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', "'")


def _decode_mime_header(value: object) -> str:
    if value is None:
        return ""
    try:
        return str(make_header(decode_header(str(value))))
    except Exception:
        return str(value)


def _extract_body(message) -> str:
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_content()
                if isinstance(payload, str):
                    return payload
        return ""
    payload = message.get_content()
    return payload if isinstance(payload, str) else ""


def _body_preview(body: str, limit: int) -> str:
    collapsed = " ".join(body.split())
    return collapsed[:limit]


def _fetch_summary(connection: imaplib.IMAP4_SSL, uid: str) -> dict[str, str]:
    full = _fetch_full_message(connection, uid)
    return {
        "id": full["id"],
        "from": full["from"],
        "subject": full["subject"],
        "date": full["date"],
        "snippet": full["body"][:SNIPPET_CHARACTER_LIMIT],
    }


def _fetch_full_message(connection: imaplib.IMAP4_SSL, uid: str) -> dict[str, str]:
    status, data = connection.uid("fetch", uid, "(BODY.PEEK[])")
    if status != "OK" or not data or data[0] is None:
        raise GmailOperationError(f"No se pudo leer el correo {uid}.")
    raw = data[0][1]
    message = BytesParser(policy=policy.default).parsebytes(raw)
    return {
        "id": uid,
        "from": _decode_mime_header(message.get("From")),
        "subject": _decode_mime_header(message.get("Subject")),
        "date": _decode_mime_header(message.get("Date")),
        "body": _body_preview(_extract_body(message), BODY_CHARACTER_LIMIT),
    }

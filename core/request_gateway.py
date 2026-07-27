"""Unified, validated request gateway for Atlas inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import re
from types import MappingProxyType
from typing import Any
from uuid import uuid4


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_LOCALE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$")
_SECRET_KEYS = ("token", "secret", "password", "api_key", "apikey", "credential")


class RequestSource(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    SYSTEM = "system"
    RESUME = "resume"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class InvalidAtlasRequestError(ValueError):
    error_code = "INVALID_ATLAS_REQUEST"


class EmptyRequestContentError(InvalidAtlasRequestError):
    error_code = "EMPTY_REQUEST_CONTENT"


class InvalidRequestMetadataError(InvalidAtlasRequestError):
    error_code = "INVALID_REQUEST_METADATA"


class InvalidRequestAttachmentError(InvalidAtlasRequestError):
    error_code = "INVALID_REQUEST_ATTACHMENT"


class InvalidRequestExecutionContextError(InvalidAtlasRequestError):
    error_code = "INVALID_REQUEST_EXECUTION_CONTEXT"


@dataclass(frozen=True, slots=True)
class RequestGatewayLimits:
    max_content_length: int = 20_000
    max_metadata_size: int = 8_192
    max_attachments: int = 16
    max_metadata_depth: int = 6

    def __post_init__(self) -> None:
        for name in (
            "max_content_length",
            "max_metadata_size",
            "max_attachments",
            "max_metadata_depth",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class RequestAttachment:
    attachment_id: str
    name: str
    media_type: str
    size_bytes: int | None = None
    local_reference: str | None = None
    external_reference: str | None = None
    checksum: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.attachment_id, "attachment_id", InvalidRequestAttachmentError)
        if not self.name.strip() or not self.media_type.strip():
            raise InvalidRequestAttachmentError("attachment name and media_type are required.")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise InvalidRequestAttachmentError("attachment size_bytes cannot be negative.")
        if self.local_reference and self.external_reference:
            raise InvalidRequestAttachmentError(
                "attachment cannot contain both local_reference and external_reference."
            )


@dataclass(frozen=True, slots=True)
class RequestExecutionContext:
    session_id: str | None = None
    resume_target: str | None = None
    confirmation_response: bool | None = None
    recovery_authorization: bool | None = None
    dry_run: bool | None = None
    requested_timeout: float | None = None
    requested_budget: float | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "resume_target"):
            value = getattr(self, name)
            if value is not None:
                _validate_id(value, name, InvalidRequestExecutionContextError)
        for name in (
            "confirmation_response",
            "recovery_authorization",
            "dry_run",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise InvalidRequestExecutionContextError(f"{name} must be a bool or None.")
        for name in ("requested_timeout", "requested_budget"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise InvalidRequestExecutionContextError(f"{name} cannot be negative.")


@dataclass(frozen=True, slots=True)
class RequestSafetyContext:
    trusted_source: bool = False
    requires_confirmation_hint: bool = False
    contains_sensitive_data: bool = False
    user_present: bool = True
    allow_side_effects: bool = False
    allow_external_calls: bool = False

    def __post_init__(self) -> None:
        for name in (
            "trusted_source",
            "requires_confirmation_hint",
            "contains_sensitive_data",
            "user_present",
            "allow_side_effects",
            "allow_external_calls",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")


@dataclass(frozen=True, slots=True)
class AtlasRequest:
    request_id: str
    content: str
    source: RequestSource
    created_at: datetime
    user_id: str | None = None
    conversation_id: str | None = None
    correlation_id: str | None = None
    locale: str = "es-ES"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    attachments: tuple[RequestAttachment, ...] = ()
    reply_expected: bool = True
    priority_hint: int = 0
    safety_context: RequestSafetyContext = field(default_factory=RequestSafetyContext)
    execution_context: RequestExecutionContext | None = None
    raw_content: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.request_id, "request_id", InvalidAtlasRequestError)
        if not isinstance(self.source, RequestSource):
            object.__setattr__(self, "source", RequestSource(self.source))
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise InvalidAtlasRequestError("created_at must be timezone-aware.")
        if not _LOCALE_PATTERN.fullmatch(self.locale):
            raise InvalidAtlasRequestError("locale is invalid.")
        for name in ("user_id", "conversation_id", "correlation_id"):
            value = getattr(self, name)
            if value is not None:
                _validate_id(value, name, InvalidAtlasRequestError)
        if not self.content:
            raise EmptyRequestContentError("content cannot be empty.")
        if type(self.reply_expected) is not bool:
            raise InvalidAtlasRequestError("reply_expected must be a bool.")
        if not isinstance(self.priority_hint, int) or isinstance(self.priority_hint, bool):
            raise InvalidAtlasRequestError("priority_hint must be an integer.")
        attachments = tuple(self.attachments)
        ids = [item.attachment_id for item in attachments]
        if len(ids) != len(set(ids)):
            raise InvalidRequestAttachmentError("attachment IDs must be unique.")
        object.__setattr__(self, "attachments", attachments)
        object.__setattr__(self, "metadata", _freeze_json_safe(self.metadata))


@dataclass(frozen=True, slots=True)
class RequestGatewayEvent:
    event_type: str
    request_id: str | None
    source: RequestSource | None
    conversation_id: str | None
    content_length: int
    attachment_count: int
    locale: str | None
    timestamp: datetime
    error_code: str | None = None


class RequestGateway:
    """Create validated AtlasRequest instances without routing or execution."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
        limits: RequestGatewayLimits | None = None,
        router: object | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._id_generator = id_generator or (lambda: str(uuid4()))
        self._limits = limits or RequestGatewayLimits()
        self._router = router
        self._events: list[RequestGatewayEvent] = []

    @property
    def events(self) -> tuple[RequestGatewayEvent, ...]:
        return tuple(self._events)

    def from_text(self, text: str, **kwargs: Any) -> AtlasRequest:
        return TextRequestAdapter(self).create(text, **kwargs)

    def from_voice(
        self,
        transcription: str,
        *,
        confidence: float | None = None,
        language: str | None = None,
        audio_device: str | None = None,
        wake_word_detected: bool | None = None,
        **kwargs: Any,
    ) -> AtlasRequest:
        return VoiceRequestAdapter(self).create(
            transcription,
            confidence=confidence,
            language=language,
            audio_device=audio_device,
            wake_word_detected=wake_word_detected,
            **kwargs,
        )

    def from_system(self, command: str, **kwargs: Any) -> AtlasRequest:
        return SystemRequestAdapter(self).create(command, **kwargs)

    def from_resume(
        self,
        session_id: str,
        *,
        confirmation_response: bool | None = None,
        recovery_authorization: bool | None = None,
        content: str | None = None,
        **kwargs: Any,
    ) -> AtlasRequest:
        return ResumeRequestAdapter(self).create(
            session_id,
            confirmation_response=confirmation_response,
            recovery_authorization=recovery_authorization,
            content=content,
            **kwargs,
        )

    def create_request(
        self,
        content: str,
        *,
        source: RequestSource,
        request_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        correlation_id: str | None = None,
        locale: str = "es-ES",
        metadata: Mapping[str, Any] | None = None,
        attachments: Sequence[RequestAttachment] = (),
        reply_expected: bool = True,
        priority_hint: int = 0,
        safety_context: RequestSafetyContext | None = None,
        execution_context: RequestExecutionContext | None = None,
    ) -> AtlasRequest:
        raw_content = content
        normalized = _normalize_content(content)
        self._record(
            "request_received",
            request_id=request_id,
            source=source,
            conversation_id=conversation_id,
            content_length=len(raw_content),
            attachment_count=len(tuple(attachments)),
            locale=locale,
        )
        try:
            if not normalized:
                raise EmptyRequestContentError("content cannot be empty.")
            if len(normalized) > self._limits.max_content_length:
                raise InvalidAtlasRequestError("content exceeds max_content_length.")
            attachments_tuple = tuple(attachments)
            if len(attachments_tuple) > self._limits.max_attachments:
                raise InvalidRequestAttachmentError("too many attachments.")
            safe_metadata = _freeze_json_safe(
                metadata or {},
                max_depth=self._limits.max_metadata_depth,
            )
            serialized_metadata = json.dumps(
                _thaw_json_safe(safe_metadata),
                sort_keys=True,
                separators=(",", ":"),
            )
            if len(serialized_metadata.encode("utf-8")) > self._limits.max_metadata_size:
                raise InvalidRequestMetadataError("metadata exceeds max_metadata_size.")
            request = AtlasRequest(
                request_id=request_id or self._id_generator(),
                content=normalized,
                source=source,
                created_at=self._clock(),
                user_id=user_id,
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                locale=locale.replace("_", "-"),
                metadata=safe_metadata,
                attachments=attachments_tuple,
                reply_expected=reply_expected,
                priority_hint=priority_hint,
                safety_context=safety_context or RequestSafetyContext(),
                execution_context=execution_context,
                raw_content=raw_content if raw_content != normalized else None,
            )
            self._record_request("request_normalized", request)
            self._record_request("request_created", request)
            return request
        except InvalidAtlasRequestError as error:
            self._record(
                "request_validation_failed",
                request_id=request_id,
                source=source,
                conversation_id=conversation_id,
                content_length=len(raw_content),
                attachment_count=len(tuple(attachments)),
                locale=locale,
                error_code=error.error_code,
            )
            raise

    def dispatch(self, request: AtlasRequest) -> Any:
        if self._router is None:
            raise InvalidAtlasRequestError("router is not configured.")
        route_request = getattr(self._router, "route_request", None)
        if not callable(route_request):
            raise InvalidAtlasRequestError("router does not support route_request.")
        result = route_request(request)
        self._record_request("request_dispatched", request)
        return result

    def _record_request(self, event_type: str, request: AtlasRequest) -> None:
        self._record(
            event_type,
            request_id=request.request_id,
            source=request.source,
            conversation_id=request.conversation_id,
            content_length=len(request.content),
            attachment_count=len(request.attachments),
            locale=request.locale,
        )

    def _record(
        self,
        event_type: str,
        *,
        request_id: str | None,
        source: RequestSource | None,
        conversation_id: str | None,
        content_length: int,
        attachment_count: int,
        locale: str | None,
        error_code: str | None = None,
    ) -> None:
        self._events.append(
            RequestGatewayEvent(
                event_type=event_type,
                request_id=request_id,
                source=source,
                conversation_id=conversation_id,
                content_length=content_length,
                attachment_count=attachment_count,
                locale=locale,
                timestamp=self._clock(),
                error_code=error_code,
            )
        )


class TextRequestAdapter:
    def __init__(self, gateway: RequestGateway) -> None:
        self._gateway = gateway

    def create(self, text: str, **kwargs: Any) -> AtlasRequest:
        return self._gateway.create_request(text, source=RequestSource.TEXT, **kwargs)


class VoiceRequestAdapter:
    def __init__(self, gateway: RequestGateway) -> None:
        self._gateway = gateway

    def create(
        self,
        transcription: str,
        *,
        confidence: float | None = None,
        language: str | None = None,
        audio_device: str | None = None,
        wake_word_detected: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
        locale: str = "es-ES",
        **kwargs: Any,
    ) -> AtlasRequest:
        voice_metadata = dict(metadata or {})
        for key, value in (
            ("confidence", confidence),
            ("language", language),
            ("audio_device", audio_device),
            ("wake_word_detected", wake_word_detected),
        ):
            if value is not None:
                voice_metadata[key] = value
        return self._gateway.create_request(
            transcription,
            source=RequestSource.VOICE,
            metadata=voice_metadata,
            locale=language or locale,
            **kwargs,
        )


class SystemRequestAdapter:
    def __init__(self, gateway: RequestGateway) -> None:
        self._gateway = gateway

    def create(
        self,
        command: str,
        *,
        safety_context: RequestSafetyContext | None = None,
        **kwargs: Any,
    ) -> AtlasRequest:
        return self._gateway.create_request(
            command,
            source=RequestSource.SYSTEM,
            safety_context=safety_context or RequestSafetyContext(user_present=False),
            **kwargs,
        )


class ResumeRequestAdapter:
    def __init__(self, gateway: RequestGateway) -> None:
        self._gateway = gateway

    def create(
        self,
        session_id: str,
        *,
        confirmation_response: bool | None = None,
        recovery_authorization: bool | None = None,
        content: str | None = None,
        execution_context: RequestExecutionContext | None = None,
        **kwargs: Any,
    ) -> AtlasRequest:
        context = execution_context or RequestExecutionContext(
            session_id=session_id,
            resume_target=session_id,
            confirmation_response=confirmation_response,
            recovery_authorization=recovery_authorization,
        )
        return self._gateway.create_request(
            content or f"resume {session_id}",
            source=RequestSource.RESUME,
            execution_context=context,
            **kwargs,
        )


def _normalize_content(content: str) -> str:
    if not isinstance(content, str):
        raise EmptyRequestContentError("content must be a string.")
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.rstrip() for line in normalized.split("\n"))


def _validate_id(value: str, name: str, error_type: type[Exception]) -> None:
    if not isinstance(value, str) or not value.strip() or not _ID_PATTERN.fullmatch(value):
        raise error_type(f"{name} is invalid.")


def _freeze_json_safe(
    value: Any,
    *,
    max_depth: int = 6,
    depth: int = 0,
) -> Any:
    if depth > max_depth:
        raise InvalidRequestMetadataError("metadata is too deep.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(key, str) or not key.strip():
                raise InvalidRequestMetadataError("metadata keys must be non-empty strings.")
            if any(secret in key.lower() for secret in _SECRET_KEYS):
                raise InvalidRequestMetadataError("metadata cannot contain credential keys.")
            frozen[key] = _freeze_json_safe(item, max_depth=max_depth, depth=depth + 1)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_safe(item, max_depth=max_depth, depth=depth + 1)
            for item in value
        )
    raise InvalidRequestMetadataError("metadata values must be JSON-safe.")


def _thaw_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_safe(item) for item in value]
    return value

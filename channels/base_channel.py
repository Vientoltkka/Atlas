"""Minimal interface for Atlas communication channels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class ChannelError(RuntimeError):
    """Base error for channel adapter operations."""


class InvalidChannelMessageError(ChannelError):
    """Raised when an inbound channel message is malformed."""


class BaseChannel(ABC):
    """Contract for inbound/outbound messaging channels.

    Channels only translate between the channel wire format and the
    existing Atlas execution contracts. They never execute agents or
    duplicate router/executor logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique channel identifier."""
        ...

    @abstractmethod
    def parse_inbound(self, message: Mapping[str, Any]) -> Any:
        """Translate a raw inbound channel message into an Atlas request contract."""
        ...

    @abstractmethod
    def format_outbound(self, result: Any) -> Mapping[str, Any]:
        """Translate an Atlas result contract into the channel outbound format."""
        ...

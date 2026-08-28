"""Minimal real inference health checks for selected Atlas models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import httpx
import ollama

from models.prompt_client import InferenceBackendError


class ModelHealthErrorCode(str, Enum):
    """Stable classifications for an unsuccessful inference health check."""

    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    UNKNOWN_BACKEND_ERROR = "UNKNOWN_BACKEND_ERROR"


@dataclass(frozen=True, slots=True)
class ModelHealthResult:
    """Controlled result of checking one inventory model through its backend."""

    logical_model_id: str
    physical_model_name: str
    provider_id: str
    healthy: bool
    error_code: ModelHealthErrorCode | None = None
    reason: str | None = None


class ModelHealthChecker(Protocol):
    """Check only the model that is about to be used for inference."""

    def check(
        self,
        *,
        logical_model_id: str,
        physical_model_name: str,
        provider_id: str,
    ) -> ModelHealthResult:
        ...


class _PromptHealthClient(Protocol):
    def check_model_health(self, model: str) -> None:
        ...


class OllamaModelHealthChecker:
    """Perform a real, minimal Ollama inference through the shared PromptClient."""

    def __init__(self, prompt_client: _PromptHealthClient) -> None:
        self._prompt_client = prompt_client

    def check(
        self,
        *,
        logical_model_id: str,
        physical_model_name: str,
        provider_id: str,
    ) -> ModelHealthResult:
        try:
            if provider_id != "ollama":
                raise InferenceBackendError(
                    physical_model_name,
                    f"Unsupported model provider '{provider_id}'.",
                )
            self._prompt_client.check_model_health(physical_model_name)
        except InferenceBackendError as error:
            return ModelHealthResult(
                logical_model_id=logical_model_id,
                physical_model_name=physical_model_name,
                provider_id=provider_id,
                healthy=False,
                error_code=self._classify(error),
                reason=error.reason,
            )
        return ModelHealthResult(
            logical_model_id=logical_model_id,
            physical_model_name=physical_model_name,
            provider_id=provider_id,
            healthy=True,
        )

    @staticmethod
    def _classify(error: InferenceBackendError) -> ModelHealthErrorCode:
        cause = error.__cause__
        if isinstance(cause, (TimeoutError, httpx.TimeoutException)):
            return ModelHealthErrorCode.TIMEOUT
        if "not found" in error.reason.lower() or "status code: 404" in error.reason.lower():
            return ModelHealthErrorCode.MODEL_UNAVAILABLE
        if isinstance(cause, ollama.ResponseError):
            status_code = getattr(cause, "status_code", None)
            if status_code == 404 or "not found" in str(cause).lower():
                return ModelHealthErrorCode.MODEL_UNAVAILABLE
        if isinstance(
            cause,
            (ConnectionError, ollama.RequestError, httpx.RequestError),
        ):
            return ModelHealthErrorCode.BACKEND_UNAVAILABLE
        return ModelHealthErrorCode.UNKNOWN_BACKEND_ERROR

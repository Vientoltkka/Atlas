"""Bounded operational fallback around one real model inference request."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TypeVar

from core.model_manager import ModelManager, ModelSelectionRequest, ModelSelectionResult
from models.prompt_client import InferenceBackendError


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ModelInferenceResult:
    """Minimal observability for the latest successful inference."""

    initial_logical_model_id: str
    initial_physical_model_name: str
    final_logical_model_id: str
    final_physical_model_name: str
    used_fallback: bool
    attempt_count: int
    fallback_reason: str | None = None


class ModelSelectionError(RuntimeError):
    """The model authority could not produce an initial compatible model."""

    def __init__(self, result: ModelSelectionResult) -> None:
        self.result = result
        super().__init__(result.reason)


class InferenceFallbackExhaustedError(RuntimeError):
    """Every authorized inference model failed, or fallback was forbidden."""

    def __init__(
        self,
        *,
        initial_logical_model_id: str,
        attempted_logical_model_ids: tuple[str, ...],
        allow_fallback: bool,
        last_error: InferenceBackendError,
    ) -> None:
        self.initial_logical_model_id = initial_logical_model_id
        self.attempted_logical_model_ids = attempted_logical_model_ids
        self.allow_fallback = allow_fallback
        self.last_error = last_error
        super().__init__(
            "Inference failed after "
            f"{len(attempted_logical_model_ids)} attempt(s); "
            "no further authorized fallback is available."
        )


class InferenceStreamInterruptedError(RuntimeError):
    """A streamed inference failed after observable content was delivered."""

    def __init__(self, model: str, error: InferenceBackendError) -> None:
        self.model = model
        self.backend_error = error
        super().__init__(
            f"Inference stream for model '{model}' failed after partial output; "
            "fallback was not attempted to avoid mixing responses."
        )


class ModelInferenceRunner:
    """Run one inference through a finite ModelManager-authorized chain."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager
        self.last_result: ModelInferenceResult | None = None

    def run(
        self,
        request: ModelSelectionRequest,
        infer: Callable[[str], T],
        *,
        initial_selection: ModelSelectionResult | None = None,
    ) -> T:
        """Run non-streaming inference, trying each authorized model at most once."""
        self.last_result = None
        initial = initial_selection or self._initial_selection(request)
        current = initial
        attempted: list[str] = []
        attempted_physical: list[str] = []
        fallback_reason: str | None = None

        while True:
            logical_id, physical_name = self._selection_identity(current)
            attempted.append(logical_id)
            attempted_physical.append(physical_name)
            try:
                value = infer(physical_name)
            except InferenceBackendError as error:
                fallback_reason = error.reason
                next_selection = self._model_manager.select_fallback(
                    request,
                    initial_model_id=self._selection_identity(initial)[0],
                    attempted_model_ids=(*attempted, *attempted_physical),
                )
                if not next_selection.success:
                    raise InferenceFallbackExhaustedError(
                        initial_logical_model_id=self._selection_identity(initial)[0],
                        attempted_logical_model_ids=tuple(attempted),
                        allow_fallback=request.allow_fallback,
                        last_error=error,
                    ) from error
                current = next_selection
                continue

            self.last_result = self._success_result(
                initial,
                current,
                len(attempted),
                fallback_reason,
            )
            return value

    def stream(
        self,
        request: ModelSelectionRequest,
        infer: Callable[[str], Iterator[str]],
        *,
        initial_selection: ModelSelectionResult | None = None,
    ) -> Iterator[str]:
        """Stream with fallback only while no fragment has become observable."""
        self.last_result = None
        initial = initial_selection or self._initial_selection(request)
        current = initial
        attempted: list[str] = []
        attempted_physical: list[str] = []
        fallback_reason: str | None = None

        while True:
            logical_id, physical_name = self._selection_identity(current)
            attempted.append(logical_id)
            attempted_physical.append(physical_name)
            emitted = False
            try:
                for fragment in infer(physical_name):
                    emitted = True
                    yield fragment
            except InferenceBackendError as error:
                if emitted:
                    raise InferenceStreamInterruptedError(physical_name, error) from error
                fallback_reason = error.reason
                next_selection = self._model_manager.select_fallback(
                    request,
                    initial_model_id=self._selection_identity(initial)[0],
                    attempted_model_ids=(*attempted, *attempted_physical),
                )
                if not next_selection.success:
                    raise InferenceFallbackExhaustedError(
                        initial_logical_model_id=self._selection_identity(initial)[0],
                        attempted_logical_model_ids=tuple(attempted),
                        allow_fallback=request.allow_fallback,
                        last_error=error,
                    ) from error
                current = next_selection
                continue

            self.last_result = self._success_result(
                initial,
                current,
                len(attempted),
                fallback_reason,
            )
            return

    def _initial_selection(
        self,
        request: ModelSelectionRequest,
    ) -> ModelSelectionResult:
        selection = self._model_manager.select_model(request)
        if not selection.success:
            raise ModelSelectionError(selection)
        self._selection_identity(selection)
        return selection

    @staticmethod
    def _selection_identity(selection: ModelSelectionResult) -> tuple[str, str]:
        logical_id = selection.logical_model_id
        physical_name = selection.physical_model_name
        if not logical_id or not physical_name:
            raise ModelSelectionError(selection)
        return logical_id, physical_name

    @classmethod
    def _success_result(
        cls,
        initial: ModelSelectionResult,
        final: ModelSelectionResult,
        attempt_count: int,
        fallback_reason: str | None,
    ) -> ModelInferenceResult:
        initial_logical, initial_physical = cls._selection_identity(initial)
        final_logical, final_physical = cls._selection_identity(final)
        used_fallback = initial_logical != final_logical
        return ModelInferenceResult(
            initial_logical_model_id=initial_logical,
            initial_physical_model_name=initial_physical,
            final_logical_model_id=final_logical,
            final_physical_model_name=final_physical,
            used_fallback=used_fallback,
            attempt_count=attempt_count,
            fallback_reason=fallback_reason if used_fallback else None,
        )

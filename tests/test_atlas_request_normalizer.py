from __future__ import annotations

import math
from types import MappingProxyType, SimpleNamespace

import pytest

from bootstrap.atlas_request_normalizer import build_core_atlas_request_normalizer
from core.atlas_request_adapter import AtlasRequestAdapter
from core.atlas_request_classifier import AtlasRequestClassifier, StructuredInput
from core.atlas_request_normalizer import (
    AtlasRequestNormalizationStatus,
    AtlasRequestNormalizer,
    InvalidAtlasRequestNormalizationInputError,
    atlas_request_normalization_signature,
)
from core.atlas_router import AtlasRouteType, AtlasRoutingResult, AtlasRoutingStatus
from core.orchestrator import AtlasOrchestrator
from core.router import Router
from memory.conversation import ConversationMemory


class CountingClassifier(AtlasRequestClassifier):
    def __init__(self) -> None:
        self.calls = 0
        self.requests = []

    def classify(self, structured_input):
        self.calls += 1
        self.requests.append(structured_input)
        return super().classify(structured_input)


class CountingAdapter(AtlasRequestAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def adapt(self, request):
        self.calls += 1
        return super().adapt(request)


class CountingRouter:
    def __init__(self, result: AtlasRoutingResult | None = None) -> None:
        self.result = result or AtlasRoutingResult(
            AtlasRoutingStatus.COMPLETED,
            AtlasRouteType.TOOL,
            output={"ok": True},
        )
        self.calls = 0
        self.requests = []

    def route(self, request):
        self.calls += 1
        self.requests.append(request)
        return self.result


def _atlas_orchestrator(*, normalizer=None, classifier=None, adapter=None, router=None):
    return AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=SimpleNamespace(get=lambda _name: None),
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        atlas_request_normalizer=normalizer,
        atlas_request_classifier=classifier,
        atlas_request_adapter=adapter,
        atlas_router=router,
    )


def _unsafe_structured_input(**overrides) -> StructuredInput:
    structured_input = object.__new__(StructuredInput)
    values = {
        "kind": None,
        "capability_id": None,
        "workflow_id": None,
        "tool_name": None,
        "route": "tool",
        "metadata": {},
        "payload": None,
        "request_id": None,
    }
    values.update(overrides)
    for name, value in values.items():
        object.__setattr__(structured_input, name, value)
    return structured_input


def test_normalizes_strings_route_kind_and_collections() -> None:
    result = AtlasRequestNormalizer().normalize(
        StructuredInput(
            kind=" TOOL ",
            route=" TOOL ",
            payload={" b ": [" x ", {" y ": " z "}]},
            metadata={" source ": " test "},
            request_id=" r-1 ",
        )
    )

    assert result.status is AtlasRequestNormalizationStatus.NORMALIZED
    assert result.structured_input is not None
    normalized = result.structured_input
    assert normalized.kind == "tool"
    assert normalized.route == "tool"
    assert normalized.request_id == "r-1"
    assert isinstance(normalized.payload, MappingProxyType)
    assert list(normalized.payload.keys()) == ["b"]  # type: ignore[union-attr]
    assert normalized.payload["b"] == ("x", MappingProxyType({"y": "z"}))  # type: ignore[index]
    assert normalized.metadata["source"] == "test"


def test_normalization_is_idempotent() -> None:
    normalizer = AtlasRequestNormalizer()
    first = normalizer.normalize(
        StructuredInput(route=" TOOL ", payload={"b": [" two "], "a": " one "}, metadata={"z": " ok "})
    )
    assert first.structured_input is not None
    second = normalizer.normalize(first.structured_input)

    assert second.structured_input == first.structured_input
    assert second.input_signature == first.input_signature


def test_payload_and_metadata_are_defensively_copied() -> None:
    payload = {"items": [" a "]}
    metadata = {"trace": " one "}
    structured_input = StructuredInput(route="tool", payload=payload, metadata=metadata)
    result = AtlasRequestNormalizer().normalize(structured_input)
    payload["items"].append("b")
    metadata["trace"] = "two"

    assert result.structured_input is not None
    assert result.structured_input.payload["items"] == ("a",)  # type: ignore[index]
    assert result.structured_input.metadata["trace"] == "one"
    with pytest.raises(TypeError):
        result.structured_input.payload["new"] = "value"  # type: ignore[index]


def test_canonical_key_order_is_stable() -> None:
    result = AtlasRequestNormalizer().normalize(
        StructuredInput(route="tool", payload={"z": 1, "a": {"b": 2, "a": 1}})
    )

    assert result.structured_input is not None
    assert list(result.structured_input.payload.keys()) == ["a", "z"]  # type: ignore[union-attr]
    assert list(result.structured_input.payload["a"].keys()) == ["a", "b"]  # type: ignore[index]


def test_structurally_equivalent_inputs_share_signature() -> None:
    first = AtlasRequestNormalizer().normalize(
        StructuredInput(
            route=" TOOL ",
            payload={"b": [" x "], "a": {"z": 1, "a": None}},
            metadata={"z": True, "a": " one "},
            request_id=" r-1 ",
        )
    )
    second = AtlasRequestNormalizer().normalize(
        StructuredInput(
            route=AtlasRouteType.TOOL,
            payload={"a": {"a": None, "z": 1}, "b": ["x"]},
            metadata={"a": "one", "z": True},
            request_id="r-1",
        )
    )

    assert first.input_signature == second.input_signature


@pytest.mark.parametrize("bad", (object(), lambda: None, AtlasRequestNormalizer))
def test_rejects_non_structured_or_unsafe_input_object(bad: object) -> None:
    result = AtlasRequestNormalizer().normalize(bad)  # type: ignore[arg-type]

    assert result.status is AtlasRequestNormalizationStatus.INVALID_INPUT
    assert result.error_code == "INVALID_INPUT"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_normalizer_rejects_nan_and_infinity_if_they_cross_input_boundary(bad: float) -> None:
    result = AtlasRequestNormalizer().normalize(_unsafe_structured_input(payload={"bad": bad}))

    assert result.status is AtlasRequestNormalizationStatus.INVALID_INPUT
    assert "finite" in result.message


@pytest.mark.parametrize("bad", (object(), lambda: None, AtlasRequestNormalizer, math))
def test_normalizer_rejects_arbitrary_objects_if_they_cross_input_boundary(bad: object) -> None:
    result = AtlasRequestNormalizer().normalize(_unsafe_structured_input(payload={"bad": bad}))

    assert result.status is AtlasRequestNormalizationStatus.INVALID_INPUT
    assert result.error_code == "INVALID_INPUT"


def test_normalizer_rejects_limits_exceeded_if_they_cross_input_boundary() -> None:
    too_long = "x" * 501
    too_many = {f"k{i}": i for i in range(65)}
    deep = {"x": {"x": {"x": {"x": {"x": {"x": {"x": {"x": {"x": "too deep"}}}}}}}}}
    normalizer = AtlasRequestNormalizer()

    assert normalizer.normalize(_unsafe_structured_input(payload={"value": too_long})).status is (
        AtlasRequestNormalizationStatus.INVALID_INPUT
    )
    assert normalizer.normalize(_unsafe_structured_input(payload=too_many)).status is (
        AtlasRequestNormalizationStatus.INVALID_INPUT
    )
    assert normalizer.normalize(_unsafe_structured_input(payload=deep)).status is (
        AtlasRequestNormalizationStatus.INVALID_INPUT
    )


def test_signature_requires_structured_input() -> None:
    with pytest.raises(InvalidAtlasRequestNormalizationInputError):
        atlas_request_normalization_signature(object())  # type: ignore[arg-type]


def test_failed_normalization_does_not_call_downstream_layers() -> None:
    classifier = CountingClassifier()
    adapter = CountingAdapter()
    router = CountingRouter()
    orchestrator = _atlas_orchestrator(
        normalizer=AtlasRequestNormalizer(),
        classifier=classifier,
        adapter=adapter,
        router=router,
    )

    result = orchestrator.route_structured_input(object())  # type: ignore[arg-type]

    assert result.status is AtlasRoutingStatus.INVALID_REQUEST
    assert result.error_code == "INVALID_INPUT"
    assert classifier.calls == 0
    assert adapter.calls == 0
    assert router.calls == 0


def test_full_integration_normalizer_classifier_adapter_router() -> None:
    classifier = CountingClassifier()
    adapter = CountingAdapter()
    router = CountingRouter()
    orchestrator = _atlas_orchestrator(
        normalizer=AtlasRequestNormalizer(),
        classifier=classifier,
        adapter=adapter,
        router=router,
    )

    result = orchestrator.route_structured_input(
        StructuredInput(route=" TOOL ", payload={" message ": " ok "}, request_id=" r-1 ")
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert classifier.calls == 1
    assert classifier.requests[0].route == "tool"
    assert classifier.requests[0].payload["message"] == "ok"
    assert adapter.calls == 1
    assert router.calls == 1
    assert router.requests[0].route_type is AtlasRouteType.TOOL


def test_compatibility_when_normalizer_is_not_injected() -> None:
    classifier = CountingClassifier()
    adapter = CountingAdapter()
    router = CountingRouter()
    orchestrator = _atlas_orchestrator(
        normalizer=None,
        classifier=classifier,
        adapter=adapter,
        router=router,
    )

    result = orchestrator.route_structured_input(StructuredInput(route="tool"))

    assert result.status is AtlasRoutingStatus.SERVICE_UNAVAILABLE
    assert result.error_code == "ATLAS_REQUEST_NORMALIZER_UNAVAILABLE"
    assert classifier.calls == 0
    assert adapter.calls == 0
    assert router.calls == 0


def test_deterministic_signature_uses_canonical_structure_only() -> None:
    normalizer = AtlasRequestNormalizer()
    first = normalizer.normalize(StructuredInput(route=" TOOL ", payload={"b": 2, "a": 1}))
    second = normalizer.normalize(StructuredInput(route="tool", payload={"a": 1, "b": 2}))
    changed = normalizer.normalize(StructuredInput(route="tool", payload={"a": 1, "b": 3}))

    assert first.structured_input is not None
    assert first.input_signature == atlas_request_normalization_signature(first.structured_input)
    assert first.input_signature == second.input_signature
    assert first.input_signature != changed.input_signature


def test_bootstrap_factory_builds_normalizer() -> None:
    assert isinstance(build_core_atlas_request_normalizer(), AtlasRequestNormalizer)

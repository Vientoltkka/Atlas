from core.model_latency import ModelLatencyTracker
from core.self_diagnosis import SelfDiagnosisService


def test_no_runtime_evidence_produces_no_finding() -> None:
    tracker = ModelLatencyTracker()

    assert SelfDiagnosisService(latency_tracker=tracker).diagnose() == ()


def test_single_slow_observation_is_insufficient() -> None:
    tracker = ModelLatencyTracker()
    tracker.record("slow-model", "test-provider", 12.0)

    assert SelfDiagnosisService(latency_tracker=tracker).diagnose() == ()


def test_repeated_fast_observations_do_not_produce_finding() -> None:
    tracker = ModelLatencyTracker()
    tracker.record("fast-model", "test-provider", 2.0)
    tracker.record("fast-model", "test-provider", 3.0)

    samples = tracker.snapshot()

    assert len(samples) == 1
    assert samples[0].observation_count == 2
    assert SelfDiagnosisService(latency_tracker=tracker).diagnose() == ()


def test_repeated_slow_observations_produce_evidence_based_finding() -> None:
    tracker = ModelLatencyTracker()
    tracker.record("slow-model", "test-provider", 10.0)
    tracker.record("slow-model", "test-provider", 12.0)

    findings = SelfDiagnosisService(latency_tracker=tracker).diagnose()

    assert len(findings) == 1
    finding = findings[0]
    assert finding.finding_id == "runtime.model-latency"
    assert "modelo=slow-model" in finding.evidence
    assert "proveedor=test-provider" in finding.evidence
    assert "observaciones=2" in finding.evidence


def test_slowest_qualifying_model_is_reported() -> None:
    tracker = ModelLatencyTracker(alpha=1.0)

    tracker.record("model-a", "provider-a", 6.0)
    tracker.record("model-a", "provider-a", 7.0)

    tracker.record("model-b", "provider-b", 9.0)
    tracker.record("model-b", "provider-b", 11.0)

    findings = SelfDiagnosisService(latency_tracker=tracker).diagnose()

    assert len(findings) == 1
    assert "modelo=model-b" in findings[0].evidence
    assert "latencia_observada=11.000s" in findings[0].evidence

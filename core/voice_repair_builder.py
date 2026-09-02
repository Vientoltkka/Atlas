"""Deterministic first supervised code repair for the voice subsystem."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.self_improvement_conversation import ImprovementDiagnosis
from core.supervised_repair import RepairProposal, RepairValidation


_VOICE_SOURCE = "use_cases/voice_conversation.py"
_VOICE_TEST = "tests/test_voice_conversation.py"
_WAIT_SECONDS = 0.25
_MEASURE_SECONDS = 0.40
_SOURCE_OLD = "    _TTS_WORKER_JOIN_TIMEOUT_SECONDS = 0.25\n"
_SOURCE_NEW = _SOURCE_OLD + "    _EXPIRED_MODEL_WORKER_WAIT_TIMEOUT_SECONDS = 0.25\n"
_WAIT_OLD = '''        with self._model_worker_condition:
            while any(worker.is_alive() for worker in self._expired_model_workers):
                self._model_worker_condition.wait()
            self._expired_model_workers.clear()
'''
_WAIT_NEW = '''        deadline = time.monotonic() + self._EXPIRED_MODEL_WORKER_WAIT_TIMEOUT_SECONDS
        with self._model_worker_condition:
            while any(worker.is_alive() for worker in self._expired_model_workers):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VoiceModelTimeoutError("un worker de modelo expirado no terminó tras la cancelación")
                self._model_worker_condition.wait(timeout=remaining)
            self._expired_model_workers.clear()
'''
_TEST = '''\n\ndef test_expired_model_worker_wait_is_bounded() -> None:
    use_case = make_use_case(FakeSpeechEngine([]))
    worker = threading.Thread(target=lambda: time.sleep(0.40), daemon=True)
    worker.start()
    with use_case._model_worker_condition:
        use_case._expired_model_workers.add(worker)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        use_case._wait_for_expired_model_workers()
    elapsed = time.monotonic() - started
    worker.join()

    assert elapsed < 0.35
'''


class VoiceCodeRepairBuilder:
    """Build exactly one reviewed voice repair; it never accepts generated code."""

    def __init__(self, project_root: Path) -> None:
        self._root = project_root.resolve()
        self._before_latency_ms: float | None = None

    def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
        return diagnosis.scope == (_VOICE_SOURCE, _VOICE_TEST)

    def build(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> RepairProposal | None:
        if not self.can_handle(diagnosis, _prompt):
            return None
        source_path, test_path = self._root / _VOICE_SOURCE, self._root / _VOICE_TEST
        try:
            source, tests = source_path.read_text(encoding="utf-8"), test_path.read_text(encoding="utf-8")
        except OSError:
            return None
        if _SOURCE_NEW in source or _SOURCE_OLD not in source or _WAIT_OLD not in source or _TEST in tests:
            return None
        before_status, before_latency_ms = self._measure_wait()
        if before_status != "completed":
            return None
        self._before_latency_ms = before_latency_ms
        return RepairProposal(
            proposal_id="repair.voice.expired-model-worker-wait",
            objective="Evitar que un worker de modelo expirado bloquee indefinidamente el siguiente turno de voz.",
            files={
                _VOICE_SOURCE: source.replace(_SOURCE_OLD, _SOURCE_NEW, 1).replace(_WAIT_OLD, _WAIT_NEW, 1),
                _VOICE_TEST: tests + _TEST,
            },
            focused_tests=(_VOICE_TEST,),
            metric_directions={"expired_model_worker_wait_ms": "decrease"},
        )

    def validator(self, _proposal: RepairProposal) -> RepairValidation:
        before = self._before_latency_ms
        if before is None:
            return RepairValidation(False, detail="No existe una medición previa confiable.")
        after_status, after = self._measure_wait()
        tests = self._run("-m", "pytest", "-q", _VOICE_TEST)
        compiled = self._run("-m", "py_compile", _VOICE_SOURCE, _VOICE_TEST)
        diff_checked = self._git("diff", "--check")
        passed = after_status == "timed_out" and after < before and all(result.returncode == 0 for result in (tests, compiled, diff_checked))
        detail = "tests focales, py_compile y git diff --check correctos." if passed else self._validation_error(after_status, tests, compiled, diff_checked)
        return RepairValidation(passed, {"expired_model_worker_wait_ms": before}, {"expired_model_worker_wait_ms": after}, detail)

    def _measure_wait(self) -> tuple[str, float]:
        """Measure the known local timeout path in a fresh interpreter."""
        script = f'''import threading, time
from use_cases.voice_conversation import VoiceConversationUseCase, VoiceModelTimeoutError
voice = VoiceConversationUseCase(object(), None)
def finish():
    time.sleep({_MEASURE_SECONDS})
    with voice._model_worker_condition:
        voice._model_worker_condition.notify_all()
worker = threading.Thread(target=finish, daemon=True)
worker.start()
with voice._model_worker_condition:
    voice._expired_model_workers.add(worker)
started = time.monotonic()
try:
    voice._wait_for_expired_model_workers()
    status = "completed"
except VoiceModelTimeoutError:
    status = "timed_out"
elapsed = (time.monotonic() - started) * 1000
worker.join()
print(status, format(elapsed, ".3f"))'''
        result = subprocess.run((sys.executable, "-c", script), cwd=self._root, capture_output=True, text=True, check=False)
        try:
            status, elapsed = result.stdout.strip().splitlines()[-1].split()
            return status, float(elapsed)
        except (IndexError, ValueError):
            return "failed", float("inf")

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run((sys.executable, *arguments), cwd=self._root, capture_output=True, text=True, check=False)

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(("git", *arguments), cwd=self._root, capture_output=True, text=True, check=False)

    @staticmethod
    def _validation_error(status: str, *results: subprocess.CompletedProcess[str]) -> str:
        labels = ("tests focales", "py_compile", "git diff --check")
        failures = [label for label, result in zip(labels, results) if result.returncode != 0]
        return "Validación fallida: medición=" + status + "; " + ", ".join(failures or ("métrica no mejoró",)) + "."

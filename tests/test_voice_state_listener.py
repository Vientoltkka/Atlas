"""Tests for the V4.3-I1 optional voice state listener."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from use_cases.voice_conversation import (
    VoiceConversationSession,
    VoiceConversationState,
    VoiceConversationUseCase,
)


def make_use_case(state_listener=None) -> VoiceConversationUseCase:
    return VoiceConversationUseCase(
        speech_engine=SimpleNamespace(),
        wake_word_engine=None,
        speech_output_engine=None,
        diagnostics_enabled=False,
        clock=lambda: 0.0,
        now_provider=lambda: datetime.now().astimezone(),
        state_listener=state_listener,
    )


def make_session() -> VoiceConversationSession:
    session = VoiceConversationSession(
        session_id="test-session",
        started_at=0.0,
    )
    # Force the logical starting point for transition assertions.
    session.state = VoiceConversationState.READY
    return session


def test_transition_calls_listener_once_per_real_change() -> None:
    calls: list[VoiceConversationState] = []
    use_case = make_use_case(state_listener=calls.append)
    session = make_session()

    use_case._set_state(session, VoiceConversationState.LISTENING)
    assert calls == [VoiceConversationState.LISTENING]

    use_case._set_state(session, VoiceConversationState.TRANSCRIBING)
    assert calls[-1] is VoiceConversationState.TRANSCRIBING

    use_case._set_state(session, VoiceConversationState.PROCESSING)
    assert calls[-1] is VoiceConversationState.PROCESSING

    use_case._set_state(session, VoiceConversationState.SPEAKING)
    assert calls[-1] is VoiceConversationState.SPEAKING

    assert len(calls) == 4


def test_stopping_to_stopped_notifies_once_each() -> None:
    calls: list[VoiceConversationState] = []
    use_case = make_use_case(state_listener=calls.append)
    session = make_session()
    session.state = VoiceConversationState.PROCESSING

    use_case._set_state(session, VoiceConversationState.STOPPING)
    use_case._set_state(session, VoiceConversationState.STOPPED)

    assert calls == [
        VoiceConversationState.STOPPING,
        VoiceConversationState.STOPPED,
    ]


def test_repeating_same_state_does_not_notify_again() -> None:
    calls: list[VoiceConversationState] = []
    use_case = make_use_case(state_listener=calls.append)
    session = make_session()

    use_case._set_state(session, VoiceConversationState.LISTENING)
    use_case._set_state(session, VoiceConversationState.LISTENING)
    use_case._set_state(session, VoiceConversationState.LISTENING)

    assert calls == [VoiceConversationState.LISTENING]


def test_none_listener_is_identical_to_current_behaviour() -> None:
    use_case = make_use_case(state_listener=None)
    session = make_session()

    use_case._set_state(session, VoiceConversationState.LISTENING)

    assert session.state is VoiceConversationState.LISTENING
    assert session.states[-1] is VoiceConversationState.LISTENING


def test_failing_listener_never_breaks_the_session() -> None:
    def broken(_state) -> None:
        raise RuntimeError("ui rota")

    use_case = make_use_case(state_listener=broken)
    session = make_session()

    use_case._set_state(session, VoiceConversationState.LISTENING)
    use_case._set_state(session, VoiceConversationState.PROCESSING)

    assert session.state is VoiceConversationState.PROCESSING
    assert session.states[-2:] == [
        VoiceConversationState.LISTENING,
        VoiceConversationState.PROCESSING,
    ]


def test_constructor_listener_used_when_no_session_override() -> None:
    calls: list[VoiceConversationState] = []
    use_case = make_use_case(state_listener=calls.append)
    session = make_session()

    use_case._set_state(session, VoiceConversationState.LISTENING)

    assert calls == [VoiceConversationState.LISTENING]


def test_execute_manual_session_scoped_listener_is_used() -> None:
    calls: list[VoiceConversationState] = []
    use_case = make_use_case()
    session = make_session()
    # Simulate the session-scoped wiring performed by execute_manual.
    use_case._session_state_listener = calls.append
    use_case._set_state(session, VoiceConversationState.PROCESSING)

    assert calls == [VoiceConversationState.PROCESSING]


def test_session_scoped_listener_overrides_constructor_one() -> None:
    constructor_calls: list[VoiceConversationState] = []
    session_calls: list[VoiceConversationState] = []
    use_case = make_use_case(state_listener=constructor_calls.append)
    use_case._session_state_listener = session_calls.append
    session = make_session()

    use_case._set_state(session, VoiceConversationState.LISTENING)

    assert session_calls == [VoiceConversationState.LISTENING]
    assert constructor_calls == []

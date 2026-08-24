"""Tests for the V4.3-I1 pure Orb state mapper."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from use_cases.ui_state_mapper import OrbVisualState, map_to_orb_state
from use_cases.voice_conversation import VoiceConversationState


@pytest.mark.parametrize(
    ("voice_state", "expected"),
    (
        (VoiceConversationState.STARTING, OrbVisualState.STARTING),
        (VoiceConversationState.READY, OrbVisualState.IDLE),
        (VoiceConversationState.LISTENING, OrbVisualState.LISTENING),
        (VoiceConversationState.TRANSCRIBING, OrbVisualState.PROCESSING),
        (VoiceConversationState.PROCESSING, OrbVisualState.PROCESSING),
        (VoiceConversationState.SPEAKING, OrbVisualState.SPEAKING),
        (VoiceConversationState.RECOVERING, OrbVisualState.RECOVERING),
        (VoiceConversationState.STOPPING, OrbVisualState.STOPPING),
        (VoiceConversationState.STOPPED, OrbVisualState.IDLE),
        (VoiceConversationState.DEGRADED, OrbVisualState.DEGRADED),
    ),
)
def test_all_ten_real_states_map_deterministically(voice_state, expected) -> None:
    assert map_to_orb_state(voice_state) is expected


def test_mapping_accepts_raw_string_values() -> None:
    assert map_to_orb_state("LISTENING") is OrbVisualState.LISTENING
    assert map_to_orb_state("STOPPED") is OrbVisualState.IDLE


def test_mapping_is_pure_and_repeatable() -> None:
    first = map_to_orb_state(VoiceConversationState.PROCESSING)
    second = map_to_orb_state(VoiceConversationState.PROCESSING)
    assert first is second is OrbVisualState.PROCESSING


def test_unknown_state_is_rejected() -> None:
    with pytest.raises(ValueError):
        map_to_orb_state("NO_EXISTE")


def test_visual_states_do_not_invent_core_states() -> None:
    # No ERROR/CLOSING duplicates: only visual derivations exist.
    visual_values = {state.value for state in OrbVisualState}
    assert visual_values == {
        "IDLE",
        "STARTING",
        "LISTENING",
        "PROCESSING",
        "SPEAKING",
        "RECOVERING",
        "DEGRADED",
        "STOPPING",
    }


def test_mapper_does_not_import_graphic_frameworks() -> None:
    code = (
        "import sys;"
        "from use_cases.ui_state_mapper import map_to_orb_state;"
        "map_to_orb_state('LISTENING');"
        "graphic = [m for m in sys.modules if m.startswith(('PySide', 'PyQt', 'tkinter'))];"
        "print(graphic)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"

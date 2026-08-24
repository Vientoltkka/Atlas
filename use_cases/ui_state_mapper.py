"""Pure mapping from Atlas voice states to future Orbe visual states (V4.3-I1).

Deterministic and side-effect free: no Qt imports, no audio access, no
orchestrator access. The Orbe will be a thin visual layer on top of the
real ``VoiceConversationState`` lifecycle; no second conversation state
machine exists.
"""

from __future__ import annotations

from enum import Enum

from use_cases.voice_conversation import VoiceConversationState


class OrbVisualState(str, Enum):
    """Visual states derived from the real Atlas session states."""

    IDLE = "IDLE"
    STARTING = "STARTING"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    RECOVERING = "RECOVERING"
    DEGRADED = "DEGRADED"
    STOPPING = "STOPPING"


_ORB_STATE_BY_VOICE_STATE = {
    VoiceConversationState.STARTING: OrbVisualState.STARTING,
    VoiceConversationState.READY: OrbVisualState.IDLE,
    VoiceConversationState.LISTENING: OrbVisualState.LISTENING,
    VoiceConversationState.TRANSCRIBING: OrbVisualState.PROCESSING,
    VoiceConversationState.PROCESSING: OrbVisualState.PROCESSING,
    VoiceConversationState.SPEAKING: OrbVisualState.SPEAKING,
    VoiceConversationState.RECOVERING: OrbVisualState.RECOVERING,
    VoiceConversationState.STOPPING: OrbVisualState.STOPPING,
    VoiceConversationState.STOPPED: OrbVisualState.IDLE,
    VoiceConversationState.DEGRADED: OrbVisualState.DEGRADED,
}


def map_to_orb_state(state: VoiceConversationState) -> OrbVisualState:
    """Return the deterministic visual state for one real session state."""
    try:
        return _ORB_STATE_BY_VOICE_STATE[VoiceConversationState(state)]
    except ValueError as error:
        raise ValueError(
            f"Estado de voz no reconocido para el Orbe: {state!r}"
        ) from error

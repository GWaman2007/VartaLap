"""
Unit tests for duplex barge-in (interruption) behavior in Vartalap.
"""

from unittest.mock import MagicMock
from vartalap.config import VartalapConfig
from vartalap.state_machine import VartalapState, VartalapStateMachine


def test_talking_barge_in_on_energy_spike():
    """Verify that when user speaks during TTS playback, TTS is stopped and state goes to MONITORING."""
    config = VartalapConfig()
    flm_mock = MagicMock()
    stt_mock = MagicMock()
    tts_mock = MagicMock()

    # Simulate user speaking via energy detection
    stt_mock.is_user_speaking.return_value = True
    stt_mock.transcript_queue.empty.return_value = True

    harness = VartalapStateMachine(config=config, flm=flm_mock, stt=stt_mock, tts=tts_mock)
    harness.current_state = VartalapState.TALKING

    barge_in_called = False
    def on_barge():
        nonlocal barge_in_called
        barge_in_called = True

    harness.on_barge_in = on_barge
    harness._handle_talking()

    assert barge_in_called is True
    assert tts_mock.stop.called
    assert harness.current_state == VartalapState.MONITORING


def test_talking_barge_in_on_words_arrived():
    """Verify that when transcribed words arrive during TTS playback, TTS stops immediately."""
    config = VartalapConfig()
    flm_mock = MagicMock()
    stt_mock = MagicMock()
    tts_mock = MagicMock()

    stt_mock.is_user_speaking.return_value = False
    stt_mock.transcript_queue.empty.return_value = False  # Words arrived!

    harness = VartalapStateMachine(config=config, flm=flm_mock, stt=stt_mock, tts=tts_mock)
    harness.current_state = VartalapState.TALKING

    harness._handle_talking()

    assert tts_mock.stop.called
    assert harness.current_state == VartalapState.MONITORING

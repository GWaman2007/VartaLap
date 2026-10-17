"""
Unit tests for Vartalap State Machine transitions.
"""

from unittest.mock import MagicMock
from vartalap.config import VartalapConfig
from vartalap.flm_client import FLMDecision
from vartalap.state_machine import VartalapState, VartalapStateMachine


def test_idle_to_monitoring_transition():
    config = VartalapConfig()
    flm_mock = MagicMock()
    stt_mock = MagicMock()
    tts_mock = MagicMock()

    harness = VartalapStateMachine(config=config, flm=flm_mock, stt=stt_mock, tts=tts_mock)
    assert harness.current_state == VartalapState.IDLE

    # Push word to transcript queue
    stt_mock.transcript_queue.get.return_value = "Hello"
    harness._handle_idle()

    assert harness.current_state == VartalapState.MONITORING


def test_monitoring_listen_stay_in_monitoring():
    config = VartalapConfig()
    flm_mock = MagicMock()
    stt_mock = MagicMock()
    tts_mock = MagicMock()

    flm_mock.evaluate_stream.return_value = FLMDecision(action="listen", utterance="")
    stt_mock.check_silence_tick.return_value = 0
    stt_mock.get_current_partial.return_value = "I am thinking about"

    harness = VartalapStateMachine(config=config, flm=flm_mock, stt=stt_mock, tts=tts_mock)
    harness.current_state = VartalapState.MONITORING

    harness._handle_monitoring()
    assert harness.current_state == VartalapState.MONITORING


def test_monitoring_talk_transition():
    config = VartalapConfig()
    flm_mock = MagicMock()
    stt_mock = MagicMock()
    tts_mock = MagicMock()

    flm_mock.evaluate_stream.return_value = FLMDecision(action="talk", utterance="Here is your answer")
    stt_mock.check_silence_tick.return_value = 2
    stt_mock.get_current_partial.return_value = "What is Vartalap?"

    harness = VartalapStateMachine(config=config, flm=flm_mock, stt=stt_mock, tts=tts_mock)
    harness.current_state = VartalapState.MONITORING

    harness._handle_monitoring()
    assert harness.current_state == VartalapState.TALKING
    assert harness.agent_last_chunk == "Here is your answer"


def test_arbitration_done_transition_to_idle():
    config = VartalapConfig()
    flm_mock = MagicMock()
    stt_mock = MagicMock()
    tts_mock = MagicMock()

    flm_mock.evaluate_stream.return_value = FLMDecision(action="done")
    stt_mock.check_silence_tick.return_value = 2
    stt_mock.get_current_partial.return_value = ""

    harness = VartalapStateMachine(config=config, flm=flm_mock, stt=stt_mock, tts=tts_mock)
    harness.current_state = VartalapState.ARBITRATION

    harness._handle_arbitration()
    assert harness.current_state == VartalapState.IDLE

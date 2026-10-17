import queue
import time
from unittest.mock import MagicMock
import numpy as np
import pytest

from vartalap.config import VartalapConfig
from vartalap.flm_client import FLMDecision
from vartalap.state_machine import VartalapState, VartalapStateMachine


def test_streaming_pieces_burst():
    config = VartalapConfig()
    flm_mock = MagicMock()
    stt_mock = MagicMock()
    tts_mock = MagicMock()

    # Simulate streaming callback emitting two pieces
    def mock_eval_stream(history, current_partial_transcript, silence_tick_count, agent_last_chunk, on_piece=None, **kwargs):
        if on_piece:
            on_piece("The President of India is Ram Nath Kovind,")
            on_piece("and the Prime Minister is Narendra Modi.")
        return FLMDecision(
            action="talk",
            utterance="The President of India is Ram Nath Kovind, and the Prime Minister is Narendra Modi.",
            latency_ms=1200.0,
            ttft_ms=900.0,
        )

    flm_mock.evaluate_stream.side_effect = mock_eval_stream
    stt_mock.check_silence_tick.return_value = 1
    stt_mock.get_current_partial.return_value = "Who is the President and Prime Minister?"

    harness = VartalapStateMachine(config=config, flm=flm_mock, stt=stt_mock, tts=tts_mock)
    harness.current_state = VartalapState.MONITORING

    harness._handle_monitoring()

    assert harness.current_state == VartalapState.TALKING
    assert len(harness.agent_turn_accumulator) == 2
    assert harness.agent_turn_accumulator[0] == "The President of India is Ram Nath Kovind,"
    assert harness.agent_turn_accumulator[1] == "and the Prime Minister is Narendra Modi."
    # Verify play_burst was called for the pieces
    assert tts_mock.play_burst.call_count == 2

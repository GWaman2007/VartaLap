import enum
import queue
import threading
import time
from typing import Callable, Optional

from .config import VartalapConfig
from .flm_client import FLMClient, FLMDecision
from .stt_stream import SherpaSTTStream
from .tts_player import PiperTTSPlayer


class VartalapState(str, enum.Enum):
    IDLE = "IDLE"
    MONITORING = "MONITORING"
    TALKING = "TALKING"
    ARBITRATION = "ARBITRATION"


class VartalapStateMachine:
    """Core 4-state conversational monitor and turn-taking harness."""

    def __init__(
        self,
        config: Optional[VartalapConfig] = None,
        flm: Optional[FLMClient] = None,
        stt: Optional[SherpaSTTStream] = None,
        tts: Optional[PiperTTSPlayer] = None,
    ):
        self.config = config or VartalapConfig()
        self.flm = flm or FLMClient(self.config)
        self.stt = stt or SherpaSTTStream(self.config)
        self.tts = tts or PiperTTSPlayer(self.config)

        self.current_state = VartalapState.IDLE
        self.history = ""
        self.agent_last_chunk = ""
        self.agent_turn_accumulator: list[str] = []

        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._playback_finished_event = threading.Event()

        # Hooks for telemetry and UI
        self.on_state_change: Optional[Callable[[VartalapState, VartalapState], None]] = None
        self.on_decision: Optional[Callable[[FLMDecision, VartalapState], None]] = None
        self.on_tts_burst: Optional[Callable[[str], None]] = None

    def set_state(self, new_state: VartalapState):
        if self.current_state != new_state:
            old_state = self.current_state
            self.current_state = new_state
            if self.on_state_change:
                self.on_state_change(old_state, new_state)

    def start(self, mic_device: Optional[int] = None):
        """Starts STT stream and state machine worker loop."""
        self._running = True
        self.stt.start(device=mic_device)
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def stop(self):
        """Stops the state machine and audio streams."""
        self._running = False
        self.tts.stop()
        self.stt.stop()
        self.flm.close()

    def _run_loop(self):
        """Main event-driven loop executing the 4 states."""
        while self._running:
            if self.current_state == VartalapState.IDLE:
                self._handle_idle()
            elif self.current_state == VartalapState.MONITORING:
                self._handle_monitoring()
            elif self.current_state == VartalapState.TALKING:
                self._handle_talking()
            elif self.current_state == VartalapState.ARBITRATION:
                self._handle_arbitration()

    def _handle_idle(self):
        """STATE: IDLE - Zero NPU/LLM inference. Waits for STT word to arrive."""
        try:
            # Wait for at least one word from microphone
            word = self.stt.transcript_queue.get(timeout=0.2)
            if word:
                # User has spoken a word -> transition to MONITORING
                self.set_state(VartalapState.MONITORING)
        except queue.Empty:
            # Remain in IDLE, zero inference load
            pass

    def _handle_monitoring(self):
        """STATE: MONITORING - Evaluates every 500ms tick."""
        t_tick_start = time.perf_counter()

        # Check silence tick status
        silence_ticks = self.stt.check_silence_tick()
        current_partial = self.stt.get_current_partial()

        # Evaluate state with FLM
        decision = self.flm.evaluate(
            history=self.history,
            current_partial_transcript=current_partial,
            silence_tick_count=silence_ticks,
            agent_last_chunk=self.agent_last_chunk,
        )

        if self.on_decision:
            self.on_decision(decision, VartalapState.MONITORING)

        if decision.action == "listen":
            # Model remains quiet, wait for next tick
            elapsed = time.perf_counter() - t_tick_start
            sleep_time = max(0.0, (self.config.tick_interval_ms / 1000.0) - elapsed)
            time.sleep(sleep_time)

        elif decision.action == "talk":
            # Stream 3-4 words to Piper and transition to TALKING
            self.agent_last_chunk = decision.utterance
            self.agent_turn_accumulator.append(decision.utterance)
            self._start_speech_burst(decision.utterance)
            self.set_state(VartalapState.TALKING)

        elif decision.action == "done":
            self._finalize_turn()
            self.set_state(VartalapState.IDLE)

    def _handle_talking(self):
        """STATE: TALKING - Wait for 3-4 words burst playback to finish."""
        # Wait until current burst completes playback or interrupted
        self._playback_finished_event.wait(timeout=3.0)
        self._playback_finished_event.clear()

        # Burst playback is complete -> transition to ARBITRATION
        self.set_state(VartalapState.ARBITRATION)

    def _handle_arbitration(self):
        """STATE: ARBITRATION - Check STT buffer & silence ticks."""
        silence_ticks = self.stt.check_silence_tick()
        current_partial = self.stt.get_current_partial()

        # Prompt Gemma to arbitrate
        decision = self.flm.evaluate(
            history=self.history,
            current_partial_transcript=current_partial,
            silence_tick_count=silence_ticks,
            agent_last_chunk=self.agent_last_chunk,
        )

        if self.on_decision:
            self.on_decision(decision, VartalapState.ARBITRATION)

        if decision.action == "talk":
            # Continue speaking next chunk
            self.agent_last_chunk = decision.utterance
            self.agent_turn_accumulator.append(decision.utterance)
            self._start_speech_burst(decision.utterance)
            self.set_state(VartalapState.TALKING)

        elif decision.action == "listen":
            # User spoke or interrupted, model backs off
            self.set_state(VartalapState.MONITORING)

        elif decision.action == "done":
            # Turn is completely done, sleep to IDLE
            self._finalize_turn()
            self.set_state(VartalapState.IDLE)

    def _start_speech_burst(self, text: str):
        """Initiates Piper TTS playback for 3-4 words."""
        self._playback_finished_event.clear()
        if self.on_tts_burst:
            self.on_tts_burst(text)

        def on_done():
            self._playback_finished_event.set()

        self.tts.play_burst(text, on_complete=on_done)

    def _finalize_turn(self):
        """Commits full exchange into conversation history and clears buffers."""
        user_utterance = self.stt.get_current_partial().strip()
        agent_utterance = " ".join(self.agent_turn_accumulator).strip()

        if user_utterance or agent_utterance:
            turn_text = ""
            if user_utterance:
                turn_text += f"User: {user_utterance}\n"
            if agent_utterance:
                turn_text += f"Agent: {agent_utterance}\n"
            self.history = (self.history + "\n" + turn_text).strip()

        self.agent_last_chunk = ""
        self.agent_turn_accumulator.clear()
        self.stt.reset_transcript()

import enum
import queue
import threading
import time
from typing import Callable, Optional

from .config import VartalapConfig
from .flm_client import FLMClient, FLMDecision
from .tts_player import PiperTTSPlayer


class VartalapState(str, enum.Enum):
    IDLE = "IDLE"
    MONITORING = "MONITORING"
    TALKING = "TALKING"
    ARBITRATION = "ARBITRATION"


class VartalapStateMachine:
    """Core 4-state conversational monitor and turn-taking harness.
    
    Supports both Whisper (FLM NPU) and Sherpa-ONNX STT backends.
    Features instant barge-in (TTS cancellation on user speech) and
    multi-burst continuation for natural flowing responses.
    """

    def __init__(
        self,
        config: Optional[VartalapConfig] = None,
        flm: Optional[FLMClient] = None,
        stt=None,  # WhisperSTTStream or SherpaSTTStream
        tts: Optional[PiperTTSPlayer] = None,
    ):
        self.config = config or VartalapConfig()
        self.flm = flm or FLMClient(self.config)

        # Lazy-load STT based on config
        if stt is not None:
            self.stt = stt
        elif self.config.stt_backend == "whisper":
            from .whisper_stt import WhisperSTTStream
            self.stt = WhisperSTTStream(self.config)
        else:
            from .stt_stream import SherpaSTTStream
            self.stt = SherpaSTTStream(self.config)

        self.tts = tts or PiperTTSPlayer(self.config)

        self.current_state = VartalapState.IDLE
        self.history = ""
        self.agent_last_chunk = ""
        self.agent_turn_accumulator: list[str] = []

        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._playback_finished_event = threading.Event()
        self._barge_in_detected = threading.Event()  # Instant barge-in flag

        # Hooks for telemetry and UI
        self.on_state_change: Optional[Callable[[VartalapState, VartalapState], None]] = None
        self.on_decision: Optional[Callable[[FLMDecision, VartalapState], None]] = None
        self.on_tts_burst: Optional[Callable[[str], None]] = None
        self.on_barge_in: Optional[Callable[[], None]] = None

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
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)
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
        """STATE: MONITORING - Evaluates every tick to decide listen/talk/done."""
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
            # Speak the full response via Piper TTS
            self.agent_last_chunk = decision.utterance
            self.agent_turn_accumulator.append(decision.utterance)
            self._start_speech_burst(decision.utterance)
            self.set_state(VartalapState.TALKING)

        elif decision.action == "done":
            self._finalize_turn()
            self.set_state(VartalapState.IDLE)

        else:
            # Handle error or transient failures
            elapsed = time.perf_counter() - t_tick_start
            sleep_time = max(0.2, (self.config.tick_interval_ms / 1000.0) - elapsed)
            time.sleep(sleep_time)


    def _handle_talking(self):
        """STATE: TALKING - Wait for TTS burst playback. Monitor for barge-in."""
        self._barge_in_detected.clear()

        # Poll for either playback completion or barge-in
        while not self._playback_finished_event.is_set():
            # Check for barge-in: user started speaking during TTS
            if not self.stt.transcript_queue.empty():
                # BARGE-IN! Instantly cancel TTS and go to MONITORING
                self.tts.stop()
                self.stt.mute_mic = False
                self._playback_finished_event.set()
                self._barge_in_detected.set()

                if self.on_barge_in:
                    self.on_barge_in()

                # Finalize what we already said, then go monitor user's new speech
                self._finalize_turn()
                self.set_state(VartalapState.MONITORING)
                return

            time.sleep(0.02)  # 20ms poll interval - responsive but not wasteful

        self._playback_finished_event.clear()

        # Normal completion -> ARBITRATION (check if we should continue talking)
        self.set_state(VartalapState.ARBITRATION)

    def _handle_arbitration(self):
        """STATE: ARBITRATION - Decide whether to continue speaking or yield turn."""
        t_tick_start = time.perf_counter()

        silence_ticks = self.stt.check_silence_tick()
        current_partial = self.stt.get_current_partial()

        # Check if user spoke any NEW words during playback
        user_spoke_during_playback = not self.stt.transcript_queue.empty()

        # If user spoke, immediately go to MONITORING for their turn
        if user_spoke_during_playback:
            self._finalize_turn()
            self.set_state(VartalapState.MONITORING)
            return

        # If user is silent, check if model wants to continue
        decision = self.flm.evaluate(
            history=self.history,
            current_partial_transcript=current_partial,
            silence_tick_count=silence_ticks,
            agent_last_chunk=self.agent_last_chunk,
        )

        if self.on_decision:
            self.on_decision(decision, VartalapState.ARBITRATION)

        if decision.action == "talk":
            new_text = decision.utterance.strip().lower()
            last_text = self.agent_last_chunk.strip().lower()
            already_spoken = [x.strip().lower() for x in self.agent_turn_accumulator]

            # Repetition guard
            if new_text == last_text or new_text in already_spoken:
                decision = FLMDecision(action="done", utterance="", raw_response="<done> (repetition-halt)")
                if self.on_decision:
                    self.on_decision(decision, VartalapState.ARBITRATION)
                self._finalize_turn()
                self.set_state(VartalapState.IDLE)
                return

            # Continue speaking next chunk
            self.agent_last_chunk = decision.utterance
            self.agent_turn_accumulator.append(decision.utterance)
            self._start_speech_burst(decision.utterance)
            self.set_state(VartalapState.TALKING)

        elif decision.action == "listen":
            # User may be about to speak, go back to monitoring
            self.set_state(VartalapState.MONITORING)

        elif decision.action == "done":
            self._finalize_turn()
            self.set_state(VartalapState.IDLE)
        else:
            elapsed = time.perf_counter() - t_tick_start
            sleep_time = max(0.2, (self.config.tick_interval_ms / 1000.0) - elapsed)
            time.sleep(sleep_time)

    def _start_speech_burst(self, text: str):
        """Initiates Piper TTS playback with mic echo muting."""
        self._playback_finished_event.clear()
        self.stt.mute_mic = True
        if self.on_tts_burst:
            self.on_tts_burst(text)

        def on_done():
            time.sleep(0.15)  # Reverb decay grace period
            self.stt.mute_mic = False
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

            # Keep history manageable (last ~10 turns)
            lines = self.history.split("\n")
            if len(lines) > 20:
                self.history = "\n".join(lines[-20:])

        self.agent_last_chunk = ""
        self.agent_turn_accumulator.clear()
        self.stt.reset_transcript()

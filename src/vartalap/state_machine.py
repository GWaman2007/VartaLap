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
    
    Supports Whisper (FLM NPU) and Sherpa-ONNX STT backends.
    Features instant hardware-level duplex barge-in (TTS cancellation
    on user voice activity) and continuous streaming evaluation.
    """

    def __init__(
        self,
        config: Optional[VartalapConfig] = None,
        flm: Optional[FLMClient] = None,
        stt=None,
        tts: Optional[PiperTTSPlayer] = None,
    ):
        self.config = config or VartalapConfig()
        self.flm = flm or FLMClient(self.config)

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
        self._barge_in_detected = threading.Event()

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
            word = self.stt.transcript_queue.get(timeout=0.2)
            if word:
                self.set_state(VartalapState.MONITORING)
        except queue.Empty:
            pass

    def _handle_monitoring(self):
        """STATE: MONITORING - Evaluates streaming audio to decide listen/talk/done."""
        t_tick_start = time.perf_counter()

        silence_ticks = self.stt.check_silence_tick()
        current_partial = self.stt.get_current_partial()

        decision = self.flm.evaluate(
            history=self.history,
            current_partial_transcript=current_partial,
            silence_tick_count=silence_ticks,
            agent_last_chunk=self.agent_last_chunk,
        )

        if self.on_decision:
            self.on_decision(decision, VartalapState.MONITORING)

        if decision.action == "listen":
            elapsed = time.perf_counter() - t_tick_start
            sleep_time = max(0.0, (self.config.tick_interval_ms / 1000.0) - elapsed)
            time.sleep(sleep_time)

        elif decision.action == "talk":
            self.agent_last_chunk = decision.utterance
            self.agent_turn_accumulator.append(decision.utterance)
            self._start_speech_burst(decision.utterance)
            self.set_state(VartalapState.TALKING)

        elif decision.action == "done":
            self._finalize_turn()
            self.set_state(VartalapState.IDLE)

        else:
            elapsed = time.perf_counter() - t_tick_start
            sleep_time = max(0.2, (self.config.tick_interval_ms / 1000.0) - elapsed)
            time.sleep(sleep_time)

    def _handle_talking(self):
        """STATE: TALKING - Monitor TTS playback and detect real-time barge-in."""
        self._barge_in_detected.clear()
        start_time = time.perf_counter()

        # Poll every 15ms for playback completion or instant user speech
        while not self._playback_finished_event.is_set():
            now = time.perf_counter()
            # 250ms grace period after starting playback to prevent DAC transients from self-interrupting
            if (now - start_time) > 0.25:
                # Check 1: User speech energy spike (RMS >= barge_in_rms_threshold)
                energy_barge_in = hasattr(self.stt, "is_user_speaking") and self.stt.is_user_speaking()
                # Check 2: Transcribed words already arrived
                words_barge_in = not self.stt.transcript_queue.empty()

                if energy_barge_in or words_barge_in:
                    # INSTANT BARGE-IN: Stop TTS immediately!
                    self.tts.stop()
                    self._playback_finished_event.set()
                    self._barge_in_detected.set()

                    if self.on_barge_in:
                        self.on_barge_in()

                    self._finalize_turn()
                    self.set_state(VartalapState.MONITORING)
                    return

            time.sleep(0.015)

        self._playback_finished_event.clear()
        self.set_state(VartalapState.ARBITRATION)

    def _handle_arbitration(self):
        """STATE: ARBITRATION - Check if user spoke or if agent should yield turn."""
        t_tick_start = time.perf_counter()

        silence_ticks = self.stt.check_silence_tick()
        current_partial = self.stt.get_current_partial()

        user_spoke = not self.stt.transcript_queue.empty()

        # If user spoke during playback, monitor their new input
        if user_spoke:
            self._finalize_turn()
            self.set_state(VartalapState.MONITORING)
            return

        # If user is silent and we already delivered an answer:
        if self.agent_turn_accumulator:
            decision = FLMDecision(action="done", utterance="", raw_response="<done> (turn-yield)")
            if self.on_decision:
                self.on_decision(decision, VartalapState.ARBITRATION)
            self._finalize_turn()
            self.set_state(VartalapState.IDLE)
            return

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

            if new_text == last_text or new_text in already_spoken:
                decision = FLMDecision(action="done", utterance="", raw_response="<done> (repetition-halt)")
                if self.on_decision:
                    self.on_decision(decision, VartalapState.ARBITRATION)
                self._finalize_turn()
                self.set_state(VartalapState.IDLE)
                return

            self.agent_last_chunk = decision.utterance
            self.agent_turn_accumulator.append(decision.utterance)
            self._start_speech_burst(decision.utterance)
            self.set_state(VartalapState.TALKING)

        elif decision.action == "listen":
            self.set_state(VartalapState.MONITORING)

        elif decision.action == "done":
            self._finalize_turn()
            self.set_state(VartalapState.IDLE)
        else:
            elapsed = time.perf_counter() - t_tick_start
            sleep_time = max(0.2, (self.config.tick_interval_ms / 1000.0) - elapsed)
            time.sleep(sleep_time)

    def _start_speech_burst(self, text: str):
        """Initiates Piper TTS playback with duplex echo control."""
        self._playback_finished_event.clear()
        if self.config.echo_suppression:
            self.stt.mute_mic = True

        if self.on_tts_burst:
            self.on_tts_burst(text)

        def on_done():
            if self.config.echo_suppression:
                time.sleep(0.12)
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

            lines = self.history.split("\n")
            if len(lines) > 20:
                self.history = "\n".join(lines[-20:])

        self.agent_last_chunk = ""
        self.agent_turn_accumulator.clear()
        self.stt.reset_transcript()

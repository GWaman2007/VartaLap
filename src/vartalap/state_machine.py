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
    
    1. IDLE: Zero inference load, waits for speech.
    2. MONITORING: Evaluates continuous streaming audio every tick.
    3. TALKING: Plays complete TTS response cleanly without interruptions.
    4. ARBITRATION: Manages turn transitions cleanly back to IDLE.
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
            from .faster_whisper_stt import FasterWhisperSTTStream
            self.stt = FasterWhisperSTTStream(self.config)
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
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)
        self.tts.stop()
        self.stt.stop()
        self.flm.close()

    def _run_loop(self):
        """Main event-driven loop executing the 4 states."""
        while self._running:
            try:
                if self.current_state == VartalapState.IDLE:
                    self._handle_idle()
                elif self.current_state == VartalapState.MONITORING:
                    self._handle_monitoring()
                elif self.current_state == VartalapState.TALKING:
                    self._handle_talking()
                elif self.current_state == VartalapState.ARBITRATION:
                    self._handle_arbitration()
            except Exception as e:
                print(f"[State Machine Loop Error]: {e}")
                time.sleep(0.1)

    def _handle_idle(self):
        """STATE: IDLE - Zero NPU/LLM inference. Waits for STT word to arrive."""
        try:
            word = self.stt.transcript_queue.get(timeout=0.2)
            if word:
                self.set_state(VartalapState.MONITORING)
        except queue.Empty:
            pass

    def _handle_monitoring(self):
        """STATE: MONITORING - Evaluates streaming audio and triggers burst pieces immediately."""
        t_tick_start = time.perf_counter()

        silence_ticks = self.stt.check_silence_tick()
        current_partial = self.stt.get_current_partial()

        # Stream pieces directly to TTS playback as soon as each sentence/clause is ready
        def on_burst_piece(piece: str):
            clean = piece.strip()
            if clean:
                self.agent_turn_accumulator.append(clean)
                self._start_speech_burst(clean)
                self.set_state(VartalapState.TALKING)

        decision = self.flm.evaluate_stream(
            history=self.history,
            current_partial_transcript=current_partial,
            silence_tick_count=silence_ticks,
            agent_last_chunk=self.agent_last_chunk,
            on_piece=on_burst_piece,
        )

        if self.on_decision:
            self.on_decision(decision, VartalapState.MONITORING)

        if decision.action == "listen":
            elapsed = time.perf_counter() - t_tick_start
            sleep_time = max(0.0, (self.config.tick_interval_ms / 1000.0) - elapsed)
            time.sleep(sleep_time)

        elif decision.action == "talk":
            self.agent_last_chunk = decision.utterance
            # If piece callback didn't trigger, play complete utterance
            if not self.agent_turn_accumulator and decision.utterance:
                self.agent_turn_accumulator.append(decision.utterance)
                self._start_speech_burst(decision.utterance)
            self.set_state(VartalapState.TALKING)

        elif decision.action == "done":
            # If the user spoke a query and agent has not answered yet, keep listening
            if not self.agent_turn_accumulator and current_partial.strip():
                elapsed = time.perf_counter() - t_tick_start
                sleep_time = max(0.1, (self.config.tick_interval_ms / 1000.0) - elapsed)
                time.sleep(sleep_time)
            else:
                self._finalize_turn()
                self.set_state(VartalapState.IDLE)

        else:
            elapsed = time.perf_counter() - t_tick_start
            sleep_time = max(0.2, (self.config.tick_interval_ms / 1000.0) - elapsed)
            time.sleep(sleep_time)

    def _handle_talking(self):
        """STATE: TALKING - Play TTS response cleanly to completion without interruptions."""
        # Wait for speech playback queue to drain and finish cleanly
        while self.tts.is_playing:
            time.sleep(0.05)
        self._playback_finished_event.wait(timeout=1.0)
        self._playback_finished_event.clear()

        # Playback is complete -> transition to ARBITRATION
        self.set_state(VartalapState.ARBITRATION)

    def _handle_arbitration(self):
        """STATE: ARBITRATION - Finalize turn and transition cleanly back to IDLE."""
        # Turn yielded cleanly to IDLE
        decision = FLMDecision(action="done", utterance="", raw_response="<done> (turn-yield)")
        if self.on_decision:
            self.on_decision(decision, VartalapState.ARBITRATION)
        self._finalize_turn()
        self.set_state(VartalapState.IDLE)

    def _start_speech_burst(self, text: str):
        """Initiates Piper TTS playback with echo suppression."""
        self._playback_finished_event.clear()
        # Suppress mic during playback to prevent echo / self-listening
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

            lines = self.history.split("\n")
            if len(lines) > 20:
                self.history = "\n".join(lines[-20:])

        self.agent_last_chunk = ""
        self.agent_turn_accumulator.clear()
        self.stt.reset_transcript()

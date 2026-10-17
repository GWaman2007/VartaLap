import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import sherpa_onnx

try:
    import sounddevice as sd
except ImportError:
    sd = None

from .config import VartalapConfig


class SherpaSTTStream:
    """Streaming Zipformer speech recognition using sherpa-onnx on a background CPU thread.
    
    Decodes continuous 100ms audio chunks in ~17ms on CPU, leaving the Snapdragon NPU
    100% free for FastFlowLM Gemma evaluation with zero mutex contention.
    """

    def __init__(self, config: Optional[VartalapConfig] = None):
        self.config = config or VartalapConfig()
        self.transcript_queue: queue.Queue[str] = queue.Queue()

        self._recognizer: Optional[sherpa_onnx.OnlineRecognizer] = None
        self._stream: Optional[sherpa_onnx.OnlineStream] = None
        self._load_recognizer()

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._input_stream = None

        self._current_partial = ""
        self._last_emitted_text = ""
        self._words_arrived_in_window = False
        self.silence_tick_count = 0
        self.last_audio_rms = 0.0
        self._recent_rms: list[float] = []
        self.mute_mic = False
        self._lock = threading.Lock()

        # Callbacks
        self.on_partial_transcript: Optional[Callable[[str], None]] = None
        self.on_energy: Optional[Callable[[float], None]] = None

        self._worker_thread: Optional[threading.Thread] = None

    def _load_recognizer(self):
        model_dir = self.config.stt_model_dir
        tokens = str(model_dir / "tokens.txt")
        encoder = str(model_dir / "encoder-epoch-99-avg-1.onnx")
        decoder = str(model_dir / "decoder-epoch-99-avg-1.onnx")
        joiner = str(model_dir / "joiner-epoch-99-avg-1.onnx")

        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=2,
            sample_rate=self.config.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        self._stream = self._recognizer.create_stream()

    def start(self, device: Optional[int] = None):
        """Starts real-time microphone capture and background STT decoding."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._decoding_worker, daemon=True)
        self._worker_thread.start()

        if sd is not None:
            def audio_callback(indata, frames, time_info, status):
                if not self._running:
                    return

                # If echo suppression enabled and mic is muted, drop frames
                if self.config.echo_suppression and self.mute_mic:
                    return

                samples = indata[:, 0].copy()
                rms = float(np.sqrt(np.mean(samples**2)))
                self.last_audio_rms = rms

                with self._lock:
                    self._recent_rms.append(rms)
                    if len(self._recent_rms) > 5:
                        self._recent_rms.pop(0)

                if self.on_energy:
                    self.on_energy(rms)

                self._audio_queue.put(samples)

            self._input_stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="float32",
                blocksize=self.config.chunk_size_samples,
                device=device,
                callback=audio_callback,
            )
            self._input_stream.start()

    def is_user_speaking(self) -> bool:
        """Returns True if recent mic audio indicates user is speaking."""
        with self._lock:
            if not self._recent_rms:
                return False
            speaking_frames = sum(1 for r in self._recent_rms if r >= self.config.barge_in_rms_threshold)
            return speaking_frames >= 2 or (self._recent_rms[-1] >= self.config.barge_in_rms_threshold * 1.5)

    def feed_audio_chunk(self, chunk: np.ndarray):
        """Manually feed audio samples (for testing / mocking)."""
        self._audio_queue.put(chunk)

    def stop(self):
        self._running = False
        if self._input_stream:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None

    def _decoding_worker(self):
        while self._running:
            try:
                samples = self._audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if self._stream is None or self._recognizer is None:
                continue

            self._stream.accept_waveform(self.config.sample_rate, samples)

            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)

            result = self._recognizer.get_result(self._stream)
            if isinstance(result, str):
                text = result.strip()
            elif hasattr(result, "text"):
                text = result.text.strip()
            else:
                text = str(result).strip()

            if text and text != self._last_emitted_text:
                with self._lock:
                    self._current_partial = text
                    self._words_arrived_in_window = True
                    old_words = self._last_emitted_text.split()
                    new_words = text.split()
                    if len(new_words) > len(old_words):
                        for word in new_words[len(old_words):]:
                            self.transcript_queue.put(word)
                    elif not old_words:
                        for word in new_words:
                            self.transcript_queue.put(word)

                self._last_emitted_text = text

                if self.on_partial_transcript:
                    self.on_partial_transcript(text)

    def check_silence_tick(self) -> int:
        """Called every tick. Increments silence count if no new words arrived."""
        with self._lock:
            if self._words_arrived_in_window:
                self.silence_tick_count = 0
                self._words_arrived_in_window = False
            else:
                self.silence_tick_count += 1
            return self.silence_tick_count

    def get_current_partial(self) -> str:
        with self._lock:
            return self._current_partial

    def reset_transcript(self):
        """Resets partial transcript when turn ends."""
        with self._lock:
            self._current_partial = ""
            self._last_emitted_text = ""
            self.silence_tick_count = 0
            self._words_arrived_in_window = False
            if self._recognizer and self._stream:
                self._recognizer.reset(self._stream)
            while not self.transcript_queue.empty():
                try:
                    self.transcript_queue.get_nowait()
                except Exception:
                    break

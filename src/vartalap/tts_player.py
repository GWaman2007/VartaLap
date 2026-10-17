import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
from piper import PiperVoice

try:
    import sounddevice as sd
except ImportError:
    sd = None

from .config import VartalapConfig


class PiperTTSPlayer:
    """Ultra-low latency burst TTS player using Piper ONNX neural synthesis and sounddevice."""

    def __init__(self, config: Optional[VartalapConfig] = None):
        self.config = config or VartalapConfig()
        self.voice: Optional[PiperVoice] = None
        self._load_voice()

        self._playback_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._is_playing = threading.Event()
        self._worker_thread = threading.Thread(target=self._player_worker, daemon=True)
        self._worker_thread.start()

    def _load_voice(self):
        onnx_path = str(self.config.tts_model_path)
        config_path = str(self.config.tts_config_path)
        self.voice = PiperVoice.load(onnx_path, config_path=config_path)

    @property
    def sample_rate(self) -> int:
        return self.voice.config.sample_rate if self.voice else 22050

    def play_burst(self, text: str, on_complete: Optional[Callable[[], None]] = None):
        """Synthesize and queue a 3-4 word burst for playback."""
        clean_text = text.strip()
        if not clean_text:
            if on_complete:
                on_complete()
            return

        self._stop_event.clear()
        self._playback_queue.put((clean_text, on_complete))

    def stop(self):
        """Interrupt and clear any active playback."""
        self._stop_event.set()
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
                self._playback_queue.task_done()
            except Exception:
                break
        self._is_playing.clear()

    @property
    def is_playing(self) -> bool:
        return self._is_playing.is_set()

    def _player_worker(self):
        while True:
            try:
                item = self._playback_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            text, on_complete = item
            self._is_playing.set()

            try:
                if self.voice and not self._stop_event.is_set():
                    audio_arrays = []
                    for chunk in self.voice.synthesize(text):
                        if self._stop_event.is_set():
                            break
                        if hasattr(chunk, "audio_float_array") and chunk.audio_float_array is not None:
                            audio_arrays.append(chunk.audio_float_array)

                    if audio_arrays and not self._stop_event.is_set():
                        samples = np.concatenate(audio_arrays)

                        if sd is not None and not self._stop_event.is_set():
                            # Play through sounddevice output stream
                            sd.play(samples, samplerate=self.sample_rate)
                            duration = len(samples) / self.sample_rate
                            start_time = time.perf_counter()
                            while (time.perf_counter() - start_time) < duration:
                                if self._stop_event.is_set():
                                    sd.stop()
                                    break
                                time.sleep(0.02)
                            sd.stop()

            except Exception as e:
                print(f"[TTS Player Error]: {e}")
            finally:
                self._is_playing.clear()
                self._playback_queue.task_done()
                if on_complete and not self._stop_event.is_set():
                    on_complete()

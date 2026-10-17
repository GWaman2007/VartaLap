"""Whisper-V3-Turbo STT via FLM's /v1/audio/transcriptions endpoint.

Uses energy-based VAD for speech boundary detection, sends completed utterances
to Whisper-V3 Turbo on the NPU, and supports instant duplex barge-in detection.
"""

import io
import queue
import struct
import threading
import time
from typing import Callable, Optional

import httpx
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from .config import VartalapConfig


class WhisperSTTStream:
    """Streaming microphone capture with VAD-chunked Whisper transcription and duplex barge-in."""

    def __init__(self, config: Optional[VartalapConfig] = None):
        self.config = config or VartalapConfig()
        self.transcript_queue: queue.Queue[str] = queue.Queue()

        # VAD Parameters from config
        self.speech_threshold = self.config.barge_in_rms_threshold
        self.silence_duration_s = self.config.silence_duration_s
        self.min_speech_duration_s = self.config.min_speech_duration_s

        # Audio capture state
        self._audio_buffer: list[np.ndarray] = []
        self._speech_started = False
        self._speech_start_time = 0.0
        self._last_speech_time = 0.0
        self._running = False
        self._input_stream = None

        # Duplex barge-in tracking
        self._recent_rms: list[float] = []
        self.last_audio_rms = 0.0
        self.mute_mic = False

        # Transcription state
        self._current_partial = ""
        self._last_emitted_text = ""
        self._words_arrived_in_window = False
        self.silence_tick_count = 0
        self._lock = threading.Lock()

        # Whisper HTTP client
        self._whisper_url = f"{self.config.flm_base_url}/audio/transcriptions"
        self._http_client = httpx.Client(timeout=httpx.Timeout(10.0, connect=3.0))

        # Background transcription worker
        self._transcribe_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_partial_transcript: Optional[Callable[[str], None]] = None
        self.on_energy: Optional[Callable[[float], None]] = None

    def start(self, device: Optional[int] = None):
        """Starts real-time microphone capture and background Whisper transcription."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self._worker_thread.start()

        if sd is not None:
            def audio_callback(indata, frames, time_info, status):
                if not self._running:
                    return

                # Echo suppression: drop frames if mic is explicitly muted during TTS
                if getattr(self.config, "echo_suppression", True) and self.mute_mic:
                    return

                samples = indata[:, 0].copy()
                rms = float(np.sqrt(np.mean(samples**2)))
                self.last_audio_rms = rms

                # Track recent RMS for instantaneous barge-in checks
                with self._lock:
                    self._recent_rms.append(rms)
                    if len(self._recent_rms) > 5:
                        self._recent_rms.pop(0)

                if self.on_energy:
                    self.on_energy(rms)

                self._process_audio_chunk(samples, rms)

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
        """Instantaneous energy-based check: returns True if recent audio indicates user is speaking."""
        with self._lock:
            if not self._recent_rms:
                return False
            # If at least 2 recent 100ms frames exceed barge-in threshold, user is actively speaking
            speaking_frames = sum(1 for r in self._recent_rms if r >= self.speech_threshold)
            return speaking_frames >= 2 or (self._recent_rms[-1] >= self.speech_threshold * 1.5)

    def _process_audio_chunk(self, samples: np.ndarray, rms: float):
        """VAD logic: detect speech start/end and queue completed utterances."""
        now = time.perf_counter()
        is_speech = rms >= self.speech_threshold

        if is_speech:
            if not self._speech_started:
                self._speech_started = True
                self._speech_start_time = now
                self._audio_buffer = []
            self._last_speech_time = now
            self._audio_buffer.append(samples)

        elif self._speech_started:
            self._audio_buffer.append(samples)
            speech_duration = now - self._speech_start_time
            silence_duration = now - self._last_speech_time

            # End of utterance: silence exceeded threshold
            if silence_duration >= self.silence_duration_s:
                if speech_duration >= self.min_speech_duration_s and self._audio_buffer:
                    full_audio = np.concatenate(self._audio_buffer)
                    self._transcribe_queue.put(full_audio)
                self._speech_started = False
                self._audio_buffer = []

            # Force-end extremely long utterances (over 12 seconds)
            elif speech_duration >= 12.0:
                if self._audio_buffer:
                    full_audio = np.concatenate(self._audio_buffer)
                    self._transcribe_queue.put(full_audio)
                self._speech_started = False
                self._audio_buffer = []

    def _transcription_worker(self):
        """Background thread that sends audio chunks to Whisper for transcription."""
        while self._running:
            try:
                audio = self._transcribe_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                text = self._transcribe_with_whisper(audio)
                if text:
                    with self._lock:
                        self._current_partial = text
                        self._words_arrived_in_window = True

                    # Emit individual words to transcript queue
                    words = text.split()
                    for word in words:
                        self.transcript_queue.put(word)

                    if self.on_partial_transcript:
                        self.on_partial_transcript(text)

            except Exception as e:
                print(f"[Whisper STT Error] {e}")

    def _transcribe_with_whisper(self, audio: np.ndarray) -> str:
        """Send audio to FLM's Whisper /v1/audio/transcriptions endpoint."""
        wav_bytes = self._audio_to_wav_bytes(audio)

        try:
            response = self._http_client.post(
                self._whisper_url,
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model": "whisper-v3:turbo"},
            )
            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception:
                    return ""

                if not isinstance(result, dict):
                    return ""

                text = (result.get("text", "") or "").strip()
                if not text:
                    return ""

                # Clean unicode replacement characters
                text = text.replace("\ufffd", "").strip()

                # Filter out repetitive character/word hallucinations (e.g. "E a E a E a")
                words = text.split()
                if len(words) >= 4 and len(set(words)) <= 2:
                    return ""

                # Filter common Whisper silence hallucinations
                hallucinations = {
                    "you", ".", "...", "thank you.", "thank you",
                    "thanks for watching!", "thank you for watching.",
                    "subscribe", "bye.", "bye", "e", "a", "so", "oh",
                }
                if text.lower() in hallucinations:
                    return ""

                return text
            else:
                return ""
        except Exception:
            return ""

    def _audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert float32 numpy audio array to 16kHz 16-bit mono WAV bytes."""
        audio_clamped = np.clip(audio, -1.0, 1.0)
        pcm_data = (audio_clamped * 32767).astype(np.int16)

        num_samples = len(pcm_data)
        data_size = num_samples * 2
        sample_rate = self.config.sample_rate
        channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8

        buf = io.BytesIO()
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))
        buf.write(struct.pack("<H", 1))
        buf.write(struct.pack("<H", channels))
        buf.write(struct.pack("<I", sample_rate))
        buf.write(struct.pack("<I", byte_rate))
        buf.write(struct.pack("<H", block_align))
        buf.write(struct.pack("<H", bits_per_sample))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(pcm_data.tobytes())

        return buf.getvalue()

    def feed_audio_chunk(self, chunk: np.ndarray):
        """Manually feed audio samples (for testing / mocking)."""
        rms = float(np.sqrt(np.mean(chunk**2)))
        self._process_audio_chunk(chunk, rms)

    def stop(self):
        self._running = False
        if self._input_stream:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None
        try:
            self._http_client.close()
        except Exception:
            pass

    def check_silence_tick(self) -> int:
        """Called on tick. Increments silence count if no new words arrived."""
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
            while not self.transcript_queue.empty():
                try:
                    self.transcript_queue.get_nowait()
                except Exception:
                    break

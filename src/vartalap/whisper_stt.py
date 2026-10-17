"""Whisper-V3-Turbo STT via FLM's /v1/audio/transcriptions endpoint.

Instead of the tiny 20M Zipformer, this sends audio chunks to the Whisper-V3
Turbo model running on the NPU via FLM. Uses simple energy-based VAD to
detect speech boundaries and sends completed utterances for transcription.
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
    """Streaming microphone capture with VAD-chunked Whisper transcription."""

    # VAD Parameters
    SPEECH_THRESHOLD = 0.02        # RMS threshold to consider as speech
    SILENCE_DURATION_S = 0.6       # Seconds of silence to end an utterance
    MIN_SPEECH_DURATION_S = 0.3    # Minimum speech duration to transcribe
    MAX_SPEECH_DURATION_S = 15.0   # Maximum single utterance duration

    def __init__(self, config: Optional[VartalapConfig] = None):
        self.config = config or VartalapConfig()
        self.transcript_queue: queue.Queue[str] = queue.Queue()

        # Audio capture state
        self._audio_buffer: list[np.ndarray] = []
        self._speech_started = False
        self._speech_start_time = 0.0
        self._last_speech_time = 0.0
        self._running = False
        self._input_stream = None

        # Transcription state
        self._current_partial = ""
        self._last_emitted_text = ""
        self._words_arrived_in_window = False
        self.silence_tick_count = 0
        self.last_audio_rms = 0.0
        self.mute_mic = False
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

                # Echo suppression: ignore audio while TTS is actively speaking
                if self.mute_mic:
                    return

                samples = indata[:, 0].copy()
                rms = float(np.sqrt(np.mean(samples**2)))
                self.last_audio_rms = rms

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

    def _process_audio_chunk(self, samples: np.ndarray, rms: float):
        """VAD logic: detect speech start/end and queue completed utterances."""
        now = time.perf_counter()
        is_speech = rms > self.SPEECH_THRESHOLD

        if is_speech:
            if not self._speech_started:
                # Speech just started
                self._speech_started = True
                self._speech_start_time = now
                self._audio_buffer = []
            self._last_speech_time = now
            self._audio_buffer.append(samples)

        elif self._speech_started:
            # Still collecting audio during short silence gaps within speech
            self._audio_buffer.append(samples)

            speech_duration = now - self._speech_start_time
            silence_duration = now - self._last_speech_time

            # End of utterance: silence exceeded threshold
            if silence_duration >= self.SILENCE_DURATION_S:
                if speech_duration >= self.MIN_SPEECH_DURATION_S and self._audio_buffer:
                    # Concatenate all audio and send to Whisper
                    full_audio = np.concatenate(self._audio_buffer)
                    self._transcribe_queue.put(full_audio)
                self._speech_started = False
                self._audio_buffer = []

            # Force-end extremely long utterances
            elif speech_duration >= self.MAX_SPEECH_DURATION_S:
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
        # Convert float32 audio to 16-bit PCM WAV bytes
        wav_bytes = self._audio_to_wav_bytes(audio)

        try:
            response = self._http_client.post(
                self._whisper_url,
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={
                    "model": "whisper-v3:turbo",
                    "language": "en",
                    "response_format": "json",
                },
            )
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                # Filter out common Whisper hallucinations on silence
                if text.lower() in ("", "you", ".", "...", "thank you.",
                                     "thanks for watching!", "thank you for watching.",
                                     "subscribe", "bye."):
                    return ""
                return text
            else:
                print(f"[Whisper HTTP {response.status_code}] {response.text[:200]}")
                return ""
        except Exception as e:
            print(f"[Whisper Request Error] {e}")
            return ""

    def _audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert float32 numpy audio array to WAV file bytes."""
        # Clamp and convert to int16
        audio_clamped = np.clip(audio, -1.0, 1.0)
        pcm_data = (audio_clamped * 32767).astype(np.int16)

        # Build WAV header manually (simpler than importing wave)
        num_samples = len(pcm_data)
        data_size = num_samples * 2  # 16-bit = 2 bytes per sample
        sample_rate = self.config.sample_rate
        channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8

        buf = io.BytesIO()
        # RIFF header
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        # fmt chunk
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))  # chunk size
        buf.write(struct.pack("<H", 1))   # PCM format
        buf.write(struct.pack("<H", channels))
        buf.write(struct.pack("<I", sample_rate))
        buf.write(struct.pack("<I", byte_rate))
        buf.write(struct.pack("<H", block_align))
        buf.write(struct.pack("<H", bits_per_sample))
        # data chunk
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
            while not self.transcript_queue.empty():
                try:
                    self.transcript_queue.get_nowait()
                except Exception:
                    break

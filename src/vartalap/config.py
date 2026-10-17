from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VartalapConfig:
    # FastFlowLM Settings
    flm_base_url: str = "http://127.0.0.1:52625/v1"
    model_name: str = "gemma4-it:e2b"
    max_tokens: int = 35  # Natural 12-25 word conversational answers
    temperature: float = 0.0  # Deterministic, lowest latency

    # STT Backend: "whisper" (faster-whisper, 220ms, recommended) or "sherpa" (Zipformer)
    stt_backend: str = "whisper"
    whisper_model_size: str = "tiny.en"
    echo_suppression: bool = True

    # State Machine Timing
    tick_interval_ms: int = 400  # 400ms evaluation ticks
    silence_threshold_ticks: int = 2

    # Audio & STT Settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size_samples: int = 1600  # 100ms audio chunks (1600 samples at 16kHz)

    # Models Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    @property
    def stt_model_dir(self) -> Path:
        return self.base_dir / "models" / "stt" / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

    @property
    def tts_model_path(self) -> Path:
        return self.base_dir / "models" / "tts" / "en_US-lessac-medium.onnx"

    @property
    def tts_config_path(self) -> Path:
        return self.base_dir / "models" / "tts" / "en_US-lessac-medium.onnx.json"

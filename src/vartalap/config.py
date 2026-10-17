from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VartalapConfig:
    # FastFlowLM Settings
    flm_base_url: str = "http://127.0.0.1:52625/v1"
    model_name: str = "gemma4-it:e2b"
    max_tokens: int = 16
    temperature: float = 0.1

    # State Machine Timing
    tick_interval_ms: int = 500  # 500ms evaluation ticks
    silence_threshold_ticks: int = 2  # 2 ticks (1000ms) silence for <done> or turn yield

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

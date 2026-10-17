"""
Model downloader for Vartalap:
- sherpa-onnx streaming Zipformer (20M parameter real-time model)
- Piper TTS voice model (en_US-lessac-medium)
"""

import os
import tarfile
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
STT_DIR = MODELS_DIR / "stt"
TTS_DIR = MODELS_DIR / "tts"

STT_TAR_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
STT_MODEL_NAME = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

PIPER_ONNX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"


def download_file(url: str, dest: Path, desc: str):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[OK] {desc} already exists at {dest}")
        return
    print(f"[INFO] Downloading {desc} from {url}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"[OK] Downloaded {desc} ({dest.stat().st_size // 1024} KB)")


def setup_stt():
    STT_DIR.mkdir(parents=True, exist_ok=True)
    target_folder = STT_DIR / STT_MODEL_NAME
    if target_folder.exists() and (target_folder / "tokens.txt").exists():
        print(f"[OK] STT model already prepared at {target_folder}")
        return target_folder

    archive_path = STT_DIR / "stt_model.tar.bz2"
    download_file(STT_TAR_URL, archive_path, "Sherpa-ONNX Zipformer archive")

    print("[INFO] Extracting STT archive...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(path=STT_DIR)
    print(f"[OK] STT model extracted to {target_folder}")

    if archive_path.exists():
        archive_path.unlink()
    return target_folder


def setup_tts():
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_file = TTS_DIR / "en_US-lessac-medium.onnx"
    json_file = TTS_DIR / "en_US-lessac-medium.onnx.json"

    download_file(PIPER_ONNX_URL, onnx_file, "Piper ONNX model")
    download_file(PIPER_JSON_URL, json_file, "Piper model JSON config")
    return onnx_file, json_file


if __name__ == "__main__":
    print("=== Downloading Vartalap Audio Models ===")
    setup_stt()
    setup_tts()
    print("=== Model Setup Complete! ===")


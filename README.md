# 🎙️ Vartalap (वार्तालाप)

**Ultra-Low Latency Real-Time Conversational Voice Assistant & Duplex Turn Monitor**

Vartalap is a voice turn-taking and real-time conversational monitor designed for ultra-low latency conversational voice interaction on edge devices (AMD Ryzen AI NPU + Zen 5 CPU).

By coupling **Faster-Whisper (INT8 CPU via CTranslate2, ~150–220ms)** with **FastFlowLM (`gemma4-it:e2b` on Ryzen AI NPU)** and **Piper Neural TTS**, Vartalap achieves instant conversational responsiveness with zero pauses and zero NPU compute overhead while idle.

---

## 🏛️ Architecture & Hardware Distribution

| Component | Engine | Hardware Tier | Latency | Role |
| :--- | :--- | :--- | :--- | :--- |
| **STT** | Faster-Whisper (`tiny.en`, INT8) | AMD Zen 5 CPU (4 threads) | ~150–220ms | Continuous rolling transcription every 400ms |
| **LLM** | FastFlowLM (`gemma4-it:e2b`) | AMD Ryzen AI NPU (port 52625) | ~1.1s TTFT | Real-time turn evaluation & streaming pieces |
| **TTS** | Piper ONNX Neural TTS (`lessac`) | CPU Neural Synthesis | ~40ms burst | Clause/sentence burst playback |
| **UI** | FastAPI + WebSockets + Canvas Orb | Browser / Localhost | < 5ms sync | Procedural voice orb, pipeline flow, audio selectors |

---

## ⚡ 4-State Turn Pipeline

```text
                         ┌─────────────────────────────┐
                         │         STATE: IDLE         │
                         │ (0% NPU / LLM inference)    │
                         │ Waiting for first speech    │
                         └──────────────┬──────────────┘
                                        │
                         [Speech detected / 400ms chunk]
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      STATE: MONITORING      │
                         │ Continuous Whisper stream   │
                         │ FastFlowLM NPU evaluation   │
                         └──────┬───────────────┬──────┘
                                │               │
                    Sentence Unfinished   Complete Query
                                │               │
                                ▼               ▼
                             <listen>        <talk>
                         (Keep listening)       │
                                                ▼
                         ┌─────────────────────────────┐
                         │       STATE: TALKING        │
                         │  Stream clause 1 to Piper   │
                         │  Piper speaks while clause  │
                         │  2 generates on NPU         │
                         └──────────────┬──────────────┘
                                        │
                               [Playback finishes]
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │     STATE: ARBITRATION      │
                         │ Turn yield check & clean    │
                         │ commit to history           │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                                  Return to IDLE
```

---

## 🎯 Conversational Rules

Decoding is strictly governed by stateful conversational rules:
1. **Safety First:** If dangerous actions are detected (e.g. live wires, chemical reactions, self-harm), outputs `<talk>` with an immediate safety warning.
2. **Unfinished / Incomplete Speech:** If the user utterance is incomplete, trailing off, or missing a main clause (*"so like"*, *"rice in relation to the"*, *"and then"*), outputs ONLY `<listen>`.
3. **Greetings & Audio Checks:** If the user says *"hello"*, *"can you hear me?"*, or *"are you there?"*, outputs `<talk>` with an immediate warm confirmation.
4. **Complete Questions & Commands:** Once the query is complete, outputs `<talk>` followed by a direct, concise 10–25 word answer.
5. **Turn Completion:** Yields with `<done>` only after the full answer has been spoken and the user is silent.

---

## 🚀 Quickstart

### 1. Prerequisites
- Windows 11 with AMD Ryzen AI processor (or modern CPU).
- Python 3.10+ (managed via `.venv` or `uv`).
- FastFlowLM installed:
  ```powershell
  flm pull gemma4-it:e2b
  flm serve gemma4-it:e2b -s 50
  ```

### 2. Launch Vartalap Modern Web Dashboard
```powershell
.\run.ps1 --ui
```
- Automatically opens **http://127.0.0.1:8000**.
- Features an interactive **Procedural Canvas Voice Orb**, real-time audio VU meters, latency telemetry, pipeline visualizer, and microphone/headphone selectors.
- Click **▶ START VARTALAP** to begin duplex voice conversation.

### 3. Launch Terminal CLI
```powershell
# List available audio devices
.\run.ps1 --list-devices

# Run with specific microphone and output device index
.\run.ps1 --mic 2 --out 4
```

---

## 🧪 Testing

Run the full pytest suite:
```powershell
pytest tests/
```
All unit tests verify state transitions, streaming piece generation, and turn-yield protections.

# 🎙️ Vartalap (वार्तालाप)

**Ultra-Low Latency Real-Time Conversational Monitor & Turn-Taking Harness**

Vartalap is a voice turn-taking and conversational monitoring architecture designed for instant conversational awareness. By coupling **NPU-accelerated edge language models (FastFlowLM / Gemma)** with streaming speech-to-text (**sherpa-onnx Zipformer**) and lightweight streaming neural speech synthesis (**Piper TTS**), Vartalap achieves sub-80ms decision latency with minimal compute overhead when idle.

---

## 🏛️ State Machine Architecture

Vartalap operates on an event-driven 4-state finite state machine with zero NPU/LLM inference while dormant.

```
                         ┌─────────────────────────────┐
                         │         STATE: IDLE         │
                         │ (Zero NPU/LLM inference)    │
                         │ Harness waits for STT word  │
                         └──────────────┬──────────────┘
                                        │
                               [User speaks a word]
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      STATE: MONITORING      │
                         │ Evaluates at 500ms ticks    │
                         └──────┬───────────────┬──────┘
                                │               │
                        User Speaking    Silence x2 Ticks
                                │               │
                     ┌──────────┴──────┐        │
                     ▼                 ▼        ▼
                 <listen>           <talk>   <talk>
                 (Remain quiet)        │        │
                                       ▼        ▼
                         ┌─────────────────────────────┐
                         │       STATE: TALKING        │
                         │  Stream 3-4 words to Piper  │
                         └──────────────┬──────────────┘
                                        │
                         [3-4 words playback complete]
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │    STATE: ARBITRATION       │
                         │  Check STT buffer & silence │
                         └──────┬───────────────┬──────┘
                                │               │
                    User spoke / Unfinished   Model finished &
                                │             User silent x2 ticks
                                ▼               │
                       Gemma evaluates:         ▼
                      <talk> or <listen>     <done>
                                                │
                                                ▼
                                         Return to IDLE
```

---

## ⚡ Core Philosophy & States

1. **`IDLE` (Dormant Sleep):**
   - Zero NPU / LLM generation.
   - Streaming Zipformer STT runs on a background thread monitoring microphone audio.
   - The harness transitions to `MONITORING` immediately when the first spoken word is detected.

2. **`MONITORING` (500ms Tick Evaluation):**
   - Ticks every 500ms.
   - Evaluates partial STT transcripts with FastFlowLM (FLM) under strict regex constraints (`^(<listen>|<talk>|<done>)`).
   - If output is `<listen>`: Discard token, remain quiet, wait for the next tick (single-token escape latency $< 80\text{ ms}$).
   - If output is `<talk>`: Collect next 3–4 tokens, stream them to Piper TTS, transition to `TALKING`.

3. **`TALKING` (Burst Speech Output):**
   - Synthesize and stream 3–4 words to the audio output device.
   - STT audio capture continues concurrently in the background.
   - Transitions to `ARBITRATION` upon completing playback of the burst.

4. **`ARBITRATION` (Interruption & Yield Check):**
   - Checks if user spoke during playback:
     - User spoke / unfinished thought: Append words to context, query Gemma whether to continue `<talk>` or back off with `<listen>`.
     - Model finished and user silent for 2 ticks: Gemma outputs `<done>` to yield turn and return to `IDLE`.

---

## 🎯 Prompt Template & Regex Constraint

Decoding is constrained to the regex: `^(<listen>|<talk>|<done>)`

```text
You are Vartalap, an ultra-low latency real-time conversational monitor.
Rules:
1. If the user is speaking correctly mid-thought, output: <listen>
2. If you need to speak (answer or correct), output: <talk> followed by at most 4 words.
3. If you have completely finished speaking and the user has been silent for 2 turns, output: <done>

Conversation History:
{history}

[Current State]
User Audio: "{current_partial_transcript}"
User Silence Ticks: {silence_tick_count}
Your Last Partial Utterance: "{agent_last_chunk}"

Output decision:
```

---

## 📦 Tech Stack & Requirements

- **Inference Runtime:** [FastFlowLM (FLM)](https://github.com/) serving NPU-optimized models (e.g., `gemma4-it:e2b`, `qwen3.5:4b`) on port `52625`.
- **Speech-to-Text (STT):** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with streaming Zipformer transducer.
- **Text-to-Speech (TTS):** [Piper](https://github.com/rhasspy/piper) ultra-fast local neural TTS.
- **Audio I/O:** `sounddevice` / `numpy` running real-time streaming circular buffers.
- **Python:** 3.10+ (managed via `uv` or `pip`).

---

## 🚀 Quickstart

### 1. Clone & Setup Environment

```bash
git clone https://github.com/GWaman2007/VartaLap.git
cd VartaLap

# Create virtual environment with uv or python
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Start FLM Model Server

```bash
# Pull model (if not already downloaded)
flm pull gemma4-it:e2b

# Start local server
flm serve gemma4-it:e2b --pmode performance
```

### 3. Run Vartalap

```bash
python -m vartalap.main
```

---

## 📊 Performance Benchmarks & Targets

| Metric | Target | Notes |
| :--- | :--- | :--- |
| **Single-token escape (`<listen>`)** | $< 80\text{ ms}$ | First-token latency on NPU; halts generation immediately |
| **Idle CPU / NPU utilization** | $\sim 0\%$ | Zero LLM calls until STT word arrives |
| **STT streaming word latency** | $< 120\text{ ms}$ | Sherpa-onnx Zipformer continuous chunking |
| **TTS burst time-to-audio** | $< 100\text{ ms}$ | 3–4 word chunk synthesized by Piper |

---

## 📜 License

MIT License.

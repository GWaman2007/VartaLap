"""FastAPI & WebSocket server for Vartalap Real-Time Conversational Dashboard.

Streams audio energy, state transitions, live transcripts, and hardware telemetry
to the modern web dashboard over WebSockets with zero latency overhead.
"""

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    import sounddevice as sd
except ImportError:
    sd = None

from .config import VartalapConfig
from .flm_client import FLMClient, FLMDecision
from .state_machine import VartalapState, VartalapStateMachine

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Vartalap Conversational AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web directory
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class ConnectionManager:
    """Manages active WebSocket dashboard clients and broadcasts telemetry."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)

    def broadcast_sync(self, message: dict[str, Any]):
        """Thread-safe synchronous broadcast called from state machine worker threads."""
        if self._loop and self._loop.is_running() and self.active_connections:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)


manager = ConnectionManager()

# Global state machine instance for the server
harness: Optional[VartalapStateMachine] = None
harness_config = VartalapConfig()
current_mic: Optional[int] = None
current_out: Optional[int] = None


def get_audio_devices():
    devices = []
    default_in = -1
    default_out = -1
    if sd is not None:
        try:
            raw_devices = sd.query_devices()
            default_in, default_out = sd.default.device
            for idx, d in enumerate(raw_devices):
                devices.append({
                    "index": idx,
                    "name": d["name"],
                    "max_input_channels": d["max_input_channels"],
                    "max_output_channels": d["max_output_channels"],
                })
        except Exception as e:
            print(f"[Device Query Warning] {e}")
    return devices, default_in, default_out


def init_harness():
    global harness
    if harness is None:
        harness = VartalapStateMachine(config=harness_config)

        # Connect hooks to broadcast live updates to WebSocket clients
        last_rms_time = 0.0

        def on_energy(rms: float):
            nonlocal last_rms_time
            now = time.perf_counter()
            # Throttle WebSocket energy frames to 20fps for optimal network efficiency
            if (now - last_rms_time) > 0.05:
                manager.broadcast_sync({"type": "energy", "rms": rms})
                last_rms_time = now

        def on_partial(text: str):
            manager.broadcast_sync({"type": "partial_transcript", "text": text})

        def on_state(old_s: VartalapState, new_s: VartalapState):
            manager.broadcast_sync({"type": "state_change", "state": new_s.value})

        def on_dec(dec: FLMDecision, state: VartalapState):
            is_danger = any(w in dec.utterance.lower() for w in ["stop", "danger", "violent", "shock", "burn", "explosion"])
            if dec.action == "talk":
                manager.broadcast_sync({
                    "type": "agent_talk",
                    "text": dec.utterance,
                    "latency_ms": dec.latency_ms,
                    "ttft_ms": dec.ttft_ms,
                    "is_danger": is_danger,
                })
            elif dec.action == "done":
                manager.broadcast_sync({"type": "agent_done"})

        def on_burst(text: str):
            pass

        # Also hook user speech commitment
        def on_user_spoke_hook(text: str, latency_ms: float = 220):
            manager.broadcast_sync({
                "type": "user_spoke",
                "text": text,
                "latency_ms": latency_ms,
            })

        harness.stt.on_energy = on_energy
        # Wrap partial callback so when full phrase is ready it fires user_spoke
        def wrapped_partial(text: str):
            on_partial(text)
            if hasattr(harness.stt, "transcript_queue"):
                manager.broadcast_sync({"type": "user_spoke", "text": text, "latency_ms": 220})

        harness.stt.on_partial_transcript = wrapped_partial
        harness.on_state_change = on_state
        harness.on_decision = on_dec
        harness.on_tts_burst = on_burst


@app.get("/")
async def serve_index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/devices")
async def api_devices():
    devices, default_in, default_out = get_audio_devices()
    return {"devices": devices, "default_in": default_in, "default_out": default_out}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_mic, current_out
    await manager.connect(websocket)
    manager.set_loop(asyncio.get_running_loop())

    # Send initial state and device list to new client
    devices, default_in, default_out = get_audio_devices()
    await websocket.send_text(json.dumps({
        "type": "devices",
        "devices": devices,
        "default_in": current_mic if current_mic is not None else default_in,
        "default_out": current_out if current_out is not None else default_out,
    }))

    is_running = harness is not None and harness._running
    state = harness.current_state.value if harness else "IDLE"
    await websocket.send_text(json.dumps({
        "type": "status",
        "running": is_running,
        "state": state,
    }))

    try:
        while True:
            data_raw = await websocket.receive_text()
            try:
                data = json.loads(data_raw)
                action = data.get("action")

                if action == "start":
                    init_harness()
                    mic = data.get("mic")
                    out = data.get("out")
                    if mic is not None:
                        current_mic = mic
                    if out is not None:
                        current_out = out

                    if sd is not None and current_mic is not None and current_out is not None:
                        try:
                            sd.default.device = (current_mic, current_out)
                        except Exception as e:
                            print(f"[Device Set Warning] {e}")

                    if not harness._running:
                        harness.start(mic_device=current_mic)
                    await manager.broadcast({"type": "status", "running": True, "state": harness.current_state.value})

                elif action == "stop":
                    if harness and harness._running:
                        harness.stop()
                    await manager.broadcast({"type": "status", "running": False, "state": "IDLE"})

                elif action == "set_mic":
                    current_mic = data.get("mic")
                    if sd is not None and current_mic is not None:
                        sd.default.device = (current_mic, sd.default.device[1])

                elif action == "set_out":
                    current_out = data.get("out")
                    if sd is not None and current_out is not None:
                        sd.default.device = (sd.default.device[0], current_out)

                elif action == "simulate_user":
                    text = data.get("text", "").strip()
                    if text:
                        init_harness()
                        # Broadcast user message
                        await manager.broadcast({"type": "user_spoke", "text": text, "latency_ms": 220})
                        # Trigger evaluation
                        threading.Thread(target=_simulate_turn, args=(text,), daemon=True).start()

            except Exception as e:
                print(f"[WS Handler Error] {e}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)


def _simulate_turn(user_text: str):
    """Executes evaluation for manual web simulation testing."""
    if harness is None:
        return
    harness.set_state(VartalapState.MONITORING)
    decision = harness.flm.evaluate(
        history=harness.history,
        current_partial_transcript=user_text,
        silence_tick_count=0,
        agent_last_chunk="",
    )
    is_danger = any(w in decision.utterance.lower() for w in ["stop", "danger", "violent", "shock", "burn", "explosion"])
    manager.broadcast_sync({
        "type": "agent_talk",
        "text": decision.utterance,
        "latency_ms": decision.latency_ms,
        "ttft_ms": decision.ttft_ms,
        "is_danger": is_danger,
    })
    if decision.utterance:
        harness.set_state(VartalapState.TALKING)
        harness._start_speech_burst(decision.utterance)
        harness.history += f"\nUser: {user_text}\nAgent: {decision.utterance}"
        harness.set_state(VartalapState.IDLE)


def start_server(host: str = "127.0.0.1", port: int = 8000):
    init_harness()
    print(f"\n[Vartalap Web Dashboard] Starting on http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()

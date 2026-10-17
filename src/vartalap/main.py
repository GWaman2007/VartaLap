import argparse
import sys
import threading
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    import sounddevice as sd
except ImportError:
    sd = None

from .config import VartalapConfig
from .flm_client import FLMClient, FLMDecision
from .state_machine import VartalapState, VartalapStateMachine
from .stt_stream import SherpaSTTStream
from .tts_player import PiperTTSPlayer

console = Console()


def list_audio_devices():
    """Lists all available input and output devices."""
    if sd is None:
        console.print("[red]sounddevice library not installed or audio unavailable[/red]")
        return

    devices = sd.query_devices()
    default_in, default_out = sd.default.device

    table = Table(title="Available Audio Devices", show_lines=True)
    table.add_column("Index", justify="right", style="cyan")
    table.add_column("Device Name", style="white")
    table.add_column("Max In/Out", justify="center")
    table.add_column("Default", justify="center", style="bold green")

    for idx, d in enumerate(devices):
        is_def = []
        if idx == default_in:
            is_def.append("DEFAULT IN")
        if idx == default_out:
            is_def.append("DEFAULT OUT")
        def_label = ", ".join(is_def) if is_def else ""

        table.add_row(
            str(idx),
            d["name"],
            f"{d['max_input_channels']} in / {d['max_output_channels']} out",
            def_label,
        )

    console.print(table)


def run_live(config: VartalapConfig, mic_device: int = None, out_device: int = None):
    console.print(Panel("[bold green]Starting Vartalap Conversational Monitor[/bold green]\n"
                        "[dim]Ultra-low latency real-time voice harness (Sherpa-ONNX + FastFlowLM + Piper)[/dim]",
                        border_style="green"))

    # Audio Device Diagnostics
    if sd is not None:
        default_in, default_out = sd.default.device
        chosen_in = mic_device if mic_device is not None else default_in
        chosen_out = out_device if out_device is not None else default_out
        try:
            in_info = sd.query_devices(chosen_in)
            out_info = sd.query_devices(chosen_out)
            console.print(f"[bold cyan]Input Mic:[/]  [{chosen_in}] {in_info['name']}")
            console.print(f"[bold cyan]Output TTS:[/] [{chosen_out}] {out_info['name']}")
        except Exception as e:
            console.print(f"[yellow]Audio device query warning:[/] {e}")

    harness = VartalapStateMachine(config=config)

    # Verbose Diagnostic Callbacks
    last_rms_log_time = 0.0

    def on_energy(rms: float):
        nonlocal last_rms_log_time
        now = time.perf_counter()
        # Log voice activity if RMS is noticeable (> 0.01) every 0.8s
        if rms > 0.015 and (now - last_rms_log_time) > 0.8:
            bars = min(20, int(rms * 150))
            meter = "|" * bars
            console.print(f"[dim green][MIC ACTIVITY][/] {meter:<20} (RMS: {rms:.4f})")
            last_rms_log_time = now

    def on_partial(text: str):
        console.print(f"[bold bright_cyan]🎙️  [USER SPOKE][/] \"{text}\"")

    def on_state(old_s, new_s):
        console.print(f"[bold yellow]🔄 [STATE CHANGE][/] {old_s.value} -> [bold green]{new_s.value}[/]")

    def on_dec(dec: FLMDecision, state: VartalapState):
        color = "cyan" if dec.action == "listen" else ("magenta" if dec.action == "talk" else "yellow")
        console.print(
            f"[{color}][DECISION - {state.value}][/] <{dec.action}> "
            f"| Latency: [bold]{dec.latency_ms:.1f}ms[/] (TTFT: {dec.ttft_ms:.1f}ms) "
            f"| Utterance: \"{dec.utterance}\""
        )

    def on_burst(text: str):
        console.print(f"[bold magenta]🔊 [PIPER TTS PLAYING][/] \"{text}\"")

    harness.stt.on_energy = on_energy
    harness.stt.on_partial_transcript = on_partial
    harness.on_state_change = on_state
    harness.on_decision = on_dec
    harness.on_tts_burst = on_burst

    console.print("[bold green]Initializing STT audio stream...[/bold green]")
    try:
        harness.start(mic_device=mic_device)
        console.print("[bold yellow]✅ Vartalap is LIVE! Speak into your microphone now (Press Ctrl+C to stop)...[/bold yellow]\n")
    except Exception as e:
        console.print(f"[bold red]Failed to start audio stream:[/] {e}")
        console.print("[yellow]Tip: Run with --list-devices to find the correct mic index and pass --mic <index>[/yellow]")
        return

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping Vartalap...[/bold red]")
    finally:
        harness.stop()
        console.print("[bold green]Stopped cleanly.[/bold green]")


def run_simulation(config: VartalapConfig):
    console.print("[bold cyan]=== Running End-to-End State Machine Simulation ===[/bold cyan]")
    stt = SherpaSTTStream(config)
    tts = PiperTTSPlayer(config)
    flm = FLMClient(config)

    harness = VartalapStateMachine(config=config, flm=flm, stt=stt, tts=tts)

    def on_state(old_s, new_s):
        console.print(f"[bold yellow][STATE CHANGE][/] {old_s.value} -> [bold green]{new_s.value}[/]")

    def on_dec(dec, state):
        console.print(
            f"[dim][{state.value}][/dim] FLM Decision: [bold]{dec.action}[/] "
            f"| Latency: {dec.latency_ms:.1f}ms | Utterance: \"{dec.utterance}\""
        )

    def on_burst(text):
        console.print(f"[bold magenta][TTS BURST PLAYBACK][/] \"{text}\"")

    harness.on_state_change = on_state
    harness.on_decision = on_dec
    harness.on_tts_burst = on_burst

    # Start harness worker loop
    harness._running = True
    loop_thread = threading.Thread(target=harness._run_loop, daemon=True)
    loop_thread.start()

    console.print(f"\n[1] Current state: [bold]{harness.current_state.value}[/]. (Waiting for user speech - 0 inference load)")
    time.sleep(1.0)

    console.print("\n[2] User starts speaking: 'What' -> feeding to transcript queue...")
    stt.transcript_queue.put("What")
    time.sleep(0.5)

    console.print(f"Harness woke up! Current state: [bold]{harness.current_state.value}[/]")
    time.sleep(0.5)

    console.print("\n[3] User continues speaking mid-thought: 'is two plus two'...")
    stt._current_partial = "What is two plus two"
    stt._words_arrived_in_window = True
    time.sleep(1.0)

    console.print("\n[4] User stops speaking. Ticking silence x 2 turns...")
    stt._words_arrived_in_window = False
    time.sleep(1.5)

    console.print("\n[5] Waiting for Piper TTS burst and arbitration...")
    time.sleep(3.0)

    console.print(f"\n[6] Cycle finished! State is now: [bold]{harness.current_state.value}[/]")
    console.print(f"Conversation History:\n{harness.history}")

    harness.stop()
    console.print("[bold green][OK] Simulation complete![/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Vartalap Conversational Monitor")
    parser.add_argument("--simulate", action="store_true", help="Run simulated end-to-end conversation flow")
    parser.add_argument("--list-devices", action="store_true", help="List all available audio input and output devices")
    parser.add_argument("--mic", type=int, default=None, help="Audio input device index")
    parser.add_argument("--out", type=int, default=None, help="Audio output device index")
    parser.add_argument("--model", type=str, default="gemma4-it:e2b", help="FastFlowLM model tag")
    parser.add_argument("--flm-url", type=str, default="http://127.0.0.1:52625/v1", help="FLM API base URL")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    config = VartalapConfig(
        model_name=args.model,
        flm_base_url=args.flm_url,
    )

    if args.simulate:
        run_simulation(config)
    else:
        run_live(config, mic_device=args.mic, out_device=args.out)


if __name__ == "__main__":
    main()

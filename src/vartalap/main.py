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
    stt_label = "Faster-Whisper (220ms, INT8 CPU)" if config.stt_backend == "whisper" else "Sherpa-ONNX Zipformer (CPU Streaming)"
    console.print(Panel(
        f"[bold green]Starting Vartalap Conversational Monitor[/bold green]\n"
        f"[dim]Continuous real-time voice harness[/dim]\n"
        f"[cyan]STT:[/] {stt_label}  •  [cyan]LLM:[/] {config.model_name}  •  [cyan]TTS:[/] Piper Neural",
        border_style="green"
    ))

    # Set audio devices in sounddevice
    if sd is not None:
        default_in, default_out = sd.default.device
        chosen_in = mic_device if mic_device is not None else default_in
        chosen_out = out_device if out_device is not None else default_out
        try:
            sd.default.device = (chosen_in, chosen_out)
            in_info = sd.query_devices(chosen_in)
            out_info = sd.query_devices(chosen_out)
            console.print(f"[bold cyan]Input Mic:[/]  [{chosen_in}] {in_info['name']}")
            console.print(f"[bold cyan]Output TTS:[/] [{chosen_out}] {out_info['name']}")
        except Exception as e:
            console.print(f"[yellow]Audio device configuration warning:[/] {e}")

    harness = VartalapStateMachine(config=config)

    # Diagnostic Callbacks
    last_rms_log_time = 0.0

    def on_energy(rms: float):
        nonlocal last_rms_log_time
        now = time.perf_counter()
        if rms > 0.040 and (now - last_rms_log_time) > 0.6:
            bars = min(20, int(rms * 150))
            meter = "|" * bars
            console.print(f"[dim green][MIC ACTIVITY][/] {meter:<20} (RMS: {rms:.4f})")
            last_rms_log_time = now

    def on_partial(text: str):
        console.print(f"[bold bright_cyan]🎙️  [USER SPOKE][/] \"{text}\"")

    def on_state(old_s, new_s):
        console.print(f"[bold yellow]🔄 [STATE CHANGE][/] {old_s.value} -> [bold green]{new_s.value}[/]")

    def on_dec(dec: FLMDecision, state: VartalapState):
        if dec.action == "error":
            console.print(
                f"[bold red][FLM ERROR - {state.value}][/] Latency: {dec.latency_ms:.1f}ms "
                f"| Detail: {dec.raw_response}"
            )
        else:
            color = "cyan" if dec.action == "listen" else ("magenta" if dec.action == "talk" else "yellow")
            console.print(
                f"[{color}][DECISION - {state.value}][/] <{dec.action}> "
                f"| Latency: [bold]{dec.latency_ms:.1f}ms[/] (TTFT: {dec.ttft_ms:.1f}ms) "
                f"| Utterance: \"{dec.utterance}\""
            )

    def on_burst(text: str):
        console.print(f"[bold magenta]🔊 [TTS PLAYING][/] \"{text}\"")

    harness.stt.on_energy = on_energy
    harness.stt.on_partial_transcript = on_partial
    harness.on_state_change = on_state
    harness.on_decision = on_dec
    harness.on_tts_burst = on_burst

    console.print("[bold green]Initializing audio streams...[/bold green]")
    try:
        harness.start(mic_device=chosen_in if sd is not None else mic_device)
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


def main():
    parser = argparse.ArgumentParser(description="Vartalap Conversational Monitor")
    parser.add_argument("--list-devices", action="store_true", help="List all available audio input and output devices")
    parser.add_argument("--mic", type=int, default=None, help="Audio input device index")
    parser.add_argument("--out", type=int, default=None, help="Audio output device index")
    parser.add_argument("--model", type=str, default="gemma4-it:e2b", help="FastFlowLM model tag")
    parser.add_argument("--flm-url", type=str, default="http://127.0.0.1:52625/v1", help="FLM API base URL")
    parser.add_argument("--stt", type=str, default="whisper", choices=["whisper", "sherpa"],
                        help="STT backend: 'whisper' (Faster-Whisper, 220ms, default) or 'sherpa' (Zipformer)")

    parser.add_argument("--ui", action="store_true", help="Launch Vartalap modern web dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Web dashboard port")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    config = VartalapConfig(
        model_name=args.model,
        flm_base_url=args.flm_url,
        stt_backend=args.stt,
    )

    if args.ui:
        import webbrowser
        from .web_server import start_server
        console.print(f"[bold green]Launching Vartalap Web Dashboard on http://127.0.0.1:{args.port}...[/bold green]")
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
        start_server(port=args.port)
    else:
        run_live(config, mic_device=args.mic, out_device=args.out)


if __name__ == "__main__":
    main()

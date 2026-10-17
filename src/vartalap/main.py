import argparse
import sys
import time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .config import VartalapConfig
from .flm_client import FLMClient, FLMDecision
from .state_machine import VartalapState, VartalapStateMachine
from .stt_stream import SherpaSTTStream
from .tts_player import PiperTTSPlayer

console = Console()


def build_status_display(
    state: VartalapState,
    partial_transcript: str,
    silence_ticks: int,
    last_decision: FLMDecision,
    last_burst: str,
    history: str,
) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="white")

    # State Badge Styling
    state_styles = {
        VartalapState.IDLE: "[bold green on black] [IDLE] (Zero Inference) [/]",
        VartalapState.MONITORING: "[bold yellow on black] [MONITORING] (500ms Ticks) [/]",
        VartalapState.TALKING: "[bold magenta on black] [TALKING] (Piper Burst) [/]",
        VartalapState.ARBITRATION: "[bold blue on black] [ARBITRATION] (Yield Check) [/]",
    }
    state_badge = state_styles.get(state, str(state))

    table.add_row("System State:", state_badge)
    table.add_row("User Partial Audio:", f'"{partial_transcript}"' if partial_transcript else "[dim]<silent>[/dim]")
    table.add_row("Silence Ticks:", f"{silence_ticks} (500ms units)")

    action_color = {
        "listen": "[cyan]<listen>[/cyan]",
        "talk": f"[green]<talk>[/green] -> \"{last_decision.utterance}\"",
        "done": "[yellow]<done>[/yellow]",
        "error": f"[red]<error: {last_decision.raw_response}>[/red]",
    }.get(last_decision.action, last_decision.action)

    table.add_row("Last Decision:", action_color)
    table.add_row("FLM Latency:", f"{last_decision.latency_ms:.1f} ms (TTFT: {last_decision.ttft_ms:.1f} ms)")
    table.add_row("Agent Last Burst:", f'"{last_burst}"' if last_burst else "[dim]None[/dim]")

    if history:
        history_preview = "\n".join(history.strip().splitlines()[-4:])
        table.add_row("Recent History:", f"[dim]{history_preview}[/dim]")

    return Panel(
        table,
        title="[bold white]Vartalap Conversational Monitor[/bold white]",
        border_style="bright_blue",
    )


def run_live(config: VartalapConfig, mic_device: int = None):
    console.print("[bold green]Starting Vartalap in Real-Time Microphone Mode...[/bold green]")
    harness = VartalapStateMachine(config=config)

    last_decision = FLMDecision(action="listen")
    last_burst = ""

    def on_state(old_s, new_s):
        console.log(f"[bold dim]State Transition:[/] {old_s.value} -> [bold underline]{new_s.value}[/]")

    def on_dec(dec, state):
        nonlocal last_decision
        last_decision = dec
        console.log(
            f"[dim][{state.value}][/dim] Decision: [bold]{dec.action}[/] "
            f"| Latency: {dec.latency_ms:.1f}ms | Raw: {dec.raw_response}"
        )

    def on_burst(text):
        nonlocal last_burst
        last_burst = text
        console.log(f"[bold magenta]Agent Output Burst:[/] \"{text}\"")

    harness.on_state_change = on_state
    harness.on_decision = on_dec
    harness.on_tts_burst = on_burst

    harness.start(mic_device=mic_device)
    console.print("[bold yellow]Vartalap is live! Speak into microphone or press Ctrl+C to stop.[/bold yellow]")

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        console.print("\n[bold red]Shutting down Vartalap...[/bold red]")
    finally:
        harness.stop()


def run_simulation(config: VartalapConfig):
    console.print("[bold cyan]=== Running End-to-End State Machine Simulation ===[/bold cyan]")
    stt = SherpaSTTStream(config)
    tts = PiperTTSPlayer(config)
    flm = FLMClient(config)

    harness = VartalapStateMachine(config=config, flm=flm, stt=stt, tts=tts)

    last_decision = FLMDecision(action="listen")
    last_burst = ""

    def on_state(old_s, new_s):
        console.print(f"[bold yellow][STATE CHANGE][/] {old_s.value} -> [bold green]{new_s.value}[/]")

    def on_dec(dec, state):
        nonlocal last_decision
        last_decision = dec
        console.print(
            f"[dim][{state.value}][/dim] FLM Decision: [bold]{dec.action}[/] "
            f"| Latency: {dec.latency_ms:.1f}ms | Utterance: \"{dec.utterance}\""
        )

    def on_burst(text):
        nonlocal last_burst
        last_burst = text
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



import threading


def main():
    parser = argparse.ArgumentParser(description="Vartalap Conversational Monitor")
    parser.add_argument("--simulate", action="store_true", help="Run simulated end-to-end conversation flow")
    parser.add_argument("--model", type=str, default="gemma4-it:e2b", help="FastFlowLM model tag")
    parser.add_argument("--flm-url", type=str, default="http://127.0.0.1:52625/v1", help="FLM API base URL")
    parser.add_argument("--mic", type=int, default=None, help="Audio input device index")

    args = parser.parse_args()

    config = VartalapConfig(
        model_name=args.model,
        flm_base_url=args.flm_url,
    )

    if args.simulate:
        run_simulation(config)
    else:
        run_live(config, mic_device=args.mic)


if __name__ == "__main__":
    main()

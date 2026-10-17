import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import httpx

from .config import VartalapConfig

DecisionType = Literal["listen", "talk", "done", "error"]


@dataclass
class FLMDecision:
    action: DecisionType
    utterance: str = ""
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    raw_response: str = ""


class FLMClient:
    """Client for FastFlowLM (FLM) running locally on NPU.
    
    Supports both ultra-fast streaming piece generation with sentence-boundary
    bursting and robust non-streaming evaluation with persistent keep-alive connections.
    """

    PROMPT_TEMPLATE = """You are Vartalap, a fast, helpful real-time conversational voice assistant.

Rules:
1. SAFETY FIRST: If user proposes something dangerous (live wires, chemicals in water, self-harm), output <talk> with an urgent warning.
   Example: <talk>Stop immediately! Sodium reacts violently with water, causing an explosion, intense heat, and severe burns.
2. UNFINISHED / INCOMPLETE SENTENCES: If the user utterance feels unfinished, incomplete, mid-thought, trailing off, or missing a main clause (e.g. "so like", "what is the...", "in relation to the", "because I was", "and then", "if I have"), output ONLY: <listen>
   Do NOT interrupt the user. Do NOT ask them to clarify or finish. Simply output <listen>.
3. GREETINGS & CHECK-INS: If user greets you ("hello", "hi", "hey") or checks if you can hear them ("can you hear me?", "you hear me?", "are you there?"), output <talk> followed by a warm, direct confirmation.
   Example: <talk>Hello! Yes, I hear you loud and clear. How can I assist you today?
4. COMPLETE QUESTIONS & STATEMENTS: ONLY if the user has fully finished speaking a complete, grammatically sound question or statement with a clear end, output <talk> followed by a concise, direct answer in 10-25 words.
5. TURN COMPLETION: Output <done> ONLY when you have already given your complete spoken answer and user has not said anything new.

Conversation History:
{history}

[Current State]
User Said: "{current_partial_transcript}"
User Speech Status: {speech_status}

Output decision:
"""

    def __init__(self, config: Optional[VartalapConfig] = None):
        self.config = config or VartalapConfig()
        self.endpoint = f"{self.config.flm_base_url}/chat/completions"
        # Persistent HTTP client with connection pooling and keep-alive
        self._http_client = httpx.Client(
            timeout=httpx.Timeout(10.0, connect=3.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    def format_prompt(
        self,
        history: str,
        current_partial_transcript: str,
        silence_tick_count: int = 0,
        agent_last_chunk: str = "",
        user_speech_status: str = "Finished speaking",
    ) -> str:
        clean_history = history.strip()
        if clean_history:
            lines = clean_history.split("\n")
            if len(lines) > 6:
                clean_history = "\n".join(lines[-6:])
        else:
            clean_history = "None"

        return self.PROMPT_TEMPLATE.format(
            history=clean_history,
            current_partial_transcript=current_partial_transcript.strip(),
            speech_status=user_speech_status,
        )

    def evaluate_stream(
        self,
        history: str,
        current_partial_transcript: str,
        silence_tick_count: int,
        agent_last_chunk: str,
        on_piece: Optional[Callable[[str], None]] = None,
        user_speech_status: str = "Finished speaking",
    ) -> FLMDecision:
        """Evaluates conversational state with streaming token reading and piece bursting.
        
        Reads stream completely to [DONE] to ensure zero socket leaks on Windows.
        As tokens arrive, cuts <talk> responses into natural sentence/clause pieces
        and emits them via on_piece for instant TTS speech initiation.
        """
        prompt = self.format_prompt(
            history=history,
            current_partial_transcript=current_partial_transcript,
            silence_tick_count=silence_tick_count,
            agent_last_chunk=agent_last_chunk,
            user_speech_status=user_speech_status,
        )

        t_start = time.perf_counter()
        first_token_time = None
        collected_tokens: list[str] = []
        current_piece: list[str] = []
        action: Optional[DecisionType] = None
        tag_buffer = ""
        in_talk = False

        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }

        try:
            with self._http_client.stream("POST", self.endpoint, json=payload) as response:
                if response.status_code != 200:
                    t_end = time.perf_counter()
                    return FLMDecision(
                        action="error",
                        raw_response=f"HTTP {response.status_code}: {response.text[:200]}",
                        latency_ms=(t_end - t_start) * 1000,
                    )

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            collected_tokens.append(delta)

                            if action is None:
                                tag_buffer += delta
                                if "<listen" in tag_buffer:
                                    action = "listen"
                                elif "<done" in tag_buffer:
                                    action = "done"
                                elif "<talk" in tag_buffer:
                                    action = "talk"
                                    in_talk = True
                                    if "<talk>" in tag_buffer:
                                        remainder = tag_buffer.split("<talk>", 1)[1]
                                    else:
                                        remainder = tag_buffer.split(">", 1)[1] if ">" in tag_buffer else ""
                                    if remainder:
                                        current_piece.append(remainder)
                                elif len(tag_buffer) > 15:
                                    # Fallback if model omitted tag
                                    action = "talk"
                                    in_talk = True
                                    current_piece.append(tag_buffer)

                            elif in_talk:
                                if "</talk>" in delta or "<done" in delta or "<listen" in delta:
                                    in_talk = False
                                    clean_delta = delta.replace("</talk>", "").replace("<done>", "").replace("<listen>", "").strip()
                                    if clean_delta:
                                        current_piece.append(clean_delta)
                                else:
                                    current_piece.append(delta)

                                piece_str = "".join(current_piece)
                                # Cut into pieces by clause/sentence boundaries or 7+ words
                                if on_piece and (any(p in delta for p in [",", ".", "!", "?", ";", "\n"]) or len(piece_str.split()) >= 7):
                                    chunk_to_speak = piece_str.strip()
                                    if chunk_to_speak:
                                        on_piece(chunk_to_speak)
                                    current_piece = []
                    except Exception:
                        pass

            # Emit any remaining text in the piece buffer
            if on_piece and current_piece:
                remaining = "".join(current_piece).replace("</talk>", "").replace("<done>", "").replace("<listen>", "").strip()
                if remaining:
                    on_piece(remaining)

            t_end = time.perf_counter()
            full_text = "".join(collected_tokens).strip()
            latency = (t_end - t_start) * 1000
            ttft = (first_token_time - t_start) * 1000 if first_token_time else latency

            if action == "listen" or "<listen" in full_text:
                return FLMDecision(action="listen", utterance="", latency_ms=latency, ttft_ms=ttft, raw_response=full_text)
            elif action == "done" or "<done" in full_text:
                return FLMDecision(action="done", utterance="", latency_ms=latency, ttft_ms=ttft, raw_response=full_text)
            else:
                utterance = full_text.replace("<talk>", "").replace("</talk>", "").replace("<done>", "").replace("<listen>", "").strip()
                return FLMDecision(action="talk", utterance=utterance, latency_ms=latency, ttft_ms=ttft, raw_response=full_text)

        except Exception as e:
            t_end = time.perf_counter()
            return FLMDecision(
                action="error",
                raw_response=f"{type(e).__name__}: {str(e)}",
                latency_ms=(t_end - t_start) * 1000,
            )

    def evaluate(
        self,
        history: str,
        current_partial_transcript: str,
        silence_tick_count: int,
        agent_last_chunk: str,
    ) -> FLMDecision:
        """Evaluates conversational state cleanly with streaming token reading."""
        return self.evaluate_stream(
            history=history,
            current_partial_transcript=current_partial_transcript,
            silence_tick_count=silence_tick_count,
            agent_last_chunk=agent_last_chunk,
            on_piece=None,
        )

    def close(self):
        try:
            self._http_client.close()
        except Exception:
            pass

import json
import re
import time
from dataclasses import dataclass
from typing import Literal, Optional

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
    
    Uses standard non-streaming HTTP requests to maintain a single persistent,
    clean connection without socket leaks or connection limit exhaustion.
    """

    PROMPT_TEMPLATE = """You are Vartalap, a fast real-time conversational voice assistant.

Rules:
1. SAFETY FIRST: If user proposes something dangerous (live wires, chemicals in water, self-harm), output <talk> with an urgent warning in 10-20 words explaining WHY it is dangerous.
   Example: <talk>Stop immediately! Sodium reacts violently with water, causing an explosion, intense heat, and severe burns.
2. If user utterance is incomplete, trailing off, or mid-thought (e.g. "so like", "what is the...", "can you"), output ONLY: <listen>
   Do NOT say "please finish your question". Simply output <listen>.
3. If user says stop, wait, or be quiet, output: <talk>I have stopped. What would you like to know?
4. If user asked a complete question, output <talk> followed by a concise, natural, direct answer in 10-20 words in complete sentences.
5. If you already answered and user is silent, output: <done>

Conversation History:
{history}

[Current State]
User Said: "{current_partial_transcript}"
User Silence Ticks: {silence_tick_count}

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
        silence_tick_count: int,
        agent_last_chunk: str,
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
            silence_tick_count=silence_tick_count,
        )

    def evaluate(
        self,
        history: str,
        current_partial_transcript: str,
        silence_tick_count: int,
        agent_last_chunk: str,
    ) -> FLMDecision:
        """Evaluates conversational state cleanly without leaking sockets."""
        prompt = self.format_prompt(
            history=history,
            current_partial_transcript=current_partial_transcript,
            silence_tick_count=silence_tick_count,
            agent_last_chunk=agent_last_chunk,
        )

        t_start = time.perf_counter()

        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }

        try:
            response = self._http_client.post(self.endpoint, json=payload)
            t_end = time.perf_counter()
            latency = (t_end - t_start) * 1000

            if response.status_code != 200:
                return FLMDecision(
                    action="error",
                    raw_response=f"HTTP {response.status_code}: {response.text[:200]}",
                    latency_ms=latency,
                )

            data = response.json()
            full_text = data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            t_end = time.perf_counter()
            return FLMDecision(
                action="error",
                raw_response=f"{type(e).__name__}: {str(e)}",
                latency_ms=(t_end - t_start) * 1000,
            )

        # Parse decision from response
        if "<listen>" in full_text:
            return FLMDecision(
                action="listen",
                utterance="",
                latency_ms=latency,
                ttft_ms=latency,
                raw_response=full_text,
            )
        elif "<done>" in full_text:
            return FLMDecision(
                action="done",
                utterance="",
                latency_ms=latency,
                ttft_ms=latency,
                raw_response=full_text,
            )
        elif "<talk>" in full_text:
            match = re.search(r"<talk>(.*?)(?:</talk>|\n|$)", full_text, re.DOTALL)
            utterance = match.group(1).strip() if match else full_text.replace("<talk>", "").strip()
            utterance = utterance.replace("<done>", "").replace("<listen>", "").replace("</talk>", "").strip()
            if not utterance:
                utterance = full_text.replace("<talk>", "").replace("<done>", "").replace("<listen>", "").replace("</talk>", "").strip()
            return FLMDecision(
                action="talk",
                utterance=utterance,
                latency_ms=latency,
                ttft_ms=latency,
                raw_response=full_text,
            )
        else:
            clean_text = full_text.strip()
            if clean_text:
                return FLMDecision(
                    action="talk",
                    utterance=clean_text,
                    latency_ms=latency,
                    ttft_ms=latency,
                    raw_response=full_text,
                )
            return FLMDecision(
                action="listen",
                utterance="",
                latency_ms=latency,
                ttft_ms=latency,
                raw_response=full_text,
            )

    def close(self):
        try:
            self._http_client.close()
        except Exception:
            pass

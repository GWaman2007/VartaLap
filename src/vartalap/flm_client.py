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
    """Client for FastFlowLM (FLM) running locally on NPU with streaming response."""

    PROMPT_TEMPLATE = """You are Vartalap, an intelligent real-time conversational voice assistant.

Rules:
1. SAFETY FIRST: If the user is about to do something DANGEROUS (mixing chemicals, touching live wires, self-harm, etc), IMMEDIATELY output <talk> followed by an urgent warning in 10-25 words explaining WHY it is dangerous.
   Example: <talk>Stop immediately! Sodium reacts violently with water causing an explosion, intense heat, and severe burns.
2. If the user tells you to stop, wait, or be quiet, acknowledge politely in 5-10 words:
   Example: <talk>I have stopped. What would you like to know?
3. If the user spoke to you or asked a question, output <talk> followed by a natural, direct, conversational answer in 10-25 words. Speak in complete sentences.
4. If you have already fully answered the user and they have not said anything new, output: <done>
5. Do NOT repeat what you already said. Build on the conversation naturally.

Conversation History:
{history}

[Current State]
User Said: "{current_partial_transcript}"
User Silence Ticks: {silence_tick_count}
Your Previous Response: "{agent_last_chunk}"

Output decision:
"""

    def __init__(self, config: Optional[VartalapConfig] = None):
        self.config = config or VartalapConfig()
        self.endpoint = f"{self.config.flm_base_url}/chat/completions"
        self._http_client = httpx.Client(timeout=httpx.Timeout(8.0, connect=3.0))

    def format_prompt(
        self,
        history: str,
        current_partial_transcript: str,
        silence_tick_count: int,
        agent_last_chunk: str,
    ) -> str:
        return self.PROMPT_TEMPLATE.format(
            history=history.strip() or "None",
            current_partial_transcript=current_partial_transcript.strip(),
            silence_tick_count=silence_tick_count,
            agent_last_chunk=agent_last_chunk.strip(),
        )

    def evaluate(
        self,
        history: str,
        current_partial_transcript: str,
        silence_tick_count: int,
        agent_last_chunk: str,
    ) -> FLMDecision:
        """Evaluates conversational state with streaming token reading."""
        prompt = self.format_prompt(
            history=history,
            current_partial_transcript=current_partial_transcript,
            silence_tick_count=silence_tick_count,
            agent_last_chunk=agent_last_chunk,
        )

        t_start = time.perf_counter()
        first_token_time = None
        collected_tokens: list[str] = []

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
                        raw_response=f"HTTP {response.status_code}",
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

                            # Early exit on single control tokens
                            partial = "".join(collected_tokens).strip()
                            if "<listen>" in partial or "<done>" in partial:
                                break
                    except Exception:
                        pass

        except Exception as e:
            t_end = time.perf_counter()
            try:
                self._http_client.close()
            except Exception:
                pass
            self._http_client = httpx.Client(timeout=httpx.Timeout(10.0, connect=3.0))
            return FLMDecision(
                action="error",
                raw_response=f"{type(e).__name__}: {str(e)}",
                latency_ms=(t_end - t_start) * 1000,
            )

        t_end = time.perf_counter()
        full_text = "".join(collected_tokens).strip()
        ttft = (first_token_time - t_start) * 1000 if first_token_time else 0.0

        # Decision extraction
        if "<listen>" in full_text:
            return FLMDecision(
                action="listen",
                utterance="",
                latency_ms=(t_end - t_start) * 1000,
                ttft_ms=ttft,
                raw_response=full_text,
            )
        elif "<done>" in full_text:
            return FLMDecision(
                action="done",
                utterance="",
                latency_ms=(t_end - t_start) * 1000,
                ttft_ms=ttft,
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
                latency_ms=(t_end - t_start) * 1000,
                ttft_ms=ttft,
                raw_response=full_text,
            )
        else:
            # If model answered directly without <talk> tag
            clean_text = full_text.strip()
            if clean_text:
                return FLMDecision(
                    action="talk",
                    utterance=clean_text,
                    latency_ms=(t_end - t_start) * 1000,
                    ttft_ms=ttft,
                    raw_response=full_text,
                )
            return FLMDecision(
                action="listen",
                utterance="",
                latency_ms=(t_end - t_start) * 1000,
                ttft_ms=ttft,
                raw_response=full_text,
            )

    def close(self):
        self._http_client.close()

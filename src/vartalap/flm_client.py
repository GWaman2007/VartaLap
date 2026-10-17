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
    """Client for FastFlowLM (FLM) running locally on NPU with single-token escape."""

    PROMPT_TEMPLATE = """You are Vartalap, an ultra-low latency real-time conversational monitor.
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
"""

    def __init__(self, config: Optional[VartalapConfig] = None):
        self.config = config or VartalapConfig()
        self.endpoint = f"{self.config.flm_base_url}/chat/completions"
        self._http_client = httpx.Client(timeout=httpx.Timeout(5.0, connect=2.0))

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
        """Evaluates the state using streaming with single-token escape."""
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
                            text = "".join(collected_tokens).strip()

                            # Single-Token Escape 1: <listen>
                            if "<listen>" in text:
                                t_end = time.perf_counter()
                                ttft = (first_token_time - t_start) * 1000 if first_token_time else 0.0
                                return FLMDecision(
                                    action="listen",
                                    utterance="",
                                    latency_ms=(t_end - t_start) * 1000,
                                    ttft_ms=ttft,
                                    raw_response=text,
                                )

                            # Escape 2: <done>
                            if "<done>" in text:
                                t_end = time.perf_counter()
                                ttft = (first_token_time - t_start) * 1000 if first_token_time else 0.0
                                return FLMDecision(
                                    action="done",
                                    utterance="",
                                    latency_ms=(t_end - t_start) * 1000,
                                    ttft_ms=ttft,
                                    raw_response=text,
                                )

                            # Escape 3: <talk> followed by words
                            if "<talk>" in text:
                                match = re.search(r"<talk>(.*?)(?:</talk>|\n|$)", text, re.DOTALL)
                                if match:
                                    talk_content = match.group(1).strip()
                                    words = talk_content.split()
                                    # Burst is limited to 3-4 words
                                    if len(words) >= 3 or ("</talk>" in text and len(words) > 0):
                                        burst_words = " ".join(words[:4])
                                        t_end = time.perf_counter()
                                        ttft = (first_token_time - t_start) * 1000 if first_token_time else 0.0
                                        return FLMDecision(
                                            action="talk",
                                            utterance=burst_words,
                                            latency_ms=(t_end - t_start) * 1000,
                                            ttft_ms=ttft,
                                            raw_response=text,
                                        )

                    except Exception:
                        pass

        except Exception as e:
            t_end = time.perf_counter()
            return FLMDecision(
                action="error",
                raw_response=str(e),
                latency_ms=(t_end - t_start) * 1000,
            )

        t_end = time.perf_counter()
        full_text = "".join(collected_tokens).strip()
        ttft = (first_token_time - t_start) * 1000 if first_token_time else 0.0

        if "<talk>" in full_text:
            match = re.search(r"<talk>(.*?)(?:</talk>|\n|$)", full_text, re.DOTALL)
            words = (match.group(1).strip().split() if match else [])
            burst_words = " ".join(words[:4]) if words else full_text.replace("<talk>", "").strip()
            return FLMDecision(
                action="talk",
                utterance=burst_words,
                latency_ms=(t_end - t_start) * 1000,
                ttft_ms=ttft,
                raw_response=full_text,
            )
        elif "<done>" in full_text:
            return FLMDecision(action="done", latency_ms=(t_end - t_start) * 1000, ttft_ms=ttft, raw_response=full_text)
        else:
            return FLMDecision(action="listen", latency_ms=(t_end - t_start) * 1000, ttft_ms=ttft, raw_response=full_text)

    def close(self):
        self._http_client.close()

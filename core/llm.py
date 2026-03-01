"""
Grotesque AI – LLM Inference Engine (Thread 5)

CUDA-accelerated LLaMA 3 8B via llama-cpp-python.

Design:
 • Model loaded ONCE at startup with max GPU layer offload.
 • Streaming token generation – tokens pushed to output queue as they emit.
 • Structured JSON output mode for intent routing.
 • KV cache stays on GPU between turns within the context window.
 • Conversation history managed in a sliding-window of n_ctx tokens.
 • Uses pinned (page-locked) memory via llama.cpp's mlock support.
 • Memory integration – accepts context from MemoryManager.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generator, List, Optional

if TYPE_CHECKING:
    from core.buffers import EventQueue

logger = logging.getLogger("grotesque.llm")

# Default structured-output system prompt
STRUCTURED_SYSTEM_PROMPT = """\
You are Grotesque, a helpful voice assistant running 100% locally.
Keep responses concise and conversational (1-3 sentences unless detail is requested).
Never mention being an AI unless directly asked.

You MUST respond in valid JSON with this exact structure:
{
  "intent": "<category>",
  "parameters": {},
  "response": "<your spoken response>"
}

Intent categories:
- "conversation": general chat, questions, explanations
- "get_time": user asks for current time
- "get_date": user asks for current date
- "system_query": user asks about CPU, memory, battery, disk, uptime
- "set_timer": user wants a timer (parameters: {"duration_seconds": N, "label": "..."})
- "set_alarm": user wants an alarm (parameters: {"time": "HH:MM", "label": "..."})
- "media_control": play/pause/next/prev/volume (parameters: {"action": "play|pause|next|previous|volume_up|volume_down|mute"})
- "change_setting": change assistant settings (parameters: {"setting": "...", "value": ...})

If no special intent applies, use "conversation" with an empty parameters object.
Always include the "response" field with what you would say out loud."""


@dataclass
class LLMToken:
    """Single streamed token from the LLM."""
    text: str
    is_final: bool = False          # True for the last token / EOS
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class LLMRequest:
    """Incoming request for the LLM."""
    user_text: str
    memory_context: Optional[list] = None   # Extra context from memory search
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class LLMResponse:
    """Complete response from LLM (for intent-routing mode)."""
    raw_text: str
    user_text: str
    intent: Optional[str] = None
    parameters: Optional[dict] = None
    spoken_response: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)


class LLMEngine:
    """
    LLaMA 3 8B Instruct inference via llama-cpp-python with CUDA offload.

    Supports two output modes:
     • Streaming mode: tokens pushed to ``token_queue`` as they generate
     • Structured mode: collects full JSON response, pushes LLMResponse to
       ``response_queue`` for intent routing

    Reads LLMRequests from ``request_queue``.
    """

    def __init__(
        self,
        request_queue: "EventQueue",
        token_queue: "EventQueue",
        response_queue: Optional["EventQueue"] = None,
        model_path: str = "models/llm/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        n_gpu_layers: int = 33,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_threads: int = 8,
        use_mmap: bool = True,
        use_mlock: bool = True,
        flash_attn: bool = True,
        seed: int = -1,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop_tokens: Optional[List[str]] = None,
        system_prompt: str = "You are a helpful voice assistant.",
        structured_output: bool = True,
    ) -> None:
        self._req_q = request_queue
        self._tok_q = token_queue
        self._resp_q = response_queue     # For intent router (structured mode)
        self._model_path = model_path
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._n_batch = n_batch
        self._n_threads = n_threads
        self._use_mmap = use_mmap
        self._use_mlock = use_mlock
        self._flash_attn = flash_attn
        self._seed = seed
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._repeat_penalty = repeat_penalty
        self._stop_tokens = stop_tokens or ["<|eot_id|>", "<|end_of_text|>"]
        self._system_prompt = system_prompt
        self._structured_output = structured_output

        # Use structured prompt if structured mode enabled and default prompt
        if self._structured_output and "JSON" not in self._system_prompt:
            self._system_prompt = STRUCTURED_SYSTEM_PROMPT

        self._model = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._heartbeat = None       # injected via set_heartbeat()

        # Conversation history for multi-turn (within context window)
        self._history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load GGUF model with CUDA offload.  Call once at startup."""
        from llama_cpp import Llama

        logger.info(
            "Loading LLM: %s (GPU layers=%d, ctx=%d, batch=%d)…",
            self._model_path,
            self._n_gpu_layers,
            self._n_ctx,
            self._n_batch,
        )
        t0 = time.monotonic()

        self._model = Llama(
            model_path=self._model_path,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
            n_batch=self._n_batch,
            n_threads=self._n_threads,
            use_mmap=self._use_mmap,
            use_mlock=self._use_mlock,
            flash_attn=self._flash_attn,
            seed=self._seed,
            verbose=False,
        )

        elapsed = time.monotonic() - t0
        logger.info("LLM loaded in %.1f s", elapsed)

    def start(self) -> None:
        if self._model is None:
            self.load_model()
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="LLM", daemon=True,
        )
        self._thread.start()
        logger.info("LLM thread started")

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("LLM thread stopped")

    def set_heartbeat(self, monitor) -> None:
        """Inject supervisor heartbeat monitor."""
        self._heartbeat = monitor

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        _last_heartbeat = time.monotonic()
        try:
            while self._running.is_set():
                req = self._req_q.get(timeout=0.2)

                # Periodic idle heartbeat so supervisor knows we're alive
                _now = time.monotonic()
                if self._heartbeat and (_now - _last_heartbeat) >= 5.0:
                    self._heartbeat.beat("LLM")
                    _last_heartbeat = _now

                if req is None:
                    continue
                if not isinstance(req, LLMRequest):
                    continue

                self._generate_streaming(req)
                if self._heartbeat:
                    self._heartbeat.beat("LLM")
                    _last_heartbeat = time.monotonic()
        except Exception:
            logger.exception("LLM thread crashed")
            self._running.clear()

    def _build_messages(self, user_text: str, memory_context: Optional[list] = None) -> list[dict[str, str]]:
        """Build LLaMA-3 Instruct chat messages with conversation history and memory."""
        messages = [{"role": "system", "content": self._system_prompt}]

        # Inject memory context if provided
        if memory_context:
            for msg in memory_context:
                messages.append(msg)

        # Append history (trimmed to fit context)
        for msg in self._history[-10:]:  # keep last 10 exchanges max
            messages.append(msg)
        messages.append({"role": "user", "content": user_text})
        return messages

    def _generate_streaming(self, req: LLMRequest) -> None:
        """Generate tokens in streaming mode, pushing each to the queue."""
        messages = self._build_messages(req.user_text, req.memory_context)

        logger.debug("LLM prompt: %s", req.user_text[:200])
        t0 = time.monotonic()
        first_token_time: Optional[float] = None
        full_response_parts: list[str] = []

        try:
            stream = self._model.create_chat_completion(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                top_k=self._top_k,
                repeat_penalty=self._repeat_penalty,
                stop=self._stop_tokens,
                stream=True,
            )

            for chunk in stream:
                if not self._running.is_set():
                    break

                delta = chunk["choices"][0].get("delta", {})
                token_text = delta.get("content", "")
                if not token_text:
                    continue

                if first_token_time is None:
                    first_token_time = time.monotonic()
                    ttft = first_token_time - t0
                    logger.info("LLM first token in %.0f ms", ttft * 1000)

                full_response_parts.append(token_text)

                # In structured mode, don't stream tokens to TTS
                # (we need the full JSON before routing)
                if not self._structured_output:
                    self._tok_q.put(LLMToken(text=token_text, is_final=False))

            full_response = "".join(full_response_parts)
            elapsed = time.monotonic() - t0
            logger.info(
                "LLM done in %.2f s (%d chars): %s",
                elapsed,
                len(full_response),
                full_response[:120],
            )

            if self._structured_output and self._resp_q:
                # Parse JSON and create structured response
                llm_resp = self._parse_structured_response(
                    full_response, req.user_text, elapsed * 1000
                )
                self._resp_q.put(llm_resp)
            else:
                # Legacy streaming mode: send final marker
                self._tok_q.put(LLMToken(text="", is_final=True))

            # Update conversation history
            self._history.append({"role": "user", "content": req.user_text})
            self._history.append({"role": "assistant", "content": full_response})

            # Trim history if it's getting too long
            if len(self._history) > 20:
                self._history = self._history[-10:]

        except Exception:
            logger.exception("LLM generation error")
            if self._structured_output and self._resp_q:
                self._resp_q.put(LLMResponse(
                    raw_text="", user_text=req.user_text,
                    intent="conversation",
                    spoken_response="I'm sorry, I had trouble processing that.",
                ))
            else:
                self._tok_q.put(LLMToken(text="", is_final=True))

    def _parse_structured_response(
        self, raw: str, user_text: str, latency_ms: float
    ) -> LLMResponse:
        """Parse the LLM's JSON response into an LLMResponse."""
        try:
            # Try to extract JSON from the response
            text = raw.strip()
            # Handle markdown code fences
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            return LLMResponse(
                raw_text=raw,
                user_text=user_text,
                intent=data.get("intent", "conversation"),
                parameters=data.get("parameters", {}),
                spoken_response=data.get("response", raw),
                latency_ms=latency_ms,
            )
        except (json.JSONDecodeError, IndexError, KeyError):
            logger.warning("LLM output is not valid JSON, treating as conversation")
            return LLMResponse(
                raw_text=raw,
                user_text=user_text,
                intent="conversation",
                parameters={},
                spoken_response=raw,
                latency_ms=latency_ms,
            )

    def clear_history(self) -> None:
        """Clear conversation history (security wipe)."""
        self._history.clear()

    def get_vram_usage_mb(self) -> float:
        """Estimate current VRAM usage."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
        except Exception:
            pass
        return 0.0

"""
Grotesque AI – Text-to-Speech via Piper (Thread 6 – synthesis part)

Piper runs on CPU (more than fast enough for real-time) to leave GPU
headroom for the LLM and STT models.

Design:
 • Piper ONNX model pre-loaded into RAM at startup.
 • Receives streamed token text from the LLM.
 • Accumulates tokens until a sentence boundary, then synthesizes.
 • Pushes raw PCM audio to the playback queue.
 • Begins speaking BEFORE the LLM finishes (sentence-level streaming).
"""

from __future__ import annotations

import io
import logging
import re
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from core.buffers import EventQueue
    from core.llm import LLMToken

logger = logging.getLogger("grotesque.tts")

# Sentence-ending punctuation for chunked synthesis
_SENTENCE_END = re.compile(r'[.!?;:\n]\s*$')
_MIN_CHUNK_LEN = 8   # don't synthesize extremely short fragments


@dataclass
class TTSAudio:
    """Synthesised audio chunk ready for playback."""
    samples: np.ndarray    # int16 PCM
    sample_rate: int
    is_final: bool = False  # last chunk for this utterance
    timestamp: float = field(default_factory=time.monotonic)


class TextToSpeech:
    """
    Piper TTS engine.

    Reads LLMToken objects from ``token_queue``, accumulates them
    into sentence-length chunks, synthesizes speech, and pushes
    TTSAudio onto ``audio_queue`` for the playback thread.
    """

    def __init__(
        self,
        token_queue: "EventQueue",
        audio_queue: "EventQueue",
        model_path: str = "models/tts/en_US-amy-medium.onnx",
        config_path: str = "models/tts/en_US-amy-medium.onnx.json",
        speaker_id: int = 0,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        sentence_silence_sec: float = 0.2,
        use_cuda: bool = False,
        output_sample_rate: int = 22050,
        leading_silence_sec: float = 0.0,
    ) -> None:
        self._tok_q = token_queue
        self._audio_q = audio_queue
        self._model_path = model_path
        self._config_path = config_path
        self._speaker_id = speaker_id
        self._length_scale = length_scale
        self._noise_scale = noise_scale
        self._noise_w = noise_w
        self._sentence_silence = sentence_silence_sec
        self._use_cuda = use_cuda
        self._output_sr = output_sample_rate
        self._leading_silence_sec = leading_silence_sec

        self._voice = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._first_chunk_of_utterance = True  # track if next chunk is first

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load Piper model into RAM."""
        try:
            from piper import PiperVoice
        except ImportError:
            raise RuntimeError("piper-tts not installed. Run: pip install piper-tts")

        logger.info("Loading Piper TTS model: %s", self._model_path)
        t0 = time.monotonic()

        self._voice = PiperVoice.load(
            self._model_path,
            config_path=self._config_path,
            use_cuda=self._use_cuda,
        )
        elapsed = time.monotonic() - t0
        logger.info("Piper TTS loaded in %.1f s", elapsed)

    def start(self) -> None:
        if self._voice is None:
            self.load_model()
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="TTS", daemon=True,
        )
        self._thread.start()
        logger.info("TTS thread started")

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("TTS thread stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal – Token accumulation + synthesis
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while self._running.is_set():
                self._process_tokens()
        except Exception:
            logger.exception("TTS thread crashed")
            self._running.clear()

    def _process_tokens(self) -> None:
        """
        Accumulate streamed tokens until sentence boundary, then synthesize.
        """
        buffer: list[str] = []
        self._first_chunk_of_utterance = True

        while self._running.is_set():
            token = self._tok_q.get(timeout=0.1)
            if token is None:
                # Timeout – if we have buffered text, synthesize it
                if buffer:
                    text = "".join(buffer)
                    if len(text.strip()) >= _MIN_CHUNK_LEN:
                        self._synthesize(text, is_final=False)
                        buffer.clear()
                continue

            if token.is_final:
                # Flush any remaining text
                if buffer:
                    text = "".join(buffer)
                    if text.strip():
                        self._synthesize(text, is_final=True)
                    buffer.clear()
                else:
                    # Send an empty final marker
                    self._audio_q.put(
                        TTSAudio(
                            samples=np.array([], dtype=np.int16),
                            sample_rate=self._output_sr,
                            is_final=True,
                        )
                    )
                return  # back to outer loop waiting for next utterance

            buffer.append(token.text)
            current = "".join(buffer)

            # Synthesize at sentence boundaries
            if _SENTENCE_END.search(current) and len(current.strip()) >= _MIN_CHUNK_LEN:
                self._synthesize(current, is_final=False)
                buffer.clear()

    def _synthesize(self, text: str, is_final: bool = False) -> None:
        """Convert text to PCM audio via Piper and push to playback queue."""
        text = text.strip()
        if not text:
            return

        t0 = time.monotonic()
        logger.debug("TTS synthesizing: %s", text[:80])

        try:
            # Build Piper SynthesisConfig for speed/quality parameters
            from piper.config import SynthesisConfig
            syn_cfg = SynthesisConfig(
                speaker_id=self._speaker_id,
                length_scale=self._length_scale,
                noise_scale=self._noise_scale,
                noise_w_scale=self._noise_w,
            )

            # Piper synthesize() yields AudioChunk objects
            audio_chunks: list[bytes] = []
            for chunk in self._voice.synthesize(text, syn_config=syn_cfg):
                audio_chunks.append(chunk.audio_int16_bytes)

            if audio_chunks:
                raw_bytes = b"".join(audio_chunks)
                samples = np.frombuffer(raw_bytes, dtype=np.int16)

                # Prepend leading silence on first chunk of each utterance
                if self._first_chunk_of_utterance and self._leading_silence_sec > 0:
                    silence_samples = int(self._output_sr * self._leading_silence_sec)
                    silence = np.zeros(silence_samples, dtype=np.int16)
                    samples = np.concatenate([silence, samples])
                    self._first_chunk_of_utterance = False

                self._audio_q.put(
                    TTSAudio(
                        samples=samples,
                        sample_rate=self._output_sr,
                        is_final=is_final,
                    )
                )
                elapsed = time.monotonic() - t0
                logger.debug(
                    "TTS done in %.0f ms (%d samples, %.2f s audio)",
                    elapsed * 1000,
                    len(samples),
                    len(samples) / self._output_sr,
                )

        except Exception:
            logger.exception("TTS synthesis error for text: %s", text[:80])

    def synthesize_direct(self, text: str) -> Optional[np.ndarray]:
        """
        Synchronous synthesis – returns PCM int16 numpy array.
        Useful for startup chime or error beep.
        """
        if self._voice is None:
            return None
        from piper.config import SynthesisConfig
        syn_cfg = SynthesisConfig(
            speaker_id=self._speaker_id,
            length_scale=self._length_scale,
            noise_scale=self._noise_scale,
            noise_w_scale=self._noise_w,
        )
        audio_chunks: list[bytes] = []
        for chunk in self._voice.synthesize(text, syn_config=syn_cfg):
            audio_chunks.append(chunk.audio_int16_bytes)
        if audio_chunks:
            return np.frombuffer(b"".join(audio_chunks), dtype=np.int16)
        return None

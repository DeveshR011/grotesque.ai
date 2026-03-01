"""
Grotesque AI – Speech-to-Text (Thread 4)

GPU-accelerated streaming transcription using Faster-Whisper (CTranslate2).

Design:
 • Model loaded ONCE at startup and kept in VRAM.
 • Receives speech audio chunks from the VAD queue.
 • Produces partial + final transcription text pushed to the STT output queue.
 • Uses float16 on CUDA for optimal throughput on RTX 4050.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from core.buffers import EventQueue

logger = logging.getLogger("grotesque.stt")


@dataclass
class TranscriptionResult:
    text: str
    is_partial: bool = False
    language: str = "en"
    confidence: float = 0.0
    duration_sec: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)
    source: str = "mic"             # "mic" or "speaker"


class SpeechToText:
    """
    Faster-Whisper based speech-to-text engine with GPU acceleration.

    Listens on ``speech_queue`` for numpy int16 audio arrays,
    transcribes them, and pushes TranscriptionResult items onto
    ``text_queue``.
    """

    def __init__(
        self,
        speech_queue: "EventQueue",
        text_queue: "EventQueue",
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 1,
        language: str = "en",
        vad_filter: bool = True,
        model_dir: str = "models/stt/",
        sample_rate: int = 16_000,
        condition_on_previous: bool = False,
        initial_prompt: Optional[str] = None,
        loopback_speech_queue: Optional["EventQueue"] = None,
        word_timestamps: bool = False,
        hotwords: Optional[list] = None,
    ) -> None:
        self._speech_q = speech_queue
        self._text_q = text_queue
        self._loopback_q = loopback_speech_queue
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._language = language
        self._vad_filter = vad_filter
        self._model_dir = model_dir
        self._sr = sample_rate
        self._condition_on_previous = condition_on_previous
        self._word_timestamps = word_timestamps

        # Merge hotwords into initial_prompt for domain-specific boost
        prompt_parts = []
        if initial_prompt:
            prompt_parts.append(initial_prompt)
        if hotwords:
            prompt_parts.extend(hotwords)
        self._initial_prompt = ", ".join(prompt_parts) if prompt_parts else None

        self._model = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._heartbeat = None       # injected via set_heartbeat()
        self._beat_counter = 0

    def set_heartbeat(self, monitor) -> None:
        """Inject supervisor heartbeat monitor."""
        self._heartbeat = monitor

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Pre-load model into GPU VRAM. Call once at startup."""
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Faster-Whisper model '%s' on %s (%s)…",
            self._model_size,
            self._device,
            self._compute_type,
        )
        t0 = time.monotonic()
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
            download_root=self._model_dir,
            cpu_threads=4,
        )
        elapsed = time.monotonic() - t0
        logger.info("Faster-Whisper loaded in %.1f s", elapsed)

    def start(self) -> None:
        if self._model is None:
            self.load_model()
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="STT", daemon=True,
        )
        self._thread.start()
        logger.info("STT thread started")

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("STT thread stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        last_beat = time.monotonic()
        try:
            while self._running.is_set():
                # Periodic heartbeat even when idle (every 5s)
                now = time.monotonic()
                if self._heartbeat and (now - last_beat) >= 5.0:
                    self._heartbeat.beat("STT")
                    last_beat = now

                # Poll mic queue (primary)
                audio = self._speech_q.get(timeout=0.1)
                if audio is not None:
                    self._transcribe(audio, source="mic")
                    if self._heartbeat:
                        self._heartbeat.beat("STT")
                        last_beat = time.monotonic()
                    continue

                # Poll loopback queue (secondary) if configured
                if self._loopback_q is not None:
                    lb_audio = self._loopback_q.get(timeout=0.1)
                    if lb_audio is not None:
                        self._transcribe(lb_audio, source="speaker")
                        if self._heartbeat:
                            self._heartbeat.beat("STT")
                            last_beat = time.monotonic()
        except Exception:
            logger.exception("STT thread crashed")
            self._running.clear()

    def _transcribe(self, audio_int16: np.ndarray, source: str = "mic") -> None:
        """Run transcription and push results."""
        # Faster-Whisper expects float32 normalised [-1, 1]
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        duration_sec = len(audio_f32) / self._sr

        t0 = time.monotonic()

        segments, info = self._model.transcribe(
            audio_f32,
            beam_size=self._beam_size,
            language=self._language,
            vad_filter=self._vad_filter,
            condition_on_previous_text=self._condition_on_previous,
            initial_prompt=self._initial_prompt,
            word_timestamps=self._word_timestamps,
        )

        full_text_parts: list[str] = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                # Push partial results for streaming feel
                partial = TranscriptionResult(
                    text=text,
                    is_partial=True,
                    language=info.language,
                    confidence=seg.avg_logprob,
                    duration_sec=seg.end - seg.start,
                    source=source,
                )
                self._text_q.put(partial)
                full_text_parts.append(text)

        elapsed = time.monotonic() - t0
        full_text = " ".join(full_text_parts)

        tag = "[MIC]" if source == "mic" else "[SPEAKER]"
        if full_text:
            final = TranscriptionResult(
                text=full_text,
                is_partial=False,
                language=info.language,
                confidence=info.language_probability,
                duration_sec=duration_sec,
                source=source,
            )
            self._text_q.put(final)
            logger.info(
                "STT %s [%.2fs audio → %.2fs proc]: %s",
                tag,
                duration_sec,
                elapsed,
                full_text[:120],
            )
        else:
            logger.debug("STT %s returned empty text for %.2fs audio", tag, duration_sec)

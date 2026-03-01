"""
Grotesque AI – Wake Word Engine (Thread 3)

Listens for a configurable wake word before activating the STT → LLM
pipeline.  Supports two backends:

1. **OpenWakeWord** – fully offline, no API key, Apache-2.0
2. **Porcupine** – Picovoice on-device, needs offline .ppn keyword file

After wake word detection the engine signals the pipeline to begin
transcription of the current speech segment.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from core.buffers import AudioRingBuffer

logger = logging.getLogger("grotesque.wake_word")


class WakeWordEngine:
    """
    Monitors audio and fires a callback on wake-word detection.

    The engine runs its own thread, reads directly from the capture ring
    buffer (independent of VAD), and uses a lightweight model that fits
    entirely on CPU.
    """

    def __init__(
        self,
        ring_buffer: "AudioRingBuffer",
        engine: str = "openwakeword",
        model_path: str = "models/wake/",
        keyword: str = "hey_jarvis",
        threshold: float = 0.5,
        sample_rate: int = 16_000,
        frame_duration_ms: int = 80,  # OWW works on 80 ms chunks
        on_wake: Optional[Callable[[], None]] = None,
        # Porcupine-specific
        porcupine_model_path: Optional[str] = None,
        porcupine_keyword_path: Optional[str] = None,
    ) -> None:
        self._ring = ring_buffer
        self._engine_name = engine.lower()
        self._model_path = model_path
        self._keyword = keyword
        self._threshold = threshold
        self._sr = sample_rate
        self._frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self._on_wake = on_wake
        self._porcupine_model = porcupine_model_path
        self._porcupine_kw = porcupine_keyword_path

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._engine = None

        # Cooldown – prevent double-trigger
        self._cooldown_sec = 2.0
        self._last_trigger = 0.0

        # Flag set when wake word detected, cleared after pipeline uses it
        self._activated = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._load_engine()
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="WakeWord", daemon=True,
        )
        self._thread.start()
        logger.info("WakeWord engine started (backend=%s)", self._engine_name)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self._unload_engine()
        logger.info("WakeWord engine stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def activated(self) -> bool:
        return self._activated.is_set()

    def clear_activation(self) -> None:
        self._activated.clear()

    # ------------------------------------------------------------------
    # Engine loading
    # ------------------------------------------------------------------

    def _load_engine(self) -> None:
        if self._engine_name == "openwakeword":
            self._load_openwakeword()
        elif self._engine_name == "porcupine":
            self._load_porcupine()
        else:
            raise ValueError(f"Unknown wake word engine: {self._engine_name}")

    def _load_openwakeword(self) -> None:
        try:
            from openwakeword.model import Model as OWWModel
        except ImportError:
            raise RuntimeError(
                "openwakeword not installed. Run: pip install openwakeword"
            )
        self._engine = OWWModel(
            wakeword_models=[self._keyword],
            inference_framework="onnx",
        )
        logger.info("OpenWakeWord model loaded: %s", self._keyword)

    def _load_porcupine(self) -> None:
        try:
            import pvporcupine
        except ImportError:
            raise RuntimeError("pvporcupine not installed")
        if not self._porcupine_kw:
            raise ValueError("porcupine_keyword_path is required")
        self._engine = pvporcupine.create(
            keyword_paths=[self._porcupine_kw],
            model_path=self._porcupine_model,
        )
        # Porcupine uses fixed 512-sample frames at 16 kHz
        self._frame_samples = self._engine.frame_length
        logger.info("Porcupine loaded with keyword: %s", self._porcupine_kw)

    def _unload_engine(self) -> None:
        if self._engine_name == "porcupine" and self._engine:
            self._engine.delete()
        self._engine = None

    # ------------------------------------------------------------------
    # Detection loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while self._running.is_set():
                frame = self._ring.peek(self._frame_samples)
                if frame is None or len(frame) < self._frame_samples:
                    time.sleep(0.02)
                    continue

                # Advance read pointer (wake word reads independently)
                self._ring.read(self._frame_samples)

                detected = self._detect(frame)
                if detected:
                    now = time.monotonic()
                    if now - self._last_trigger > self._cooldown_sec:
                        self._last_trigger = now
                        self._activated.set()
                        logger.info("Wake word detected!")
                        if self._on_wake:
                            self._on_wake()

        except Exception:
            logger.exception("WakeWord thread crashed")
            self._running.clear()

    def _detect(self, frame: np.ndarray) -> bool:
        if self._engine_name == "openwakeword":
            return self._detect_oww(frame)
        elif self._engine_name == "porcupine":
            return self._detect_porcupine(frame)
        return False

    def _detect_oww(self, frame: np.ndarray) -> bool:
        # OpenWakeWord expects float32 in [-1, 1] or int16
        prediction = self._engine.predict(frame)
        for kw, score in prediction.items():
            if score >= self._threshold:
                logger.debug("OWW score %.3f for '%s'", score, kw)
                return True
        return False

    def _detect_porcupine(self, frame: np.ndarray) -> bool:
        result = self._engine.process(frame.tolist())
        return result >= 0

"""
Grotesque AI – WebRTC Voice Activity Detection (Thread 2)

Reads 20 ms frames from the capture ring buffer, runs WebRTC VAD,
and emits speech segments to the downstream queue.

Implements:
 • Aggressive filtering (mode 3)
 • Speech padding (prepend + append)
 • Minimum speech duration gate
 • Silence-based end-of-utterance detection
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

if TYPE_CHECKING:
    from core.buffers import AudioRingBuffer, EventQueue

logger = logging.getLogger("grotesque.vad")

# Allowed frame durations for WebRTC VAD
_ALLOWED_FRAME_MS = (10, 20, 30)


class VoiceActivityDetector:
    """
    Continuously reads from the audio ring buffer, applies WebRTC VAD,
    and pushes complete speech chunks to an EventQueue for downstream.
    """

    def __init__(
        self,
        ring_buffer: "AudioRingBuffer",
        speech_queue: "EventQueue",
        sample_rate: int = 16_000,
        frame_duration_ms: int = 20,
        aggressiveness: int = 3,
        speech_pad_ms: int = 300,
        min_speech_ms: int = 250,
        silence_limit_ms: int = 1200,
    ) -> None:
        assert frame_duration_ms in _ALLOWED_FRAME_MS
        assert 0 <= aggressiveness <= 3

        self._ring = ring_buffer
        self._queue = speech_queue
        self._sr = sample_rate
        self._frame_ms = frame_duration_ms
        self._frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self._aggressiveness = aggressiveness
        self._pad_frames = int(speech_pad_ms / frame_duration_ms)
        self._min_speech_frames = int(min_speech_ms / frame_duration_ms)
        self._silence_limit_frames = int(silence_limit_ms / frame_duration_ms)

        self._vad: webrtcvad.Vad | None = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        # State
        self._is_speaking = False
        self._speech_frames: list[np.ndarray] = []
        self._silence_count = 0
        self._ring_history: list[np.ndarray] = []  # pre-speech padding
        self._heartbeat = None  # set via set_heartbeat()
        self._beat_counter = 0

    def set_heartbeat(self, monitor) -> None:
        """Inject supervisor heartbeat monitor."""
        self._heartbeat = monitor

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if webrtcvad is None:
            raise RuntimeError("webrtcvad not installed")
        self._vad = webrtcvad.Vad(self._aggressiveness)
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="VAD", daemon=True
        )
        self._thread.start()
        logger.info("VAD started (mode=%d)", self._aggressiveness)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        logger.info("VAD stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while self._running.is_set():
                frame = self._ring.read(self._frame_samples)
                if frame is None or len(frame) < self._frame_samples:
                    time.sleep(self._frame_ms / 1000 * 0.5)  # back-off
                    continue

                is_speech = self._vad.is_speech(
                    frame.tobytes(), self._sr
                )

                if is_speech:
                    self._handle_speech(frame)
                else:
                    self._handle_silence(frame)

                # Send heartbeat every ~100 frames (~2 s at 20 ms)
                self._beat_counter += 1
                if self._beat_counter >= 100 and self._heartbeat:
                    self._heartbeat.beat("VAD")
                    self._beat_counter = 0

        except Exception:
            logger.exception("VAD thread crashed")
            self._running.clear()

    def _handle_speech(self, frame: np.ndarray) -> None:
        if not self._is_speaking:
            self._is_speaking = True
            # Pre-pad with recent non-speech frames
            self._speech_frames = list(self._ring_history[-self._pad_frames:])
            logger.debug("Speech start detected")

        self._speech_frames.append(frame)
        self._silence_count = 0

    def _handle_silence(self, frame: np.ndarray) -> None:
        if self._is_speaking:
            self._speech_frames.append(frame)  # pad trailing silence
            self._silence_count += 1

            if self._silence_count >= self._silence_limit_frames:
                self._finalize_utterance()
        else:
            # Keep a rolling history for pre-padding
            self._ring_history.append(frame)
            if len(self._ring_history) > self._pad_frames + 5:
                self._ring_history.pop(0)

    def _finalize_utterance(self) -> None:
        if len(self._speech_frames) >= self._min_speech_frames:
            audio = np.concatenate(self._speech_frames)
            self._queue.put(audio)
            logger.debug(
                "Utterance finalized: %.2f sec",
                len(audio) / self._sr,
            )
        else:
            logger.debug("Utterance too short, discarded")

        self._speech_frames.clear()
        self._silence_count = 0
        self._is_speaking = False

    def clear_state(self) -> None:
        """Reset all internal state (security wipe)."""
        self._speech_frames.clear()
        self._ring_history.clear()
        self._silence_count = 0
        self._is_speaking = False

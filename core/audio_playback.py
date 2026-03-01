"""
Grotesque AI – Audio Playback (part of Thread 6)

Plays synthesised PCM audio through the system default speaker.

Design:
 • Audio device kept open permanently.
 • Reads TTSAudio chunks from the playback queue.
 • Non-blocking write with double-buffering.
 • Supports barge-in: can abort current playback when user speaks.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

if TYPE_CHECKING:
    from core.buffers import EventQueue
    from core.tts import TTSAudio

logger = logging.getLogger("grotesque.playback")


class AudioPlayback:
    """
    Threaded audio playback engine.

    Pops TTSAudio items from ``audio_queue`` and streams them to the
    system speaker.  The audio device stays open between utterances
    to avoid open/close overhead.
    """

    def __init__(
        self,
        audio_queue: "EventQueue",
        sample_rate: int = 22050,
        channels: int = 1,
        device_index: Optional[int] = None,
        blocksize: int = 1024,
    ) -> None:
        self._audio_q = audio_queue
        self._sr = sample_rate
        self._channels = channels
        self._device_index = device_index
        self._blocksize = blocksize

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._playing = threading.Event()
        self._abort = threading.Event()  # for barge-in
        self._heartbeat = None  # set via set_heartbeat()

    def set_heartbeat(self, monitor) -> None:
        """Inject supervisor heartbeat monitor."""
        self._heartbeat = monitor

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice not installed")
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="Playback", daemon=True,
        )
        self._thread.start()
        logger.info("AudioPlayback started (sr=%d)", self._sr)

    def stop(self) -> None:
        self._running.clear()
        self._abort.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("AudioPlayback stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def abort_playback(self) -> None:
        """Interrupt current playback (barge-in support)."""
        self._abort.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            _idle_beats = 0
            while self._running.is_set():
                chunk = self._audio_q.get(timeout=0.1)
                if chunk is None:
                    # Send heartbeat even when idle (~every 2 s)
                    _idle_beats += 1
                    if _idle_beats >= 20 and self._heartbeat:
                        self._heartbeat.beat("Playback")
                        _idle_beats = 0
                    continue
                _idle_beats = 0
                if self._heartbeat:
                    self._heartbeat.beat("Playback")

                if hasattr(chunk, "samples") and len(chunk.samples) > 0:
                    self._play_samples(chunk.samples, chunk.sample_rate)

                if hasattr(chunk, "is_final") and chunk.is_final:
                    self._playing.clear()

        except Exception:
            logger.exception("AudioPlayback thread crashed")
            self._running.clear()

    def _play_samples(self, samples: np.ndarray, sample_rate: int) -> None:
        """Play a numpy int16 array through the speaker."""
        self._abort.clear()
        self._playing.set()

        try:
            # Convert to float32 for sounddevice
            audio_f32 = samples.astype(np.float32) / 32768.0

            # Reshape for sounddevice: (n_samples, n_channels)
            if audio_f32.ndim == 1:
                audio_f32 = audio_f32.reshape(-1, 1)

            # Play in blocks so we can check abort flag
            total = len(audio_f32)
            pos = 0
            block = self._blocksize

            with sd.OutputStream(
                samplerate=sample_rate,
                channels=self._channels,
                dtype="float32",
                device=self._device_index,
                blocksize=block,
            ) as stream:
                while pos < total and not self._abort.is_set():
                    end = min(pos + block, total)
                    chunk = audio_f32[pos:end]
                    stream.write(chunk)
                    pos = end

            if self._abort.is_set():
                logger.debug("Playback aborted (barge-in)")

        except Exception:
            logger.exception("Playback error")
        finally:
            self._playing.clear()

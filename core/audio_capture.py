"""
Grotesque AI – Audio Capture (Thread 1)

Continuously reads from the system microphone in 20 ms frames and
pushes raw PCM int16 samples into the shared ring buffer.

The audio device is opened ONCE at startup and kept open permanently
to avoid per-request latency.  PyAudio callback runs in a C-level
thread for minimal jitter.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import pyaudio
except ImportError:
    pyaudio = None

if TYPE_CHECKING:
    from core.buffers import AudioRingBuffer

logger = logging.getLogger("grotesque.audio_capture")


class AudioCapture:
    """
    Microphone capture thread.  Writes into an AudioRingBuffer.

    Uses sounddevice (PortAudio) by default; falls back to PyAudio.
    """

    def __init__(
        self,
        ring_buffer: "AudioRingBuffer",
        sample_rate: int = 16_000,
        channels: int = 1,
        frame_duration_ms: int = 20,
        device_index: int | None = None,
    ) -> None:
        self._ring = ring_buffer
        self._sr = sample_rate
        self._channels = channels
        self._frame_size = int(sample_rate * frame_duration_ms / 1000)
        self._device_index = device_index
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._heartbeat = None  # set via set_heartbeat()
        self._beat_counter = 0

    def set_heartbeat(self, monitor) -> None:
        """Inject supervisor heartbeat monitor."""
        self._heartbeat = monitor

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="AudioCapture", daemon=True
        )
        self._thread.start()
        logger.info("AudioCapture started (sr=%d, frame=%d)", self._sr, self._frame_size)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self._close_stream()
        logger.info("AudioCapture stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main capture loop using sounddevice (blocking read)."""
        try:
            if sd is not None:
                self._run_sounddevice()
            elif pyaudio is not None:
                self._run_pyaudio()
            else:
                raise RuntimeError("No audio backend available (install sounddevice or pyaudio)")
        except Exception:
            logger.exception("AudioCapture thread crashed")
            self._running.clear()

    def _run_sounddevice(self) -> None:
        with sd.InputStream(
            samplerate=self._sr,
            channels=self._channels,
            dtype="int16",
            blocksize=self._frame_size,
            device=self._device_index,
        ) as stream:
            self._stream = stream
            while self._running.is_set():
                data, overflowed = stream.read(self._frame_size)
                if overflowed:
                    logger.debug("Audio input overflow")
                # data shape: (frame_size, channels) – flatten to 1-D
                samples = data[:, 0] if self._channels == 1 else data.flatten()
                self._ring.write(samples)
                # Send heartbeat every ~100 frames (~2 s at 20 ms)
                self._beat_counter += 1
                if self._beat_counter >= 100 and self._heartbeat:
                    self._heartbeat.beat("AudioCapture")
                    self._beat_counter = 0
        self._stream = None

    def _run_pyaudio(self) -> None:
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._sr,
                input=True,
                frames_per_buffer=self._frame_size,
                input_device_index=self._device_index,
            )
            self._stream = stream
            while self._running.is_set():
                raw = stream.read(self._frame_size, exception_on_overflow=False)
                samples = np.frombuffer(raw, dtype=np.int16)
                self._ring.write(samples)
                # Send heartbeat every ~100 frames (~2 s at 20 ms)
                self._beat_counter += 1
                if self._beat_counter >= 100 and self._heartbeat:
                    self._heartbeat.beat("AudioCapture")
                    self._beat_counter = 0
        finally:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
            pa.terminate()

    def _close_stream(self) -> None:
        s = self._stream
        if s is not None:
            try:
                if hasattr(s, "stop_stream"):
                    s.stop_stream()
                    s.close()
                elif hasattr(s, "abort"):
                    s.abort()
            except Exception:
                pass
            self._stream = None

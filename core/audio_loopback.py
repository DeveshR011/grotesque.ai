"""
Grotesque AI – Audio Loopback Capture (Speaker Output)

Captures system audio output (what the PC is playing) via Windows
WASAPI loopback using pyaudiowpatch.  This enables Parakeet-style
awareness of music, videos, calls, and other PC audio.

Design:
 • Uses WASAPI loopback to capture the default speaker output
 • Resamples from native device rate (e.g. 48 000 Hz) to 16 000 Hz
 • Writes 20 ms int16 PCM frames into a dedicated ring buffer
 • Runs on its own daemon thread named "LoopbackCapture"
 • Gracefully degrades: logs a warning and exits if no loopback device
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

try:
    import pyaudiowpatch as pyaudio_wp
except ImportError:
    pyaudio_wp = None

if TYPE_CHECKING:
    from core.buffers import AudioRingBuffer

logger = logging.getLogger("grotesque.loopback")


class AudioLoopback:
    """
    WASAPI loopback capture thread.  Writes system audio into an
    AudioRingBuffer at 16 000 Hz mono int16 – same format as
    AudioCapture for seamless VAD/STT downstream processing.
    """

    def __init__(
        self,
        ring_buffer: "AudioRingBuffer",
        sample_rate: int = 16_000,
        channels: int = 1,
        frame_duration_ms: int = 20,
        device_index: Optional[int] = None,
    ) -> None:
        self._ring = ring_buffer
        self._target_sr = sample_rate
        self._target_channels = channels
        self._frame_duration_ms = frame_duration_ms
        self._frame_size = int(sample_rate * frame_duration_ms / 1000)
        self._device_index = device_index

        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
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
        if pyaudio_wp is None:
            logger.warning(
                "pyaudiowpatch not installed – loopback capture disabled. "
                "Install with: pip install pyaudiowpatch"
            )
            return
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="LoopbackCapture", daemon=True,
        )
        self._thread.start()
        logger.info("LoopbackCapture started (target sr=%d)", self._target_sr)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self._close_stream()
        logger.info("LoopbackCapture stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    def _find_loopback_device(self, pa: "pyaudio_wp.PyAudio") -> Optional[dict]:
        """
        Auto-detect the default WASAPI loopback device.
        Returns the device info dict, or None if not found.
        """
        if self._device_index is not None:
            try:
                info = pa.get_device_info_by_index(self._device_index)
                logger.info(
                    "Using explicit loopback device %d: %s",
                    self._device_index, info.get("name", "?"),
                )
                return info
            except Exception:
                logger.warning(
                    "Specified loopback device_index %d not found, "
                    "falling back to auto-detect",
                    self._device_index,
                )

        # --- Auto-detect via WASAPI loopback ---
        try:
            # pyaudiowpatch provides get_default_wasapi_loopback()
            wasapi_info = pa.get_host_api_info_by_type(
                pyaudio_wp.paWASAPI
            )
            default_speakers = pa.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )

            # Search for the loopback device matching the default speakers
            for i in range(pa.get_device_count()):
                dev = pa.get_device_info_by_index(i)
                if (
                    dev.get("name", "").startswith(default_speakers["name"])
                    and dev.get("isLoopbackDevice", False)
                ):
                    logger.info(
                        "Auto-detected WASAPI loopback device %d: %s (%.0f Hz, %d ch)",
                        i, dev["name"],
                        dev["defaultSampleRate"],
                        dev["maxInputChannels"],
                    )
                    return dev

            logger.warning(
                "No WASAPI loopback device found for default speakers '%s'",
                default_speakers.get("name", "?"),
            )
        except Exception:
            logger.warning("WASAPI loopback auto-detection failed", exc_info=True)

        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loopback capture loop."""
        pa = pyaudio_wp.PyAudio()
        try:
            device = self._find_loopback_device(pa)
            if device is None:
                logger.warning(
                    "No loopback device available – LoopbackCapture thread exiting. "
                    "Ensure a WASAPI-compatible speaker is active."
                )
                self._running.clear()
                return

            device_sr = int(device["defaultSampleRate"])
            device_ch = int(device["maxInputChannels"])
            device_idx = int(device["index"])

            # Compute device-native frame size so we read ~20 ms of audio
            device_frame_size = int(device_sr * self._frame_duration_ms / 1000)

            # We may need to resample from device_sr → 16 000 Hz
            need_resample = (device_sr != self._target_sr)
            resample_fn = None
            if need_resample:
                resample_fn = self._make_resampler(device_sr, self._target_sr)

            stream = pa.open(
                format=pyaudio_wp.paInt16,
                channels=device_ch,
                rate=device_sr,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=device_frame_size,
            )
            self._stream = stream

            logger.info(
                "Loopback stream opened: device=%d, sr=%d, ch=%d, framesize=%d",
                device_idx, device_sr, device_ch, device_frame_size,
            )

            while self._running.is_set():
                try:
                    raw = stream.read(device_frame_size, exception_on_overflow=False)
                except Exception:
                    logger.debug("Loopback read error", exc_info=True)
                    time.sleep(0.01)
                    continue

                samples = np.frombuffer(raw, dtype=np.int16)

                # Downmix to mono if multi-channel
                if device_ch > 1:
                    samples = samples.reshape(-1, device_ch)
                    samples = samples.mean(axis=1).astype(np.int16)

                # Resample to target rate
                if resample_fn is not None:
                    samples = resample_fn(samples)

                self._ring.write(samples)

                # Heartbeat every ~50 frames (~1 s at 20 ms)
                self._beat_counter += 1
                if self._beat_counter >= 50 and self._heartbeat:
                    self._heartbeat.beat("LoopbackCapture")
                    self._beat_counter = 0

        except Exception:
            logger.exception("LoopbackCapture thread crashed")
        finally:
            self._close_stream()
            if pa:
                pa.terminate()
            self._running.clear()

    @staticmethod
    def _make_resampler(src_sr: int, dst_sr: int):
        """
        Return a callable that resamples int16 audio from src_sr to dst_sr.
        Uses simple integer-ratio decimation when possible; falls back to
        resampy for arbitrary ratios.
        """
        ratio = src_sr / dst_sr

        # Fast path: integer decimation (e.g. 48000→16000 = 3:1)
        if src_sr % dst_sr == 0:
            step = src_sr // dst_sr

            def _decimate(samples: np.ndarray) -> np.ndarray:
                return samples[::step].copy()

            return _decimate

        # Slow path: arbitrary resampling via resampy
        try:
            import resampy

            def _resample(samples: np.ndarray) -> np.ndarray:
                f32 = samples.astype(np.float32) / 32768.0
                resampled = resampy.resample(f32, src_sr, dst_sr)
                return (resampled * 32768.0).clip(-32768, 32767).astype(np.int16)

            return _resample
        except ImportError:
            logger.warning(
                "resampy not installed; using naive decimation for %d→%d Hz "
                "(quality may be reduced). Install with: pip install resampy",
                src_sr, dst_sr,
            )
            step = max(1, round(ratio))

            def _naive(samples: np.ndarray) -> np.ndarray:
                return samples[::step].copy()

            return _naive

    def _close_stream(self) -> None:
        s = self._stream
        if s is not None:
            try:
                s.stop_stream()
                s.close()
            except Exception:
                pass
            self._stream = None

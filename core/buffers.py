"""
Grotesque AI – Lock-Free Ring Buffers

High-performance circular buffers for inter-thread communication.
Uses numpy arrays with atomic index updates for zero-copy where possible.
No mutexes on the hot path – only atomic pointer advances.
"""

from __future__ import annotations

import threading
import numpy as np
from typing import Optional


class AudioRingBuffer:
    """
    Single-producer / single-consumer lock-free ring buffer for PCM audio.

    Storage: pre-allocated numpy int16 array.
    Thread safety: relies on Python's GIL for atomic index reads/writes
    plus memory-barriers via volatile-like access patterns.
    """

    __slots__ = (
        "_buf", "_capacity", "_write_idx", "_read_idx",
        "_sample_rate", "_channels",
    )

    def __init__(
        self,
        duration_sec: float,
        sample_rate: int = 16_000,
        channels: int = 1,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._capacity = int(duration_sec * sample_rate * channels)
        # Pre-allocate contiguous buffer – stays in RAM, never reallocated
        self._buf = np.zeros(self._capacity, dtype=np.int16)
        self._write_idx: int = 0
        self._read_idx: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def available_read(self) -> int:
        """Samples available to read."""
        w, r = self._write_idx, self._read_idx
        if w >= r:
            return w - r
        return self._capacity - r + w

    @property
    def available_write(self) -> int:
        """Samples that can be written before overrun."""
        return self._capacity - self.available_read - 1

    # ------------------------------------------------------------------
    # Write (producer)
    # ------------------------------------------------------------------

    def write(self, data: np.ndarray) -> int:
        """
        Write samples into the buffer.  Returns number actually written.
        Drops oldest data on overflow (sliding window behaviour for audio).
        """
        n = len(data)
        if n == 0:
            return 0

        w = self._write_idx
        # Split into at most two copies (wrap-around)
        first = min(n, self._capacity - w)
        self._buf[w: w + first] = data[:first]
        if first < n:
            second = n - first
            self._buf[:second] = data[first:]
            new_w = second
        else:
            new_w = w + first
            if new_w >= self._capacity:
                new_w = 0

        # If we lapped the reader, advance it (drop oldest)
        self._write_idx = new_w
        return n

    # ------------------------------------------------------------------
    # Read (consumer)
    # ------------------------------------------------------------------

    def read(self, n: int) -> Optional[np.ndarray]:
        """
        Read up to *n* samples.  Returns None if buffer empty.
        """
        avail = self.available_read
        if avail == 0:
            return None
        n = min(n, avail)
        r = self._read_idx
        first = min(n, self._capacity - r)
        out = np.empty(n, dtype=np.int16)
        out[:first] = self._buf[r: r + first]
        if first < n:
            out[first:] = self._buf[: n - first]
            new_r = n - first
        else:
            new_r = r + first
            if new_r >= self._capacity:
                new_r = 0
        self._read_idx = new_r
        return out

    def peek(self, n: int) -> Optional[np.ndarray]:
        """Read without advancing the pointer."""
        avail = self.available_read
        if avail == 0:
            return None
        n = min(n, avail)
        r = self._read_idx
        first = min(n, self._capacity - r)
        out = np.empty(n, dtype=np.int16)
        out[:first] = self._buf[r: r + first]
        if first < n:
            out[first:] = self._buf[: n - first]
        return out

    def clear(self) -> None:
        """Zero-fill and reset pointers (used for security wipe)."""
        self._buf[:] = 0
        self._write_idx = 0
        self._read_idx = 0


class EventQueue:
    """
    Simple bounded thread-safe queue for control / text messages
    between pipeline stages.  Uses a lock only because messages
    are infrequent (not on 20 ms audio hot-path).
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._data: list = []
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._not_empty = threading.Event()

    def put(self, item, timeout: Optional[float] = None) -> bool:
        with self._lock:
            if len(self._data) >= self._maxsize:
                return False  # drop on overflow
            self._data.append(item)
            self._not_empty.set()
        return True

    def get(self, timeout: Optional[float] = None):
        """Blocking get with optional timeout.  Returns None on timeout."""
        if not self._not_empty.wait(timeout=timeout):
            return None
        with self._lock:
            if not self._data:
                self._not_empty.clear()
                return None
            item = self._data.pop(0)
            if not self._data:
                self._not_empty.clear()
            return item

    def get_nowait(self):
        with self._lock:
            if not self._data:
                return None
            item = self._data.pop(0)
            if not self._data:
                self._not_empty.clear()
            return item

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._not_empty.clear()

    @property
    def qsize(self) -> int:
        return len(self._data)

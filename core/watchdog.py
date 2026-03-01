"""
Grotesque AI – Watchdog Thread (Thread 7)

Monitors health of all pipeline threads and restarts them if they crash.

Also handles:
 • Periodic VRAM usage check
 • Memory leak detection (RSS growth)
 • Idle buffer wiping (security)
 • Graceful shutdown coordination
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger("grotesque.watchdog")


class ComponentHandle:
    """Tracks a single pipeline component for the watchdog."""

    __slots__ = ("name", "obj", "start_fn", "max_restarts", "restart_count", "last_restart")

    def __init__(
        self,
        name: str,
        obj,
        start_fn: Callable,
        max_restarts: int = 10,
    ) -> None:
        self.name = name
        self.obj = obj
        self.start_fn = start_fn
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.last_restart = 0.0


class Watchdog:
    """
    Continuously monitors pipeline components and system resources.
    Restarts crashed threads and logs health metrics.
    """

    def __init__(
        self,
        check_interval_sec: float = 5.0,
        idle_clear_timeout_sec: float = 300.0,
        max_rss_mb: float = 8192.0,      # 8 GB max RSS
        max_vram_mb: float = 7000.0,      # 7 GB VRAM ceiling
        on_fatal: Optional[Callable] = None,
    ) -> None:
        self._interval = check_interval_sec
        self._idle_clear_timeout = idle_clear_timeout_sec
        self._max_rss_mb = max_rss_mb
        self._max_vram_mb = max_vram_mb
        self._on_fatal = on_fatal

        self._components: Dict[str, ComponentHandle] = {}
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_activity = time.monotonic()

        # Buffers to wipe on idle (registered by pipeline)
        self._wipeable_buffers: list = []
        self._wipe_callbacks: list[Callable] = []

        self._process = psutil.Process(os.getpid())

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        obj,
        start_fn: Callable,
        max_restarts: int = 10,
    ) -> None:
        self._components[name] = ComponentHandle(name, obj, start_fn, max_restarts)

    def register_wipeable(self, buffer) -> None:
        self._wipeable_buffers.append(buffer)

    def register_wipe_callback(self, fn: Callable) -> None:
        self._wipe_callbacks.append(fn)

    def report_activity(self) -> None:
        """Called by pipeline when user interaction detected."""
        self._last_activity = time.monotonic()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="Watchdog", daemon=True,
        )
        self._thread.start()
        logger.info("Watchdog started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("Watchdog stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running.is_set():
            try:
                self._check_components()
                self._check_memory()
                self._check_idle_wipe()
            except Exception:
                logger.exception("Watchdog check error")

            # Sleep in small increments so stop() is responsive
            for _ in range(int(self._interval * 10)):
                if not self._running.is_set():
                    return
                time.sleep(0.1)

    def _check_components(self) -> None:
        for name, handle in self._components.items():
            alive = False
            if hasattr(handle.obj, "is_alive"):
                alive = handle.obj.is_alive
            elif hasattr(handle.obj, "is_alive") and callable(handle.obj.is_alive):
                alive = handle.obj.is_alive()

            if not alive:
                if handle.restart_count >= handle.max_restarts:
                    logger.critical(
                        "Component '%s' exceeded max restarts (%d). FATAL.",
                        name, handle.max_restarts,
                    )
                    if self._on_fatal:
                        self._on_fatal()
                    return

                logger.warning(
                    "Component '%s' is dead. Restarting (%d/%d)…",
                    name, handle.restart_count + 1, handle.max_restarts,
                )
                try:
                    handle.start_fn()
                    handle.restart_count += 1
                    handle.last_restart = time.monotonic()
                except Exception:
                    logger.exception("Failed to restart '%s'", name)

    def _check_memory(self) -> None:
        try:
            mem = self._process.memory_info()
            rss_mb = mem.rss / (1024 * 1024)
            if rss_mb > self._max_rss_mb:
                logger.warning(
                    "RSS %.0f MB exceeds limit %.0f MB – forcing GC",
                    rss_mb, self._max_rss_mb,
                )
                gc.collect()
        except Exception:
            pass

        # VRAM check
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                vram_used = gpus[0].memoryUsed
                if vram_used > self._max_vram_mb:
                    logger.warning(
                        "VRAM %.0f MB exceeds target %.0f MB",
                        vram_used, self._max_vram_mb,
                    )
        except ImportError:
            pass
        except Exception:
            pass

    def _check_idle_wipe(self) -> None:
        """Wipe audio/text buffers after idle period (security)."""
        idle_sec = time.monotonic() - self._last_activity
        if idle_sec < self._idle_clear_timeout:
            return

        for buf in self._wipeable_buffers:
            if hasattr(buf, "clear"):
                buf.clear()

        for fn in self._wipe_callbacks:
            try:
                fn()
            except Exception:
                logger.debug("Wipe callback error", exc_info=True)

    def get_health_report(self) -> dict:
        """Return a snapshot of system health metrics."""
        mem = self._process.memory_info()
        report = {
            "rss_mb": mem.rss / (1024 * 1024),
            "threads": self._process.num_threads(),
            "cpu_percent": self._process.cpu_percent(interval=0.1),
            "components": {},
        }
        for name, handle in self._components.items():
            alive = getattr(handle.obj, "is_alive", False)
            if callable(alive):
                alive = alive()
            report["components"][name] = {
                "alive": alive,
                "restarts": handle.restart_count,
            }
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                g = gpus[0]
                report["gpu"] = {
                    "name": g.name,
                    "vram_used_mb": g.memoryUsed,
                    "vram_total_mb": g.memoryTotal,
                    "gpu_load": g.load,
                    "temperature": g.temperature,
                }
        except Exception:
            pass
        return report

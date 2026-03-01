"""
Grotesque AI – Supervisor Process

A higher-level monitor that sits above the Watchdog and provides:

 • Heartbeat system between all threads
 • GPU failure detection and safe fallback
 • RSS/VRAM trend monitoring (memory leak detection)
 • Automatic module restart with backoff
 • Fatal error handling and graceful degradation
 • Health report aggregation

The Supervisor is designed so that every module can be replaced
at runtime without rewriting the whole system (hot-swap ready).
"""

from __future__ import annotations

import gc
import logging
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import psutil

logger = logging.getLogger("grotesque.supervisor")


# ======================================================================
# Heartbeat system
# ======================================================================

@dataclass
class HeartbeatRecord:
    """Tracks heartbeats from a single component."""
    name: str
    last_beat: float = 0.0
    timeout_sec: float = 15.0
    missed_count: int = 0
    max_misses: int = 3
    alive: bool = True


class HeartbeatMonitor:
    """
    Components periodically call ``beat(name)`` to signal liveness.
    The supervisor checks for stale heartbeats and flags dead components.
    """

    def __init__(self) -> None:
        self._records: Dict[str, HeartbeatRecord] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        timeout_sec: float = 15.0,
        max_misses: int = 3,
    ) -> None:
        with self._lock:
            self._records[name] = HeartbeatRecord(
                name=name,
                last_beat=time.monotonic(),
                timeout_sec=timeout_sec,
                max_misses=max_misses,
            )

    def beat(self, name: str) -> None:
        """Called by a component to signal it's alive."""
        with self._lock:
            rec = self._records.get(name)
            if rec:
                rec.last_beat = time.monotonic()
                rec.missed_count = 0
                rec.alive = True

    def check_all(self) -> List[str]:
        """
        Check all heartbeats.  Returns list of names that have expired.
        """
        now = time.monotonic()
        dead: List[str] = []

        with self._lock:
            for name, rec in self._records.items():
                elapsed = now - rec.last_beat
                if elapsed > rec.timeout_sec:
                    rec.missed_count += 1
                    if rec.missed_count >= rec.max_misses:
                        rec.alive = False
                        dead.append(name)
                        logger.warning(
                            "Heartbeat expired for '%s' (missed %d, elapsed %.1fs)",
                            name, rec.missed_count, elapsed,
                        )

        return dead

    def is_alive(self, name: str) -> bool:
        rec = self._records.get(name)
        return rec.alive if rec else False

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        now = time.monotonic()
        status = {}
        with self._lock:
            for name, rec in self._records.items():
                status[name] = {
                    "alive": rec.alive,
                    "last_beat_ago_sec": now - rec.last_beat,
                    "missed_count": rec.missed_count,
                }
        return status


# ======================================================================
# GPU Health Monitor
# ======================================================================

class GPUMonitor:
    """
    Monitors GPU health including:
     • VRAM usage trends (detect leaks)
     • Temperature
     • Utilisation
     • Driver errors
     • Fallback signalling when GPU fails
    """

    def __init__(
        self,
        max_vram_mb: float = 5500.0,
        max_temperature: float = 90.0,
        vram_history_size: int = 60,
    ) -> None:
        self._max_vram = max_vram_mb
        self._max_temp = max_temperature
        self._vram_history: deque = deque(maxlen=vram_history_size)
        self._gpu_available = True
        self._fallback_active = False

    def check(self) -> Dict[str, Any]:
        """Run GPU health check. Returns status dict."""
        report = {
            "available": self._gpu_available,
            "fallback_active": self._fallback_active,
            "vram_used_mb": 0,
            "vram_total_mb": 0,
            "temperature": 0,
            "load": 0,
            "vram_trend": "stable",
        }

        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if not gpus:
                self._gpu_available = False
                report["available"] = False
                return report

            g = gpus[0]
            report["vram_used_mb"] = g.memoryUsed
            report["vram_total_mb"] = g.memoryTotal
            report["temperature"] = g.temperature
            report["load"] = g.load

            self._vram_history.append(g.memoryUsed)

            # Check VRAM ceiling
            if g.memoryUsed > self._max_vram:
                logger.warning("VRAM usage %.0f MB exceeds limit %.0f MB", g.memoryUsed, self._max_vram)

            # Check temperature
            if g.temperature > self._max_temp:
                logger.warning("GPU temperature %.0f°C exceeds limit %.0f°C", g.temperature, self._max_temp)

            # Detect VRAM leak (monotonic increase over last N checks)
            report["vram_trend"] = self._detect_vram_trend()

            self._gpu_available = True

        except ImportError:
            pass
        except Exception:
            logger.debug("GPU check error", exc_info=True)
            self._gpu_available = False
            report["available"] = False

        return report

    def _detect_vram_trend(self) -> str:
        """Analyse VRAM history for leaks."""
        if len(self._vram_history) < 10:
            return "insufficient_data"

        recent = list(self._vram_history)
        # Simple linear regression check
        half = len(recent) // 2
        mean_first = statistics.mean(recent[:half])
        mean_second = statistics.mean(recent[half:])
        delta = mean_second - mean_first

        if delta > 100:  # 100 MB growth
            logger.warning("VRAM trend: growing (Δ%.0f MB) – possible leak", delta)
            return "growing"
        elif delta < -100:
            return "shrinking"
        return "stable"

    def should_fallback_to_cpu(self) -> bool:
        """True if GPU is unavailable and CPU fallback should be used."""
        return not self._gpu_available

    def activate_fallback(self) -> None:
        self._fallback_active = True
        logger.warning("GPU fallback activated – switching to CPU mode")

    def deactivate_fallback(self) -> None:
        self._fallback_active = False
        logger.info("GPU fallback deactivated – returning to GPU mode")


# ======================================================================
# Memory Leak Detector
# ======================================================================

class MemoryLeakDetector:
    """
    Tracks RSS over time and detects monotonic growth patterns.
    """

    def __init__(self, history_size: int = 120) -> None:
        self._rss_history: deque = deque(maxlen=history_size)
        self._process = psutil.Process(os.getpid())

    def sample(self) -> float:
        rss_mb = self._process.memory_info().rss / (1024 * 1024)
        self._rss_history.append(rss_mb)
        return rss_mb

    def detect_leak(self, threshold_mb_per_hour: float = 100.0) -> Optional[str]:
        """Returns a warning message if a leak is detected, else None."""
        if len(self._rss_history) < 20:
            return None

        history = list(self._rss_history)
        half = len(history) // 2
        mean_first = statistics.mean(history[:half])
        mean_second = statistics.mean(history[half:])
        delta = mean_second - mean_first

        if delta > threshold_mb_per_hour:
            return (
                f"Possible memory leak: RSS grew {delta:.0f} MB "
                f"(first half avg: {mean_first:.0f} MB, second half avg: {mean_second:.0f} MB)"
            )
        return None

    def get_current_rss_mb(self) -> float:
        return self._process.memory_info().rss / (1024 * 1024)


# ======================================================================
# Supervisor
# ======================================================================

class Supervisor:
    """
    Top-level process supervisor.

    Aggregates HeartbeatMonitor, GPUMonitor, and MemoryLeakDetector.
    Runs on its own thread, checking health at a configurable interval.
    Coordinates automatic restarts with exponential backoff.
    """

    def __init__(
        self,
        check_interval_sec: float = 10.0,
        on_component_dead: Optional[Callable[[str], None]] = None,
        on_gpu_fallback: Optional[Callable[[], None]] = None,
        on_fatal: Optional[Callable[[], None]] = None,
        gpu_max_vram_mb: float = 7000.0,
        gpu_max_temperature: float = 90.0,
    ) -> None:
        self._interval = check_interval_sec
        self._on_component_dead = on_component_dead
        self._on_gpu_fallback = on_gpu_fallback
        self._on_fatal = on_fatal

        self.heartbeat = HeartbeatMonitor()
        self.gpu = GPUMonitor(
            max_vram_mb=gpu_max_vram_mb,
            max_temperature=gpu_max_temperature,
        )
        self.memory = MemoryLeakDetector()

        self._restart_fns: Dict[str, Callable] = {}
        self._restart_counts: Dict[str, int] = {}
        self._max_restarts = 10

        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_component(
        self,
        name: str,
        restart_fn: Callable,
        heartbeat_timeout: float = 15.0,
    ) -> None:
        self.heartbeat.register(name, timeout_sec=heartbeat_timeout)
        self._restart_fns[name] = restart_fn
        self._restart_counts[name] = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="Supervisor", daemon=True,
        )
        self._thread.start()
        logger.info("Supervisor started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=15)
            self._thread = None
        logger.info("Supervisor stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running.is_set():
            try:
                self._check_heartbeats()
                self._check_gpu()
                self._check_memory()
            except Exception:
                logger.exception("Supervisor check error")

            for _ in range(int(self._interval * 10)):
                if not self._running.is_set():
                    return
                time.sleep(0.1)

    def _check_heartbeats(self) -> None:
        dead = self.heartbeat.check_all()
        for name in dead:
            self._handle_dead_component(name)

    def _handle_dead_component(self, name: str) -> None:
        count = self._restart_counts.get(name, 0)

        if count >= self._max_restarts:
            logger.critical(
                "Component '%s' exceeded max restarts (%d). Fatal escalation.",
                name, self._max_restarts,
            )
            if self._on_fatal:
                self._on_fatal()
            return

        # Exponential backoff
        backoff = min(2 ** count, 60)
        logger.warning(
            "Restarting '%s' (attempt %d/%d, backoff %ds)",
            name, count + 1, self._max_restarts, backoff,
        )
        time.sleep(backoff)

        restart_fn = self._restart_fns.get(name)
        if restart_fn:
            try:
                restart_fn()
                self._restart_counts[name] = count + 1
                self.heartbeat.beat(name)  # reset heartbeat
                logger.info("Component '%s' restarted successfully", name)
            except Exception:
                logger.exception("Failed to restart '%s'", name)

        if self._on_component_dead:
            try:
                self._on_component_dead(name)
            except Exception:
                pass

    def _check_gpu(self) -> None:
        report = self.gpu.check()

        if not report["available"] and not self.gpu._fallback_active:
            self.gpu.activate_fallback()
            if self._on_gpu_fallback:
                try:
                    self._on_gpu_fallback()
                except Exception:
                    pass

        if report.get("vram_trend") == "growing":
            logger.warning("GPU VRAM leak detected – consider restart")

    def _check_memory(self) -> None:
        rss = self.memory.sample()

        leak_msg = self.memory.detect_leak()
        if leak_msg:
            logger.warning(leak_msg)
            gc.collect()

    # ------------------------------------------------------------------
    # Health report
    # ------------------------------------------------------------------

    def get_health_report(self) -> Dict[str, Any]:
        return {
            "heartbeats": self.heartbeat.get_status(),
            "gpu": self.gpu.check(),
            "rss_mb": self.memory.get_current_rss_mb(),
            "restart_counts": dict(self._restart_counts),
        }

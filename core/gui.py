"""
Grotesque AI – Covert System Tray Interface

Provides a minimal, inconspicuous Windows system-tray icon that:
 • Blends with native OS icons (generic speaker/audio icon)
 • Names itself like a standard Windows audio subsystem process
 • Runs the Pipeline on a background daemon thread
 • Exposes status, pause/resume, config reload, and graceful exit
 • Thread-safe communication via atomic state variables

Designed to be launched via  pythonw.exe main.py --gui  (no console)
or from the included run_covert.vbs helper script.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PIL import Image, ImageDraw, ImageFont

import pystray

from core.monitor_window import MonitorWindow

if TYPE_CHECKING:
    from core.pipeline import Pipeline

logger = logging.getLogger("grotesque.gui")

# ======================================================================
# Constants – keep everything looking like a stock Windows process
# ======================================================================

_TRAY_TITLE      = "Windows Audio Subsystem Host"
_PROCESS_NAME    = "audiodg"  # cosmetic; actual rename requires frozen exe

# Pipeline state labels
STATE_STARTING   = "Starting…"
STATE_SLEEPING   = "Sleeping"
STATE_LISTENING  = "Listening"
STATE_THINKING   = "Processing"
STATE_SPEAKING   = "Speaking"
STATE_PAUSED     = "Paused"
STATE_STOPPING   = "Shutting down…"


# ======================================================================
# Icon generation – generic speaker icon drawn with Pillow (no files)
# ======================================================================

def _create_speaker_icon(size: int = 64, colour: str = "#AAAAAA") -> Image.Image:
    """
    Draw a simple speaker-shaped glyph on a transparent canvas.
    Looks identical to the stock Windows volume icon at tray size.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Speaker body (rectangle)
    bx0 = int(size * 0.18)
    bx1 = int(size * 0.38)
    by0 = int(size * 0.32)
    by1 = int(size * 0.68)
    d.rectangle([bx0, by0, bx1, by1], fill=colour)

    # Cone (triangle)
    cx = int(size * 0.58)
    cy0 = int(size * 0.18)
    cy1 = int(size * 0.82)
    d.polygon([(bx1, by0), (cx, cy0), (cx, cy1), (bx1, by1)], fill=colour)

    # Sound waves (arcs)
    for i, radius_frac in enumerate([0.30, 0.42]):
        r = int(size * radius_frac)
        cx_arc = int(size * 0.58)
        cy_arc = int(size * 0.50)
        bbox = [cx_arc - r, cy_arc - r, cx_arc + r, cy_arc + r]
        d.arc(bbox, start=-45, end=45, fill=colour, width=max(2, size // 24))

    return img


def _create_paused_icon(size: int = 64) -> Image.Image:
    """Speaker icon with a dim/grey tint indicating microphone muted."""
    return _create_speaker_icon(size, colour="#666666")


# ======================================================================
# Console hider – aggressively prevent any console window from showing
# ======================================================================

def hide_console() -> None:
    """Force-hide the console window on Windows (idempotent)."""
    if platform.system() != "Windows":
        return
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        u32 = ctypes.WinDLL("user32", use_last_error=True)
        hwnd = k32.GetConsoleWindow()
        if hwnd:
            u32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass


# ======================================================================
# TrayApp – main class
# ======================================================================

class TrayApp:
    """
    System-tray controller.  Owns the pystray event loop (must run on
    the main thread on Windows) and launches the Pipeline on a daemon
    thread.
    """

    def __init__(self, config_path: Optional[str] = None, debug: bool = False) -> None:
        self._config_path = config_path
        self._debug = debug

        # Thread-safe state
        self._state: str = STATE_STARTING
        self._paused: bool = False
        self._lock = threading.Lock()

        # Pipeline reference (set once the bg thread creates it)
        self._pipeline: Optional["Pipeline"] = None
        self._pipeline_thread: Optional[threading.Thread] = None

        # pystray icon (set in run())
        self._icon: Optional[pystray.Icon] = None

        # Icons (generated once)
        self._icon_normal = _create_speaker_icon()
        self._icon_paused = _create_paused_icon()

        # Monitor window
        self._monitor: Optional[MonitorWindow] = None
        self._monitor_visible: bool = True

    # ------------------------------------------------------------------
    # State helpers (thread-safe)
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @state.setter
    def state(self, value: str) -> None:
        with self._lock:
            self._state = value
        # Propagate state to monitor window
        if self._monitor:
            self._monitor.set_status(value)
        # Update tooltip live (include wake mode if pipeline is loaded)
        if self._icon:
            try:
                mode_label = ""
                if self._pipeline and hasattr(self._pipeline, 'cfg'):
                    wm = (self._pipeline.cfg or {}).get("wake_word", {}).get("mode", "")
                    if wm:
                        mode_label = f"  ({wm})"
                self._icon.title = f"{_TRAY_TITLE}  [{value}]{mode_label}"
            except Exception:
                pass

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    # ------------------------------------------------------------------
    # Menu builder
    # ------------------------------------------------------------------

    def _build_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem(
                lambda _text: f"Status: {self.state}",
                action=None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda _text: "Resume Listening" if self.paused else "Pause Listening",
                self._on_toggle_pause,
            ),
            pystray.MenuItem(
                lambda _text: "Hide Monitor" if self._monitor_visible else "Show Monitor",
                self._on_toggle_monitor,
            ),
            pystray.MenuItem(
                "Reload Configuration",
                self._on_reload_config,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", self._on_exit),
        )

    # ------------------------------------------------------------------
    # Menu callbacks
    # ------------------------------------------------------------------

    def _on_toggle_pause(self, icon, item) -> None:
        with self._lock:
            self._paused = not self._paused
        if self.paused:
            self.state = STATE_PAUSED
            icon.icon = self._icon_paused
            logger.info("Listening paused by user")
            # Mute capture – stop feeding the ring buffer
            if self._pipeline and self._pipeline._audio_capture:
                self._pipeline._audio_capture.stop()
        else:
            self.state = STATE_SLEEPING
            icon.icon = self._icon_normal
            logger.info("Listening resumed by user")
            if self._pipeline and self._pipeline._audio_capture:
                self._pipeline._audio_capture.start()

    def _on_reload_config(self, icon, item) -> None:
        logger.info("Hot-reloading configuration…")
        if self._pipeline:
            try:
                self._pipeline._load_config()
                self.state = STATE_SLEEPING
                logger.info("Configuration reloaded successfully")
            except Exception:
                logger.exception("Config reload failed")

    def _on_toggle_monitor(self, icon, item) -> None:
        """Toggle the monitor window visibility from the tray menu."""
        if self._monitor:
            if self._monitor_visible:
                self._monitor.hide()
                self._monitor_visible = False
            else:
                self._monitor.show()
                self._monitor_visible = True

    def _on_exit(self, icon, item) -> None:
        logger.info("Exit requested from tray menu")
        self.state = STATE_STOPPING
        # Stop monitor window
        if self._monitor:
            self._monitor.stop()
        # Signal pipeline shutdown
        if self._pipeline:
            self._pipeline.request_shutdown()
        # Wait for pipeline thread to finish (bounded)
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            self._pipeline_thread.join(timeout=15)
        # Destroy tray icon (exits pystray loop)
        icon.stop()

    # ------------------------------------------------------------------
    # Pipeline background thread
    # ------------------------------------------------------------------

    def _pipeline_worker(self) -> None:
        """Runs the full Pipeline lifecycle on a daemon thread."""
        try:
            from core.pipeline import Pipeline

            pipeline = Pipeline(config_path=self._config_path)
            self._pipeline = pipeline

            # Start monitor window
            self._monitor = MonitorWindow()
            self._monitor.start()
            self._monitor.post("system", "Loading models…")

            # Run the individual stages so we can update state between them
            pipeline._load_config()
            pipeline._setup_logging()
            pipeline._apply_security()
            pipeline._build()

            # Inject monitor into pipeline
            pipeline._monitor = self._monitor

            self.state = STATE_STARTING
            pipeline._preload_models()

            self.state = STATE_SLEEPING
            pipeline._start_all()

            # Skip pipeline._install_signals() – signals must be registered
            # from the main thread.  The tray Exit button handles shutdown.

            logger.info("═══ Grotesque AI pipeline running (tray mode) ═══")

            # --- live-status poller ---
            # Poll pipeline internals to update tray state
            while not pipeline._shutdown.is_set():
                if self.paused:
                    pass  # keep STATE_PAUSED
                elif (pipeline._wake_word and
                      hasattr(pipeline._wake_word, 'activated') and
                      pipeline._wake_word.activated):
                    self.state = STATE_LISTENING
                elif (pipeline._playback and
                      hasattr(pipeline._playback, 'is_playing') and
                      pipeline._playback.is_playing):
                    self.state = STATE_SPEAKING
                else:
                    if self.state not in (STATE_PAUSED, STATE_STOPPING):
                        self.state = STATE_SLEEPING

                # Also push state to monitor status bar
                if self._monitor:
                    self._monitor.set_status(self.state)

                pipeline._shutdown.wait(timeout=0.5)

            # Clean shutdown
            self.state = STATE_STOPPING
            pipeline._shutdown_all()
            logger.info("═══ Grotesque AI pipeline stopped (tray mode) ═══")

        except Exception:
            logger.exception("Pipeline worker crashed")
            self.state = "Error"

    # ------------------------------------------------------------------
    # Entry point (must be called from the main thread)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the tray icon on the main thread and the Pipeline on a
        background daemon thread.  Blocks until the user selects Exit.
        """
        hide_console()

        # Start pipeline in background
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_worker,
            name="PipelineWorker",
            daemon=True,
        )
        self._pipeline_thread.start()

        # Build and run the tray icon (blocks until icon.stop())
        self._icon = pystray.Icon(
            name=_PROCESS_NAME,
            icon=self._icon_normal,
            title=f"{_TRAY_TITLE}  [{self.state}]",
            menu=self._build_menu(),
        )

        logger.info("System tray started as '%s'", _TRAY_TITLE)
        self._icon.run()   # blocks here

        # After icon.stop() returns – ensure pipeline is down
        if self._pipeline and not self._pipeline._shutdown.is_set():
            self._pipeline.request_shutdown()
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            self._pipeline_thread.join(timeout=10)

        logger.info("Tray application exited cleanly")

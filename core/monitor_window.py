"""
Grotesque AI – Stealth Real-Time Text Monitor Window

Floating, borderless, always-on-top overlay styled like Parakeet AI.
Completely undetectable to casual observation:

  • No taskbar button, no Alt+Tab entry (WS_EX_TOOLWINDOW)
  • No title bar, no window border (overrideredirect)
  • Does not steal focus when clicked (WS_EX_NOACTIVATE)
  • 60 % opacity at all times (WS_EX_LAYERED / -alpha)
  • Draggable via the thin grip bar at the top
  • Hidden (withdraw) not destroyed on close — re-shown from tray

No external dependencies — built-in tkinter + ctypes only.
"""

from __future__ import annotations

import ctypes
import platform
import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import scrolledtext

# ---------------------------------------------------------------------------
# Windows-only stealth helpers
# ---------------------------------------------------------------------------

_GWL_EXSTYLE      = -20
_WS_EX_TOOLWINDOW = 0x00000080   # removes from taskbar & Alt+Tab
_WS_EX_NOACTIVATE = 0x08000000   # doesn't steal keyboard focus
_WS_EX_APPWINDOW  = 0x00040000   # force-shows in taskbar (must be CLEARED)
_WS_EX_LAYERED    = 0x00080000   # required for SetLayeredWindowAttributes
_WS_EX_TOPMOST    = 0x00000008


def _apply_stealth_flags(hwnd: int) -> None:
    """Set WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, clear WS_EX_APPWINDOW."""
    if platform.system() != "Windows":
        return
    try:
        u32 = ctypes.windll.user32
        ex = u32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        ex |= _WS_EX_TOOLWINDOW | _WS_EX_NOACTIVATE | _WS_EX_LAYERED | _WS_EX_TOPMOST
        ex &= ~_WS_EX_APPWINDOW
        u32.SetWindowLongW(hwnd, _GWL_EXSTYLE, ex)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# MonitorWindow
# ---------------------------------------------------------------------------

class MonitorWindow:
    """
    Stealth floating overlay for Grotesque AI.

    Thread-safe: ``post()`` and ``set_status()`` can be called from any thread.
    The tkinter mainloop runs on its own dedicated daemon thread.
    """

    _ALPHA = 0.92          # global window opacity (0.0–1.0)
    _WIN_W = 480
    _WIN_H = 540
    _DRAG_H = 18           # height of the drag handle strip at the top

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._root: tk.Tk | None = None
        self._text: scrolledtext.ScrolledText | None = None
        self._status_var: tk.StringVar | None = None
        self._thread: threading.Thread | None = None
        self._visible: bool = True
        # drag state
        self._drag_x: int = 0
        self._drag_y: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the tkinter mainloop on a daemon thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="MonitorTk",
        )
        self._thread.start()

    def stop(self) -> None:
        """Destroy the window cleanly from any thread."""
        if self._root:
            self._root.after(0, self._root.destroy)
            self._root = None

    def show(self) -> None:
        """Un-hide the overlay."""
        if self._root:
            self._root.after(0, self._root.deiconify)
            self._visible = True

    def hide(self) -> None:
        """Withdraw (hide) without destroying."""
        if self._root:
            self._root.after(0, self._root.withdraw)
            self._visible = False

    def post(self, role: str, text: str) -> None:
        """
        Append a message.  Thread-safe.

        role: ``"user"`` | ``"assistant"`` | ``"speaker"`` | ``"system"``
        """
        self._queue.put((role, text))

    def set_status(self, state: str) -> None:
        """Update the status bar text.  Thread-safe."""
        self._queue.put(("__status__", state))

    # ------------------------------------------------------------------
    # Internal – build UI
    # ------------------------------------------------------------------

    def _run(self) -> None:
        root = tk.Tk()
        self._root = root

        # ── Stealth window setup ──────────────────────────────────────
        root.overrideredirect(True)          # no title bar, no border
        root.attributes("-topmost", True)    # always on top
        root.attributes("-alpha", self._ALPHA)

        # Position: top-right corner with a small margin
        sw = root.winfo_screenwidth()
        root.geometry(f"{self._WIN_W}x{self._WIN_H}+{sw - self._WIN_W - 12}+40")
        root.configure(bg="#0f0f0f")

        # Apply Windows stealth flags after the window has an HWND
        root.update_idletasks()
        try:
            hwnd = ctypes.windll.user32.FindWindowW(None, "")
            # Use the tkinter frame id
            hwnd = root.winfo_id()
            # On Windows, get the real top-level HWND via ctypes
            import ctypes.wintypes
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()
        except Exception:
            hwnd = 0
        if hwnd:
            _apply_stealth_flags(hwnd)
        # Fallback: re-apply after mainloop starts (handles any HWND reparenting)
        root.after(200, self._reapply_stealth)

        # ── Drag handle (top strip) ───────────────────────────────────
        drag_bar = tk.Frame(root, bg="#1e1e1e", height=self._DRAG_H, cursor="fleur")
        drag_bar.pack(fill=tk.X, side=tk.TOP)

        # Grip dots
        grip = tk.Label(drag_bar, text="· · · · ·", bg="#1e1e1e",
                        fg="#444444", font=("Consolas", 7))
        grip.pack(side=tk.LEFT, padx=8)

        # Close button (hides, doesn't destroy)
        close_btn = tk.Label(drag_bar, text="✕", bg="#1e1e1e",
                             fg="#555555", font=("Consolas", 9), cursor="hand2")
        close_btn.pack(side=tk.RIGHT, padx=6)
        close_btn.bind("<Button-1>", lambda _e: self.hide())

        # Bind drag on the bar and the grip label
        for widget in (drag_bar, grip):
            widget.bind("<ButtonPress-1>",   self._on_drag_start)
            widget.bind("<B1-Motion>",       self._on_drag_motion)

        # ── Scrollable chat area ──────────────────────────────────────
        self._text = scrolledtext.ScrolledText(
            root,
            bg="#0f0f0f",
            fg="#cccccc",
            font=("Consolas", 11),
            wrap=tk.WORD,
            state=tk.DISABLED,
            borderwidth=0,
            highlightthickness=0,
            insertbackground="#cccccc",
            selectbackground="#2a2a2a",
        )
        self._text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 0))

        # Colour tags
        self._text.tag_config("user",      foreground="#5bc8f5")
        self._text.tag_config("assistant", foreground="#b5e853")
        self._text.tag_config("speaker",   foreground="#f0a050")
        self._text.tag_config("system",    foreground="#666666")

        # ── Status bar ────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Starting\u2026")
        status_bar = tk.Label(
            root,
            textvariable=self._status_var,
            bg="#090909",
            fg="#444444",
            font=("Consolas", 8),
            anchor="w",
            padx=8,
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # ── Start queue poll ──────────────────────────────────────────
        root.after(100, self._poll)
        root.mainloop()

    def _reapply_stealth(self) -> None:
        """Re-apply stealth flags once the HWND is fully initialised."""
        try:
            if self._root and platform.system() == "Windows":
                import ctypes.wintypes
                # Walk up to the actual top-level window owned by tkinter
                child_hwnd = self._root.winfo_id()
                hwnd = ctypes.windll.user32.GetAncestor(child_hwnd, 2)  # GA_ROOT=2
                if hwnd:
                    _apply_stealth_flags(hwnd)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Drag-to-move
    # ------------------------------------------------------------------

    def _on_drag_start(self, event: tk.Event) -> None:
        self._drag_x = event.x_root - self._root.winfo_x()
        self._drag_y = event.y_root - self._root.winfo_y()

    def _on_drag_motion(self, event: tk.Event) -> None:
        if self._root:
            x = event.x_root - self._drag_x
            y = event.y_root - self._drag_y
            self._root.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # Queue poll (on tkinter thread)
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        PREFIXES = {
            "user":      "\U0001f3a4  ",   # 🎤
            "assistant": "\U0001f916  ",   # 🤖
            "speaker":   "\U0001f50a  ",   # 🔊
            "system":    "\u2699  ",       # ⚙
        }
        try:
            while True:
                item = self._queue.get_nowait()
                if item[0] == "__status__":
                    if self._status_var:
                        self._status_var.set(item[1])
                else:
                    role, text = item
                    ts = datetime.now().strftime("%H:%M:%S")
                    prefix = PREFIXES.get(role, "")
                    line = f"{ts}  {prefix}{text}\n"
                    if self._text:
                        self._text.configure(state=tk.NORMAL)
                        self._text.insert(tk.END, line, role)
                        self._text.configure(state=tk.DISABLED)
                        self._text.see(tk.END)
        except Exception:
            pass
        if self._root:
            self._root.after(100, self._poll)

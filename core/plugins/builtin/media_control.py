"""
Grotesque AI – Built-in Media Control Plugin

Handles:
 • volume_up / volume_down / volume_set
 • play_music / pause_music
 • media_control (generic)

Uses Windows Core Audio (pycaw) or platform key simulation.
Sandboxed: only simulates media key presses, no arbitrary commands.
"""

from __future__ import annotations

import ctypes
import logging
import platform
from typing import Any, Dict, Optional

from core.plugins.engine import Plugin, PluginPermission

logger = logging.getLogger("grotesque.plugins.media")

# Virtual key codes (Windows)
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT = 0xB0
VK_MEDIA_PREV = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

KEYEVENTF_KEYUP = 0x0002


def _press_key_windows(vk_code: int) -> None:
    """Simulate a key press on Windows via user32."""
    if platform.system() != "Windows":
        return
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.keybd_event(vk_code, 0, 0, 0)             # key down
    user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)  # key up


class MediaControlPlugin(Plugin):
    name = "media_control"
    version = "1.0.0"
    description = "Control system media playback and volume"
    intents = [
        "media_control", "play_music", "pause_music",
        "volume_up", "volume_down", "volume_mute",
        "media_next", "media_previous",
    ]
    required_permissions = PluginPermission.MEDIA_CONTROL

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        action = parameters.get("action", "play_pause").lower()

        actions = {
            "play": self._play_pause,
            "pause": self._play_pause,
            "play_pause": self._play_pause,
            "toggle": self._play_pause,
            "next": self._next_track,
            "previous": self._prev_track,
            "volume_up": self._volume_up,
            "volume_down": self._volume_down,
            "mute": self._mute,
        }

        handler = actions.get(action)
        if handler is None:
            return {
                "success": False,
                "error": f"Unknown media action: {action}",
            }

        return handler(parameters)

    def _play_pause(self, params: Dict) -> Dict[str, Any]:
        _press_key_windows(VK_MEDIA_PLAY_PAUSE)
        return {"success": True, "spoken_override": "Done."}

    def _next_track(self, params: Dict) -> Dict[str, Any]:
        _press_key_windows(VK_MEDIA_NEXT)
        return {"success": True, "spoken_override": "Next track."}

    def _prev_track(self, params: Dict) -> Dict[str, Any]:
        _press_key_windows(VK_MEDIA_PREV)
        return {"success": True, "spoken_override": "Previous track."}

    def _volume_up(self, params: Dict) -> Dict[str, Any]:
        steps = min(int(params.get("steps", 5)), 20)
        for _ in range(steps):
            _press_key_windows(VK_VOLUME_UP)
        return {"success": True, "spoken_override": "Volume up."}

    def _volume_down(self, params: Dict) -> Dict[str, Any]:
        steps = min(int(params.get("steps", 5)), 20)
        for _ in range(steps):
            _press_key_windows(VK_VOLUME_DOWN)
        return {"success": True, "spoken_override": "Volume down."}

    def _mute(self, params: Dict) -> Dict[str, Any]:
        _press_key_windows(VK_VOLUME_MUTE)
        return {"success": True, "spoken_override": "Muted."}

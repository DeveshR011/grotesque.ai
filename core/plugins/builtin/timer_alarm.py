"""
Grotesque AI – Built-in Timer & Alarm Plugin

Handles:
 • set_timer      – countdown timer in seconds/minutes
 • cancel_timer   – cancel a named/default timer
 • set_alarm      – one-shot alarm at a specific time

Timers run in background threads.  Alarm notification is pushed
to the TTS audio queue (spoken alert).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from core.plugins.engine import Plugin, PluginPermission

logger = logging.getLogger("grotesque.plugins.timer")


class TimerAlarmPlugin(Plugin):
    name = "timer_alarm"
    version = "1.0.0"
    description = "Set and cancel timers and alarms"
    intents = ["set_timer", "cancel_timer", "set_alarm"]
    required_permissions = PluginPermission.TIMER

    def __init__(self) -> None:
        super().__init__()
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._on_timer_complete: Optional[Callable[[str], None]] = None

    def set_timer_callback(self, callback: Callable[[str], None]) -> None:
        """Set a callback that fires when a timer completes (spoken alert)."""
        self._on_timer_complete = callback

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        action = parameters.get("action", "set")
        if action == "cancel":
            return self._cancel_timer(parameters)
        elif action == "set_alarm":
            return self._set_alarm(parameters)
        else:
            return self._set_timer(parameters)

    def validate_parameters(self, parameters: Dict[str, Any]) -> Optional[str]:
        action = parameters.get("action", "set")
        if action in ("set", "set_timer"):
            duration = parameters.get("duration_seconds", parameters.get("duration", 0))
            if not isinstance(duration, (int, float)) or duration <= 0:
                return "Timer duration must be a positive number"
            if duration > 86400:
                return "Timer duration cannot exceed 24 hours"
        return None

    def _set_timer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name", "default")
        seconds = float(params.get("duration_seconds", params.get("duration", 60)))

        with self._lock:
            # Cancel existing timer with same name
            if name in self._timers:
                self._timers[name].cancel()

            timer = threading.Timer(seconds, self._timer_fired, args=[name])
            timer.daemon = True
            timer.start()
            self._timers[name] = timer

        if seconds >= 60:
            mins = int(seconds // 60)
            spoken = f"Timer set for {mins} minute{'s' if mins != 1 else ''}."
        else:
            spoken = f"Timer set for {int(seconds)} seconds."

        logger.info("Timer '%s' set for %.0f seconds", name, seconds)
        return {
            "success": True,
            "data": {"timer_name": name, "duration_seconds": seconds},
            "spoken_override": spoken,
        }

    def _cancel_timer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name", "default")
        with self._lock:
            timer = self._timers.pop(name, None)
            if timer:
                timer.cancel()
                return {
                    "success": True,
                    "data": {"timer_name": name},
                    "spoken_override": f"Timer {name} cancelled.",
                }
            return {
                "success": False,
                "error": f"No timer named '{name}' found.",
                "spoken_override": f"No active timer named {name}.",
            }

    def _set_alarm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Simple: convert alarm time to seconds from now
        import datetime
        time_str = params.get("time", "")
        if not time_str:
            return {"success": False, "error": "No alarm time specified"}

        try:
            # Parse HH:MM format
            parts = time_str.replace(".", ":").split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0

            now = datetime.datetime.now()
            alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if alarm_time <= now:
                alarm_time += datetime.timedelta(days=1)

            seconds = (alarm_time - now).total_seconds()
            params["duration_seconds"] = seconds
            params["name"] = f"alarm_{hour}_{minute}"
            return self._set_timer(params)
        except (ValueError, IndexError):
            return {"success": False, "error": f"Invalid time format: '{time_str}'"}

    def _timer_fired(self, name: str) -> None:
        logger.info("Timer '%s' completed!", name)
        with self._lock:
            self._timers.pop(name, None)
        if self._on_timer_complete:
            try:
                self._on_timer_complete(f"Timer {name} is done!")
            except Exception:
                logger.debug("Timer callback error", exc_info=True)

    def cleanup(self) -> None:
        with self._lock:
            for name, timer in self._timers.items():
                timer.cancel()
            self._timers.clear()

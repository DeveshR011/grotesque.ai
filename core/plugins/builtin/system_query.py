"""
Grotesque AI – Built-in System Query Plugin

Handles:
 • get_time
 • get_date
 • get_battery
 • system_query (general system info)
"""

from __future__ import annotations

import datetime
import logging
import platform
from typing import Any, Dict, Optional

import psutil

from core.plugins.engine import Plugin, PluginPermission

logger = logging.getLogger("grotesque.plugins.system")


class SystemQueryPlugin(Plugin):
    name = "system_query"
    version = "1.0.0"
    description = "Provides system information: time, date, battery, CPU/RAM usage"
    intents = ["get_time", "get_date", "get_battery", "system_query"]
    required_permissions = PluginPermission.READ_SYSTEM_INFO

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        query_type = parameters.get("type", parameters.get("query", "time")).lower()

        handlers = {
            "time": self._get_time,
            "date": self._get_date,
            "battery": self._get_battery,
            "cpu": self._get_cpu,
            "memory": self._get_memory,
            "ram": self._get_memory,
            "disk": self._get_disk,
            "uptime": self._get_uptime,
            "system": self._get_system_info,
        }

        handler = handlers.get(query_type, self._get_time)
        return handler()

    def _get_time(self) -> Dict[str, Any]:
        now = datetime.datetime.now()
        spoken = now.strftime("It's %I:%M %p")
        return {
            "success": True,
            "data": {"time": now.isoformat()},
            "spoken_override": spoken,
        }

    def _get_date(self) -> Dict[str, Any]:
        now = datetime.datetime.now()
        spoken = now.strftime("Today is %A, %B %d, %Y")
        return {
            "success": True,
            "data": {"date": now.isoformat()},
            "spoken_override": spoken,
        }

    def _get_battery(self) -> Dict[str, Any]:
        batt = psutil.sensors_battery()
        if batt is None:
            return {
                "success": True,
                "data": {"battery": None},
                "spoken_override": "No battery detected on this system.",
            }
        pct = batt.percent
        plugged = "plugged in" if batt.power_plugged else "on battery"
        spoken = f"Battery is at {pct:.0f} percent, {plugged}."
        return {
            "success": True,
            "data": {
                "percent": pct,
                "plugged": batt.power_plugged,
                "secs_left": batt.secsleft,
            },
            "spoken_override": spoken,
        }

    def _get_cpu(self) -> Dict[str, Any]:
        usage = psutil.cpu_percent(interval=0.5)
        return {
            "success": True,
            "data": {"cpu_percent": usage, "cores": psutil.cpu_count()},
            "spoken_override": f"CPU usage is at {usage:.0f} percent.",
        }

    def _get_memory(self) -> Dict[str, Any]:
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        return {
            "success": True,
            "data": {"used_gb": used_gb, "total_gb": total_gb, "percent": mem.percent},
            "spoken_override": f"Memory usage is {mem.percent:.0f} percent. "
                               f"{used_gb:.1f} of {total_gb:.1f} gigabytes used.",
        }

    def _get_disk(self) -> Dict[str, Any]:
        usage = psutil.disk_usage("/")
        used_gb = usage.used / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        return {
            "success": True,
            "data": {"used_gb": used_gb, "total_gb": total_gb, "percent": usage.percent},
            "spoken_override": f"Disk usage is {usage.percent:.0f} percent.",
        }

    def _get_uptime(self) -> Dict[str, Any]:
        import time
        boot = psutil.boot_time()
        uptime_sec = time.time() - boot
        hours = int(uptime_sec // 3600)
        mins = int((uptime_sec % 3600) // 60)
        return {
            "success": True,
            "data": {"uptime_seconds": uptime_sec},
            "spoken_override": f"System has been up for {hours} hours and {mins} minutes.",
        }

    def _get_system_info(self) -> Dict[str, Any]:
        return {
            "success": True,
            "data": {
                "platform": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "spoken_override": f"Running {platform.system()} {platform.release()} "
                               f"on {platform.processor()}.",
        }

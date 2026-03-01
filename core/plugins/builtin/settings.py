"""
Grotesque AI – Built-in Settings Plugin

Handles:
 • change_setting – modify runtime assistant settings

Currently supports:
 • wake_timeout   – how long to listen after wake word
 • tts_speed      – speech rate
 • volume_level   – TTS output volume
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core.plugins.engine import Plugin, PluginPermission

logger = logging.getLogger("grotesque.plugins.settings")


class SettingsPlugin(Plugin):
    name = "settings"
    version = "1.0.0"
    description = "Modify assistant runtime settings"
    intents = ["change_setting"]
    required_permissions = PluginPermission.SETTINGS

    # In-memory runtime overrides (no persistence)
    _overrides: Dict[str, Any] = {}

    # Allowed settings and their range/type validators
    ALLOWED_SETTINGS = {
        "wake_timeout": {"type": float, "min": 5.0, "max": 120.0},
        "tts_speed": {"type": float, "min": 0.5, "max": 2.0},
        "volume_level": {"type": float, "min": 0.0, "max": 1.0},
        "vad_aggressiveness": {"type": int, "min": 0, "max": 3},
        "silence_timeout": {"type": float, "min": 0.5, "max": 5.0},
    }

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        setting = parameters.get("setting", "").lower()
        value = parameters.get("value")

        if setting not in self.ALLOWED_SETTINGS:
            return {
                "success": False,
                "error": f"Unknown setting: {setting}",
                "spoken_override": f"I don't know that setting.",
            }

        spec = self.ALLOWED_SETTINGS[setting]
        try:
            typed_value = spec["type"](value)
        except (ValueError, TypeError):
            return {
                "success": False,
                "error": f"Invalid value for {setting}",
                "spoken_override": f"That's not a valid value for {setting}.",
            }

        if typed_value < spec["min"] or typed_value > spec["max"]:
            return {
                "success": False,
                "error": f"Value out of range for {setting}",
                "spoken_override": f"Value must be between {spec['min']} and {spec['max']}.",
            }

        self._overrides[setting] = typed_value
        logger.info("Setting '%s' changed to %s", setting, typed_value)

        return {
            "success": True,
            "data": {"setting": setting, "value": typed_value},
            "spoken_override": f"Setting {setting.replace('_', ' ')} changed to {typed_value}.",
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> Optional[str]:
        if "setting" not in parameters:
            return "Missing 'setting' parameter"
        if "value" not in parameters:
            return "Missing 'value' parameter"
        return None

    @classmethod
    def get_override(cls, setting: str, default=None):
        return cls._overrides.get(setting, default)

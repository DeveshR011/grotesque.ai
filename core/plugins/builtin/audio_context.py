"""
Grotesque AI – Built-in Audio Context Plugin

Handles questions about what is currently playing or being said
through the PC speakers by retrieving recent [SPEAKER] context
from memory.

Intents:
 • audio_context  – generic "what's playing?"
 • identify_audio – "identify this song / sound"
 • what_song      – "what song is this?"
 • who_said       – "who said that?" / "what did they say?"
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from core.plugins.engine import Plugin, PluginPermission

logger = logging.getLogger("grotesque.plugins.audio_context")


class AudioContextPlugin(Plugin):
    name = "audio_context"
    version = "1.0.0"
    description = (
        "Answers questions about audio currently playing through "
        "the PC speakers using recent [SPEAKER] transcription context."
    )
    intents = ["audio_context", "identify_audio", "what_song", "who_said"]
    required_permissions = PluginPermission.READ_SYSTEM_INFO

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        The heavy lifting is done by the LLM itself – the dispatcher
        already injects recent [SPEAKER] context into the prompt.
        This plugin exists so the intent router can tag these queries
        and the LLM receives an explicit nudge to use speaker context.
        """
        query_type = parameters.get("type", "general").lower()

        # Build a nudge message that will be appended to the spoken response
        # if the LLM's own answer is not adequate.
        nudge = {
            "what_song": "Based on the recent speaker audio context, ",
            "who_said": "Based on what was recently heard through the speakers, ",
            "identify_audio": "Analysing the recent speaker audio transcription, ",
        }.get(query_type, "")

        return {
            "success": True,
            "data": {"query_type": query_type, "nudge": nudge},
            # No spoken_override – let the LLM answer naturally using
            # the [SPEAKER] context already injected into its prompt.
        }

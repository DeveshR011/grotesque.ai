"""
Grotesque AI – Intent Router

Sits between STT output and action dispatch. The LLM returns structured
JSON with an intent + parameters, and the router validates then
dispatches to the appropriate plugin or falls back to conversational TTS.

Flow:
  STT text → LLM (structured JSON mode) → IntentRouter → Plugin Engine
                                                        → TTS (spoken reply)

The router ensures:
 • ALL OS/system actions go through the Plugin Engine (never direct exec)
 • Malformed JSON falls back to conversational response
 • Unknown intents are handled gracefully
 • Input validation before plugin dispatch
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from core.buffers import EventQueue
    from core.plugins.engine import PluginEngine

logger = logging.getLogger("grotesque.intent_router")


# ======================================================================
# Intent categories
# ======================================================================

class IntentCategory(str, Enum):
    """Top-level intent categories recognised by the router."""
    CONVERSATION = "conversation"       # plain spoken reply
    SYSTEM_QUERY = "system_query"       # time, date, battery, etc.
    MEDIA_CONTROL = "media_control"     # play/pause/volume
    TIMER_ALARM = "timer_alarm"         # set timer, alarm
    REMINDER = "reminder"               # set/list reminders
    AUTOMATION = "automation"           # OS automation (future)
    FILE_CONTROL = "file_control"       # file operations (future)
    CLIPBOARD = "clipboard"             # clipboard access (future)
    SETTINGS = "settings"              # change assistant settings
    UNKNOWN = "unknown"


@dataclass
class ParsedIntent:
    """Structured representation of a parsed LLM response."""
    intent: str
    category: IntentCategory
    parameters: Dict[str, Any]
    spoken_response: str                # text the assistant should speak
    confidence: float = 1.0
    raw_json: str = ""
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def has_action(self) -> bool:
        """True if this intent requires plugin execution."""
        return self.category not in (
            IntentCategory.CONVERSATION,
            IntentCategory.UNKNOWN,
        )


@dataclass
class RouterResult:
    """Result after routing: what to speak + any action output."""
    spoken_text: str
    action_result: Optional[Dict[str, Any]] = None
    intent: Optional[ParsedIntent] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


# ======================================================================
# JSON extraction patterns
# ======================================================================

# Matches ```json ... ``` blocks or bare { ... } objects
_JSON_BLOCK_RE = re.compile(
    r'```json\s*(.*?)\s*```|(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
    re.DOTALL,
)


def extract_json(text: str) -> Optional[dict]:
    """
    Extract JSON from LLM output.  Handles:
     • Clean JSON
     • JSON wrapped in markdown code fences
     • JSON embedded in conversational text
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fence or embedded object
    match = _JSON_BLOCK_RE.search(text)
    if match:
        json_str = match.group(1) or match.group(2)
        if json_str:
            try:
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                pass

    return None


# ======================================================================
# Intent Router
# ======================================================================

class IntentRouter:
    """
    Parses structured JSON from the LLM, validates it, and routes
    actionable intents to the Plugin Engine.  Non-actionable intents
    (conversation) pass through as spoken text to TTS.
    """

    def __init__(
        self,
        plugin_engine: Optional["PluginEngine"] = None,
        fallback_to_conversation: bool = True,
    ) -> None:
        self._plugin_engine = plugin_engine
        self._fallback = fallback_to_conversation

        # Map of intent strings → IntentCategory
        self._intent_map: Dict[str, IntentCategory] = {
            "conversation": IntentCategory.CONVERSATION,
            "system_query": IntentCategory.SYSTEM_QUERY,
            "get_time": IntentCategory.SYSTEM_QUERY,
            "get_date": IntentCategory.SYSTEM_QUERY,
            "get_battery": IntentCategory.SYSTEM_QUERY,
            "media_control": IntentCategory.MEDIA_CONTROL,
            "play_music": IntentCategory.MEDIA_CONTROL,
            "pause_music": IntentCategory.MEDIA_CONTROL,
            "volume_up": IntentCategory.MEDIA_CONTROL,
            "volume_down": IntentCategory.MEDIA_CONTROL,
            "set_timer": IntentCategory.TIMER_ALARM,
            "set_alarm": IntentCategory.TIMER_ALARM,
            "cancel_timer": IntentCategory.TIMER_ALARM,
            "set_reminder": IntentCategory.REMINDER,
            "list_reminders": IntentCategory.REMINDER,
            "open_application": IntentCategory.AUTOMATION,
            "close_application": IntentCategory.AUTOMATION,
            "file_search": IntentCategory.FILE_CONTROL,
            "clipboard_read": IntentCategory.CLIPBOARD,
            "clipboard_write": IntentCategory.CLIPBOARD,
            "change_setting": IntentCategory.SETTINGS,
        }

        # Stats
        self._total_routed = 0
        self._total_fallbacks = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, llm_output: str) -> RouterResult:
        """
        Parse LLM output and dispatch.

        Returns a RouterResult with the text to speak and any action
        results from plugins.
        """
        t0 = time.monotonic()

        # Try to parse structured JSON
        parsed = self._parse_intent(llm_output)

        if parsed is None:
            # Fallback: treat entire output as conversational text
            self._total_fallbacks += 1
            return RouterResult(
                spoken_text=llm_output.strip(),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        self._total_routed += 1

        # If it's just conversation, return spoken response
        if not parsed.has_action:
            return RouterResult(
                spoken_text=parsed.spoken_response,
                intent=parsed,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        # Dispatch to plugin engine
        action_result = self._dispatch(parsed)

        # Compose spoken response
        spoken = parsed.spoken_response
        if action_result and action_result.get("spoken_override"):
            spoken = action_result["spoken_override"]

        result = RouterResult(
            spoken_text=spoken,
            action_result=action_result,
            intent=parsed,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

        if action_result and action_result.get("error"):
            result.error = action_result["error"]

        logger.info(
            "Routed intent '%s' (category=%s) in %.1f ms",
            parsed.intent, parsed.category.value, result.latency_ms,
        )
        return result

    def register_intent(self, intent_name: str, category: IntentCategory) -> None:
        """Register a custom intent → category mapping."""
        self._intent_map[intent_name] = category

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_intent(self, text: str) -> Optional[ParsedIntent]:
        """
        Attempt to parse structured intent JSON from LLM output.

        Expected format:
        {
            "intent": "conversation",
            "parameters": { ... },
            "response": "Spoken text here"
        }
        """
        data = extract_json(text)
        if data is None:
            logger.debug("No JSON found in LLM output, falling back")
            return None

        intent_str = data.get("intent", "").lower().strip()
        if not intent_str:
            logger.debug("JSON has no 'intent' field")
            return None

        category = self._intent_map.get(intent_str, IntentCategory.UNKNOWN)
        parameters = data.get("parameters", {})
        spoken = data.get("response", data.get("spoken_response", ""))

        if not isinstance(parameters, dict):
            parameters = {}

        # Validate parameters (sanitize)
        parameters = self._sanitize_parameters(parameters)

        return ParsedIntent(
            intent=intent_str,
            category=category,
            parameters=parameters,
            spoken_response=spoken or "Done.",
            raw_json=json.dumps(data, ensure_ascii=False),
        )

    def _sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize plugin parameters.
        - Remove any keys that look like injection attempts
        - Limit string lengths
        - Disallow nested executable content
        """
        sanitized = {}
        MAX_KEY_LEN = 64
        MAX_VAL_LEN = 1024
        FORBIDDEN_KEYS = {"__import__", "eval", "exec", "system", "os",
                          "subprocess", "cmd", "shell", "powershell"}

        for k, v in params.items():
            key = str(k)[:MAX_KEY_LEN]

            if key.lower() in FORBIDDEN_KEYS:
                logger.warning("Blocked forbidden parameter key: %s", key)
                continue

            if isinstance(v, str):
                sanitized[key] = v[:MAX_VAL_LEN]
            elif isinstance(v, (int, float, bool)):
                sanitized[key] = v
            elif isinstance(v, list):
                sanitized[key] = [
                    str(item)[:MAX_VAL_LEN] if isinstance(item, str) else item
                    for item in v[:50]  # limit list length
                ]
            elif isinstance(v, dict):
                # One level of nesting allowed
                sanitized[key] = self._sanitize_parameters(v)
            else:
                sanitized[key] = str(v)[:MAX_VAL_LEN]

        return sanitized

    # ------------------------------------------------------------------
    # Plugin dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, parsed: ParsedIntent) -> Optional[Dict[str, Any]]:
        """Send validated intent to the plugin engine."""
        if self._plugin_engine is None:
            logger.debug("No plugin engine configured, skipping dispatch")
            return None

        try:
            result = self._plugin_engine.execute(
                intent=parsed.intent,
                parameters=parsed.parameters,
                category=parsed.category.value,
            )
            return result
        except Exception:
            logger.exception("Plugin execution failed for intent '%s'", parsed.intent)
            return {"error": f"Plugin error for '{parsed.intent}'"}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "total_routed": self._total_routed,
            "total_fallbacks": self._total_fallbacks,
            "registered_intents": len(self._intent_map),
        }

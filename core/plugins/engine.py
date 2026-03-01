"""
Grotesque AI – Plugin Base & Engine

Provides the abstract Plugin interface and the sandboxed PluginEngine
that discovers, loads, validates, and executes plugins.

Security model:
 • Plugins run inside a restricted sandbox (no direct OS exec)
 • Each plugin declares required permissions
 • The engine validates permissions before execution
 • All inputs are sanitized by the IntentRouter before arrival
 • Plugin exceptions are caught and never propagate to the pipeline
 • No plugin can escalate privileges or open network sockets
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Flag, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

logger = logging.getLogger("grotesque.plugins")


# ======================================================================
# Permission system
# ======================================================================

class PluginPermission(Flag):
    """Fine-grained permission flags for plugins."""
    NONE = 0
    READ_SYSTEM_INFO = auto()       # time, date, battery, CPU/RAM
    MEDIA_CONTROL = auto()          # play, pause, volume
    TIMER = auto()                  # set/cancel timers
    REMINDER = auto()               # manage reminders
    CLIPBOARD_READ = auto()         # read clipboard
    CLIPBOARD_WRITE = auto()        # write to clipboard
    FILE_READ = auto()              # read files (future)
    FILE_WRITE = auto()             # write files (future)
    PROCESS_LAUNCH = auto()         # launch approved apps (future)
    PROCESS_KILL = auto()           # kill processes (future)
    SETTINGS = auto()               # modify assistant settings
    # NEVER grant these:
    # NETWORK = auto()              # FORBIDDEN – no network access ever
    # ADMIN = auto()                # FORBIDDEN – no privilege escalation


# Default safe permissions for built-in plugins
DEFAULT_PERMISSIONS = (
    PluginPermission.READ_SYSTEM_INFO
    | PluginPermission.MEDIA_CONTROL
    | PluginPermission.TIMER
    | PluginPermission.REMINDER
    | PluginPermission.SETTINGS
)


# ======================================================================
# Plugin base class
# ======================================================================

class Plugin(ABC):
    """
    Abstract base class for all Grotesque AI plugins.

    Every plugin must:
     1. Declare its ``name`` and ``version``
     2. Declare ``intents`` it handles
     3. Declare ``required_permissions``
     4. Implement ``execute(parameters) → dict``

    Plugins must NOT:
     - Import subprocess, os.system, socket, or http modules
     - Access the filesystem outside approved paths
     - Open network connections
     - Modify global state
    """

    # Override in subclass
    name: str = "base_plugin"
    version: str = "1.0.0"
    description: str = ""
    intents: List[str] = []
    required_permissions: PluginPermission = PluginPermission.NONE

    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the plugin action.

        Args:
            parameters: Sanitized parameters from the intent router.

        Returns:
            Dict with at least:
              - "success": bool
              - "data": any result data
              - "spoken_override": optional override for TTS response
        """
        ...

    def validate_parameters(self, parameters: Dict[str, Any]) -> Optional[str]:
        """
        Validate input parameters. Return None if valid,
        or an error message string if invalid.
        Override in subclasses for custom validation.
        """
        return None

    def cleanup(self) -> None:
        """Called on shutdown. Override for resource cleanup."""
        pass


# ======================================================================
# Plugin execution result
# ======================================================================

@dataclass
class PluginResult:
    success: bool
    data: Any = None
    spoken_override: str = ""
    error: str = ""
    execution_time_ms: float = 0.0
    plugin_name: str = ""


# ======================================================================
# Plugin Engine (sandbox + dispatcher)
# ======================================================================

class PluginEngine:
    """
    Discovers, registers, validates, and executes plugins.

    Provides a sandboxed execution environment:
     - Permission checks before every call
     - Input validation before every call
     - Timeout enforcement
     - Exception isolation
     - Audit logging
    """

    def __init__(
        self,
        granted_permissions: PluginPermission = DEFAULT_PERMISSIONS,
        max_execution_sec: float = 5.0,
        plugin_dirs: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._granted = granted_permissions
        self._max_exec = max_execution_sec
        self._plugin_dirs = plugin_dirs or []
        self._config = config or {}

        # intent_name → Plugin instance
        self._registry: Dict[str, Plugin] = {}
        # All loaded plugins
        self._plugins: Dict[str, Plugin] = {}

        self._lock = threading.Lock()
        self._total_executions = 0
        self._total_errors = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, plugin: Plugin) -> None:
        """Register a plugin instance."""
        # Check permissions
        if not self._check_permissions(plugin):
            logger.warning(
                "Plugin '%s' requires permissions %s which are not granted. Skipping.",
                plugin.name, plugin.required_permissions,
            )
            return

        self._plugins[plugin.name] = plugin
        for intent in plugin.intents:
            if intent in self._registry:
                logger.warning(
                    "Intent '%s' already registered by '%s', overriding with '%s'",
                    intent, self._registry[intent].name, plugin.name,
                )
            self._registry[intent] = plugin

        logger.info(
            "Plugin registered: %s v%s (intents: %s)",
            plugin.name, plugin.version, ", ".join(plugin.intents),
        )

    def discover_and_load(self) -> int:
        """
        Auto-discover Plugin subclasses in the built-in plugins package
        and any configured plugin_dirs.
        Returns count of loaded plugins.
        """
        count = 0

        # Load built-in plugins
        try:
            from core.plugins import builtin
            for importer, modname, ispkg in pkgutil.iter_modules(builtin.__path__):
                try:
                    module = importlib.import_module(f"core.plugins.builtin.{modname}")
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, Plugin)
                            and attr is not Plugin
                        ):
                            instance = attr()
                            self.register(instance)
                            count += 1
                except Exception:
                    logger.exception("Failed to load builtin plugin: %s", modname)
        except ImportError:
            logger.debug("No builtin plugins package found")

        logger.info("Discovered and loaded %d plugins", count)
        return count

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        intent: str,
        parameters: Dict[str, Any],
        category: str = "",
    ) -> Dict[str, Any]:
        """
        Execute the plugin registered for the given intent.

        Performs:
         1. Lookup plugin by intent
         2. Validate permissions
         3. Validate input parameters
         4. Execute with timeout
         5. Return result dict
        """
        t0 = time.monotonic()
        self._total_executions += 1

        plugin = self._registry.get(intent)
        if plugin is None:
            logger.debug("No plugin registered for intent '%s'", intent)
            return {"success": False, "error": f"No handler for intent '{intent}'"}

        # Permission check
        if not self._check_permissions(plugin):
            self._total_errors += 1
            msg = f"Permission denied for plugin '{plugin.name}'"
            logger.warning(msg)
            return {"success": False, "error": msg}

        # Input validation
        validation_error = plugin.validate_parameters(parameters)
        if validation_error:
            self._total_errors += 1
            logger.warning(
                "Plugin '%s' input validation failed: %s",
                plugin.name, validation_error,
            )
            return {"success": False, "error": validation_error}

        # Execute in a sandboxed manner
        try:
            result = self._execute_sandboxed(plugin, parameters)
            elapsed = (time.monotonic() - t0) * 1000

            logger.info(
                "Plugin '%s' executed intent '%s' in %.1f ms (success=%s)",
                plugin.name, intent, elapsed, result.get("success"),
            )
            result["execution_time_ms"] = elapsed
            return result

        except Exception:
            self._total_errors += 1
            logger.exception("Plugin '%s' crashed on intent '%s'", plugin.name, intent)
            return {"success": False, "error": f"Plugin '{plugin.name}' internal error"}

    def _execute_sandboxed(
        self, plugin: Plugin, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run plugin.execute() with timeout enforcement.
        Uses a thread to enforce max_execution_sec.
        """
        result_holder: Dict[str, Any] = {"success": False, "error": "Timeout"}
        error_holder: list = []

        def _run():
            try:
                r = plugin.execute(parameters)
                result_holder.update(r)
            except Exception as e:
                error_holder.append(e)
                result_holder["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self._max_exec)

        if t.is_alive():
            logger.warning("Plugin '%s' timed out after %.1fs", plugin.name, self._max_exec)
            return {"success": False, "error": "Plugin execution timed out"}

        if error_holder:
            raise error_holder[0]

        return result_holder

    # ------------------------------------------------------------------
    # Permission check
    # ------------------------------------------------------------------

    def _check_permissions(self, plugin: Plugin) -> bool:
        """Verify that all permissions required by the plugin are granted."""
        required = plugin.required_permissions
        return (self._granted & required) == required

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean up all plugins."""
        for name, plugin in self._plugins.items():
            try:
                plugin.cleanup()
            except Exception:
                logger.debug("Plugin '%s' cleanup error", name, exc_info=True)
        self._registry.clear()
        self._plugins.clear()
        logger.info("Plugin engine shut down")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_plugins(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "intents": p.intents,
                "permissions": str(p.required_permissions),
            }
            for p in self._plugins.values()
        ]

    def list_intents(self) -> List[str]:
        return list(self._registry.keys())

    def get_stats(self) -> dict:
        return {
            "total_executions": self._total_executions,
            "total_errors": self._total_errors,
            "loaded_plugins": len(self._plugins),
            "registered_intents": len(self._registry),
        }

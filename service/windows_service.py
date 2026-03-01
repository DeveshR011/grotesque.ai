"""
Grotesque AI – Windows Service Layer (Hardened)

Implements the assistant as a proper Windows Service using pywin32.
This allows:
 • Auto-start at boot (delayed auto-start)
 • Hidden process (no console window)
 • Controlled via sc.exe / services.msc
 • Graceful stop/restart from SCM
 • Recovery options: restart on all failures (1st, 2nd, subsequent)
 • High process priority
 • No visible console window ever created

For Linux: see docs/DEPLOYMENT.md for systemd unit file.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("grotesque.service")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _is_windows() -> bool:
    return sys.platform == "win32"


if _is_windows():
    try:
        import servicemanager
        import win32event
        import win32service
        import win32serviceutil
    except ImportError:
        logger.error(
            "pywin32 not installed. Run: pip install pywin32\n"
            "Then run: python Scripts/pywin32_postinstall.py -install"
        )
        raise


class GrotesqueService(win32serviceutil.ServiceFramework if _is_windows() else object):
    """
    Windows Service wrapper around the Grotesque AI pipeline.
    Hardened with recovery options, delayed auto-start, and process priority.
    """

    _svc_name_ = "GrotesqueAI"
    _svc_display_name_ = "Grotesque AI Voice Assistant"
    _svc_description_ = (
        "Fully local, GPU-accelerated voice assistant. "
        "Runs LLaMA 3 8B + Faster-Whisper + Piper TTS in real-time."
    )
    _svc_deps_ = ["Audiosrv"]  # depend on Windows Audio service

    def __init__(self, args=None):
        if _is_windows():
            win32serviceutil.ServiceFramework.__init__(self, args)
            self._stop_event = win32event.CreateEvent(None, 0, 0, None)
        self._pipeline = None

    def SvcStop(self):
        """Called by SCM when the service is being stopped."""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        logger.info("Service stop requested by SCM")
        if self._pipeline:
            self._pipeline.request_shutdown()
        if _is_windows():
            win32event.SetEvent(self._stop_event)

    def SvcDoRun(self):
        """Called by SCM when the service starts."""
        if _is_windows():
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, ""),
            )

        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        logger.info("Service starting…")

        try:
            # Change to project directory
            os.chdir(str(PROJECT_ROOT))

            # Set high process priority
            _set_service_process_priority()

            # Hide any console window
            _hide_service_console()

            from core.pipeline import Pipeline

            self._pipeline = Pipeline()
            self._pipeline.run()  # blocks until shutdown
        except Exception:
            logger.exception("Service crashed")
            if _is_windows():
                servicemanager.LogErrorMsg(f"{self._svc_name_} crashed")
        finally:
            logger.info("Service exiting")


# ======================================================================
# Service hardening helpers
# ======================================================================

def _set_service_process_priority() -> None:
    """Set the service process to HIGH priority class."""
    if not _is_windows():
        return
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32")
        handle = kernel32.GetCurrentProcess()
        # HIGH_PRIORITY_CLASS = 0x00000080
        kernel32.SetPriorityClass(handle, 0x00000080)
        logger.info("Service process priority set to HIGH")
    except Exception:
        logger.warning("Failed to set service process priority")


def _hide_service_console() -> None:
    """Ensure no console window is visible."""
    if not _is_windows():
        return
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass


def _configure_recovery_options(service_name: str = "GrotesqueAI") -> bool:
    """
    Configure service recovery options using sc.exe:
     • 1st failure: restart after 5 seconds
     • 2nd failure: restart after 10 seconds
     • Subsequent failures: restart after 30 seconds
     • Reset failure count after 86400 seconds (24 hours)
    """
    if not _is_windows():
        return False

    cmd = [
        "sc", "failure", service_name,
        "reset=", "86400",
        "actions=", "restart/5000/restart/10000/restart/30000",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode == 0:
            logger.info("Service recovery options configured")
            return True
        else:
            logger.warning("Failed to set recovery options: %s", result.stderr.strip())
            return False
    except Exception:
        logger.exception("Failed to configure recovery options")
        return False


def _configure_delayed_auto_start(service_name: str = "GrotesqueAI") -> bool:
    """
    Set the service to delayed auto-start.
    This ensures the service starts after critical system services.
    """
    if not _is_windows():
        return False

    cmd = ["sc", "config", service_name, "start=", "delayed-auto"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode == 0:
            logger.info("Service set to delayed auto-start")
            return True
        else:
            logger.warning("Failed to set delayed auto-start: %s", result.stderr.strip())
            return False
    except Exception:
        logger.exception("Failed to configure delayed auto-start")
        return False


def _set_service_description(service_name: str = "GrotesqueAI") -> bool:
    """Set a detailed service description."""
    desc = (
        "Grotesque AI - Fully local, GPU-accelerated voice assistant. "
        "Runs LLaMA 3 8B + Faster-Whisper STT + Piper TTS. "
        "100%% offline, zero telemetry, all data stays on-device."
    )
    cmd = ["sc", "description", service_name, desc]
    try:
        subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return True
    except Exception:
        return False


# ======================================================================
# Helpers for service install/uninstall
# ======================================================================

def install_service() -> None:
    """Install the Windows service with hardened configuration."""
    if not _is_windows():
        print("Windows service installation is only supported on Windows.")
        print("For Linux, use the systemd unit file in docs/DEPLOYMENT.md")
        return

    # Ensure running as admin
    import ctypes
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("ERROR: Must run as Administrator to install service.")
        sys.exit(1)

    # Build the service command pointing to this file
    service_exe = sys.executable
    service_script = str(Path(__file__).resolve())

    argv = [
        "GrotesqueAI",
        "--startup", "auto",
        "install",
    ]

    print(f"Installing service '{GrotesqueService._svc_name_}'…")
    print(f"  Python: {service_exe}")
    print(f"  Script: {service_script}")

    win32serviceutil.HandleCommandLine(GrotesqueService, argv=argv)

    # Apply hardening after installation
    print("\nApplying service hardening…")
    _configure_delayed_auto_start()
    _configure_recovery_options()
    _set_service_description()

    print("\n✓ Service installed and hardened.")
    print("  Recovery: restart on 1st (5s), 2nd (10s), subsequent (30s) failures")
    print("  Startup:  delayed auto-start")
    print("  Priority: HIGH")
    print("\nStart with:")
    print("  net start GrotesqueAI")
    print("  - or -")
    print("  sc start GrotesqueAI")


def uninstall_service() -> None:
    """Remove the Windows service."""
    if not _is_windows():
        print("Not on Windows.")
        return

    import ctypes
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("ERROR: Must run as Administrator.")
        sys.exit(1)

    print(f"Removing service '{GrotesqueService._svc_name_}'…")
    win32serviceutil.HandleCommandLine(GrotesqueService, argv=["GrotesqueAI", "remove"])
    print("Service removed.")


def start_service() -> None:
    """Start the service via SCM."""
    if _is_windows():
        os.system("net start GrotesqueAI")


def stop_service() -> None:
    """Stop the service via SCM."""
    if _is_windows():
        os.system("net stop GrotesqueAI")


# ======================================================================
# Direct execution (for pywin32 service host)
# ======================================================================

if __name__ == "__main__":
    if _is_windows():
        if len(sys.argv) == 1:
            # Started by Windows SCM
            servicemanager.Initialize()
            servicemanager.PrepareToHostSingle(GrotesqueService)
            servicemanager.StartServiceCtrlDispatcher()
        else:
            # Command-line management (install/remove/start/stop)
            win32serviceutil.HandleCommandLine(GrotesqueService)
    else:
        print("This module is for Windows services.")
        print("On Linux, run:  python main.py")

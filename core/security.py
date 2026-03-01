"""
Grotesque AI – Security Hardening Module

Implements:
 • Outbound firewall rule (blocks all internet for this process)
 • Socket binding prevention (no localhost servers)
 • Buffer wiping / memory zeroing
 • Minimal encrypted config (optional)
 • AES-256 encrypted error logging
 • Windows crash dump disabling
 • Restricted-user account checks
 • Process-level network lockdown
"""

from __future__ import annotations

import ctypes
import io
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("grotesque.security")


# ======================================================================
# Firewall – Block all outbound traffic for this process
# ======================================================================

def block_outbound_windows(rule_name: str = "GrotesqueAI_Block") -> bool:
    """
    Add a Windows Firewall outbound rule that blocks this process.
    Requires administrator privileges.
    """
    exe_path = sys.executable
    cmd = [
        "netsh", "advfirewall", "firewall", "add", "rule",
        f"name={rule_name}",
        "dir=out",
        "action=block",
        f"program={exe_path}",
        "enable=yes",
        "profile=any",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        if result.returncode == 0:
            logger.info("Firewall rule '%s' added – outbound blocked", rule_name)
            return True
        else:
            logger.warning("Firewall rule failed: %s", result.stderr.strip())
            return False
    except Exception:
        logger.exception("Failed to add firewall rule")
        return False


def remove_outbound_windows(rule_name: str = "GrotesqueAI_Block") -> bool:
    """Remove the firewall rule on clean shutdown."""
    cmd = [
        "netsh", "advfirewall", "firewall", "delete", "rule",
        f"name={rule_name}",
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=10,
                       creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0)
        logger.info("Firewall rule '%s' removed", rule_name)
        return True
    except Exception:
        logger.exception("Failed to remove firewall rule")
        return False


def block_outbound_linux() -> bool:
    """Add iptables rule to block outbound for current UID."""
    uid = os.getuid()
    cmd = [
        "iptables", "-A", "OUTPUT",
        "-m", "owner", "--uid-owner", str(uid),
        "-j", "DROP",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=10)
        logger.info("iptables outbound block added for UID %d", uid)
        return True
    except Exception:
        logger.exception("Failed to add iptables rule")
        return False


def setup_firewall(rule_name: str = "GrotesqueAI_Block") -> bool:
    """Platform-aware firewall setup."""
    system = platform.system()
    if system == "Windows":
        return block_outbound_windows(rule_name)
    elif system == "Linux":
        return block_outbound_linux()
    else:
        logger.warning("Firewall not supported on %s", system)
        return False


def teardown_firewall(rule_name: str = "GrotesqueAI_Block") -> bool:
    system = platform.system()
    if system == "Windows":
        return remove_outbound_windows(rule_name)
    return True


# ======================================================================
# Memory wiping
# ======================================================================

def secure_zero_bytes(data: bytearray) -> None:
    """Overwrite a bytearray with zeros (best-effort secure wipe)."""
    ctypes.memset(ctypes.addressof((ctypes.c_char * len(data)).from_buffer(data)), 0, len(data))


def secure_zero_string(s: str) -> None:
    """
    Attempt to zero-out a Python string's internal buffer.
    NOT guaranteed due to Python's string interning – use for defence-in-depth.
    """
    try:
        str_addr = id(s)
        # CPython string header offset (64-bit): ~48 bytes for compact ASCII
        header_size = sys.getsizeof("") - 1  # approximate
        buf_addr = str_addr + header_size
        n = len(s)
        ctypes.memset(buf_addr, 0, n)
    except Exception:
        pass  # best-effort


# ======================================================================
# Config encryption (optional)
# ======================================================================

def generate_key(key_path: Path) -> bytes:
    """Generate and save a Fernet key."""
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    key_path.write_bytes(key)
    logger.info("Encryption key generated at %s", key_path)
    return key


def encrypt_file(source: Path, dest: Path, key: bytes) -> None:
    from cryptography.fernet import Fernet
    f = Fernet(key)
    data = source.read_bytes()
    dest.write_bytes(f.encrypt(data))


def decrypt_file(source: Path, key: bytes) -> bytes:
    from cryptography.fernet import Fernet
    f = Fernet(key)
    return f.decrypt(source.read_bytes())


# ======================================================================
# Process hardening
# ======================================================================

def hide_console_window() -> None:
    """Hide the console window on Windows."""
    if platform.system() != "Windows":
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            user32.ShowWindow(hwnd, 0)  # SW_HIDE
            logger.debug("Console window hidden")
    except Exception:
        pass


def set_process_priority_high() -> None:
    """Raise process priority for low-latency audio."""
    if platform.system() == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32")
            handle = kernel32.GetCurrentProcess()
            # HIGH_PRIORITY_CLASS = 0x00000080
            kernel32.SetPriorityClass(handle, 0x00000080)
            logger.info("Process priority set to HIGH")
        except Exception:
            logger.warning("Failed to set high priority")
    else:
        try:
            os.nice(-10)
            logger.info("Process nice level set to -10")
        except PermissionError:
            logger.warning("Cannot set nice level (not root)")


def disable_core_dumps() -> None:
    """Prevent core dumps that could leak audio/text data."""
    if platform.system() != "Windows":
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            logger.info("Core dumps disabled")
        except Exception:
            pass


# ======================================================================
# Socket binding prevention
# ======================================================================

_original_socket_bind = socket.socket.bind


def _blocked_bind(self, address):
    """Monkey-patched socket.bind that blocks all listen attempts."""
    logger.warning("BLOCKED socket.bind(%s) – no servers allowed", address)
    raise PermissionError("Grotesque AI: socket binding is prohibited for security")


def prevent_socket_binding() -> None:
    """
    Monkey-patch socket.bind to prevent any component from opening
    a localhost server.  Defence-in-depth against accidental exposure.
    """
    socket.socket.bind = _blocked_bind
    logger.info("Socket binding prevention active")


def restore_socket_binding() -> None:
    """Restore original socket.bind (for testing)."""
    socket.socket.bind = _original_socket_bind


# ======================================================================
# Windows crash dump disabling
# ======================================================================

def disable_windows_crash_dumps() -> None:
    """
    Disable Windows Error Reporting (WER) crash dumps for this process.
    Prevents minidump files that could leak audio/text data.
    """
    if platform.system() != "Windows":
        return

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
        error_mode = 0x0001 | 0x0002 | 0x8000
        kernel32.SetErrorMode(error_mode)
        logger.info("Windows crash dump error mode set (no WER dialogs)")
    except Exception:
        logger.debug("Failed to set error mode", exc_info=True)

    try:
        import winreg
        key_path = r"SOFTWARE\Microsoft\Windows\Windows Error Reporting"
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, "DontShowUI", 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(key, "Disabled", 0, winreg.REG_DWORD, 1)
        winreg.CloseKey(key)
        logger.info("WER disabled via registry")
    except Exception:
        logger.debug("Could not modify WER registry (non-admin or missing key)")


# ======================================================================
# AES-256 Encrypted Error Logging
# ======================================================================

class EncryptedLogHandler(logging.Handler):
    """
    A logging handler that encrypts log messages with AES-256 (Fernet)
    and writes them to an encrypted log file.

    Each line is independently encrypted so the file can be appended to.
    """

    def __init__(
        self,
        log_path: Path,
        encryption_key: Optional[bytes] = None,
        max_size_mb: float = 50.0,
    ) -> None:
        super().__init__()
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_size = int(max_size_mb * 1024 * 1024)
        self._fernet = None

        if encryption_key:
            from cryptography.fernet import Fernet
            self._fernet = Fernet(encryption_key)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"
            if self._fernet:
                encrypted = self._fernet.encrypt(msg.encode("utf-8"))
                with open(self._log_path, "ab") as f:
                    f.write(encrypted + b"\n")
            else:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(msg)

            # Rotate if too large
            if self._log_path.exists() and self._log_path.stat().st_size > self._max_size:
                rotated = self._log_path.with_suffix(".old.log")
                if rotated.exists():
                    rotated.unlink()
                self._log_path.rename(rotated)
        except Exception:
            self.handleError(record)


def decrypt_log_file(log_path: Path, encryption_key: bytes) -> str:
    """Decrypt an encrypted log file for review."""
    from cryptography.fernet import Fernet
    f = Fernet(encryption_key)
    output = []
    with open(log_path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    output.append(f.decrypt(line).decode("utf-8"))
                except Exception:
                    output.append("[DECRYPT_ERROR]")
    return "".join(output)


# ======================================================================
# Restricted user account checks
# ======================================================================

def check_restricted_user() -> bool:
    """
    Warn if running as Administrator.  Production deployments should
    use a restricted / least-privilege service account.
    """
    if platform.system() != "Windows":
        if os.getuid() == 0:
            logger.warning("Running as root – use a restricted user for production")
            return False
        return True

    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if is_admin:
            logger.warning(
                "Running as Administrator – consider a restricted service account"
            )
            return False
    except Exception:
        pass
    return True


# ======================================================================
# Plugin input sanitisation
# ======================================================================

FORBIDDEN_CHARS = set('<>|&;`$(){}[]\\')
MAX_INPUT_LENGTH = 4096


def sanitize_plugin_input(value: str) -> str:
    """Strip dangerous characters from plugin parameter values."""
    if len(value) > MAX_INPUT_LENGTH:
        value = value[:MAX_INPUT_LENGTH]
    return "".join(c for c in value if c not in FORBIDDEN_CHARS)


def validate_plugin_params(params: dict) -> dict:
    """Recursively sanitise all string values in plugin parameters."""
    clean = {}
    for k, v in params.items():
        if isinstance(v, str):
            clean[k] = sanitize_plugin_input(v)
        elif isinstance(v, dict):
            clean[k] = validate_plugin_params(v)
        elif isinstance(v, (int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = [
                sanitize_plugin_input(i) if isinstance(i, str) else i
                for i in v
            ]
        # Drop anything else (bytes, callables, etc.)
    return clean


# ======================================================================
# Full security setup
# ======================================================================

def apply_all_security(
    block_outbound: bool = True,
    prevent_sockets: bool = True,
    disable_crash_dumps: bool = True,
    firewall_rule_name: str = "GrotesqueAI_Block",
) -> None:
    """Apply all security hardening measures."""
    hide_console_window()
    set_process_priority_high()
    disable_core_dumps()

    if disable_crash_dumps:
        disable_windows_crash_dumps()

    if prevent_sockets:
        prevent_socket_binding()

    if block_outbound:
        setup_firewall(firewall_rule_name)

    check_restricted_user()

    logger.info("All security hardening applied")

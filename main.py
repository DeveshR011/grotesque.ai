"""
Grotesque AI – Main Entry Point

Supports four modes:
  1. Direct execution:     python main.py
  2. Windows service:      Handled via service/windows_service.py
  3. Debug/test mode:      python main.py --debug
  4. Covert tray mode:     pythonw main.py --gui       (no console)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Ensure CUDA runtime DLLs are discoverable (pip-installed nvidia packages)
# ---------------------------------------------------------------------------
def _add_cuda_dll_dirs() -> None:
    """Add nvidia pip-package bin dirs to DLL search path / PATH."""
    site_pkgs = PROJECT_ROOT / "venv" / "Lib" / "site-packages" / "nvidia"
    if not site_pkgs.exists():
        return
    for sub in site_pkgs.iterdir():
        bin_dir = sub / "bin"
        if bin_dir.is_dir():
            dll_str = str(bin_dir)
            import os
            if dll_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = dll_str + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(dll_str)
            except (OSError, AttributeError):
                pass

_add_cuda_dll_dirs()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="grotesque-ai",
        description="Fully local, GPU-accelerated voice assistant",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch in covert system-tray mode (no console window)",
    )
    parser.add_argument(
        "--install-service",
        action="store_true",
        help="Install as Windows service (requires admin)",
    )
    parser.add_argument(
        "--uninstall-service",
        action="store_true",
        help="Uninstall Windows service (requires admin)",
    )
    args = parser.parse_args()

    # Service management shortcuts
    if args.install_service:
        from service.windows_service import install_service
        install_service()
        return

    if args.uninstall_service:
        from service.windows_service import uninstall_service
        uninstall_service()
        return

    # Logging setup
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # ── GUI / Tray mode ──────────────────────────────────────────────
    if args.gui:
        from core.gui import TrayApp, hide_console
        hide_console()
        app = TrayApp(config_path=args.config, debug=args.debug)
        app.run()
        return

    # ── Terminal / headless mode ─────────────────────────────────────
    from core.pipeline import Pipeline

    pipeline = Pipeline(config_path=args.config)
    pipeline.run()


if __name__ == "__main__":
    main()

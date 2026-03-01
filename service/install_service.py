"""
Grotesque AI – Service Installer Script

Usage (run as Administrator):
    python service/install_service.py install
    python service/install_service.py uninstall
    python service/install_service.py start
    python service/install_service.py stop
    python service/install_service.py status
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python install_service.py <install|uninstall|start|stop|status>")
        sys.exit(1)

    command = sys.argv[1].lower()

    if sys.platform == "win32":
        from service.windows_service import (
            install_service,
            start_service,
            stop_service,
            uninstall_service,
        )

        actions = {
            "install": install_service,
            "uninstall": uninstall_service,
            "start": start_service,
            "stop": stop_service,
            "status": lambda: __import__("os").system("sc query GrotesqueAI"),
        }
    else:
        print("On Linux/macOS, use systemd/launchd directly.")
        print("See docs/DEPLOYMENT.md for instructions.")
        sys.exit(0)

    fn = actions.get(command)
    if fn is None:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(actions.keys())}")
        sys.exit(1)

    fn()


if __name__ == "__main__":
    main()

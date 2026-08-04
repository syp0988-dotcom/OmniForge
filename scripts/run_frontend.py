"""Start the OmniForge frontend dev server as a detached background process (Windows).

Launches the Vite dev server in its own detached process group so it keeps
running after the launching shell exits.  Idempotent: if something is already
listening on the configured port, it prints that and exits.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
HOST, PORT = "127.0.0.1", 5174


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    if _port_in_use(HOST, PORT):
        print(f"frontend already running on port {PORT}")
        return

    node = shutil.which("node") or r"G:\nodejs\node.exe"
    flags = 0
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    stdout = open(LOGS / "frontend.log", "a", encoding="utf-8")
    stderr = open(LOGS / "frontend.err.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        [
            node,
            "node_modules/vite/bin/vite.js",
            "--host", HOST, "--port", str(PORT),
        ],
        cwd=str(ROOT / "frontend"),
        stdout=stdout,
        stderr=stderr,
        creationflags=flags,
    )
    print(f"frontend started, pid={proc.pid}, logs in {LOGS}", flush=True)


if __name__ == "__main__":
    main()

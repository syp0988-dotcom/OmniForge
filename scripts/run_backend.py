"""Start the OmniForge backend as a detached background process (Windows).

Launches uvicorn in its own detached process group so it keeps running
after the launching shell exits.  Idempotent: if something is already
listening on port 8000, it prints that and exits.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
HOST, PORT = "0.0.0.0", 8000


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    if _port_in_use("127.0.0.1", PORT):
        print(f"backend already running on port {PORT}")
        return

    flags = 0
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    stdout = open(LOGS / "backend.log", "a", encoding="utf-8")
    stderr = open(LOGS / "backend.err.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "agentflow.app.main:app",
            "--host", HOST, "--port", str(PORT),
        ],
        cwd=str(ROOT),
        stdout=stdout,
        stderr=stderr,
        creationflags=flags,
    )
    print(f"backend started, pid={proc.pid}, logs in {LOGS}", flush=True)


if __name__ == "__main__":
    main()

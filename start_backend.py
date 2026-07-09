"""Launch uvicorn as a fully detached process on Windows."""
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
venv_python = project_root / ".venv" / "Scripts" / "python.exe"

env = {
    **__import__("os").environ,
    "PYTHONIOENCODING": "utf-8",
}

uvicorn = subprocess.Popen(
    [
        str(venv_python), "-m", "uvicorn",
        "agentflow.app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ],
    cwd=str(project_root),
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    env=env,
)

print(f"Backend started: PID={uvicorn.pid}")
print("http://localhost:8000")

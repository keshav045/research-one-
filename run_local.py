"""
ResearchLens — Local Runner
============================
Launches both the FastAPI backend (port 8000) and the Streamlit UI (port 8501)
concurrently for local development or usage.
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = ROOT_DIR / "venv" / "Scripts" / "python.exe"

if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

def main():
    print("=" * 70)
    print("           RESEARCHLENS — LOCAL APPLICATION RUNNER")
    print("=" * 70)
    print(f"Working Directory: {ROOT_DIR}")
    print(f"Python:            {VENV_PYTHON}")
    print()

    # 1. Start FastAPI backend
    print("🚀 [1/2] Starting FastAPI backend on http://127.0.0.1:8000 ...")
    backend_cmd = [
        str(VENV_PYTHON),
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--log-level",
        "info",
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(ROOT_DIR),
        stdout=None,
        stderr=None,
    )

    # Allow backend a moment to spin up
    time.sleep(3)

    # 2. Start Streamlit UI
    print("🚀 [2/2] Starting Streamlit interface on http://localhost:8501 ...")
    streamlit_cmd = [
        str(VENV_PYTHON),
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.port",
        "8501",
        "--server.headless",
        "false",
    ]
    frontend_proc = subprocess.Popen(
        streamlit_cmd,
        cwd=str(ROOT_DIR),
        stdout=None,
        stderr=None,
    )

    print()
    print("=" * 70)
    print(" ResearchLens is now running locally!")
    print("   • Web Interface:    http://localhost:8501")
    print("   • Backend API:      http://127.0.0.1:8000")
    print("   • API Health Check: http://127.0.0.1:8000/health")
    print("   • API Swagger Docs: http://127.0.0.1:8000/docs")
    print("=" * 70)
    print("Press Ctrl+C to stop both servers.")
    print()

    try:
        # Keep runner alive while processes run
        while True:
            if backend_proc.poll() is not None:
                print("⚠️ FastAPI backend exited unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("⚠️ Streamlit UI exited.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ResearchLens local servers...")
    finally:
        if backend_proc.poll() is None:
            backend_proc.terminate()
        if frontend_proc.poll() is None:
            frontend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    main()


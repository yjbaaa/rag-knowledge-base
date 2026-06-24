"""
RAG Knowledge Base - Startup Entry Point
Usage:
    python run.py api       # Start FastAPI backend
    python run.py ui        # Start Streamlit frontend
    python run.py all       # Start both
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def start_api():
    """Start FastAPI backend."""
    print("Starting FastAPI server...")
    from app.core.config import API_HOST, API_PORT
    import uvicorn
    uvicorn.run(
        "app.api.routes:app",
        host=API_HOST,
        port=API_PORT,
    )


def start_ui():
    """Start Streamlit frontend."""
    print("Starting Streamlit UI...")
    from app.core.config import STREAMLIT_PORT
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "frontend" / "app.py"),
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", "0.0.0.0",
    ])


if __name__ == "__main__":
    # config.py auto-loads .env via dotenv, just import it early
    from app.core.config import API_HOST, API_PORT  # noqa: F401

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "api":
        start_api()
    elif cmd == "ui":
        start_ui()
    elif cmd == "all":
        from multiprocessing import Process
        p1 = Process(target=start_api)
        p2 = Process(target=start_ui)
        p1.start()
        p2.start()
        p1.join()
        p2.join()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)

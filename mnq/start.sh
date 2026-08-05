#!/usr/bin/env bash
# Start the MNQ dashboard: update, check dependencies, run.
#
# The macOS/Linux twin of start.bat. It handles the directory juggling:
# `git pull` needs the repository root while `run.py` needs the mnq folder.

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo
echo "=== Checking for updates ==="
if ! git -C "$here/.." pull --ff-only; then
    echo
    echo "  Update failed or skipped - starting the version you already have."
    echo
fi

cd "$here"

python3 -c "import fastapi, uvicorn, httpx" >/dev/null 2>&1 || {
    echo
    echo "=== Installing dependencies (first run only) ==="
    python3 -m pip install -r requirements.txt || {
        echo
        echo "  Could not install dependencies. Is Python 3.10+ available?"
        exit 1
    }
}

echo
echo "=== Starting dashboard - open http://127.0.0.1:8765 ==="
echo "    Press Ctrl+C to stop."
echo
exec python3 run.py

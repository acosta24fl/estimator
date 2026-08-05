#!/usr/bin/env bash
# Start the MNQ dashboard: update, check dependencies, run.
#
# The macOS/Linux twin of start.bat. It handles the directory juggling:
# `git pull` needs the repository root while `run.py` needs the mnq folder.

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==========================================================================
#  SETTINGS - delete the leading # to switch one on. All optional.
# ==========================================================================

# Colour the page green/red on every call, however weak. By default the page
# only commits when the projected move is at least 10% of a typical move, so
# it shows NO CALL most of the time. Set to 0 to see it react constantly.
# export MNQ_SIGNAL_MIN_RATIO=0

# How far ahead the green/red call looks, in minutes. Must be one of the
# chart's timeframes: 1, 5, 10, 15, 30, 60, 240.
# export MNQ_SIGNAL_HORIZON_MINUTES=10

# Simulated trading. On by default; logged to mnq/data/trades.jsonl and drawn
# on the chart as arrows. Nothing is ever sent to a broker. 0 turns it off.
# export MNQ_PAPER_TRADING=0
# export MNQ_PAPER_COST_POINTS=0.75

# Offline demo mode - generated prices instead of Yahoo.
# export MNQ_FEED=synthetic

# ==========================================================================

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

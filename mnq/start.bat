@echo off
REM Start the MNQ dashboard: update, check dependencies, run.
REM
REM Double-click this file, or run `start.bat` from a Command Prompt.
REM It handles the directory juggling: `git pull` needs the repository root
REM while `run.py` needs the mnq folder, which is the usual thing to get wrong.

setlocal
title MNQ Dashboard

REM ==========================================================================
REM  SETTINGS - delete the word REM at the start of a line to switch it on.
REM  Everything here is optional; the dashboard runs fine with all of it off.
REM ==========================================================================

REM --- Colour the page green/red on every call, however weak -----------------
REM  By default the page only commits when the projected move is at least 10%%
REM  of a typical move, so it shows NO CALL most of the time. That is honest,
REM  but if you want to see the colour react constantly, set this to 0.
REM set MNQ_SIGNAL_MIN_RATIO=0

REM --- How far ahead the green/red call looks, in minutes --------------------
REM  Must be one of the chart's timeframes: 1, 5, 10, 15, 30, 60, 240.
REM set MNQ_SIGNAL_HORIZON_MINUTES=10

REM --- Simulated trading ----------------------------------------------------
REM  On by default. Trades are logged to mnq\data\trades.jsonl and drawn on
REM  the chart as arrows. Nothing is ever sent to a broker. Set to 0 for off.
REM set MNQ_PAPER_TRADING=0
REM  Round-turn cost charged per simulated trade, in points (0.75 = ~$1.50).
REM set MNQ_PAPER_COST_POINTS=0.75

REM --- Offline demo mode ----------------------------------------------------
REM  Runs on generated prices instead of Yahoo. Useful when the market is
REM  closed, or to watch the trade markers appear without waiting on the tape.
REM set MNQ_FEED=synthetic

REM ==========================================================================

REM %~dp0 is this script's folder (…\estimator\mnq\), so ".." is the repo root.
cd /d "%~dp0.."

echo.
echo === Checking for updates ===
git pull --ff-only
if errorlevel 1 (
    echo.
    echo   Update failed or skipped - starting the version you already have.
    echo.
)

cd /d "%~dp0"

REM Fast no-op once the dependencies are installed.
python -c "import fastapi, uvicorn, httpx" >nul 2>&1
if errorlevel 1 (
    echo.
    echo === Installing dependencies ^(first run only^) ===
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo   Could not install dependencies. Is Python 3.10+ on your PATH?
        echo   Check with:  python --version
        pause
        exit /b 1
    )
)

echo.
echo === Starting dashboard - open http://127.0.0.1:8765 ===
echo     Press Ctrl+C in this window to stop.
echo.
python run.py

REM Keep the window open if it exited because of an error rather than Ctrl+C.
if errorlevel 1 (
    echo.
    echo   The dashboard stopped with an error. The message above says why.
    pause
)
endlocal

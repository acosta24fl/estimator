@echo off
REM Start the MNQ dashboard: update, check dependencies, run.
REM
REM Double-click this file, or run `start.bat` from a Command Prompt.
REM To EDIT it: right-click -> Edit (double-clicking runs it instead).
REM
REM This file must keep Windows (CRLF) line endings. cmd.exe seeks through a
REM batch file by byte offset, and with Unix line endings that arithmetic
REM drifts until it lands mid-word and tries to run a fragment of a comment.
REM .gitattributes pins it; do not "fix" the line endings.

setlocal
title MNQ Dashboard

REM ==========================================================================
REM  SETTINGS - delete the word REM at the start of a "set" line to switch it
REM  on, then save. All optional; the dashboard runs fine with all of it off.
REM ==========================================================================

REM --- Colour the page green/red on every call, however weak -----------------
REM  By default the page only commits when the projected move reaches 10 pct
REM  of a typical move, so it shows NO CALL most of the time. That is honest,
REM  but if you want to see the colour react constantly, set this to 0.
REM set MNQ_SIGNAL_MIN_RATIO=0

REM --- How far ahead the green/red call looks, in minutes --------------------
REM  Must be one of the chart timeframes: 1, 5, 10, 15, 30, 60, 240.
REM set MNQ_SIGNAL_HORIZON_MINUTES=10

REM --- Simulated trading ----------------------------------------------------
REM  On by default. Trades are logged to mnq\data\trades.jsonl and drawn on
REM  the chart as arrows. Nothing is ever sent to a broker. Set 0 for off.
REM set MNQ_PAPER_TRADING=0
REM  Round-turn cost per simulated trade, in points (0.75 is about $1.50).
REM set MNQ_PAPER_COST_POINTS=0.75

REM --- Offline demo mode ----------------------------------------------------
REM  Runs on generated prices instead of Yahoo. Useful when the market is
REM  closed, or to watch the trade markers appear without waiting on the tape.
REM set MNQ_FEED=synthetic

REM --- Serve on a different port --------------------------------------------
REM set MNQ_PORT=8766

REM ==========================================================================

REM %~dp0 is this script folder (...\estimator\mnq\), so ".." is the repo root.
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

REM --- Is the port already taken? -------------------------------------------
REM Almost always this same dashboard still running in another window. Binding
REM would fail with WinError 10048, so catch it here where we can explain it.
set "PORT=8765"
if defined MNQ_PORT set "PORT=%MNQ_PORT%"
set "BUSY="
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /r /c:":%PORT% .*LISTENING"') do set "BUSY=%%P"
if defined BUSY (
    echo.
    echo   Port %PORT% is already in use by process %BUSY%.
    echo   That is almost always this dashboard still open in another window.
    echo.
    choice /c YN /n /m "  Stop it and start fresh? [Y/N] "
    if errorlevel 2 (
        echo.
        echo   Left it running. Close the other window, or pick another port
        echo   by uncommenting the MNQ_PORT line at the top of this file.
        echo.
        pause
        exit /b 1
    )
    taskkill /pid %BUSY% /f >nul 2>&1
    echo   Stopped process %BUSY%.
)

echo.
echo === Starting dashboard - open http://127.0.0.1:%PORT% ===
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

@echo off
REM Start the MNQ dashboard: update, check dependencies, run.
REM
REM Double-click this file, or run `start.bat` from a Command Prompt.
REM It handles the directory juggling: `git pull` needs the repository root
REM while `run.py` needs the mnq folder, which is the usual thing to get wrong.

setlocal
title MNQ Dashboard

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

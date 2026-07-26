@echo off
REM ===================================================================
REM  MNQ Trading System - one-click setup and first run (Windows)
REM
REM  Just double-click this file. It will:
REM    1. check Python is installed
REM    2. create a private Python environment in this folder
REM    3. install the required packages (a few minutes, one time only)
REM    4. download real MNQ price data from Yahoo Finance
REM    5. train the models and print the result
REM
REM  Safe to run again later - it skips steps that are already done.
REM ===================================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   MNQ Trading System - setup
echo ============================================================
echo.

REM ---- Step 1: find Python -------------------------------------------
set PYCMD=
py --version >nul 2>&1
if %errorlevel%==0 set PYCMD=py
if "%PYCMD%"=="" (
    python --version >nul 2>&1
    if %errorlevel%==0 set PYCMD=python
)

if "%PYCMD%"=="" (
    echo [X] Python is not installed, or it was installed without
    echo     "Add python.exe to PATH" ticked.
    echo.
    echo     Fix: install Python from https://www.python.org/downloads/
    echo     and TICK the box "Add python.exe to PATH" on the first screen.
    echo     Then run this file again.
    echo.
    pause
    exit /b 1
)

echo [1/4] Found Python:
%PYCMD% --version
echo.

REM ---- Step 2: create the private environment ------------------------
if exist ".venv\Scripts\python.exe" (
    echo [2/4] Environment already exists - skipping.
) else (
    echo [2/4] Creating the Python environment...
    %PYCMD% -m venv .venv
    if errorlevel 1 (
        echo [X] Could not create the environment.
        pause
        exit /b 1
    )
)
echo.

REM Use the environment's Python directly. This deliberately avoids
REM "activating" it, which PowerShell often blocks on a default install.
set VPY=.venv\Scripts\python.exe

REM ---- Step 3: install packages --------------------------------------
echo [3/4] Installing required packages. First run takes a few minutes...
echo.
%VPY% -m pip install --upgrade pip --quiet
%VPY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [X] Package installation failed. Check your internet connection
    echo     and run this file again.
    pause
    exit /b 1
)
echo.
echo     Packages installed.
echo.

REM ---- Step 4: fetch data and train ----------------------------------
echo [4/4] Downloading MNQ price data from Yahoo Finance...
echo.
%VPY% -m mnq.cli fetch
if errorlevel 1 (
    echo.
    echo [X] Could not download data. This is usually a network or
    echo     firewall problem. Check your connection and try again.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Training the models. This takes several minutes.
echo   Leave this window open.
echo ============================================================
echo.
%VPY% -m mnq.cli train

echo.
echo ============================================================
echo   DONE
echo.
echo   Scroll up and find the two lines that look like:
echo.
echo       long: OOS AUC 0.5xxx ...
echo      short: OOS AUC 0.4xxx ...
echo.
echo   That number is the answer to "does this predict MNQ?"
echo     around 0.50 = predicts nothing, do not trade it
echo     0.55 or more = there is something real, worth continuing
echo.
echo   Copy those two lines and send them to Claude.
echo ============================================================
echo.
pause

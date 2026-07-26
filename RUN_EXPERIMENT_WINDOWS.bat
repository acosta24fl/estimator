@echo off
REM ===================================================================
REM  MNQ - THE DECISIVE EXPERIMENT
REM
REM  Your first run tested one thing: 60 days of 5-minute bars using
REM  only MNQ's own price. It found nothing (AUC 0.50).
REM
REM  That test had two weaknesses, and this script fixes both:
REM
REM    1. 60 days is ONE market regime. Your base rates (long 27%%,
REM       short 36%%) show it was a falling market. Nothing validated
REM       on one regime generalises. This adds 730 days of hourly
REM       bars, which spans many regimes.
REM
REM    2. EMAs and RSI on MNQ alone are the most heavily arbitraged
REM       numbers in markets. This adds cross-asset context - bonds,
REM       the dollar, credit, semiconductors, volatility - all free
REM       from the same Yahoo source, no new accounts or API keys.
REM
REM  It runs FOUR configurations and prints one comparison table, so
REM  the timeframe change and the data change cannot be confused with
REM  each other.
REM
REM  Takes 20-60 minutes. Leave it running.
REM ===================================================================

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [X] Setup has not been run yet.
    echo     Double-click RUN_ME_WINDOWS.bat first, then run this file.
    echo.
    pause
    exit /b 1
)

set VPY=.venv\Scripts\python.exe

echo.
echo ============================================================
echo   Downloading data
echo.
echo   MNQ hourly history plus the cross-asset basket. The first
echo   run pulls a lot; later runs reuse the cache and are faster.
echo ============================================================
echo.

%VPY% -m mnq.cli fetch --profile wide
if errorlevel 1 (
    echo.
    echo [X] Could not download price data. Check your connection.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Running four configurations. This is the slow part.
echo   Leave this window open.
echo ============================================================
echo.

%VPY% -m mnq.cli experiment

echo.
echo ============================================================
echo   DONE
echo.
echo   Scroll up to the RESULTS table and read the VERDICT.
echo.
echo   The number to look at is AUC:
echo     0.50        = predicts nothing
echo     0.52 - 0.55 = marginal, probably eaten by costs
echo     0.55+       = worth pursuing
echo.
echo   Copy the whole RESULTS table and send it to Claude.
echo ============================================================
echo.
pause

@echo off
REM ===================================================================
REM  MNQ - DOES THE EDGE SURVIVE COSTS?
REM
REM  The experiment found a real signal in the winning configuration
REM  (1h/730d + cross-asset): AUC 0.560 long / 0.563 short, with all
REM  five short folds above 0.55.
REM
REM  AUC is not money. A model can be genuinely predictive and still
REM  lose after spread, slippage and commission. This script answers
REM  the next question, in three steps:
REM
REM    1. Train the winning configuration and save the models
REM    2. Backtest it with realistic costs
REM    3. Search entry thresholds, scored on the WEAKER half of the
REM       sample so a curve fit cannot win
REM
REM  Step 3 is legitimate here only because step 1 measured a real
REM  edge first. Threshold-hunting on a coin flip is how people fool
REM  themselves; this is not that.
REM
REM  Takes 30-90 minutes.
REM ===================================================================

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [X] Setup has not been run. Double-click RUN_ME_WINDOWS.bat first.
    pause
    exit /b 1
)

set VPY=.venv\Scripts\python.exe

echo.
echo ============================================================
echo   STEP 1 of 3 - training the winning configuration
echo ============================================================
echo.
%VPY% -m mnq.cli train --profile wide
if errorlevel 1 (
    echo [X] Training failed. Send the error above to Claude.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   STEP 2 of 3 - backtest with realistic costs
echo ============================================================
echo.
%VPY% -m mnq.cli backtest --profile wide

echo.
echo ============================================================
echo   STEP 3 of 3 - threshold search, validated on both halves
echo ============================================================
echo.
%VPY% -m mnq.cli sweep --profile wide --min-trades 15

echo.
echo ============================================================
echo   DONE - what to look for
echo.
echo   STEP 2: "Net P and L" and "Profit factor".
echo           Profit factor above 1.0 means it made money.
echo.
echo   STEP 3: the columns is_net_usd and oos_net_usd.
echo           BOTH must be positive. A row that is positive in
echo           one half and negative in the other is a curve fit,
echo           no matter how good its total looks.
echo.
echo   Copy the STEP 2 report and the top few STEP 3 rows,
echo   and send them to Claude.
echo ============================================================
echo.
pause
